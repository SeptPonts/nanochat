"""
Simplified GPT model (learning purpose only)
features:
    - rotary embeddings (and no positional embeddings)
    - QK norm
    - untied weights for token embedding and lm_head
    - relu^2 activation in MLP
    - norm after token embedding
    - no learnable params in rmsnorm
    - no bias in linear layers
    - Multi-Query Attention (MQA) support for more efficient inference
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of kv heads (MQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    # normalized_shape: 匹配 x 的最后若干维的形状, 例如 (C,) 或 (H, W).
    # 假设 x 的 dimention 是 [B, T, C], 则当前的写法指的是在最后一维 C 上做归一化
    # 直观想象一下, 对于 [B, T, C] 就相当于 (b, t) 在一个平面上"索引" 长度为 c 的向量; 做归一化就是
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2  # 向下取整
    # 在此情景下维度的典型配置是 [B, T, n_head, head_dim]
    # B - batch 一次喂给模型多少个序列 (32, 64, 128, ...)
    # T - time/sequence 每个序列多少 token (512, 1024, 2048, ...)
    # n_head - MHA 多少个头 (12, 16, 32, ...)
    # head_dim - 每个头的向量维度 (64, 128, ...)
    # 沿着最后一个 dimension 一分为二做旋转(联想一下 2d 平面上向量的旋转)
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    # 沿着最后一个 dimension 原样拼回去
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    """
    在 MQA 里, kv 头会共享. 根据上面的配置 n_head = 12, n_kv_head = 2
    需要把 2 组 K/V 复制成 12 组, 让每组 K/V 被多个 Q 共享
    """
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    # 先扩展 expand 一个维度, 再 reshape 而不是直接 repeat 的原因是我们对复制出来的头的排列有要求
    # 我们要 head_0, head_0_copy1, head_0_copy2, head_1, head_1_copy1, head_1_copy2 这样的排列, 而不是交错的
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx  # 每个 layer 都有个唯一的 index 来标识, 从 0 开始
        self.n_head = config.n_head  # number of query(Q) heads
        self.n_kv_head = config.n_kv_head  # number of kv heads (MQA)
        self.n_embd = (
            config.n_embd
        )  # model hidden dimension (每个 token 的向量表示维度)
        self.head_dim = (
            self.n_embd // self.n_head
        )  # 向下取整, 每个 head 关注 embd 不同的部分
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, _C = x.size()

        # Project the input to get queries, keys, and values
        # qkv 向量来自于对同一输入的而不同变换
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 对 q 和 k 做 RoPE, 编码相对位置信息进去
        cos, sin = cos_sin
        q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        # B - batch size, T - number of tokens, H - number of heads, D - head dimension, 其中 H * D = n_embd, 是 model hidden dimension 也是每个 token 的向量表示维度
        # batch 和 head 组成 batch * head 个独立任务并行计算, 有利于提高算法的执行效率(这是从 head 和 batch 的定义出发的, head 定义了 subspace, batch 本就是不同的 seq)

        # Apply KV cahce: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)
        # Tq 和 Tk 取的维度 2 对应 T 也就是本次处理的 token 数量

        # Apply MQA: replicate the key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        if kv_cache is None or Tq == Tk:
            # 训练或首次推理, 直接走 causal 逻辑
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # 推理, 一次生成一个 token
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # 推理, 一次生成多个 token
            # 手动构建 mask, 因为 prefix(历史)全可见, chunk(正在生成)走 causal
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True  # prefix 全可见
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )  # 用 1 填充下三角区
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Re-assemble the heads side by side and project back to residual stream
        # 转置 transpose 是为了把刚才为了加速计算倒置的 H 和 T 再倒回去
        # view 要求 contiguous, 也就是内存连续
        # view 的 -1 指的是维度 2 自动计算
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        # 这是个可学习的线性变换, 我们期待模型在数据中学习如何融合各个 heads 拼接后的信息
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ModuleDict 使得我们可以通过名字来管理多个模块, 比如 self.transformer[wte].
        # wte - Word Token Embedding, h - Hidden Layers
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        # lm_head - Language Model Head 语言模型输出层
        # 用于将 transfomer 输出的隐藏向量映射到词汇表空间, 用于预测下一个 token
        # 具体来说, lm_head 的输入是 heads 拼接的结果, 输出是 logits. 从维度的变化上来看, 是个 n_embd ➡️ vocab_size 的映射
        # 接下来经过 softmax 会被映射成概率分布, 用于从词表中选取 token(采样策略? 可能不会仅仅选取概率最高的)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def init_weights(self):
        pass

    def _init_weights(self, module):
        pass

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        pass

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        pass

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        pass

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        pass

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        pass
