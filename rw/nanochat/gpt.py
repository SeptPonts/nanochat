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

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from adamw import DistAdamW
from common import get_dist_info
from engine import KVCache
from muon import DistMuon, Muon


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

# MLP - multi-layer perceptron
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
    def __init__(self, config: GPTConfig):
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
        # Rotary embedding 很小 (seq_len * head_dim * 0.5 个 float), 所以与其每次生成时动态扩展, 不如一次性分配 10 倍长度(要是真的超了就让程序崩了吧)
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        # 因为 Pytorch 的 meta device 允许初始化模型但不实际分配内存(用于大模型分布式加载), 我们在这用的是"假的" cos/sin, 后面在 init_weights() 方法中才做真实计算(因为到那会才会有真实数据)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # persistent=False 意味着此 buffer不会和 model parameters 一样被保存到 checkpoint
        # 唯一的判断标准是"这个 buffer 能否从模型的其他部分廉价且确定性地重新计算?如果答案是 yes, 那就不要保存。要保存的一个例子是 BatchNorm's running mean, 这个值是训练过程中积累的统计量, 不能重算
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # wte - word token embedding: embedding layer (size: (vocab_size, embd_size))
        # 从 fp32 转 bf16 ➡️ 节约内存(model 和 activations)
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            # 具体可以看 notion, 文章要求对权重初始化和梯度更新学习率都根据 fanin fanout 做缩放
            # 这样才能保证 hideen size 极大的模型的特征学习能力
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    # 函数生成了旋转角度速查表, 在真正需要 cos/sin 时直接从表里取数
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        # head_dim = 64 向量 会被拆成 32 对 2D 向量(每对做独立旋转), 每一对都有不同的旋转频率
        # 0th pair 频率最高, 旋转快, 编码短期位置差; 31th pair 频率最低, 旋转慢, 编码长期位置差
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        # [0, 1, ..., seq_len-1] 代表每个 token 位置
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the roation frequencies at each (time, channel) pair
        # outer 外积 freqs 是个矩阵, 包含了时间 * 频率的所有组合: 每个元素的含义是第 pos 个 token 在第 channel_pair 维度上的旋转角度(弧度)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        # 加维度 broadcasting 的目的是:
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        ) # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        n_layers, n_heads, head_dim, seq_len = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        num_flops_per_token = (
            6 * (nparams - nparams_embedding) + 12 * n_layers * n_heads * head_dim * seq_len
        )
        return num_flops_per_token

    # 注意到不同层使用的 lr (learning rate) 是不一样的
    # unembedding layer 指的是将最后的隐藏向量映射回词汇表空间的层 (lm_head 线性层)
    # embedding layer (wte) 是将 token id 映射为向量表示的层
    # matrix layer 包括所有的 Transformer blocks (mlp/ffn + attention)
    # weight_decay 用于 AdamW optimizer
    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        # get 分布式训练参数
        ddp, rank, _local_rank, _world_size = get_dist_info()
        # 把所有参数分进 3 groups 里 (matrix, embedding, lm_head)
        # h - hidden layers, 去看 config 的话不难看出每个 layer 都是 Block
        # 而 Block 由 CausalSelfAttention 和 MLP 构成
        matrix_params = list[nn.Parameter](self.transformer.h.parameters())
        # wte - Word Token Emebdding (nn.Embedding 层)
        # 用于将 token id 转为向量
        embedding_params = list[nn.Parameter](self.transformer.wte.parameters())
        # MHA - multi-head attention
        lm_head_params = list[nn.Parameter](self.lm_head.parameters())
        assert len(list[nn.Parameter](self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # 创建 AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model) - 这里 scale 的依据是 init weights 那里引用的论文 (初始权重和learning rate都需要按照 layer 的 fanin fanout 做 scaling 来确保网络能有效学习)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            # 对全局第一个进程打印此消息
            print(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
        adam_groups = (
            {"params": lm_head_params, "lr": unembedding_lr * dmodel_lr_scale},
            {"params": embedding_params, "lr": embedding_lr * dmodel_lr_scale},
        )
        adamw_kwargs = {
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "weight_decay": weight_decay,
        }
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = {"lr": matrix_lr, "momentum": 0.95}
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, 
                idx : torch.Tensor, 
                targets : torch.Tensor = None, 
                kv_cache : KVCache = None, 
                loss_reduction="mean"):
        _B, T = idx.size()
        
        # 为当前 sequence length 获取 rotary embeddings (shape: (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # 如果 kv cache 存在, 我们需要把 rotary embeddings offset 到当前 cache 的位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        ) # 把 cache 截断到 seq length T
        
        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        
        # Forward the lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)
        # 处理之后将 logits 限定到 [-softcap, softcap] 之间
        # 作用: 1. 防止极端值 2. 训练更稳定 3. 避免数值溢出
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            # traning mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference mode: compute and return the logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive straming infefrence.
        To make it super simple, lets assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            # naive 方案, 每次都 forward 计算了所有 tokens
            logits = self.forward(ids) # (B, T, vocab_size)
            # 只取最新token的logits
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                # 选 top_k 大的 logits 进入下一筛选阶段
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                # temperature 越大越有创造性 
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                # 贪心选择: 直接选概率最大的
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
