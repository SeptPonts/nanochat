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
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of kv heads (MQA)
    n_embd: int = 768

def norm(x):
    # Purely functional rmsnorm with no learnable params
    # normalized_shape: 匹配 x 的最后若干维的形状，例如 (C,) 或 (H, W)。
    # 假设 x 的 dimention 是 [B, T, C]，则当前的写法指的是在最后一维 C 上做归一化
    # 直观想象一下，对于 [B, T, C] 就相当于 (b, t) 在一个平面上“索引” 长度为 c 的向量；做归一化就是
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3] // 2 # 向下取整
    # 在此情景下维度的典型配置是 [B, T, n_head, head_dim]
    # B - batch 一次喂给模型多少个序列 (32, 64, 128, ...)
    # T - time/sequence 每个序列多少 token (512, 1024, 2048, ...)
    # n_head - MHA 多少个头 (12, 16, 32, ...)
    # head_dim - 每个头的向量维度 (64, 128, ...)
    # 沿着最后一个 dimension 一分为二做旋转（联想一下 2d 平面上向量的旋转）
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
    在 MQA 里，kv 头会共享。根据上面的配置 n_head = 12, n_kv_head = 2
    需要把 2 组 K/V 复制成 12 组，让每组 K/V 被多个 Q 共享
    """
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    # 先扩展 expand 一个维度，再 reshape 而不是直接 repeat 的原因是我们对复制出来的头的排列有要求
    # 我们要 head_0, head_0_copy1, head_0_copy2, head_1, head_1_copy1, head_1_copy2 这样的排列，而不是交错的
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class CausalSelfAttention(nn.Module):
    pass

class MLP(nn.Module):
    pass

class Block(nn.Module):
    pass

class GPT(nn.Module):
    pass
