import torch


class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # 每个 K/V 的 shape 都是 (B, H, T, D), 典型值可能是 (1, 6, 2048, 128)
        # 对于 Transformer 的每一层, 我们都需要一对 K/V - 所以 size(0) == num_layers, size(1) == 2
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        # current position in time in the cache
        # pos 是一个指针, 追踪当前已经缓存了多少 token: 每个 token 的生成时都会向 kv cache 中保存 kv
        # pos 存在的意义是避免重复计算,
        self.pos = 0 
    
    def reset(self):
        self.pos = 0
    
    def get_pos(self):
        return self.pos

    def prefill(self, other):
        pass

    def insert_kv(self, layer_idx, k, v):
        pass
    
    @torch.inference_mode()
    def sample_next_token(logits, rng, temperature=1.0, top_k=None):
        pass