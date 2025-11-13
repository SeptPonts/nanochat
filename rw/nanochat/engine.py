from contextlib import contextmanager
import torch
import torch.nn.functional as F

import signal
import warnings
from collections import deque

from .gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")
    
    # 等效于不同 context manager，但定义 __enter__ 方法（准备资源）
    # 注册 signal handler ➡️ 当收到 SIGALRM 信号时，调用 timeout_handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # 在 duration 秒之后，向当前进程发送 SIGALRM 信号
    signal.alarm(duration)
    # 暂停执行，让出控制权给 with block 里的逻辑（使用资源）
    # 控制权 = 程序执行的主导权
    yield
    # 等效于不用 context manager，但定义 __exit__ 方法（清理资源）
    # 清理资源的逻辑即使 with block 抛出异常也会执行（先执行清理资源，再捕获异常）
    # 取消之前设置的定时器（参数 0 就是这个含义）
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        # 带超时保护的 eval
        with timeout(max_time, formula):
            # 创建一个临时警告环境，with block 内的警告只在 block 内生效
            with warnings.catch_warnings():
                # 对于 syntax warning 不显示（不打印到控制台）
                warnings.simplefilter("ignore", SyntaxWarning)
                # eval 过程中其他的 warning 会显示
                return eval(formula)
    except Exception:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """Evaluate a math expression safely."""
    expr = expr.replace(",", "")
    if any(x not in "0123456789*+-/.() " for x in expr):  # disallow non-numeric chars
        return None
    if "**" in expr:  # for now disallow power operator, could be very expensive
        return None
    return eval_with_timeout(expr)


class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # 每个 K/V 的 shape 都是 (B, H, T, D), 典型值可能是 (1, 6, 2048, 128)
        # 回忆一下 hidden layer dimension：n_embd = n_heads * head_dim，每个头独立处理（并行）最后再拼接回来
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
        """
        Prefill KV cache. 可选沿着 batch dim 做 expand. why?
        这是个常见的优化场景：
        1. 先用 batch 1 prefill (处理输入的 prompt) 生成 kv cache
        2. 复制/扩展这个 kv cache 到 batch_size = N
        3. 并行生成 N 个 samples (避免每个样本都对同样的 prompt 做重复处理)
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        # zip - 给 self.kv_shape 和 other.kv_shape 配上对，每个维度配成一组
        # 按维度遍历
        for ix, (dim1, dim2) in enumerate(
            zip(self.kv_shape, other.kv_shape, strict=True)
        ):
            # 检查第 0 1 3 5 维，要求 dim match
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size (疑似 typo 应为 2 K/V), num_heads, head_dim must match
                assert dim1 == dim2, f"Batch dim mismatch: {dim1} != {dim2}"
            # 检查第 2 维也就是 batch size
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, (
                    f"Batch dim mismatch: {dim1} != {dim2}"
                )
            # 检查第 4 维也就是 seq length
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        # a tensor with uninitialized data
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, : other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        _B, _H, T_add, _D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            # 虽然增长到 t1 长度已经够了，但是额外留出 1024 个位置 ➡️ 避免频繁扩展，类似 python list 的预分配策略
            t_needed = t1 + 1024
            # 向上对齐到 1024 的倍数：内存对齐优化（gpu/cpu 访存高效） + 减少碎片（固定的块大小）+简化管理（容量总是 1024 的倍数，便于调试和理解）
            # 一些对齐的例子：100 ➡️ 1024，1024 ➡️ 2048，1025 ➡️ 2048，5000 ➡️ 6144
            t_needed = (
                t_needed + 1023
            ) & ~1023
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)
        # Insert k, v into the cache (尾部的 head_dim 省略掉，pytorch 的行为是自动处理：全选)
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        # pytorch 中基本索引操作默认返回 view，要返回 copy 得用高级索引(.clone(), 表达式, ...)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes
        # 翻译一下：只在最后一层才更新 pos，因为 Transformer 每层都会调用 insert_kv
        # 所以更新的 pattern 是所有层都处理同一位置的 token，全部完成后才推进 pos
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    从 logits 中采样下一个 token. (B, vocab_size) ➡️ (B, 1)
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice) # 映射回没经过 topk 选取的原始 index
    else:
        # temperature 越小会把 logits 的分布形状拉得越陡峭，进而让 sample 出的结果确定性越高
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------
class RowState:
    # 生成过程中 Per-row 状态跟踪
    def __init__(self, current_tokens=None):
        self.current_tokens = (
            current_tokens or []
        ) # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # 是否在 python block 中
        self.python_expr_tokens = [] # 当前 python 表达式的 tokens
        self.completed = False # 是否 this row 已经完成了生成

class Engine:
    def __init__(self, model: GPT, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
    
    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42  
    ):
        """Same as generate, but does single prefill and then clones the KV cahce."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), (
            "expecting list of ints"
        )
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        
        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row
        
        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        # tokens 是 1d list，但是模型期望输入是 (batch_size, seq_len)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        # 只保留最后一个位置的 logits，前面位置的logits只有训练时的损失计算才 care
        # logits 的 shape 是 (batch_size, num_tokens, vocab_size)
        logits = logits[:, -1, :]
        # 处理之后是 (batch_size, vocab_size)
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        # next_ids.shape = (batch_size, 1)，取 [:, 0] 后变成 (batch_size,)
        # .tolist() 后得到 [token1, token2, ...] 而不是 [[token1], [token2], ...]
        sampled_tokens = next_ids[:, 0].tolist()
        
        # 2) Replace the KV cache for each sample/row
        # 完成 prefill 准备 decode
        kv_length_hint = (
            # len(tokens) - prompt, max_tokens - 要生成的最大 token 数
            (len(tokens) + max_tokens) 
            if max_tokens is not None
            else self.model.config.sequence_len
        )
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        # 复制 prefill 阶段的 kvcache 到新的 cache
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # 不再需要了
        
        # 3) Initialize states for each sample
        # 多样本并行生成 ➡️ 本质上是生成 n 个不同的回复
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        
        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max_tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break
            
            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                sampled_tokens = [
                    sampled_tokens[0]
                ] * num_samples # Broadcast first token to all rows
                # TODO: we should sample a token for each row instead of broadcasting
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self.model.forward(
                    ids, kv_cache=kv_cache_decode
                ) # (B, T, vocab_size)
                logits = logits[:, -1, :] # (B, vocab_size) at last time step
                next_ids = sample_next_token(logits, rng, temperature, top_k) # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()
            
            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = (
                    len(state.forced_tokens) > 0
                ) # are there tokens waiting to be forced in deque?
                token_masks.append(
                    0 if is_forced else 1
                ) # mask is 0 if forced, 1 if sampled
                next_token = (
                    state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                )
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
            
            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next generation
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
    
    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of list of ints).
        Terminal tokens (assisntant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, _token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(
                zip(token_column, _token_masks, strict=True)
            ):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    from nanochat.checkpoint_manager import load_model
    from nanochat.common import compute_init

    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hperparameters
    kwargs = {"max_tokens": 64, "temperature": 0.0}
    # set the starting prompt
    prompt_tokens = tokenizer.encode(
        "The chemical formula of water is", prepend=bos_token_id
    )
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(
        prompt_tokens, num_samples=1, **kwargs
    ) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, _token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")