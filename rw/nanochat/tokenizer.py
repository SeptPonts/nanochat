import os

from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    # 预训练阶段只用到 bos 就够了
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    # sft 阶段会使用所有的 special tokens: 训练数据是对话格式 (conversation), 需要转成 token ids ("一句话的形式")
    "<|user_start|>", # use messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# 注: \p{L} - 任何语言的字母 (Letter)
# \p{N} - 任何数字 (Number)
# \p{N}{1,2} - 1到2个连续数字
# 作者做的这个优化未经验证, 他的目的是词汇表更小, 为文本token留出更多空间; 缺点是长数字会被切得更碎 (123 会被切成 12 和 3 两个 tokens 来表达), 可能影响数字理解能力
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""
    
    def __init__(self, tokenizer : HFTokenizer):
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)
    
    @classmethod
    def from_directoy(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)
    
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(
            BPE(
                byte_fallback=True, # needed!
                unk_token=None,
                fuse_unk=False
            )
        )
        # Normalizer: None
        tokenizer.normalizer = None
        # pretokenizer: 用预定义好的 regex partern 在 BPE 之前把 text 分成 groups
        gpt4_split_regex = Regex(
            SPLIT_PATTERN
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                # 基于正则的文本分割, text ➡️ text groups
                pre_tokenizers.Split(
                    pattern=gpt4_split_regex, behavior="isolated", invert=False
                ), # isolated - 每个匹配单独处理; invert - 保留匹配到的内容 (invert 为 True 的话会丢弃匹配到的内容)
                # 字节级编码: text groups ➡️ unicode 表示
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False), # 文本开头不自动加空格; 禁用额外的正则处理 (第一阶段已完成分割)
            ]
        )
        # Decoder: Bytelevel (it pairs together with the Bytelevel pre-tokenizer) 编码/解码要配对
        tokenizer.decoder = decoders.ByteLevel()
        # portprocessor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequence
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens
    
    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
    
    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)        
            )
            ids.append(prepend_id)
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )
            ids.append(append_id)
        return ids
    
    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)
    
    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos
    
    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
        