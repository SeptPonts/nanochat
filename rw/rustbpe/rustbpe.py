from __future__ import annotations

import heapq
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import regex as re

# ------------------------ constants & type aliases ------------------------

# Default GPT-4 style regex pattern for splitting text (identical to Rust)
GPT4_PATTERN: str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

Pair = tuple[int, int]

# ------------------------ internal helpers ------------------------


class Word:
    """
    A single word/chunk, 表示为一个 token IDs 组成的列表 (bytes as ints)

    在 rust 原版实现中用 Vec<u32> 存储 ids
    """

    def __init__(self, ids: list[int]):
        self.ids: list[int] = ids

    def pairs(self) -> Iterator[Pair]:
        """
        相邻 token id 成对输出(简单的滑动窗口)

        BPE (Byte Pair Encoding) 算法的核心逻辑:
        1. 找出文本中最频繁的相邻 token 对
        2. 把这对合并成一个新的 token
        3. 重复直到词汇表大小

        注意我们不需要有预先设定的词表, BPE 的理念是从GB级的数据中学习(学习合并规则表), 这个过程就是所谓的 tokenizer training; 对应的, tokenizer inference 指的是应用这个规则表到新文本

        前者可能跑几个小时, 后者就是毫秒级的。所以 tokenizer 的 training 实现必须高效

        假如我们要编码这段文本 "low low low lower", 先把文本拆成最小单位(字符)
        词 1: [l, o, w]        # "low"
        词 2: [l, o, w]        # "low"
        词 3: [l, o, w]        # "low"
        词 4: [l, o, w, e, r]  # "lower"

        对每个词调用 pairs() 函数
        词 1.pairs() → (l,o), (o,w)
        词 2.pairs() → (l,o), (o,w)
        词 3.pairs() → (l,o), (o,w)
        词 4.pairs() → (l,o), (o,w), (w,e), (e,r)

        统计频率:
        (l,o): 4 次  ← 最高频!
        (o,w): 4 次  ← 并列最高
        (w,e): 1 次
        (e,r): 1 次

        假设我们选择合并 (l,o) → 创建新 token lo

        循环这个过程, 直到词表的大小达到目标: 在上面的例子中, 可能 "low" 被编码成了词表中的一个 token; lower 则是两个 token 的组合
        """
        for i in range(len(self.ids) - 1):
            yield (self.ids[i], self.ids[i + 1])

    def merge_pair(self, pair: Pair, new_id: int) -> list[tuple[Pair, int]]:
        """
        这个函数用于词表更新时, 更新 word 的 token ids 表示

        比如初始状态是 self.ids = [5, 3, 7, 3, 2], 现在词表更新 (3,7) 会被合并为一个新的 token 100. 此时调用 merge_pair(pair=(3, 7), new_id=100)

        过程:
        位置:  0  1  2  3  4
        初始: [5, 3, 7, 3, 2]
                  ^^^^
               找到 (3,7)

        结果: [5, 100, 3, 2]

        合并前的 pairs: (5,3), (3,7), (7,3), (3,2)
        合并后的 pairs: (5,100), (100, 3), (3,2)
        生成 deltas list 作为返回值, 供 bpe 进行全局统计, 继续合并最高频率的 pair
        1. ((5, 3), -1)      # 左边的 (5,3) 被破坏
        2. ((5, 100), +1)    # 新出现 (5,100)
        3. ((3, 7), -1)      # 被合并的 pair 本身消失
        4. ((100, 3), +1)
        5. ((7, 3), -1)      # 右边的 (7,3) 被破坏
        显然, (3,2)不受影响
        """
        a, b = pair
        n = len(self.ids)
        if n < 2:
            return []

        out: list[int] = []
        deltas: list[tuple[Pair, int]] = []

        i = 0
        while i < n:
            if i + 1 < n and self.ids[i] == a and self.ids[i + 1] == b:
                # 准备合并, 先看会被影响的 neighbours
                left = out[-1] if out else None
                right = self.ids[i + 2] if i + 2 < n else None

                # 移除 (a,b) 旁边的 old pairs; 然后添加新的 pairs
                if left is not None:
                    deltas.append(((left, a), -1))
                    deltas.append(((left, new_id), 1))
                deltas.append(((a, b), -1))
                if right is not None:
                    deltas.append(((b, right), -1))
                    deltas.append(((new_id, right), 1))

                # 写入结果
                out.append(new_id)
                # 这个是 non-overlapping 的精髓
                # 比如有 [5,6,6,6,2], 合并(6,6), 如果不 + 2 会导致合并两次(6,6)
                i += 2
            else:
                # 最后一个元素, 没有 pair; 或者不是要更新的 pair 中的 token
                # ➡️ 直接从 ids copy to out, 无需更新
                out.append(self.ids[i])
                i += 1

        self.ids = out
        return deltas


# 自动生成 __eq__ 方法, 允许 object 创建后修改字段
@dataclass(eq=True, frozen=False)
class MergeJob:
    """
    Heap item describing a candidate pair merge job

    排序规则:
    - 按频率排序, 频序高的优先 Max-heap by count
    - 频率相同时, 按 pair 值升序 Tie-break to ascending pair order
    BPE 训练时, 每轮要选频率最高的 pair 合并. 如果有多个 pair 频率相同, 需要一个确定性规则来打破平局, 避免每次运行结果不同.

    Rust 的 heap 是 max-heap, 但是 python 的 heapq 是 min-heap ➡️ 需要反转 trick

    pos: set[int]  # 这个字段存储哪些 word 包含这个 pair, pos 不影响排序
    """

    pair: Pair
    count: int
    pos: set[int]

    def __lt__(self, other: MergeJob) -> bool:  # heapq uses this
        if self.count != other.count:
            # Reverse: higher count considered "smaller" for min-heap, 这样 higher count 就能被排到更接近 heap top 的地方了
            return self.count > other.count
        # pair 值小的排的更靠 heap top
        return self.pair < other.pair


def count_pairs_sequential(
    words: list[Word], counts: list[int]
) -> tuple[dict[Pair, int], dict[Pair, set[int]]]:
    """
    words - 所有 unique word
    counts - 每个 word 在语料库中出现的次数
    返回值 1 - pair -> 全局频率
    返回值 2 - pair -> 含有此 pair 的 word 的索引
    """
    pair_counts: dict[Pair, int] = {}
    where_to_update: dict[Pair, set[int]] = {}

    # enumerate()为可迭代对象添加索引, 返回 (index, value) tuple
    # zip 用于将多个 iterables 缝在一起, 返回元组的迭代器
    # strict 在 zip 的 iterables 不等长时抛错 ValueError
    for i, (w, c) in enumerate(zip(words, counts, strict=True)):
        if c == 0 or len(w.ids) < 2:
            continue
        """
        对 word 的 ids 进行合并处理
        如 ids = [5,3,3,2]
        w.pairs 依次返回 [5,3], [3,3], [3,2]
        """
        for pair in w.pairs():
            pair_counts[pair] = pair_counts.get(pair, 0) + c
            s = where_to_update.get(pair)
            if s is None:
                s = set()
                where_to_update[pair] = s
            s.add(i)

    return pair_counts, where_to_update


class Tokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer
    """

    def __init__(self) -> None:
        # Maps paris of token IDs to their merged token ID
        self.merges: dict[Pair, int] = {}
        # Regex pattern used for text splitting (string form)
        self.pattern: str = ""
        # Compilied version
        self.compiled_pattern: re.Pattern | None = None

    def train_core_incremental(
        self, words: list[Word], counts: list[int], vocab_size: int
    ) -> None:
        """
        Core incremental BPE training logic
        给定列表 [(word1, count1), ..., (wordk, countk)]
        
        从调用方的逻辑可以看出，本方法传入的是对语料进行统计之后，得到的 unique words 及其对应的词频。根据词频进行 pair 的合并
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        logging.info("Starting BPE training: %d merges to compute", num_merges)
        self.merges.clear()
        
        # ---- initial Pair counting ----
        logging.info(
            "Computing initial pair counts from %d unique sequences", len(words)
        )
        # pair_counts ➡️ dict[Pair, int] ➡️ value 为 Pair 出现的频率 (全局 word 统计)
        # where_to_update ➡️ dict[Pair, set[int]] ➡️ value 为含有此 Pair 的 Word 的索引
        pair_counts, where_to_update = count_pairs_sequential(words, counts)
        
        # ---- Build heap ----
        logging.info("Building heap with %d unique pairs", len(pair_counts))
        heap: list[MergeJob] = []
        for pair, pos in where_to_update.items():
            c = pair_counts.get(pair, 0)
            if c > 0:
                heapq.heappush(heap, MergeJob(pair=pair, count=c, pos=set(pos)))
        # MergeJob 自定义了 __lt__ 方法，让 heapq 最小堆做出了最大堆的效果：count 越大越靠近 heap top，count 一样 pair 值越小越靠近 heap top
        
        # ---- Merge loop ----
        logging.info("Starting merge loop")
        merges_done = 0
        last_log_percent = 0
        
        while merges_done < num_merges:
            if not heap:
                break
            top = heapq.heappop(heap)
            
            # Lazy refresh: If outdated, refresh to current count and reinsert
            # 注意构成 heap 的 MergeJob 是在训练初始阶段构造的（来源于对语料的全面统计），随着训练开始 pairs 被 merge，有些 pair 的 count 会被影响（甚至消失）；我们出于效率考虑没有选择立即更新整个 heap 被影响的 pairs，而是碰到了再更新
            current = pair_counts.get(top.pair, 0)
            if top.count != current:
                top.count = current
                if top.count > 0:
                    heapq.heappush(heap, top)
                # If refreshed to 0, dont re-insert, but dont break either
                continue
            if top.count == 0:
                break
            
            # Record merge: assgin next token id
            new_id = 256 + merges_done
            self.merges[top.pair] = new_id

            # Merge this pair in all words where it occurs
            local_pos_updates: dict[Pair, set[int]] = {}
            for word_idx in top.pos:
                changes = words[word_idx].merge_pair(top.pair, new_id)
                # Update global pair counts based on this word's count
                c_word = counts[word_idx]
                for pair, delta in changes:
                    delta_total = delta * c_word
                    if delta_total != 0:
                        pair_counts[pair] = pair_counts.get(pair, 0) + delta_total
                        if delta > 0:
                            s = local_pos_updates.get(pair)
                            if s is None:
                                s = set()
                                local_pos_updates[pair] = s
                            s.add(word_idx)
            
            # Add the updated pair counts back to the heap
            for pair, pos in local_pos_updates.items():
                cnt = pair_counts.get(pair, 0)
                if cnt > 0:
                    heapq.heappush(heap, MergeJob(pair=pair, count=cnt, pos=pos))

            merges_done += 1

            # Log progress every 1%
            if num_merges > 0:
                current_percent = (merges_done * 100) // num_merges
                if current_percent > last_log_percent:
                    logging.info(
                        "Progress: %d%% (%d/%d merges) - Last merge: %s -> %d (frequency: %d)",
                        current_percent,
                        merges_done,
                        num_merges,
                        top.pair,
                        new_id,
                        top.count,
                    )
                    last_log_percent = current_percent

        logging.info("Finished training: %d merges completed", merges_done)


    # ---------------- public API ----------------
    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int,
        buffer_size: int = 8192,
        pattern: str | None = None,
    ) -> None:
        """
        Trarin BPE tokenizeer from a streaming iterator of strings.
        """
        pattern_str = pattern if pattern is not None else GPT4_PATTERN
        self.pattern = pattern_str

        try:
            self.compiled_pattern = re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        # Global chunk counts
        counts: dict[str, int] = {}

        # Temporary buffer filled from the input iterator
        buf: list[str] = []
        it = iter(iterator)

        logging.info(
            "Processing sequences from iterator (buffer_size: %d)", buffer_size
        )
        total_sequences = 0

        def refill() -> bool:
            """
            Refill `buf` from `it` up to `buffer_size`

            Returns True if the iterator is exhausted; False otherwise
            """
            buf.clear()
            while len(buf) < buffer_size:
                try:
                    buf.append(next(it))
                except StopIteration:
                    return True
            return False

        # Stream ingestion loop: refill buffer, then process sequentially
        while True:
            exhausted = refill()
            if not buf and exhausted:
                break

            total_sequences += len(buf)

            # Rust 版本中通过 release GIL 实现了并行，这里实现串行版本
            pattern_obj = self.compiled_pattern
            assert pattern_obj is not None
            local: dict[str, int] = {}
            for s in buf:
                for m in pattern_obj.finditer(s):
                    piece = m.group(0)
                    local[piece] = local.get(piece, 0) + 1
            
            # Merge local into global (single-threaded)
            for k, v in local.items():
                counts[k] = counts.get(k, 0) + v

            if exhausted:
                break
        
        logging.info("Processed %d sequences total, %d unique", total_sequences, len(counts))
        
        # Materialize words & counts (byte-level like Rust)
        words: list[Word] = []
        cvec: list[int] = []
        for chunk, c in counts.items():
            words.append(Word(list(chunk.encode("utf-8"))))
            cvec.append(c)
        
        self.train_core_incremental(words, cvec, vocab_size)

    def get_pattern(self) -> str:
        """
        Return 分割文本用的 regex pattern
        """
        return self.pattern

    def get_mergeable_ranks(self) -> list[tuple[bytes, int]]:
        """
        Return mergable ranks: token bytes -> token id (rank)
        """
        raise NotImplementedError("Not implemented")

    def encode(self, text: str) -> list[int]:
        """
        将 string encode 为一连串的 token IDs

        byte-level BPE encoding:
        1. 使用 compiled regex pattern 分割 text
        2. 将每个 chunk 转换为 UTF-8 bytes -> list of ints
        3. 重复合并最低 new_id 的 pair 直到没有可合并的 pair
        4. 将所有 resulting ids 跨 chunk 拼接在一起
        """
        assert self.compiled_pattern is not None, (
            "Tokenizer not trained: call train_from_iterator first"
        )
        all_ids: list[int] = []

        # 正则匹配将给定 text 分割成一系列的 chunk
        """
        第 1 次循环

        # finditer 找到第一个匹配
        m = <Match object>
        m.span() = (0, 1)  # 匹配位置：从索引 0 到 1
        m.group(0) = "I"   # 匹配到的内容

        chunk = "I"
        ids = list(chunk.encode("utf-8"))
        ids = list(b'I')
        ids = [73]  # 字符 'I' 的 ASCII 码

        此时 all_ids:
        all_ids = [73]

        第 2 次循环

        # finditer 找到第二个匹配
        m = <Match object>
        m.span() = (1, 3)  # 从索引 1 到 3
        m.group(0) = "'m"  # 匹配到缩写

        chunk = "'m"
        ids = list(chunk.encode("utf-8"))
        ids = list(b"'m")
        ids = [39, 109]  # 单引号 39, 字母 m 是 109

        此时 all_ids:
        all_ids = [73, 39, 109] # 假设没有成功合并
        """
        for m in self.compiled_pattern.finditer(text):
            chunk = m.group(0)  # group(0) 返回整个匹配到的字符串
            ids: list[int] = list(chunk.encode("utf-8"))

            # 贪心算法合并
            while len(ids) >= 2:
                best_idx: int | None = None
                best_new_id: int | None = None

                # 找到最佳 pair 来合并：拥有最小 merged token id 的 pair
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    new_id = self.merges.get(pair)
                    if new_id is None:
                        continue
                    if best_new_id is None or new_id < best_new_id:
                        best_idx = i
                        best_new_id = new_id

                # 如果找到适合 merge 的 pair
                if best_idx is not None and best_new_id is not None:
                    ids[best_idx] = best_new_id
                    # 移除下一个 element（被合并消失的那个？）
                    del ids[best_idx + 1]
                else:
                    break

            # 将 ids 拼接到后面
            all_ids.extend(ids)

        return all_ids
