from __future__ import annotations

from collections.abc import Iterator

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