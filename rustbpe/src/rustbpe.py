"""
Educational, logic-equivalent Python port of rustbpe/src/lib.rs.

This file intentionally mirrors the design and logic of the Rust version
as closely as is idiomatic in Python for learning purposes. Where Python
differs from Rust (e.g., no rayon parallelism, different heap semantics),
we call that out explicitly in comments.

Key goals:
- Keep public API identical to the Rust Tokenizer: new (via __init__),
  train_from_iterator, get_pattern, get_mergeable_ranks, encode.
- Replicate the core incremental BPE training loop and merging logic.
- Split text using the same GPT-4 style regex pattern and operate on raw
  UTF-8 bytes (exactly like the Rust version).
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# The third-party "regex" library is used to match Rust's FancyRegex features
# (e.g., \p{L}, \p{N}, possessive quantifiers). pyproject.toml pins it.
import regex as re

# ------------------------ constants & type aliases ------------------------

# Default GPT-4 style regex pattern for splitting text (identical to Rust)
GPT4_PATTERN: str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

Pair = tuple[int, int]


# ------------------------ internal helpers ------------------------


class Word:
    """A single word/chunk represented as a list of token IDs (bytes as ints).

    This mirrors the Rust struct Word { ids: Vec<u32> }.
    We keep logic equivalent to the Rust version for clarity and learning.
    """

    def __init__(self, ids: list[int]):
        self.ids: list[int] = ids

    def pairs(self) -> Iterator[Pair]:
        # Equivalent to Rust's `self.ids.windows(2)` iterator.
        for i in range(len(self.ids) - 1):
            yield (self.ids[i], self.ids[i + 1])

    def merge_pair(self, pair: Pair, new_id: int) -> list[tuple[Pair, int]]:
        """Merge all non-overlapping occurrences of `pair` -> `new_id`.

        Returns a list of local pair-count deltas for THIS word only:
        - (old_pair, -1) for removed pairs
        - (new_pair, +1) for newly created pairs

        This mirrors the Rust implementation's approach and data shape.
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
                # Determine neighboring tokens around the merged pair
                left = out[-1] if out else None
                right = self.ids[i + 2] if i + 2 < n else None

                # Remove old pairs around (a,b); add new pairs involving new_id
                if left is not None:
                    deltas.append(((left, a), -1))
                    deltas.append(((left, new_id), 1))
                deltas.append(((a, b), -1))
                if right is not None:
                    deltas.append(((b, right), -1))
                    deltas.append(((new_id, right), 1))

                # Write merged token
                out.append(new_id)
                i += 2  # skip 'a' and 'b'
            else:
                out.append(self.ids[i])
                i += 1

        self.ids = out
        return deltas


@dataclass(eq=True, frozen=False)
class MergeJob:
    """Heap item describing a candidate pair merge job.

    - Ordering matches Rust's Ord impl:
      1) Max-heap by count
      2) Tie-break to ascending pair order (deterministic)

    Python's heapq is a min-heap on __lt__, so we define __lt__ accordingly.
    We do NOT include 'pos' in comparisons, matching Rust behavior where
    positions do not influence ordering when counts tie.
    """

    pair: Pair
    count: int
    pos: set[int]

    def __lt__(self, other: MergeJob) -> bool:  # heapq uses this
        if self.count != other.count:
            # Reverse: higher count considered "smaller" for min-heap
            return self.count > other.count
        # Ascending order on the pair when counts tie
        return self.pair < other.pair


def count_pairs_sequential(
    words: list[Word], counts: list[int]
) -> tuple[dict[Pair, int], dict[Pair, set[int]]]:
    """Compute initial pair counts and the word indices where pairs occur.

    NOTE: Unlike Rust (which uses rayon for parallelism), this Python version
    computes counts sequentially for simplicity and determinism. The logic and
    the outputs are equivalent; only the lack of parallelism differs.
    """
    pair_counts: dict[Pair, int] = {}
    where_to_update: dict[Pair, set[int]] = {}

    for i, (w, c) in enumerate(zip(words, counts)):
        if c == 0 or len(w.ids) < 2:
            continue
        for pair in w.pairs():
            pair_counts[pair] = pair_counts.get(pair, 0) + c
            s = where_to_update.get(pair)
            if s is None:
                s = set()
                where_to_update[pair] = s
            s.add(i)

    return pair_counts, where_to_update


# ------------------------ Tokenizer ------------------------


class Tokenizer:
    """A Byte Pair Encoding tokenizer matching the Rust implementation.

    Public API mirrors Rust:
    - __init__()  -> Rust's new()
    - train_from_iterator(iterator, vocab_size, buffer_size=8192, pattern=None)
    - get_pattern() -> str
    - get_mergeable_ranks() -> list[tuple[bytes, int]]
    - encode(text: str) -> list[int]

    Design notes for learners:
    - We split text with the GPT-4 style regex pattern and operate on raw
      UTF-8 bytes of those pieces, exactly like the Rust version.
    - Training uses an incremental BPE algorithm with a heap of candidate
      merges; each step merges the most frequent pair.
    - Python differences are explicitly annotated (e.g., no rayon, heapq).
    """

    def __init__(self) -> None:
        # Maps pairs of token IDs to their merged token ID (Rust: StdHashMap<Pair, u32>)
        self.merges: dict[Pair, int] = {}

        # Regex pattern used for text splitting (string form)
        self.pattern: str = ""

        # Compiled regex for efficiency (Rust caches a compiled FancyRegex)
        # Python: use the third-party 'regex' module to support \p{...} etc.
        self.compiled_pattern: re.Pattern | None = None

    # ---------------- internal core training (mirrors Rust) ----------------
    def train_core_incremental(
        self, words: list[Word], counts: list[int], vocab_size: int
    ) -> None:
        """Core incremental BPE training given unique words and their counts.

        This mirrors the Rust method Tokenizer::train_core_incremental(). The
        loop structure, heap behavior, updates of pair_counts, and logging are
        kept as close as possible to the original for learning.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        logging.info("Starting BPE training: %d merges to compute", num_merges)
        self.merges.clear()

        # ---- Initial pair_counts and where_to_update (sequential in Python) ----
        # Rust uses rayon to parallelize count_pairs; we do it sequentially.
        logging.info(
            "Computing initial pair counts from %d unique sequences", len(words)
        )
        pair_counts, where_to_update = count_pairs_sequential(words, counts)

        # ---- Build heap ----
        # Rust uses OctonaryHeap; we use Python's heapq with a custom comparator
        # in MergeJob. Behavior is equivalent: max-heap by count, tie-break by pair.
        logging.info("Building heap with %d unique pairs", len(pair_counts))
        heap: list[MergeJob] = []
        for pair, pos in where_to_update.items():
            c = pair_counts.get(pair, 0)
            if c > 0:
                heapq.heappush(heap, MergeJob(pair=pair, count=c, pos=set(pos)))

        # ---- Merge loop ----
        logging.info("Starting merge loop")
        merges_done = 0
        last_log_percent = 0

        while merges_done < num_merges:
            if not heap:
                break
            top = heapq.heappop(heap)

            # Lazy refresh: If outdated, refresh to current count and reinsert
            current = pair_counts.get(top.pair, 0)
            if top.count != current:
                top.count = current
                if top.count > 0:
                    heapq.heappush(heap, top)
                # If refreshed to 0, don't re-insert, but don't break either
                continue
            if top.count == 0:
                break

            # Record merge: assign next token id
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
        """Train BPE tokenizer from a streaming iterator of strings.

        Parameters mirror Rust's signature and behavior. Major differences:
        - Python version does not manage the GIL explicitly or parallelize the
          splitting/counting (Rust uses rayon + FancyRegex). This version is
          sequential but logically equivalent.
        - Errors from regex compilation are raised as Python exceptions.
        """
        # Use provided pattern or default to GPT-4 pattern
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
            """Refill `buf` from `it` up to `buffer_size`.

            Returns True if the iterator is exhausted; False otherwise.
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

            # In Rust this block runs without GIL and in parallel over buf.
            # Python: run sequentially but keep logic identical.
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

        logging.info(
            "Processed %d sequences total, %d unique", total_sequences, len(counts)
        )

        # Materialize words & counts (byte-level like Rust)
        words: list[Word] = []
        cvec: list[int] = []
        # For determinism, iterate in the insertion order of `counts`, which is
        # deterministic in Python 3.7+. Rust iterates over a HashMap then collects;
        # order does not affect correctness since we store (word, count) aligned.
        for chunk, c in counts.items():
            words.append(Word(list(chunk.encode("utf-8"))))
            cvec.append(c)

        self.train_core_incremental(words, cvec, vocab_size)

    def get_pattern(self) -> str:
        """Return the regex pattern string used for splitting text."""
        return self.pattern

    def get_mergeable_ranks(self) -> list[tuple[bytes, int]]:
        """Return the mergeable ranks: token bytes -> token id (rank).

        The build process mirrors the Rust version: we construct token byte
        sequences for IDs 0..255, then append merged tokens in order of their
        assigned token ID (ascending). This allows reconstructing the merged
        byte sequences progressively.
        """
        mergeable_ranks: list[tuple[bytes, int]] = []

        # Build vocabulary incrementally from low to high token IDs
        token_bytes: list[bytes] = [bytes([i]) for i in range(256)]

        for i, b in enumerate(token_bytes):
            mergeable_ranks.append((b, i))

        # Sort merges by token id (so we can reconstruct bytes progressively)
        # Rust stores merges in a map: Pair -> merged_id. We produce a list of
        # (pair, merged_id) sorted by merged_id, then compute byte sequences.
        sorted_merges = sorted(self.merges.items(), key=lambda kv: kv[1])

        for (left, right), merged_id in sorted_merges:
            # Ensure token_bytes has space for merged_id
            if len(token_bytes) <= merged_id:
                token_bytes.extend([b""] * (merged_id - len(token_bytes) + 1))

            merged_bytes = token_bytes[left] + token_bytes[right]
            token_bytes[merged_id] = merged_bytes
            mergeable_ranks.append((merged_bytes, merged_id))

        return mergeable_ranks

    def encode(self, text: str) -> list[int]:
        """Encode a string into token IDs by applying learned merges.

        This is a byte-level BPE encoding:
        1) Split text using the compiled regex pattern (same as Rust)
        2) Convert each chunk to UTF-8 bytes -> list of ints
        3) Repeatedly merge the best pair (lowest new_id) until no more applies
        4) Concatenate all resulting ids across chunks
        """
        assert self.compiled_pattern is not None, (
            "Tokenizer not trained: call train_from_iterator first"
        )
        all_ids: list[int] = []

        for m in self.compiled_pattern.finditer(text):
            chunk = m.group(0)
            ids: list[int] = list(chunk.encode("utf-8"))

            # Apply merges iteratively
            while len(ids) >= 2:
                best_idx: int | None = None
                best_new_id: int | None = None

                # Find the best pair to merge: choose the pair with the smallest merged token id
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    new_id = self.merges.get(pair)
                    if new_id is None:
                        continue
                    if best_new_id is None or new_id < best_new_id:
                        best_idx = i
                        best_new_id = new_id

                # If we found a pair to merge, apply it; otherwise, stop
                if best_idx is not None and best_new_id is not None:
                    ids[best_idx] = best_new_id
                    # remove the next element (ids[best_idx + 1])
                    del ids[best_idx + 1]
                else:
                    break

            all_ids.extend(ids)

        return all_ids
