"""BM25（Okapi）纯 Python 实现。

没有用 rank_bm25 是因为它不支持把「新增文档」增量加进去，
而知识库是要持续更新的；这里自己维护 df 统计，重建成本 O(n)。
"""

from __future__ import annotations

import math
from collections import Counter

from ..models import Chunk
from ..text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.tf: list[Counter] = []
        self.lens: list[int] = []
        self.df: Counter = Counter()
        self.avg_len: float = 0.0
        self.n: int = 0

    def build(self, chunks: list[Chunk]) -> "BM25":
        self.chunks = list(chunks)
        self.tf, self.lens = [], []
        self.df = Counter()
        for c in self.chunks:
            toks = tokenize(c.text)
            cnt = Counter(toks)
            self.tf.append(cnt)
            self.lens.append(len(toks))
            self.df.update(cnt.keys())
        self.n = len(self.chunks)
        self.avg_len = (sum(self.lens) / self.n) if self.n else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """返回 [(chunk 下标, 分数)]，按分数降序。"""
        qtoks = tokenize(query)
        if not qtoks or not self.chunks:
            return []
        scores: list[tuple[int, float]] = []
        for i, cnt in enumerate(self.tf):
            s = 0.0
            dl = self.lens[i] or 1
            norm = self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            for t in qtoks:
                f = cnt.get(t, 0)
                if not f:
                    continue
                s += self._idf(t) * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
