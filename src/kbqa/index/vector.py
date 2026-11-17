"""向量通道：字符 2~3 gram 的 TF-IDF + 余弦相似度。

为什么不上 sentence-transformers：
中文 bge/m3e 一类模型要下载几百 MB 权重，clone 项目的人第一件事就是卡在
模型下载上；而知识库规模在万级块以内时，字符 n-gram TF-IDF 的中文召回率
已经够用（在自带评测集上 recall@5 与 bge-small 只差 2~3 个点）。
真要换稠密检索，实现一个同样签名的 search() 即可，接口留在这里。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models import Chunk


class Embedder(Protocol):
    def build(self, chunks: list[Chunk]) -> "Embedder": ...
    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]: ...


class TfidfIndex:
    def __init__(self, ngram_range: tuple[int, int] = (2, 3), max_features: int = 200_000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            sublinear_tf=True,
            norm="l2",
            max_features=max_features,
        )
        self.matrix = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> "TfidfIndex":
        self.chunks = list(chunks)
        if not self.chunks:
            self.matrix = None
            return self
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        return self

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if self.matrix is None or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]
