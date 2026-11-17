"""混合检索：BM25（词法）+ TF-IDF 向量（语义近似），用 RRF 融合。

为什么用 RRF（Reciprocal Rank Fusion）而不是加权求和：
两路分数量纲完全不同（BM25 是无上界实数，余弦是 0~1），加权求和必须做归一化，
而归一化会被单条极高分带偏。RRF 只看名次，只用 1/(k+rank) 累加，不需要调权重，
Elasticsearch / Vespa 默认都是这个做法。

邻块扩展：一个小节被切成两块时，答案往往横跨边界，所以命中的块会把
上下相邻块按比例拉进候选池（neighbor_bonus = 继承父块分数的比例）。
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ..models import Chunk, Hit
from ..text import bigrams
from .bm25 import BM25
from .rerank import rerank
from .vector import TfidfIndex


class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()
        self.vector = TfidfIndex()
        self._by_doc: dict[str, list[int]] = {}
        self._idx: dict[str, int] = {}

    # ---------------- 构建 ----------------
    def build(self, chunks: list[Chunk]) -> "HybridRetriever":
        self.chunks = list(chunks)
        self.bm25.build(self.chunks)
        self.vector.build(self.chunks)
        self._by_doc = {}
        for i, c in enumerate(self.chunks):
            self._by_doc.setdefault(c.doc_id, []).append(i)
        self._idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        return self

    # ---------------- 检索 ----------------
    def _fuse(self, q: str) -> tuple[list[int], dict[int, float], dict[int, str]]:
        rrf_k = int(self.cfg.get("retrieval.rrf_k", 60))
        bm_hits = self.bm25.search(q, int(self.cfg.get("retrieval.bm25_k", 20)))
        vc_hits = self.vector.search(q, int(self.cfg.get("retrieval.vector_k", 20)))

        fused: dict[int, float] = {}
        src: dict[int, set[str]] = {}
        for name, hits in (("bm25", bm_hits), ("vector", vc_hits)):
            for rank, (idx, _) in enumerate(hits):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
                src.setdefault(idx, set()).add(name)

        # 邻块扩展：命中块按比例把上下邻块带进候选池
        ratio = float(self.cfg.get("retrieval.neighbor_bonus", 0.0))
        if ratio > 0 and fused:
            for idx in sorted(fused, key=lambda i: fused[i], reverse=True)[:3]:
                c = self.chunks[idx]
                for j in self._by_doc.get(c.doc_id, []):
                    if abs(self.chunks[j].order - c.order) != 1:
                        continue
                    cand = fused[idx] * ratio
                    if fused.get(j, 0.0) < cand:
                        fused[j] = cand
                        src.setdefault(j, set()).add("neighbor")

        order = sorted(fused, key=lambda i: fused[i], reverse=True)
        mx = fused[order[0]] if order else 1.0
        norm = {i: (fused[i] / mx if mx else 0.0) for i in order}
        labels = {i: "+".join(sorted(src.get(i, ()))) for i in order}
        return order, norm, labels

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        if not self.chunks or not query.strip():
            return []
        pool = int(self.cfg.get("retrieval.top_k", 5)) if top_k is None else top_k
        order, norm, labels = self._fuse(query)
        hits = [
            Hit(chunk=self.chunks[i], retrieval_score=norm[i], score=norm[i], source=labels[i])
            for i in order[: max(pool * 4, 20)]
        ]
        hits = rerank(query, hits, dict(self.cfg.get("rerank", {})))
        self._attach_coverage(query, hits)
        return hits[:pool]

    def _attach_coverage(self, query: str, hits: list[Hit]) -> None:
        """可回答性 = 问题原文的字符 bigram 被这个块覆盖了多少。

        为什么不用归一化后的融合分：RRF 归一化后第一名恒为 1.0，
        库里根本没有相关内容时它也是 1.0，没法判断「该不该答」。
        字符 bigram 覆盖率是绝对量纲，与库里有没有对应内容直接相关。
        IDF 加权词覆盖率也试过，会被「多久」「之内」这类高频虚词带偏，不如 bigram 稳。
        """
        qb = bigrams(query)
        if not qb:
            return
        for h in hits:
            cb = bigrams(h.chunk.text)
            h.coverage = round(len(qb & cb) / len(qb), 4)

    # ---------------- 持久化 ----------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25, "vector": self.vector}, f)

    @classmethod
    def load(cls, path: Path, cfg) -> "HybridRetriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        r = cls(cfg)
        r.chunks = data["chunks"]
        r.bm25 = data["bm25"]
        r.vector = data["vector"]
        for i, c in enumerate(r.chunks):
            r._by_doc.setdefault(c.doc_id, []).append(i)
        r._idx = {c.chunk_id: i for i, c in enumerate(r.chunks)}
        return r

    # ---------------- 辅助 ----------------
    def docs(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in self.chunks:
            d = seen.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "chunks": 0})
            d["chunks"] += 1
        return list(seen.values())

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None
