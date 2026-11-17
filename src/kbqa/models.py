"""贯穿全链路的轻量数据结构：文档 → 段落块 → 检索命中 → 回答与引用。

刻意不用 pydantic：索引要能 pickle 落盘，dataclass 更省心；
只有 HTTP 层才需要校验，那边单独用 pydantic。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Document:
    doc_id: str          # 文件名（去扩展名），用作引用里的稳定标识
    path: str            # 相对 corpus 的路径
    title: str
    text: str
    mtime: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str        # doc_id#0007
    doc_id: str
    title: str
    section: str         # 标题路径，如「检索策略 > 混合检索」
    order: int
    text: str
    n_chars: int = 0

    def __post_init__(self) -> None:
        self.n_chars = len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    retrieval_score: float = 0.0   # 召回（BM25 / 向量 / RRF 融合）分，已归一化
    rerank_score: float = 0.0      # 重排分
    score: float = 0.0             # 最终排序分
    coverage: float = 0.0          # 查询词 IDF 覆盖率：绝对可回答性，用来判定「该不该答」
    source: str = ""               # bm25 / vector / both

    @property
    def citation(self) -> str:
        return f"{self.chunk.title}｜{self.chunk.section or '正文'}"


@dataclass
class Citation:
    index: int
    doc_id: str
    title: str
    section: str
    chunk_id: str
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AskResult:
    query: str
    rewritten: str = ""
    answer: str = ""
    status: str = "answered"        # answered | caveat | refused | capability | greeting
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    grounded_ratio: float = 0.0
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        d["hits"] = [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc_id": h.chunk.doc_id,
                "section": h.chunk.section,
                "score": round(h.score, 4),
                "retrieval_score": round(h.retrieval_score, 4),
                "rerank_score": round(h.rerank_score, 4),
                "source": h.source,
            }
            for h in self.hits
        ]
        return d
