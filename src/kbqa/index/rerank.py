"""重排：把「召回到了」变成「排得对」。

BM25 和向量都只看局部匹配，容易把一段讲同一堆术语但答非所问的内容排前面。
这里用三个可解释的特征做加权重排，权重写在配置里，便于做消融：
- 查询词覆盖率：问题里的实词块里出现了几个（最重要）
- 标题命中：命中的是「发布流程 > 紧急发布」这种小节路径，说明上下文对上了
- 长度惩罚：太短信息不足，太长注意力被稀释
"""

from __future__ import annotations

from ..models import Chunk, Hit
from ..text import keyword_overlap, tokenize


def _length_score(n: int, ideal: int = 340, span: int = 340) -> float:
    return max(0.0, 1.0 - abs(n - ideal) / span * 0.5)


def rerank(query: str, hits: list[Hit], weights: dict[str, float]) -> list[Hit]:
    qtoks = tokenize(query)
    for h in hits:
        cov, _ = keyword_overlap(qtoks, h.chunk.text)
        head = h.chunk.section + " " + h.chunk.title
        head_cov, _ = keyword_overlap(qtoks, head)
        h.rerank_score = (
            weights.get("weight_term_coverage", 0.55) * cov
            + weights.get("weight_heading", 0.25) * min(1.0, head_cov * 1.5)
            + weights.get("weight_length", 0.20) * _length_score(h.chunk.n_chars)
        )
    rw = weights.get("retrieval_weight", 0.5)
    for h in hits:
        h.score = rw * h.retrieval_score + (1 - rw) * h.rerank_score
    return sorted(hits, key=lambda x: x.score, reverse=True)


def explain(hit: Hit, query: str) -> str:
    qtoks = tokenize(query)
    cov, words = keyword_overlap(qtoks, hit.chunk.text)
    return (
        f"{hit.chunk.chunk_id} 分={hit.score:.3f}（召回 {hit.retrieval_score:.3f} "
        f"+ 重排 {hit.rerank_score:.3f}，词覆盖 {cov:.0%}：{'/'.join(words[:6]) or '—'}）"
    )
