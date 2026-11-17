"""回答生成：抽取式（默认）与生成式（可选）两条路，共用同一份上下文。

抽取式的关键不是「找到和问题词重合的句子」，而是「在正确的块里找句子」。
纯看句子和问题的词重合会翻车：问「北京住宿能报多少」，正确句是
「一线城市（北京、上海、广州、深圳）600 元」——它不含「住宿」「报销」两个词，
但所在的块分最高。所以这里把块分作为主权重，句子相关度只做微调。
"""

from __future__ import annotations

import re

from ..models import Citation, Hit
from ..text import bigrams, tokenize
from . import prompts

CIT = re.compile(r"\[(\d{1,2})\]")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [p.strip() for p in parts if p and p.strip()]


def _overlap(query_bg: set[str], sent: str) -> float:
    """字符 bigram 双向覆盖：既看查询被覆盖多少，也看句子有多少是噪声。"""
    s = bigrams(sent)
    if not s or not query_bg:
        return 0.0
    inter = len(query_bg & s)
    return 0.7 * inter / len(query_bg) + 0.3 * inter / len(s)


def _sent_score(sent: str, query_bg: set[str], chunk_score: float, pos: int,
                want_number: bool = False) -> float:
    pos_bonus = 0.10 if pos == 0 else (0.04 if pos == 1 else 0.0)
    len_pen = 0.0 if 12 <= len(sent) <= 170 else -0.10
    # 制度类问答里，「5 天」这种阈值必须落在同一档条款上，带数字的句子更可能是答案
    num_bonus = 0.06 if (want_number and re.search(r"\d", sent)) else 0.0
    return 0.55 * _overlap(query_bg, sent) + 0.35 * chunk_score + pos_bonus + len_pen + num_bonus


def build_citations(hits: list[Hit]) -> list[Citation]:
    out: list[Citation] = []
    for i, h in enumerate(hits, 1):
        snippet = h.chunk.text[:120].replace("\n", " ")
        out.append(
            Citation(
                index=i,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                section=h.chunk.section,
                chunk_id=h.chunk.chunk_id,
                score=round(h.score, 4),
                snippet=snippet + ("…" if len(h.chunk.text) > 120 else ""),
            )
        )
    return out


def extractive_answer(
    query: str,
    hits: list[Hit],
    max_sentences: int = 3,
    max_per_chunk: int = 2,
    dominance: float = 0.90,
):
    """返回 (答案文本, 被引用的 hit 下标列表)。答案句尾直接带 [n]。

    抽取策略是「主块优先」：
    只从 top1 块里挑句子；其他块只有在分数达到 top1 的 90% 以上、
    且句子分不输给主块最佳句时，才允许补一两句。
    宁可少答一句，也不要把「看起来相关」的旁支段落拼进来——
    拼错一句，用户对整个知识库的信任就没了。
    """
    if not hits:
        return "", []
    qbg = bigrams(query)
    want_number = bool(re.search(r"\d", query))

    def rank_sentences(h: Hit) -> list[tuple[float, int, str]]:
        sents = split_sentences(h.chunk.text)
        scored = [(_sent_score(s, qbg, h.score, i, want_number), i, s) for i, s in enumerate(sents)]
        return sorted(scored, reverse=True)

    top = hits[0]
    top_scored = rank_sentences(top)
    picked: list[tuple[int, int, str]] = [(0, si, s) for _, si, s in top_scored[:max_per_chunk]]
    best = top_scored[0][0] if top_scored else 0.0

    for hi, h in enumerate(hits[1:], 1):
        if len(picked) >= max_sentences:
            break
        if h.score < dominance * top.score:
            continue
        for sc, si, s in rank_sentences(h):
            if sc < best or len(picked) >= max_sentences:
                break
            picked.append((hi, si, s))

    picked = picked[:max_sentences]
    picked.sort(key=lambda x: (x[0], x[1]))

    used: list[int] = []
    parts: list[str] = []
    for hi, _, sent in picked:
        if hi not in used:
            used.append(hi)
        body = sent.rstrip("。；;，, ")
        parts.append(f"{body}[{used.index(hi) + 1}]。")
    return " ".join(parts), used


def llm_answer(query: str, hits: list[Hit], client, max_context_chars: int = 1800) -> str:
    ctx_parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        block = f"【{i}】{h.chunk.title}｜{h.chunk.section}\n{h.chunk.text}"
        if used + len(block) > max_context_chars and ctx_parts:
            break
        ctx_parts.append(block)
        used += len(block)
    context = "\n\n".join(ctx_parts)
    text, _ = client.chat(prompts.SYSTEM, prompts.user_prompt(query, context))
    return text


def cited_indices(answer: str) -> list[int]:
    return [int(m) for m in CIT.findall(answer)]
