"""自检：回答里的每一句，是否真的能从它引用的原文里找到依据？

这是 RAG 最容易翻车的地方——模型把「15 个工作日」写成「两周」，
检索没错、答案读起来也对，但数字错了。这里用字符 bigram 覆盖率做一次
事后核对，不依赖第二个模型（省钱、确定性强），低于阈值就降级成
「依据较弱，请核对原文」，完全无引用则直接拒答。
"""

from __future__ import annotations

import re

from ..models import Hit
from ..text import containment
from .answerer import CIT

SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
LEAD_CIT = re.compile(r"^(\s*\[\d{1,2}\]\s*)+")


def _merge_leading_citations(sents: list[str]) -> list[str]:
    """模型常写成「……600 元。[1]」——标记落在下一句开头，这里并回上一句。"""
    out: list[str] = []
    for s in sents:
        if out and LEAD_CIT.match(s):
            out[-1] = out[-1] + s
        else:
            out.append(s)
    return out


def check(answer: str, hits: list[Hit], threshold: float = 0.34) -> dict:
    raw = [s.strip() for s in SENT_SPLIT.split(answer) if s.strip()]
    sents = _merge_leading_citations(raw)
    if not sents:
        return {"ratio": 0.0, "details": [], "invalid": [], "unsupported": 0, "sentences": 0}

    details = []
    invalid: list[int] = []
    unsupported = 0
    for s in sents:
        ids = [int(x) for x in CIT.findall(s)]
        body = CIT.sub("", s).strip()
        if not ids or not body:
            details.append({"sentence": s, "support": 0.0, "cited": []})
            unsupported += 1
            continue
        valid = [i for i in ids if 1 <= i <= len(hits)]
        invalid.extend([i for i in ids if i not in valid])
        support = 0.0
        for i in valid:
            support = max(support, containment(body, hits[i - 1].chunk.text))
        if support < threshold:
            unsupported += 1
        details.append({"sentence": s, "support": round(support, 3), "cited": valid})

    ratio = sum(d["support"] for d in details) / len(details)
    return {
        "ratio": round(ratio, 4),
        "details": details,
        "invalid": sorted(set(invalid)),
        "unsupported": unsupported,
        "sentences": len(details),
    }
