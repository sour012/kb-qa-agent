"""切分：先按标题树切「语义段」，再在段内按句子打包。

为什么不直接按固定长度硬切：
固定窗口会把「第 3 条：……；第 4 条：……」这类条款拦腰截断，检索命中后
上下文里只剩半条规则，答案就是错的。所以这里三道约束：
1) 标题层级优先——一个小节的内容尽量留在同一块；
2) 句子边界对齐——绝不在句号中间切；
3) 重叠区按句回退——上一块末尾 1~2 句带进下一块，避免边界问题答不出来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=\n)")


@dataclass
class Section:
    path: str      # 「检索策略 > 混合检索」
    text: str


def split_sections(text: str, doc_title: str) -> list[Section]:
    lines = text.split("\n")
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []      # (level, title)
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(t for _, t in stack if t) or doc_title
            sections.append(Section(path=path, text=body))
        buf.clear()

    for line in lines:
        m = HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def split_sentences(text: str) -> list[str]:
    parts = [p for p in SENT_SPLIT.split(text) if p is not None]
    # 保留句末标点：正则用了 lookbehind，split 后标点留在前一段
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents


def pack_sentences(sents: list[str], max_chars: int, min_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for s in sents:
        if cur and cur_len + len(s) > max_chars:
            chunks.append("".join(cur))
            # 回退若干句做重叠，但不超过 overlap 预算，也不吃掉整块
            tail: list[str] = []
            acc = 0
            for prev in reversed(cur):
                if acc + len(prev) > overlap or len(tail) >= 2:
                    break
                tail.insert(0, prev)
                acc += len(prev)
            cur = tail
            cur_len = sum(len(x) for x in cur)
        cur.append(s)
        cur_len += len(s)

    if cur:
        chunks.append("".join(cur))

    # 尾块过短就并回前一块，避免「只有一句话」的碎片参与打分
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(
    docs: list[Document],
    max_chars: int = 460,
    min_chars: int = 120,
    overlap: int = 80,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        sections = split_sections(doc.text, doc.title)
        order = 0
        for sec in sections:
            sents = split_sentences(sec.text)
            for piece in pack_sentences(sents, max_chars, min_chars, overlap):
                order += 1
                out.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{order:04d}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section=sec.path,
                        order=order,
                        text=piece,
                    )
                )
    return out
