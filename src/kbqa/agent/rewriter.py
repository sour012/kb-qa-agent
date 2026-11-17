"""多轮问题改写：把「它要多久」补成「紧急发布要多久」。

真实用户不会每句话都带主语，指代消解做不好，第二轮的检索就会直接崩。
这里不调模型，用代价最低的规则：
1. 去掉客套话（请问/帮我/麻烦/谢谢），它们会污染 BM25 的词频；
2. 识别指代词与超短问句，从上一轮问题里取实词补进查询。

同义词扩展不放在这里：无条件扩词会把「住宿能报多少」扩成「住宿 差旅 费用 发票」，
反而稀释查询主题。扩词只在检索置信度不足时由 pipeline 触发。
"""

from __future__ import annotations

import re

from ..text import STOPWORDS, tokenize

POLITE = re.compile(r"^(请问一下|请问|麻烦问一下|麻烦|帮我看看|帮我|我想知道|想问一下|我想问|想了解)[，,：:\s]*")
TAIL = re.compile(r"(谢谢|谢啦|感谢)[!！。.\s]*$")

PRONOUNS = ("它", "他", "这个", "那个", "该", "上述", "刚才", "前面", "这条", "那条", "这些", "那些")

# 领域同义词：只在检索分数偏低时作为扩展词使用
SYNONYMS = {
    "报销": ["差旅", "费用", "发票"],
    "请假": ["年假", "事假", "病假", "审批"],
    "上线": ["发布", "部署", "审批"],
    "发布": ["上线", "灰度", "回滚"],
    "故障": ["应急", "处理", "响应", "值班"],
    "权限": ["申请", "账号", "授权"],
    "评测": ["评估", "指标", "召回"],
    "切分": ["分块", "chunk", "重叠"],
}


def _topics(text: str, n: int = 3) -> list[str]:
    toks = [t for t in tokenize(text) if t not in STOPWORDS]
    # 优先长词：长词更具体（「知识库」优于「库」）
    toks.sort(key=len, reverse=True)
    out: list[str] = []
    for t in toks:
        if t not in out:
            out.append(t)
        if len(out) >= n:
            break
    return out


def rewrite(query: str, history: list[dict] | None = None) -> tuple[str, list[str]]:
    notes: list[str] = []
    q = query.strip()
    q2 = POLITE.sub("", q)
    q2 = TAIL.sub("", q2).strip()
    if q2 != q:
        notes.append("去除客套词")
    q = q2 or q

    last_user = ""
    for turn in reversed(history or []):
        if turn.get("role", "user") == "user" and turn.get("content"):
            last_user = str(turn["content"])
            break

    if last_user:
        need = any(p in q for p in PRONOUNS) or len(q) <= 8
        if need:
            topics = _topics(last_user, 3)
            if topics:
                q = "".join(topics) + " " + q
                notes.append(f"补全指代（上一轮主题：{'/'.join(topics)}）")

    return q.strip(), notes
