"""意图路由：知识问答 / 能力咨询 / 寒暄。

路由要放在检索之前——「你好」也去检索一遍，只会给出一堆莫名其妙的引用。
检索之后的「依据够不够」由 pipeline 依据 min_score 判断，这里只管纯文本规则。
"""

from __future__ import annotations

import re

GREETING = re.compile(r"^(你好|您好|hi|hello|在吗|哈喽|早上好|下午好)[!！。.\s]*$", re.I)
CAPABILITY = re.compile(r"(你能做什么|能干什么|会什么|支持哪些|怎么用|使用说明|help|帮助|你是谁)")
THANKS = re.compile(r"^(谢谢|感谢|好的|ok|嗯)[!！。.\s]*$", re.I)


def route(query: str) -> str:
    q = query.strip()
    if GREETING.match(q) or THANKS.match(q):
        return "greeting"
    if CAPABILITY.search(q):
        return "capability"
    return "qa"
