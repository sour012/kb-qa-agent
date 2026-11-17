"""中文文本工具：分词、停用词、字符 bigram 相似度。

jieba 用搜索引擎模式（cut_for_search）而不是精确模式：
「知识库问答」被切成「知识库/问答」而不是整词，召回明显更好；
代价是索引变大，这里语料不大，可以接受。
"""

from __future__ import annotations

import re
from functools import lru_cache

import jieba

# 只保留真正会干扰排序的功能词，别把实词误杀
STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "有", "被", "把", "对", "为",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "这个", "那个",
    "一个", "一下", "什么", "怎么", "如何", "哪些", "哪个", "为什么", "是否", "可以",
    "需要", "应该", "吗", "呢", "吧", "啊", "呀", "请问", "请", "帮", "帮我", "想",
    "要", "会", "能", "能不能", "有没有", "以及", "并且", "但是", "如果", "就", "都",
    "也", "还", "很", "更", "最", "再", "又", "才", "而", "其", "之", "于", "上", "下",
    "中", "内", "外", "后", "前", "时", "时候", "个", "些", "等", "等等",
}

PUNCT = re.compile(r"[\s\u3000-\u303f\uff00-\uffef`~!@#$%^&*()_\-+=\[\]{};:'\",.<>/?\\|]+")


def clean(text: str) -> str:
    return PUNCT.sub(" ", text).strip()


@lru_cache(maxsize=8192)
def tokenize(text: str) -> tuple[str, ...]:
    """返回去停用词后的词序列（tuple 以便缓存）。"""
    if not text:
        return tuple()
    toks: list[str] = []
    for t in jieba.cut_for_search(clean(text.lower())):
        t = t.strip()
        if not t or t in STOPWORDS:
            continue
        if len(t) == 1 and not ("\u4e00" <= t <= "\u9fff"):
            continue          # 单个英文字母/数字没信息量
        toks.append(t)
    return tuple(toks)


def bigrams(text: str) -> set[str]:
    """字符二元组集合，用于「答案句是否被原文支持」的判定。

    中文没有词边界，词级重合会被分词差异带偏；字符 bigram 更稳。
    """
    s = clean(text)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def containment(part: str, whole: str) -> float:
    """part 的字符 bigram 有多大比例出现在 whole 里。"""
    p = bigrams(part)
    if not p:
        return 0.0
    return len(p & bigrams(whole)) / len(p)


def keyword_overlap(query_tokens: tuple[str, ...], text: str) -> tuple[float, list[str]]:
    """查询词在文本中的覆盖率，以及命中了哪些词（用于解释）。"""
    if not query_tokens:
        return 0.0, []
    low = text.lower()
    hit = [t for t in dict.fromkeys(query_tokens) if t in low]
    uniq = list(dict.fromkeys(query_tokens))
    return len(hit) / len(uniq), hit
