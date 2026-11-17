from kbqa.index.bm25 import BM25
from kbqa.text import bigrams, containment, tokenize


def test_tokenizer_drops_stopwords():
    toks = tokenize("请问住宿费可以报销吗")
    assert "请问" not in toks
    assert "住宿费" in toks or "住宿" in toks


def test_bm25_prefers_matching_chunk(chunks):
    bm = BM25().build(chunks)
    hits = bm.search("P0 故障响应时间", k=3)
    assert hits, "应至少召回一个块"
    top = chunks[hits[0][0]]
    assert top.doc_id == "incident"


def test_bm25_unknown_terms_return_nothing(chunks):
    bm = BM25().build(chunks)
    assert bm.search("量子纠缠拓扑绝缘体", k=3) == []


def test_idf_favours_rare_terms(chunks):
    bm = BM25().build(chunks)
    # 「住宿」只出现在报销文档，「响应」只出现在故障文档，两者 df 都应小于总块数
    assert bm.df["住宿"] < bm.n


def test_containment_is_independent_of_length():
    whole = "按城市分三档，一线城市 600 元，省会城市 450 元。"
    assert containment("一线城市 600 元", whole) == 1.0
    assert containment("省会城市 450 元", whole) == 1.0
    # 改一个数字也要能被检出：支持度必须掉到 1.0 以下
    assert containment("一线城市 900 元", whole) < 1.0


def test_bigrams_ignore_punctuation():
    assert bigrams("A，B。") == bigrams("A B")
