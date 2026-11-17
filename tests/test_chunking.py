from kbqa.ingest.chunker import pack_sentences, split_sections, split_sentences


def test_sections_follow_heading_tree():
    text = "# 标题\n\n## 一 > 甲\n\n内容甲\n\n### 细\n\n内容细\n"
    secs = split_sections(text, "标题")
    paths = [s.path for s in secs]
    assert paths == ["标题 > 一 > 甲", "标题 > 一 > 甲 > 细"]
    assert secs[0].text == "内容甲"


def test_pack_never_breaks_a_sentence():
    sents = ["第一句话内容比较长。" * 3, "第二句。" * 3, "第三句。" * 3]
    chunks = pack_sentences(sents, max_chars=120, min_chars=40, overlap=40)
    joined = "".join(chunks)
    for s in sents:
        assert s in joined, "句子被切断或丢失"
    # 每个块都应以句末标点结尾，说明没有在半句处截断
    for c in chunks:
        assert c.rstrip()[-1] in "。；！？"


def test_pack_has_overlap_between_neighbours():
    sents = [f"第{i}条规则的内容大概这么长，需要占一点位置。" for i in range(6)]
    chunks = pack_sentences(sents, max_chars=80, min_chars=20, overlap=40)
    assert len(chunks) >= 2
    # 前一块的末句应出现在下一块开头（句子级重叠）
    tail = split_sentences(chunks[0])[-1]
    assert tail in chunks[1]


def test_short_tail_chunk_is_merged():
    sents = ["甲。" * 20, "乙。"]
    chunks = pack_sentences(sents, max_chars=60, min_chars=40, overlap=10)
    assert len(chunks) == 1


def test_split_sentences_keeps_punctuation():
    out = split_sentences("第一句。第二句！第三句？")
    assert out == ["第一句。", "第二句！", "第三句？"]
