import pytest

from kbqa.agent.answerer import extractive_answer
from kbqa.agent.pipeline import QAAgent
from kbqa.agent.rewriter import rewrite
from kbqa.agent import verifier


def test_rewrite_strips_polite_words():
    q, notes = rewrite("请问一下报销单多久内提交？")
    assert "请问" not in q
    assert "去除客套词" in "".join(notes)


def test_rewrite_resolves_pronoun_with_history():
    hist = [{"role": "user", "content": "紧急发布怎么审批"}]
    q, notes = rewrite("它需要几个人", hist)
    assert "紧急发布" in q or "发布" in q
    assert notes


def test_rewrite_keeps_question_without_history():
    q, _ = rewrite("住宿标准是多少")
    assert "住宿" in q


def test_verifier_accepts_verbatim_answer(retriever):
    hits = retriever.search("住宿标准 一线城市")
    sent = hits[0].chunk.text.split("。")[0]
    chk = verifier.check(f"{sent}[1]。", hits, threshold=0.3)
    assert chk["ratio"] > 0.8
    assert chk["unsupported"] == 0


def test_verifier_merges_citation_after_period(retriever):
    hits = retriever.search("住宿标准")
    chk = verifier.check("这句话和原文无关。[1]", hits, threshold=0.3)
    # 只有一句，标记应被并入而不是单独成句
    assert chk["sentences"] == 1


def test_verifier_flags_unsupported_and_invalid(retriever):
    hits = retriever.search("住宿标准")
    chk = verifier.check("一线城市住宿上限是 2000 元[1]。另外[9]。", hits, threshold=0.6)
    assert chk["unsupported"] >= 1
    assert chk["invalid"] == [9]


def test_extractive_answer_stays_in_top_chunk(retriever):
    hits = retriever.search("住宿标准一线城市多少钱")
    ans, used = extractive_answer("住宿标准一线城市多少钱", hits, max_sentences=2, max_per_chunk=2)
    assert used and used[0] == 0
    assert "600" in ans


def test_agent_refuses_unrelated_question(cfg, retriever):
    agent = QAAgent(cfg, retriever)
    res = agent.ask("食堂几点开门")
    assert res.status == "refused"
    assert not res.citations


def test_agent_answers_and_cites(cfg, retriever):
    agent = QAAgent(cfg, retriever)
    res = agent.ask("P0 故障多久内响应")
    assert res.status in ("answered", "caveat")
    assert "5 分钟" in res.answer
    assert res.citations
    assert res.grounded_ratio > 0.3


def test_agent_greeting_skips_retrieval(cfg, retriever):
    agent = QAAgent(cfg, retriever)
    res = agent.ask("你好")
    assert res.status == "greeting"
    assert res.hits == []


def test_multiturn_uses_history(cfg, retriever):
    agent = QAAgent(cfg, retriever)
    hist = [
        {"role": "user", "content": "P0 故障要求多久响应"},
        {"role": "assistant", "content": "5 分钟"},
    ]
    res = agent.ask("那 P1 呢", hist)
    assert res.status in ("answered", "caveat")
    assert "15" in res.answer
