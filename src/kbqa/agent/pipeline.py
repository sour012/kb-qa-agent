"""问答 Agent 主流程：改写 → 路由 → 混合检索 →（低置信时）查询扩展 → 生成/抽取 → 自检。

这里刻意把「要不要拒答」做成显式判断而不是交给模型：
检索最高分低于阈值就拒答，命中但依据弱就加免责声明。
线上最贵的错误是「一本正经地编」，宁可少答一句。
"""

from __future__ import annotations

import time

from ..models import AskResult, Hit
from ..text import tokenize
from . import answerer, router, verifier
from .rewriter import SYNONYMS, rewrite


class QAAgent:
    def __init__(self, cfg, retriever=None):
        from ..index.retriever import HybridRetriever
        from ..llm.client import LLMClient

        self.cfg = cfg
        self.retriever = retriever or HybridRetriever.load(cfg.index_file, cfg)
        self.llm = LLMClient(cfg)
        self.min_cov = float(cfg.get("retrieval.min_coverage", 0.14))
        self.expand_cov = float(cfg.get("retrieval.expand_coverage", 0.28))
        self.ground_thr = float(cfg.get("answer.ground_threshold", 0.34))
        tokenize("预热分词器，避免首次调用把 jieba 词典加载算进检索耗时")

    # ------------------------------------------------------------------
    def _capability_text(self) -> str:
        docs = self.retriever.docs()
        names = "、".join(d["title"] for d in docs[:8])
        return (
            f"我是这个知识库的问答助手，目前收录了 {len(docs)} 篇文档（{names} 等）。\n"
            "可以问我制度、流程、技术方案类的具体问题，我会先检索原文再作答，并标出出处。\n"
            "如果知识库里没有依据，我会直接说没有，而不是编一个给你。"
        )

    def _refusal_text(self, best: float, hits: list[Hit]) -> str:
        docs = self.retriever.docs()
        names = "、".join(d["title"] for d in docs[:6])
        hint = ""
        if hits:
            hint = f"\n最接近的是《{hits[0].chunk.title}》里的「{hits[0].chunk.section}」，可以直接问那一节的内容。"
        return (
            f"知识库中没有找到与该问题直接相关的依据（问题覆盖率 {best:.0%}，低于阈值 {self.min_cov:.0%}）。"
            f"\n当前知识库覆盖：{names}。{hint}\n"
            "建议换个说法、补充具体名词（比如制度名称、系统名称、流程环节），或确认该内容是否已经入库。"
        )

    # ------------------------------------------------------------------
    def _expand(self, query: str, hits: list[Hit]) -> tuple[str, list[str]]:
        """低置信时的查询扩展：领域同义词 + 首轮 top1 块里的高 IDF 词。"""
        added: list[str] = []
        for key, extra in SYNONYMS.items():
            if key in query and not any(e in query for e in extra):
                added.extend(extra[:2])
                break
        if not hits:
            return (" ".join([query] + added)).strip(), added
        chunk = hits[0].chunk
        toks = tokenize(chunk.text)
        df = self.retriever.bm25.df
        n = max(1, self.retriever.bm25.n)
        scored = sorted(
            set(toks),
            key=lambda t: (n / (1 + df.get(t, n))),
            reverse=True,
        )
        q_toks = set(tokenize(query))
        for t in scored:
            if t in q_toks or len(t) < 2 or t in added:
                continue
            added.append(t)
            if len(added) >= 4:
                break
        return (query + " " + " ".join(added)).strip(), added

    def _merge(self, a: list[Hit], b: list[Hit]) -> list[Hit]:
        seen: dict[str, Hit] = {}
        for h in a + b:
            key = h.chunk.chunk_id
            if key not in seen or h.score > seen[key].score:
                seen[key] = h
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    # ------------------------------------------------------------------
    def ask(self, query: str, history: list[dict] | None = None) -> AskResult:
        t_all = time.perf_counter()
        trace: list[dict] = []
        res = AskResult(query=query)

        intent = router.route(query)
        trace.append({"step": "route", "detail": intent})
        if intent == "greeting":
            res.status = "greeting"
            res.answer = "你好，问点什么？制度、流程、技术方案都可以，我会给出处。"
            res.latency_ms = (time.perf_counter() - t_all) * 1000
            res.trace = trace
            return res
        if intent == "capability":
            res.status = "capability"
            res.answer = self._capability_text()
            res.latency_ms = (time.perf_counter() - t_all) * 1000
            res.trace = trace
            return res

        # 1) 改写
        t0 = time.perf_counter()
        rewritten, notes = rewrite(query, history)
        res.rewritten = rewritten
        res.notes = notes
        trace.append({
            "step": "rewrite",
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "detail": rewritten + (f"（{'；'.join(notes)}）" if notes else ""),
        })

        # 2) 检索
        t0 = time.perf_counter()
        hits = self.retriever.search(rewritten)
        best = hits[0].coverage if hits else 0.0
        trace.append({
            "step": "retrieve",
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "detail": f"混合检索 top{len(hits)}，问题覆盖率 {best:.2f}",
        })

        # 3) 低置信 → 查询扩展再检索一次
        if hits and best < self.expand_cov:
            t0 = time.perf_counter()
            expanded, added = self._expand(rewritten, hits)
            if added:
                more = self.retriever.search(expanded)
                hits = self._merge(hits, more)[: int(self.cfg.get("retrieval.top_k", 5))]
                # 拒答判断必须用「用户原话」的覆盖率：扩展词是检索手段，
                # 用扩展后的词去算覆盖率会自我抬高，把本该拒答的题答出来。
                self.retriever._attach_coverage(rewritten, hits)
                best = hits[0].coverage if hits else best
                trace.append({
                    "step": "expand",
                    "ms": round((time.perf_counter() - t0) * 1000, 2),
                    "detail": f"加入扩展词 {'/'.join(added)}，覆盖率 {best:.2f}",
                })

        res.hits = hits

        # 4) 依据不足 → 拒答
        if best < self.min_cov:
            res.status = "refused"
            res.answer = self._refusal_text(best, hits)
            res.citations = []
            res.latency_ms = (time.perf_counter() - t_all) * 1000
            trace.append({"step": "answer", "detail": "问题覆盖率低于阈值，拒答"})
            res.trace = trace
            return res

        # 5) 生成
        t0 = time.perf_counter()
        if self.llm.available:
            answer = answerer.llm_answer(
                rewritten, hits, self.llm,
                int(self.cfg.get("answer.max_context_chars", 1800)),
            )
            used = sorted({i for i in answerer.cited_indices(answer)})
            used_hits = [hits[i - 1] for i in used if 1 <= i <= len(hits)] or hits[:1]
            mode = "llm"
        else:
            answer, used_idx = answerer.extractive_answer(
                rewritten, hits,
                int(self.cfg.get("answer.max_sentences", 3)),
                int(self.cfg.get("answer.max_per_chunk", 2)),
            )
            used_hits = [hits[i] for i in used_idx] or hits[:1]
            mode = "extractive"
        trace.append({
            "step": "answer",
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "detail": f"{mode} 模式，引用 {len(used_hits)} 个片段",
        })

        # 6) 自检
        chk = verifier.check(answer, hits, self.ground_thr)
        res.grounded_ratio = chk["ratio"]
        trace.append({
            "step": "verify",
            "ms": 0,
            "detail": (
                f"引用支持度 {chk['ratio']:.0%}，"
                f"{chk['unsupported']}/{chk['sentences']} 句依据不足"
                + (f"，非法引用 {chk['invalid']}" if chk["invalid"] else "")
            ),
        })

        if chk["ratio"] < self.ground_thr:
            res.status = "caveat"
            answer = (
                f"（提示：以下回答的原文依据较弱，支持度 {chk['ratio']:.0%}，"
                f"关键数字请对照文末出处核对）\n{answer}"
            )
        else:
            res.status = "answered"

        res.answer = answer
        res.citations = answerer.build_citations(used_hits)
        res.latency_ms = (time.perf_counter() - t_all) * 1000
        res.trace = trace
        return res
