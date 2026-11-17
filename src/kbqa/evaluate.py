"""离线评测：召回、排序、引用、拒答四类指标，一条命令复现。

指标口径写在 docs/DESIGN.md，这里只负责算：
- recall@k / MRR 只看 hits（检索层），与生成无关；
- 引用准确率看 citations（生成层挑中了哪几个出处）；
- 拒答准确率只在不可回答集上统计。
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def load_qa(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate(agent, qa_items: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    recall = {k: 0 for k in ks}
    mrr = 0.0
    cite_hit = 0
    cite_total = 0
    kw_hit = 0
    kw_total = 0
    refuse_ok = 0
    refuse_total = 0
    lat: list[float] = []
    rows: list[dict] = []

    for item in qa_items:
        t0 = time.perf_counter()
        res = agent.ask(item["q"])
        lat.append((time.perf_counter() - t0) * 1000)

        hits = res.hits
        docs = [h.chunk.doc_id for h in hits]
        row = {
            "q": item["q"],
            "expected": item.get("doc"),
            "status": res.status,
            "top_docs": docs[:5],
            "citations": [c.doc_id for c in res.citations],
            "grounded": res.grounded_ratio,
        }

        if item.get("answerable"):
            for k in ks:
                if item["doc"] in docs[:k]:
                    recall[k] += 1
            rank = next((i for i, d in enumerate(docs) if d == item["doc"]), None)
            mrr += 1.0 / (rank + 1) if rank is not None else 0.0

            cited = [c.doc_id for c in res.citations]
            if cited:
                cite_total += 1
                if item["doc"] in cited:
                    cite_hit += 1
            kws = item.get("keywords") or []
            if kws:
                kw_total += 1
                if all(str(k) in res.answer for k in kws):
                    kw_hit += 1
        else:
            refuse_total += 1
            if res.status == "refused":
                refuse_ok += 1
                row["ok"] = True
            else:
                row["ok"] = False
        rows.append(row)

    n = max(1, len(qa_items))
    n_ans = max(1, sum(1 for i in qa_items if i.get("answerable")))
    return {
        "n": len(qa_items),
        "recall": {k: round(recall[k] / n_ans, 4) for k in ks},
        "mrr": round(mrr / n_ans, 4),
        "citation_accuracy": round(cite_hit / max(1, cite_total), 4),
        "answer_keyword_hit": round(kw_hit / max(1, kw_total), 4),
        "refusal_accuracy": round(refuse_ok / max(1, refuse_total), 4),
        "latency_ms_avg": round(sum(lat) / n, 1),
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
        "rows": rows,
    }


def to_markdown(result: dict, cfg) -> str:
    lines = [
        "# 评测报告",
        "",
        f"- 语料：`{cfg.corpus_dir.name}/`（索引 `{cfg.index_file.name}`）",
        f"- 评测集：`{cfg.eval_file.name}`，共 {result['n']} 条（含不可回答题）",
        f"- 生成模式：{'LLM' if cfg.llm_enabled else '抽取式（未配置 API Key）'}",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
    ]
    for k, v in result["recall"].items():
        lines.append(f"| recall@{k} | {v:.1%} | 正确文档是否进入前 {k} 条 |")
    lines += [
        f"| MRR | {result['mrr']:.3f} | 首个正确结果的平均倒数排名 |",
        f"| 引用准确率 | {result['citation_accuracy']:.1%} | 回答引用的出处是否正确 |",
        f"| 答案关键词命中 | {result['answer_keyword_hit']:.1%} | 关键条件（金额/时限）是否出现在答案里 |",
        f"| 拒答准确率 | {result['refusal_accuracy']:.1%} | 不可回答题是否被正确拒答 |",
        f"| 平均延迟 | {result['latency_ms_avg']} ms | p95 {result['latency_ms_p95']} ms |",
        "",
        "## 失败样本",
        "",
    ]
    bad = [
        r for r in result["rows"]
        if (r.get("expected") and r["expected"] not in (r["top_docs"] or [])[:5])
        or (r.get("expected") is None and not r.get("ok"))
    ]
    if not bad:
        lines.append("无。")
    for r in bad[:12]:
        lines.append(
            f"- `{r['q']}` → 期望 `{r['expected']}`，实际 top5 `{r['top_docs']}`，状态 `{r['status']}`"
        )
    lines.append("")
    return "\n".join(lines)
