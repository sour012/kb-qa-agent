"""命令行入口：build-index / ask / chat / eval / serve / docs。

用法示例：
    PYTHONPATH=src python -m kbqa build-index
    PYTHONPATH=src python -m kbqa ask "北京出差住宿能报多少" --trace
    PYTHONPATH=src python -m kbqa eval
    PYTHONPATH=src python -m kbqa serve
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from .config import Config
from .ingest.chunker import chunk_documents
from .ingest.loader import load_corpus
from .models import AskResult


def _print(res: AskResult, trace: bool = False) -> None:
    tag = {
        "answered": "已回答",
        "caveat": "依据较弱",
        "refused": "拒答",
        "capability": "能力咨询",
        "greeting": "寒暄",
    }.get(res.status, res.status)
    print(f"\n[{tag}] {res.latency_ms:.0f} ms｜引用支持度 {res.grounded_ratio:.0%}")
    if res.rewritten and res.rewritten != res.query:
        print(f"改写后问题：{res.rewritten}")
    print("-" * 68)
    print(res.answer)
    if res.citations:
        print("-" * 68)
        for c in res.citations:
            print(f"[{c.index}] {c.title}｜{c.section}（score {c.score:.3f}）")
    if trace:
        print("-" * 68)
        for step in res.trace:
            ms = f"{step['ms']}ms " if "ms" in step else ""
            print(f"  · {step['step']:<9}{ms}{step.get('detail', '')}")


def cmd_build(args: argparse.Namespace, cfg: Config) -> None:
    from .index.retriever import HybridRetriever

    t0 = time.perf_counter()
    docs = load_corpus(cfg.corpus_dir)
    chunks = chunk_documents(
        docs,
        int(cfg.get("chunk.max_chars", 460)),
        int(cfg.get("chunk.min_chars", 120)),
        int(cfg.get("chunk.overlap", 80)),
    )
    retriever = HybridRetriever(cfg).build(chunks)
    retriever.save(cfg.index_file)
    print(f"文档 {len(docs)} 篇 → 切分 {len(chunks)} 块 → 索引已保存 {cfg.index_file}"
          f"（{time.perf_counter() - t0:.1f}s）")
    lens = [c.n_chars for c in chunks]
    print(f"块长度：min {min(lens)} / 平均 {sum(lens) // len(lens)} / max {max(lens)} 字")
    for d in retriever.docs():
        print(f"  - {d['doc_id']}: {d['chunks']} 块｜{d['title']}")


def _agent(cfg: Config):
    from .agent.pipeline import QAAgent

    if not cfg.index_file.exists():
        sys.exit(f"索引不存在：{cfg.index_file}\n先执行 python -m kbqa build-index")
    return QAAgent(cfg)


def cmd_ask(args: argparse.Namespace, cfg: Config) -> None:
    agent = _agent(cfg)
    res = agent.ask(args.question)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print(res, trace=args.trace)


def cmd_chat(args: argparse.Namespace, cfg: Config) -> None:
    agent = _agent(cfg)
    history: list[dict] = []
    print("知识库问答（空行退出）")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        res = agent.ask(q, history)
        _print(res, trace=args.trace)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": res.answer})


def cmd_eval(args: argparse.Namespace, cfg: Config) -> None:
    from .evaluate import evaluate, load_qa, to_markdown

    agent = _agent(cfg)
    items = load_qa(cfg.eval_file)
    result = evaluate(agent, items)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(result, cfg)
    (cfg.report_dir / "eval.md").write_text(md, encoding="utf-8")
    (cfg.report_dir / "eval.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(md)
    print(f"已写入 {cfg.report_dir / 'eval.md'}")


def cmd_serve(args: argparse.Namespace, cfg: Config) -> None:
    import uvicorn

    from .api import create_app

    app = create_app(cfg)
    host = args.host or str(cfg.get("server.host", "127.0.0.1"))
    port = args.port or int(cfg.get("server.port", 8020))
    print(f"API: http://{host}:{port}/  （打开浏览器可直接对话）")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_docs(args: argparse.Namespace, cfg: Config) -> None:
    agent = _agent(cfg)
    for d in agent.retriever.docs():
        print(f"{d['doc_id']:<28} {d['chunks']:>3} 块  {d['title']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kbqa", description="知识库问答 Agent")
    p.add_argument("--config", default=None, help="配置文件路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-index", help="加载语料、切分并建索引")
    sub.add_parser("docs", help="列出知识库文档")

    a = sub.add_parser("ask", help="单轮提问")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    a.add_argument("--trace", action="store_true", help="打印每步耗时与决策")

    c = sub.add_parser("chat", help="多轮对话")
    c.add_argument("--trace", action="store_true")

    sub.add_parser("eval", help="跑评测集并生成报告")

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    handlers = {
        "build-index": cmd_build,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "eval": cmd_eval,
        "serve": cmd_serve,
        "docs": cmd_docs,
    }
    handlers[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
