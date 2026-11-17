"""测试用的最小语料与夹具：不依赖仓库里真实的 data/，可独立跑。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kbqa.config import Config  # noqa: E402
from kbqa.ingest.chunker import chunk_documents  # noqa: E402
from kbqa.index.retriever import HybridRetriever  # noqa: E402
from kbqa.models import Document  # noqa: E402

DOC_A = """# 差旅报销制度

## 住宿标准

按城市分三档：一线城市 600 元，省会城市 450 元，其他城市 350 元。
超标部分自理。同城出差当天可返回的，不报销住宿费。

## 报销时限

费用发生后 30 个自然日内提交，跨季度不予受理。
每年 12 月 20 日为当年费用截止日。
"""

DOC_B = """# 线上故障响应

## 分级

P0 故障要求 5 分钟内响应，30 分钟内给出初步定位结论。
P1 故障要求 15 分钟内响应。

## 止损

发现故障后的第一步是止损，不是查原因。止损手段优先回滚最近一次变更。
"""


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def chunks():
    docs = [
        Document(doc_id="expense", path="expense.md", title="差旅报销制度", text=DOC_A),
        Document(doc_id="incident", path="incident.md", title="线上故障响应", text=DOC_B),
    ]
    return chunk_documents(docs, max_chars=200, min_chars=60, overlap=40)


@pytest.fixture(scope="session")
def retriever(cfg, chunks) -> HybridRetriever:
    return HybridRetriever(cfg).build(chunks)
