"""配置加载：点号路径取值 + 相对项目根目录的绝对路径。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else ROOT / "config" / "config.yaml"
        with open(p, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    # ---- 常用快捷方式：配置里全部是相对路径，这里统一转绝对路径 ----
    def path(self, key: str) -> Path:
        return (ROOT / str(self.get(f"paths.{key}"))).resolve()

    @property
    def corpus_dir(self) -> Path:
        return self.path("corpus_dir")

    @property
    def index_file(self) -> Path:
        return self.path("index_file")

    @property
    def eval_file(self) -> Path:
        return self.path("eval_file")

    @property
    def report_dir(self) -> Path:
        return self.path("report_dir")

    @property
    def llm_enabled(self) -> bool:
        if not self.get("llm.enabled", False):
            return False
        return bool(os.environ.get(str(self.get("llm.api_key_env", "")), "").strip())
