"""LLM 接入：OpenAI 兼容的 /chat/completions，只用 requests，不引 SDK。

设计上 LLM 是「可选增强」而不是硬依赖：
没有 Key 时整条链路退化成抽取式回答，效果弱一点但完全离线可跑，
面试时也能解释清楚「生成模型不是 RAG 的必要条件，检索质量才是」。
"""

from __future__ import annotations

import os
import time

import requests


class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_url = str(cfg.get("llm.base_url", "")).rstrip("/")
        self.model = str(cfg.get("llm.model", ""))
        self.key_env = str(cfg.get("llm.api_key_env", "OPENAI_API_KEY"))
        self.timeout = int(cfg.get("llm.timeout", 60))
        self.temperature = float(cfg.get("llm.temperature", 0.2))

    @property
    def available(self) -> bool:
        return bool(self.cfg.get("llm.enabled", False) and os.environ.get(self.key_env, "").strip())

    def chat(self, system: str, user: str) -> tuple[str, float]:
        if not self.available:
            raise RuntimeError("LLM 未启用或缺少 API Key")
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ[self.key_env]}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip(), (time.perf_counter() - t0) * 1000
