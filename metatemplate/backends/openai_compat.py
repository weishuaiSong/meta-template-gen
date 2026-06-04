"""OpenAI-compatible Chat Completions backend.

One backend, two roles — switch purely via ``base_url``:

- **Cloud API**: leave ``base_url`` unset (or point at OpenAI), set ``OPENAI_API_KEY``.
- **Local vLLM on the A800**: start ``vllm serve <model> --port 8000`` and set
  ``base_url: http://localhost:8000/v1`` (api_key can be any non-empty string).

This is the cheapest path to "switchable backend": the local high-throughput
A800 server and a cloud API expose the same wire protocol.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .base import BaseLLMBackend, GenRequest
from .registry import register_backend


@register_backend("openai")
class OpenAICompatBackend(BaseLLMBackend):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use the 'openai' backend") from e

        base_url = self.config.get("base_url")  # None -> real OpenAI; set for vLLM serve
        key_env = self.config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(key_env) or self.config.get("api_key") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.max_tokens = int(self.config.get("max_tokens", 4096))
        self.concurrency = int(self.config.get("concurrency", 8))

    def _one(self, r: GenRequest) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": r.system},
                {"role": "user", "content": r.user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def complete(self, requests: list[GenRequest]) -> list[str]:
        if self.concurrency <= 1 or len(requests) <= 1:
            return [self._one(r) for r in requests]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._one, requests))
