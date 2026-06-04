"""Anthropic (Claude) backend."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .base import BaseLLMBackend, GenRequest
from .registry import register_backend


@register_backend("anthropic")
class AnthropicBackend(BaseLLMBackend):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic to use the 'anthropic' backend") from e

        key_env = self.config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(key_env) or self.config.get("api_key")
        self.client = Anthropic(api_key=api_key)
        self.model = self.config.get("model", "claude-sonnet-4-6")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.max_tokens = int(self.config.get("max_tokens", 4096))
        self.concurrency = int(self.config.get("concurrency", 4))

    def _one(self, r: GenRequest) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=r.system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": r.user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def complete(self, requests: list[GenRequest]) -> list[str]:
        if self.concurrency <= 1 or len(requests) <= 1:
            return [self._one(r) for r in requests]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._one, requests))
