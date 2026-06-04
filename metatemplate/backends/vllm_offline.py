"""Offline vLLM backend — high-throughput local generation on the A800.

Loads the model in-process (no server) and batches the whole request list through
``LLM.chat``. This is the fastest path for generating large meta-template pools on
8x A800-80GB. Use the ``openai`` backend with a ``base_url`` instead if you prefer
to run ``vllm serve`` as a separate process.

Config keys: ``model`` (HF id or local path), ``tensor_parallel_size``,
``gpu_memory_utilization``, ``max_model_len``, ``temperature``, ``top_p``,
``max_tokens``, ``trust_remote_code``.
"""
from __future__ import annotations

from typing import Any

from .base import BaseLLMBackend, GenRequest
from .registry import register_backend


@register_backend("vllm")
class VLLMOfflineBackend(BaseLLMBackend):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install vllm to use the 'vllm' backend") from e

        self.model = self.config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=int(self.config.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_utilization", 0.90)),
            max_model_len=self.config.get("max_model_len"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
        )
        self._SamplingParams = SamplingParams
        self.sampling = SamplingParams(
            temperature=float(self.config.get("temperature", 1.0)),
            top_p=float(self.config.get("top_p", 0.95)),
            max_tokens=int(self.config.get("max_tokens", 4096)),
            seed=self.config.get("seed"),
        )

    def complete(self, requests: list[GenRequest]) -> list[str]:
        conversations = [
            [
                {"role": "system", "content": r.system},
                {"role": "user", "content": r.user},
            ]
            for r in requests
        ]
        outputs = self.llm.chat(conversations, self.sampling, use_tqdm=len(requests) > 4)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]
