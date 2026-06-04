"""Local HuggingFace ``transformers`` backend.

In-process model load + batched ``generate``. Useful when vLLM is not installed
(e.g. on NVIDIA NGC images whose pinned torch vLLM would clobber): this backend
reuses whatever torch is already present.

Point ``model`` at a HF id or a local path. In networks where huggingface.co is
unreachable, pre-download with ModelScope and pass the local snapshot path.

Config: ``model``, ``device_map`` (default "auto"), ``dtype`` ("bfloat16"),
``temperature``, ``top_p``, ``max_tokens``, ``batch_size``, ``trust_remote_code``.
"""
from __future__ import annotations

from typing import Any

from .base import BaseLLMBackend, GenRequest
from .registry import register_backend


@register_backend("transformers")
class TransformersBackend(BaseLLMBackend):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install transformers accelerate to use the 'transformers' backend") from e

        self._torch = torch
        model = self.config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        trust = bool(self.config.get("trust_remote_code", True))
        dtype = getattr(torch, self.config.get("dtype", "bfloat16"), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust)
        # Left padding is required for correct batched decoder-only generation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=trust,
        )
        self.model.eval()
        self.max_new_tokens = int(self.config.get("max_tokens", 2048))
        self.temperature = float(self.config.get("temperature", 1.0))
        self.top_p = float(self.config.get("top_p", 0.95))
        self.batch_size = int(self.config.get("batch_size", 8))

    def complete(self, requests: list[GenRequest]) -> list[str]:
        torch = self._torch
        outs: list[str] = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": r.system},
                        {"role": "user", "content": r.user},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in chunk
            ]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
            do_sample = self.temperature > 0
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    top_p=self.top_p if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                new_tokens = gen[j][prompt_len:]
                outs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outs
