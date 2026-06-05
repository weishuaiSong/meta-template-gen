"""Parallel-generation fan-out tests — stub backend, no real LLM."""
from __future__ import annotations

import json

from metatemplate import MetaTemplateGenerator
from metatemplate.backends.base import BaseLLMBackend


class RecordingBackend(BaseLLMBackend):
    """Returns one unique meta-template per request; records request batches."""

    name = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.user_prompts: list[list[str]] = []
        self._i = 0

    def complete(self, requests):
        self.batch_sizes.append(len(requests))
        self.user_prompts.append([r.user for r in requests])
        outs = []
        for _ in requests:
            self._i += 1
            outs.append(json.dumps([{
                "skeleton": f"Please look at the ⟨img⟩ angle {self._i} and answer: {{question}}",
                "placeholders": {"img": {"synonyms": ["image", "picture", "photo", "photograph"]}},
                "syntax": {"mood": "imperative", "complexity": "simple", "clause": "none"},
            }]))
        return outs


def test_parallel_fans_out_requests():
    be = RecordingBackend()
    result = MetaTemplateGenerator(be).generate(target_count=6, batch_size=6, parallel=3, max_rounds=5)
    assert be.batch_sizes[0] == 3          # 3 concurrent requests per round
    assert len(result.pool) == 6           # reaches the target (2 rounds x 3)
    assert result.rounds == 2


def test_parallel_default_is_single_request():
    be = RecordingBackend()
    MetaTemplateGenerator(be).generate(target_count=2, batch_size=2, max_rounds=3)
    assert all(n == 1 for n in be.batch_sizes)  # back-compat: one request per round


def test_parallel_rotates_syntax_targets():
    be = RecordingBackend()
    MetaTemplateGenerator(be).generate(target_count=4, batch_size=4, parallel=4, max_rounds=3)
    prompts = be.user_prompts[0]
    bias_lines = {next((ln for ln in p.split("\n") if "Bias the syntax" in ln), "") for p in prompts}
    assert len(bias_lines) >= 3            # distinct syntax targets across the fan-out


def test_parallel_respects_explicit_syntax_target():
    be = RecordingBackend()
    MetaTemplateGenerator(be).generate(
        target_count=2, batch_size=2, parallel=2, max_rounds=3,
        syntax_target={"mood": "imperative"},
    )
    for p in be.user_prompts[0]:
        assert "mood=imperative" in p      # explicit target applies to every request


def test_parallel_dedups_across_requests():
    class SameOutput(BaseLLMBackend):
        name = "same"
        def complete(self, requests):
            tpl = json.dumps([{
                "skeleton": "Please look at the ⟨img⟩ and answer: {question}",
                "placeholders": {"img": {"synonyms": ["image", "picture"]}},
                "syntax": {"mood": "imperative", "complexity": "simple", "clause": "none"},
            }])
            return [tpl] * len(requests)

    result = MetaTemplateGenerator(SameOutput()).generate(
        target_count=4, batch_size=4, parallel=4, max_rounds=2, stall_patience=1,
    )
    assert len(result.pool) == 1           # identical outputs collapse to one
    assert result.n_duplicate >= 3
