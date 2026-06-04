"""LLM-based quality gate for meta-templates.

The mechanism: for each *meta-template*, fill its ⟨...⟩
placeholders to produce a few concrete *templates* (instruction shells), fill the
``{question}`` content slot with a bracketed example so the judge focuses on the
wrapper, and ask a (typically larger) judge model whether the phrasings are
reasonable — grammatical, natural, and image-agnostic. A meta-template is kept
only if its sampled templates pass.

The judge backend is independent of the generator backend, so you can generate
with a small model and judge with a big one (e.g. Qwen2.5-32B/72B-Instruct).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any

from .backends.base import BaseLLMBackend, GenRequest
from .expand import _substitute, _syn, normalize_shell
from .schema import MetaTemplate, TemplatePool

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

DEFAULT_EXAMPLE_QUESTIONS = [
    "What color is the object on the left?",
    "How many people are in the scene?",
    "Is there a clock in the room?",
    "What is the person on the right doing?",
]

JUDGE_SYSTEM = """You are a strict linguistic quality judge for visual-question-answering (VQA) instruction templates.

You are shown several concrete phrasings produced by ONE template. Each phrasing wraps a question; the actual question is fixed content supplied later, so here it is filled with an EXAMPLE shown in [square brackets]. Judge the WRAPPER (the words around the bracketed question), NOT the example question itself.

A phrasing is reasonable only if ALL of these hold:
- Grammatical and natural English.
- English only: reject if it contains any non-English characters (e.g. Chinese/Japanese/Korean).
- No redundancy: reject repeated or doubled words (e.g. "please please", "the the").
- It is a coherent instruction to answer a question about an image.
- Image-agnostic: it assumes no specific object/color/answer and works for ANY image. Reject wording that presumes a particular medium such as "chart", "diagram", "figure", "graph".
- The bracketed text is a STANDALONE complete question dropped in verbatim. Reject phrasings that use it as a verb's object/complement, e.g. "would you be able to [What color is the car?]" or "proceed to [What color is the car?]" — it must read as a full question appended to the wrapper.
- It contains no answer of its own and no second question.

Return ONLY a JSON object, no prose:
{"reasonable": <true|false>, "score": <integer 1-5>, "issues": [<short strings>]}
score: 5 = fluent and natural; 3 = acceptable but awkward; 1 = ungrammatical, leaks content, or not a valid wrapper. Mark reasonable=false if any sampled phrasing is broken."""


def fill_for_judge(shell: str, question: str, options: str = "(A) yes (B) no") -> str:
    """Fill content slots with bracketed examples so the judge ignores them."""
    return shell.replace("{question}", f"[{question}]").replace("{options}", f"[{options}]")


def random_variants(mt: MetaTemplate, k: int, rng: random.Random, include_extremes: bool = True) -> list[str]:
    """Up to k distinct expansions (templates) of one meta-template.

    With ``include_extremes`` (default), the all-first-synonym and all-last-synonym
    combos are included first. Pure random sampling can miss the "loud" combos that
    expose defects — e.g. a hardcoded "please" next to a ⟨polite⟩={Please,Kindly,""}
    slot only reads as "please please" when the slot picks "Please", which a random
    draw may skip. The extremes make such redundancy visible to the judge.
    """
    names = mt.unique_placeholder_names()
    target = min(k, mt.variant_count())
    out: list[str] = []
    seen: set[str] = set()

    def add(combo: tuple[str, ...]) -> None:
        shell = _substitute(mt.skeleton, names, combo)
        if shell not in seen:
            seen.add(shell)
            out.append(shell)

    if include_extremes and names:
        add(tuple(_syn(mt, n)[0] for n in names))
        add(tuple(_syn(mt, n)[-1] for n in names))

    attempts = 0
    while len(out) < target and attempts < target * 30 + 50:
        attempts += 1
        add(tuple(rng.choice(_syn(mt, n)) for n in names))
    return out[:target] if target else out


@dataclass
class Verdict:
    meta_template_id: str
    reasonable: bool
    score: float
    issues: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasonable": self.reasonable,
            "score": self.score,
            "issues": self.issues,
            "samples": self.samples,
        }


def _parse_verdict(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    candidates = [text]
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        candidates.append(text[s : e + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


class TemplateJudge:
    def __init__(
        self,
        backend: BaseLLMBackend,
        system_prompt: str = JUDGE_SYSTEM,
        example_questions: list[str] | None = None,
        n_variants: int = 3,
    ) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.example_questions = example_questions or DEFAULT_EXAMPLE_QUESTIONS
        self.n_variants = n_variants

    def _build_request(self, mt: MetaTemplate, rng: random.Random) -> tuple[GenRequest, list[str]]:
        variants = random_variants(mt, self.n_variants, rng)
        filled = [fill_for_judge(v, q) for v, q in zip(variants, cycle(self.example_questions))]
        listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(filled))
        user = (
            "Template skeleton (⟨...⟩ are word-choice slots; {question} is the fixed content slot):\n"
            f"{mt.skeleton}\n\n"
            "Example phrasings from this template (the [bracketed] part is the example question — ignore its content):\n"
            f"{listing}\n\n"
            "Judge the wrapper. Return ONLY the JSON object."
        )
        return GenRequest(system=self.system_prompt, user=user, metadata={"id": mt.id}), filled

    def judge_pool(self, pool: TemplatePool, seed: int = 0) -> list[Verdict]:
        """One batched judge call over the whole pool (backend handles concurrency)."""
        reqs: list[GenRequest] = []
        sample_map: list[list[str]] = []
        for i, mt in enumerate(pool.meta_templates):
            req, samples = self._build_request(mt, random.Random(seed + i))
            reqs.append(req)
            sample_map.append(samples)
        outs = self.backend.complete(reqs) if reqs else []
        verdicts: list[Verdict] = []
        for mt, out, samples in zip(pool.meta_templates, outs, sample_map):
            obj = _parse_verdict(out)
            try:
                score = float(obj.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            reasonable = bool(obj.get("reasonable", False))
            issues = obj.get("issues") or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            # If parsing failed entirely, mark unreasonable with a flag (fail-closed).
            if not obj:
                reasonable, issues = False, ["judge returned unparseable output"]
            verdicts.append(
                Verdict(
                    meta_template_id=mt.id,
                    reasonable=reasonable,
                    score=score,
                    issues=[str(x) for x in issues],
                    samples=samples,
                    raw=out or "",
                )
            )
        return verdicts

    def judge_one(self, mt: MetaTemplate, seed: int = 0) -> Verdict:
        return self.judge_pool(TemplatePool(meta_templates=[mt]), seed=seed)[0]


def filter_pool(
    pool: TemplatePool,
    verdicts: list[Verdict],
    min_score: float = 3.0,
    require_reasonable: bool = True,
) -> tuple[TemplatePool, list[Verdict]]:
    """Split a pool into (kept, rejected) by verdict. Annotates each kept
    meta-template's ``meta['judge']`` with its verdict for provenance."""
    by_id = {v.meta_template_id: v for v in verdicts}
    kept: list[MetaTemplate] = []
    rejected: list[Verdict] = []
    for mt in pool.meta_templates:
        v = by_id.get(mt.id)
        ok = v is not None and v.score >= min_score and (v.reasonable or not require_reasonable)
        if ok:
            mt.meta["judge"] = v.to_dict()
            kept.append(mt)
        elif v is not None:
            rejected.append(v)
    return TemplatePool(meta_templates=kept, meta=dict(pool.meta)), rejected
