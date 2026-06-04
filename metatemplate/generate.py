"""Drive an LLM backend to produce a validated, de-duplicated pool of meta-templates.

The generator runs a *diversity feedback loop*: each round it asks for a batch,
keeps the valid + novel ones, and feeds the accumulated skeletons back into the
prompt so the next round is pushed toward genuinely new syntax — repeating until
the target count is reached or progress stalls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .backends.base import BaseLLMBackend, GenRequest
from .expand import expansion_issues
from .prompts import DEFAULT_SYSTEM, build_user_prompt
from .schema import MetaTemplate, Placeholder, TemplatePool

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _norm_skeleton(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array of objects from a model response, tolerating fences
    and leading/trailing prose."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Try direct parse, else carve out the outermost [...] span.
    for candidate in (text, _carve_array(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict) and isinstance(obj.get("meta_templates"), list):
            return [x for x in obj["meta_templates"] if isinstance(x, dict)]
    return []


def _carve_array(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _coerce_placeholders(raw: Any) -> dict[str, Placeholder]:
    out: dict[str, Placeholder] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((p.get("name"), p) for p in raw if isinstance(p, dict))
    else:
        return out
    for name, spec in items:
        if not name or not isinstance(spec, dict):
            continue
        syns = spec.get("synonyms") or spec.get("synonym") or []
        if isinstance(syns, str):
            syns = [syns]
        out[name] = Placeholder(name=name, synonyms=[str(s) for s in syns], pos=spec.get("pos", "function"))
    return out


@dataclass
class GenerationResult:
    pool: TemplatePool
    n_requested: int
    n_raw: int = 0          # total objects returned by the LLM
    n_invalid: int = 0      # dropped on structural validation
    n_duplicate: int = 0    # dropped as dup of an existing/earlier skeleton
    n_judged: int = 0       # candidates sent to the LLM judge
    n_rejected_judge: int = 0  # dropped by the LLM judge quality gate
    rounds: int = 0
    stalled: bool = False
    invalid_samples: list[dict] = field(default_factory=list)
    rejected_samples: list[dict] = field(default_factory=list)


class MetaTemplateGenerator:
    def __init__(
        self,
        backend: BaseLLMBackend,
        system_prompt: str = DEFAULT_SYSTEM,
        id_prefix: str = "mt",
    ) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.id_prefix = id_prefix

    def generate(
        self,
        target_count: int,
        batch_size: int = 20,
        seed_pool: TemplatePool | None = None,
        avoid_shells: list[str] | None = None,
        syntax_target: dict[str, Any] | None = None,
        max_rounds: int = 50,
        stall_patience: int = 3,
        judge: Any | None = None,
        judge_min_score: float = 3.0,
        judge_require_reasonable: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        """Generate until ``target_count`` distinct valid meta-templates exist.

        ``seed_pool``    : start from an existing pool (its skeletons count as taken).
        ``avoid_shells`` : extra skeletons/instruction shells to stay disjoint from
                           (e.g. a held-out OOD set — keeps the generator space clean).
        ``stall_patience``: stop after this many consecutive rounds adding nothing new.
        ``judge``        : optional ``TemplateJudge`` — each round's structurally-valid
                           candidates are LLM-judged and only those scoring
                           ``>= judge_min_score`` (and reasonable, if required) are kept.
        """
        log = progress or (lambda _m: None)
        kept: list[MetaTemplate] = list(seed_pool.meta_templates) if seed_pool else []
        seen: set[str] = {_norm_skeleton(mt.skeleton) for mt in kept}
        forbidden = {_norm_skeleton(s) for s in (avoid_shells or [])}

        result = GenerationResult(pool=TemplatePool(meta_templates=kept), n_requested=target_count)
        counter = len(kept)
        stale_rounds = 0

        while len(kept) < target_count and result.rounds < max_rounds:
            result.rounds += 1
            need = target_count - len(kept)
            ask = min(batch_size, need)
            # show the model a sample of what already exists to push for novelty
            avoid_list = [mt.skeleton for mt in kept[-80:]] + list(avoid_shells or [])[:40]
            user = build_user_prompt(ask, avoid_skeletons=avoid_list, syntax_target=syntax_target)

            responses = self.backend.complete([GenRequest(system=self.system_prompt, user=user)])
            objs = parse_json_array(responses[0]) if responses else []
            result.n_raw += len(objs)

            # Structurally validate this round's candidates first.
            candidates: list[MetaTemplate] = []
            for obj in objs:
                skeleton = obj.get("skeleton")
                if not isinstance(skeleton, str) or not skeleton.strip():
                    result.n_invalid += 1
                    continue
                key = _norm_skeleton(skeleton)
                if key in seen or key in forbidden:
                    result.n_duplicate += 1
                    continue
                counter += 1
                mt = MetaTemplate(
                    id=f"{self.id_prefix}_{counter:04d}",
                    skeleton=skeleton.strip(),
                    placeholders=_coerce_placeholders(obj.get("placeholders")),
                    syntax=dict(obj.get("syntax", {})),
                    meta={"generated": True, "backend": self.backend.name, "round": result.rounds},
                )
                # Structural validity first; only run the (substitution-based)
                # expansion guards on a well-formed template so a skeleton that
                # references an undefined placeholder is reported, not crashed on.
                errs = mt.validate()
                if not errs:
                    errs = expansion_issues(mt)
                if errs:
                    result.n_invalid += 1
                    if len(result.invalid_samples) < 25:
                        result.invalid_samples.append({"skeleton": skeleton, "errors": errs})
                    counter -= 1
                    continue
                seen.add(key)  # mark seen even if the judge later rejects, so we don't re-judge it
                candidates.append(mt)

            # Optional LLM quality gate: keep only candidates that pass the judge.
            if judge is not None and candidates:
                from .judge import filter_pool  # local import avoids a hard dependency cycle

                verdicts = judge.judge_pool(TemplatePool(meta_templates=candidates), seed=result.rounds)
                result.n_judged += len(candidates)
                passed_pool, rejected = filter_pool(
                    TemplatePool(meta_templates=candidates),
                    verdicts,
                    min_score=judge_min_score,
                    require_reasonable=judge_require_reasonable,
                )
                result.n_rejected_judge += len(rejected)
                for v in rejected:
                    if len(result.rejected_samples) < 25:
                        result.rejected_samples.append(
                            {"id": v.meta_template_id, "score": v.score, "issues": v.issues, "samples": v.samples}
                        )
                candidates = passed_pool.meta_templates

            added_this_round = 0
            for mt in candidates:
                if len(kept) >= target_count:
                    break
                kept.append(mt)
                added_this_round += 1

            log(f"round {result.rounds}: +{added_this_round} (have {len(kept)}/{target_count})")
            stale_rounds = stale_rounds + 1 if added_this_round == 0 else 0
            if stale_rounds >= stall_patience:
                result.stalled = True
                log(f"stalled after {stale_rounds} empty rounds; stopping at {len(kept)}")
                break

        result.pool = TemplatePool(
            meta_templates=kept,
            meta={"target_count": target_count, "produced": len(kept), "backend": self.backend.name},
        )
        return result
