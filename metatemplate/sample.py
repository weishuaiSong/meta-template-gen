"""Sampling a template pool of size N from the meta-templates.

Implements "four-level sentence-pattern tree, weighted top-down" sampling
(mood -> complexity -> clause -> meta-template). We expose three weighting modes;
the default ``"variant"`` is mathematically identical to a top-down scheme where
each node's weight equals the number of templates its subtree can generate
(which yields a *uniform* distribution over the final variants).

Sampling is lazy: we never materialise the full ∏|s_j| product. We draw a
meta-template by weight, then draw one synonym per placeholder uniformly — this
is exactly a uniform draw over that template's variants. Dedup is applied so the
returned pool has N *distinct* shells. Everything is seeded for reproducibility.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .expand import _substitute, _syn
from .schema import MetaTemplate, TemplatePool

WeightMode = str  # "variant" | "template" | "balanced"


def _syntax_key(mt: MetaTemplate) -> tuple[str, str, str]:
    s = mt.syntax
    return (
        str(s.get("mood", "unknown")),
        str(s.get("complexity", "unknown")),
        str(s.get("clause", "none")),
    )


def _draw_variant(mt: MetaTemplate, rng: random.Random) -> str:
    names = mt.unique_placeholder_names()
    combo = tuple(rng.choice(_syn(mt, n)) for n in names)
    return _substitute(mt.skeleton, names, combo)


def compute_weights(meta_templates: list[MetaTemplate], mode: WeightMode) -> list[float]:
    """Per-meta-template sampling weight under the chosen mode.

    - ``variant``  : weight = variant_count  -> uniform over all expanded shells
                     (equivalent to "subtree size" top-down weighting).
    - ``template`` : weight = 1              -> uniform over meta-templates.
    - ``balanced`` : uniform across (mood, complexity, clause) nodes, then
                     uniform across templates within a node (de-biases skeletons
                     that happen to have huge synonym sets).
    """
    if mode == "template":
        return [1.0] * len(meta_templates)
    if mode == "variant":
        return [float(mt.variant_count()) for mt in meta_templates]
    if mode == "balanced":
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, mt in enumerate(meta_templates):
            groups[_syntax_key(mt)].append(i)
        n_groups = len(groups)
        weights = [0.0] * len(meta_templates)
        for idxs in groups.values():
            per = 1.0 / (n_groups * len(idxs))
            for i in idxs:
                weights[i] = per
        return weights
    raise ValueError(f"unknown weight mode {mode!r}; use variant|template|balanced")


def sample_pool(
    pool: TemplatePool,
    n: int,
    seed: int = 0,
    mode: WeightMode = "variant",
    max_attempts_factor: int = 50,
) -> list[dict]:
    """Sample N distinct instruction shells. Records carry provenance + syntax.

    If the pool cannot supply N distinct shells (``N > total_variants``), returns
    every distinct shell it can and the caller can detect the shortfall via the
    returned length.
    """
    rng = random.Random(seed)
    mts = pool.meta_templates
    if not mts:
        return []
    weights = compute_weights(mts, mode)
    target = min(n, pool.total_variants())
    results: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    cap_attempts = target * max_attempts_factor + 1000
    while len(results) < target and attempts < cap_attempts:
        attempts += 1
        mt = rng.choices(mts, weights=weights, k=1)[0]
        shell = _draw_variant(mt, rng)
        if shell in seen:
            continue
        seen.add(shell)
        results.append({"shell": shell, "meta_template_id": mt.id, "syntax": mt.syntax})
    return results


def syntax_histogram(pool: TemplatePool) -> dict[str, dict[str, int]]:
    """Counts of meta-templates and theoretical variants per syntax level."""
    out: dict[str, dict[str, int]] = {"mood": defaultdict(int), "complexity": defaultdict(int), "clause": defaultdict(int)}
    var: dict[str, dict[str, int]] = {"mood": defaultdict(int), "complexity": defaultdict(int), "clause": defaultdict(int)}
    for mt in pool.meta_templates:
        mood, cx, clause = _syntax_key(mt)
        vc = mt.variant_count()
        for level, val in (("mood", mood), ("complexity", cx), ("clause", clause)):
            out[level][val] += 1
            var[level][val] += vc
    return {
        "templates_per_level": {k: dict(v) for k, v in out.items()},
        "variants_per_level": {k: dict(v) for k, v in var.items()},
    }
