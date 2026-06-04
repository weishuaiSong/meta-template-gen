"""Pool-level validation, leakage checking, and diversity reporting.

- *leakage check* keeps the generated pool disjoint from any held-out / OOD
  template set (train/eval separation for the template space).
- *diversity report* exposes an embedding hook (Vendi Score) so pool diversity
  can be measured rather than approximated by naive template counting.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .expand import expansion_issues, normalize_shell
from .sample import syntax_histogram
from .schema import TemplatePool


def structural_report(pool: TemplatePool) -> dict[str, Any]:
    errs = pool.validate()
    # Deterministic post-substitution grammar guards (doubled words, emptied slots).
    exp_errs: list[str] = []
    for mt in pool.meta_templates:
        exp_errs.extend(expansion_issues(mt))
    return {
        "n_meta_templates": len(pool),
        "total_variants": pool.total_variants(),
        "n_structural_errors": len(errs),
        "errors": errs[:100],
        "n_expansion_errors": len(exp_errs),
        "expansion_errors": exp_errs[:100],
        "syntax": syntax_histogram(pool),
    }


def leakage_report(shells: Sequence[str], held_out: Sequence[str]) -> dict[str, Any]:
    """Normalised-shell overlap between a generated set and a held-out set."""
    gen = {normalize_shell(s).lower() for s in shells}
    held = {normalize_shell(s).lower() for s in held_out}
    overlap = sorted(gen & held)
    return {
        "n_generated": len(gen),
        "n_held_out": len(held),
        "n_overlap": len(overlap),
        "overlap_rate": (len(overlap) / len(gen)) if gen else 0.0,
        "examples": overlap[:25],
    }


def diversity_report(
    shells: Sequence[str],
    embed_fn: Callable[[Sequence[str]], Any] | None = None,
) -> dict[str, Any]:
    """Lexical diversity now; semantic diversity (Vendi/NovelSum) via ``embed_fn``.

    ``embed_fn`` maps a list of strings to an (n, d) embedding matrix; when given,
    we compute the Vendi Score (exp of the Shannon entropy of the normalised
    Gram-matrix eigenvalues) as a measured diversity score for the pool. Without
    it, only cheap surface statistics are returned.
    """
    n = len(shells)
    distinct = len({normalize_shell(s).lower() for s in shells})
    tokens = [len(s.split()) for s in shells]
    report: dict[str, Any] = {
        "n": n,
        "n_distinct": distinct,
        "distinct_ratio": (distinct / n) if n else 0.0,
        "avg_tokens": (sum(tokens) / n) if n else 0.0,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "vendi_score": None,
    }
    if embed_fn is not None and n > 1:
        report["vendi_score"] = _vendi_score(embed_fn(list(shells)))
    return report


def _vendi_score(embeddings: Any) -> float:
    """Vendi Score = exp(H) of the eigenvalue distribution of the normalised
    similarity Gram matrix (cosine kernel). Requires numpy."""
    import numpy as np

    x = np.asarray(embeddings, dtype="float64")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    xn = x / norms
    n = xn.shape[0]
    k = (xn @ xn.T) / n  # normalised so trace == 1
    eigvals = np.linalg.eigvalsh(k)
    eigvals = eigvals[eigvals > 1e-12]
    entropy = -float(np.sum(eigvals * np.log(eigvals)))
    return float(np.exp(entropy))
