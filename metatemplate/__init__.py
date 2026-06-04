"""metatemplate — LLM-driven generation of instruction meta-templates.

Meta-templates are syntactic skeletons with ⟨placeholder⟩ synonym sets, expanded
combinatorially and sampled via a weighted four-level sentence-pattern tree. The
skeletons themselves are authored by a switchable LLM backend instead of by hand.
"""
from __future__ import annotations

from .backends import build_backend, list_backends
from .expand import (
    expand_meta_template,
    expand_pool,
    expand_pool_report,
    expansion_issues,
    iter_variants,
    normalize_shell,
)
from .generate import GenerationResult, MetaTemplateGenerator, parse_json_array
from .judge import TemplateJudge, Verdict, filter_pool, random_variants
from .sample import compute_weights, sample_pool, syntax_histogram
from .schema import MetaTemplate, Placeholder, TemplatePool
from .validate import diversity_report, leakage_report, structural_report

__version__ = "0.0.1"

__all__ = [
    "MetaTemplate",
    "Placeholder",
    "TemplatePool",
    "MetaTemplateGenerator",
    "GenerationResult",
    "parse_json_array",
    "build_backend",
    "list_backends",
    "expand_meta_template",
    "expand_pool",
    "expand_pool_report",
    "expansion_issues",
    "iter_variants",
    "normalize_shell",
    "sample_pool",
    "compute_weights",
    "syntax_histogram",
    "structural_report",
    "leakage_report",
    "diversity_report",
    "TemplateJudge",
    "Verdict",
    "filter_pool",
    "random_variants",
]
