"""Combinatorial expansion: meta-template -> instruction shells.

Each shell still carries the ``{question}`` content slot — only the ``⟨...⟩``
placeholders are substituted. Whitespace/punctuation is normalised so that an
optional (empty-string) synonym does not leave a double space or a " ?" before
the content slot.
"""
from __future__ import annotations

import itertools
import random
import re
from typing import Iterator

from .schema import MetaTemplate, TemplatePool

_WS = re.compile(r"\s+")

# --- Deterministic "does it read after substitution?" guards -----------------
# Cheap, high-precision defect detectors run over EXPANDED shells. They attack
# the root cause of one-shot generation: synonyms emitted by lexical association
# that don't actually drop into the syntactic slot. They do NOT replace the LLM
# judge — subtle errors that need a parser ("think about upon", "answer to") fall
# through to the judge. These only catch the unambiguous, regex-checkable breakage.
_DOUBLE_WORD_RE = re.compile(r"\b(\w+)\s+\1\b")  # run on lower-cased text
_BROKEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    # auxiliary + subject + object-pronoun, e.g. "would you me by ..." — left by an
    # empty-string synonym dropped from a grammatically required verb slot.
    (re.compile(r"\b(would|could|can|will|do|does|did|shall|should)\s+(you|we|i|they)\s+(me|us|him|her|them|it)\b", re.I),
     "auxiliary directly followed by an object pronoun (likely an emptied verb slot)"),
    # imperative whose verb was emptied, e.g. "please on the image", "kindly to ...".
    (re.compile(r"^(please|kindly)\s+(on|to|of|in|at|by|for|with|the|a|an)\b", re.I),
     "imperative missing its verb (likely an emptied verb slot)"),
]


def expansion_issues(mt: MetaTemplate, max_check: int = 128, seed: int = 0) -> list[str]:
    """Detect unambiguous grammatical breakage in a meta-template's expansions.

    Checks every expansion when the product is small (<= ``max_check``); otherwise
    checks the two extremes (all-first / all-last synonym) plus a seeded random
    sample. Returns one issue string per distinct defect, with an example shell.
    """
    shells = _shells_to_check(mt, max_check, seed)
    found: dict[str, str] = {}
    for sh in shells:
        # Use non-word sentinels for content slots so a literal word before the slot
        # (e.g. "the question {question}") is not mis-flagged as a doubled word.
        text = sh.replace("{question}", "Qslot0").replace("{options}", "Oslot0")
        m = _DOUBLE_WORD_RE.search(text.lower())
        if m:
            found.setdefault(f"repeated adjacent word {m.group(1)!r}", sh)
        for rx, msg in _BROKEN_PATTERNS:
            if rx.search(text):
                found.setdefault(msg, sh)
    return [f"{mt.id}: {msg} (e.g. {ex!r})" for msg, ex in found.items()]


def _shells_to_check(mt: MetaTemplate, max_check: int, seed: int) -> list[str]:
    names = mt.unique_placeholder_names()
    if not names:
        return [normalize_shell(mt.skeleton)]
    if mt.variant_count() <= max_check:
        return list(iter_variants(mt))
    rng = random.Random(seed)
    combos = [
        tuple(_syn(mt, n)[0] for n in names),   # all-first
        tuple(_syn(mt, n)[-1] for n in names),  # all-last
    ]
    for _ in range(max_check):
        combos.append(tuple(rng.choice(_syn(mt, n)) for n in names))
    seen: set[str] = set()
    out: list[str] = []
    for combo in combos:
        sh = _substitute(mt.skeleton, names, combo)
        if sh not in seen:
            seen.add(sh)
            out.append(sh)
    return out


def normalize_shell(s: str) -> str:
    """Collapse whitespace and tidy spacing around punctuation."""
    s = _WS.sub(" ", s).strip()
    for p in (",", ".", "?", "!", ":", ";"):
        s = s.replace(f" {p}", p)
    return s


def _substitute(skeleton: str, names: list[str], combo: tuple[str, ...]) -> str:
    s = skeleton
    for name, val in zip(names, combo):
        s = s.replace(f"⟨{name}⟩", val)
    return normalize_shell(s)


def _syn(mt: MetaTemplate, name: str) -> list[str]:
    """Synonyms for a placeholder, robust to a skeleton referencing an undefined
    one (a malformed meta-template) — returns [""] so callers never KeyError.
    Structural validity is enforced separately by ``MetaTemplate.validate``."""
    ph = mt.placeholders.get(name)
    return (ph.synonyms or [""]) if ph else [""]


def iter_variants(mt: MetaTemplate) -> Iterator[str]:
    """Lazily yield every distinct expanded shell of a single meta-template."""
    names = mt.unique_placeholder_names()
    choices = [_syn(mt, n) for n in names]
    seen: set[str] = set()
    for combo in itertools.product(*choices):
        shell = _substitute(mt.skeleton, names, combo)
        if shell in seen:
            continue
        seen.add(shell)
        yield shell


def expand_meta_template(mt: MetaTemplate, cap: int | None = None) -> list[str]:
    out: list[str] = []
    for shell in iter_variants(mt):
        if cap is not None and len(out) >= cap:
            break
        out.append(shell)
    return out


def expand_pool(pool: TemplatePool, cap_per_template: int | None = None) -> list[dict]:
    """Materialise the whole pool.

    Returns one record per shell carrying provenance so downstream code (and the
    sampling tree) knows which skeleton / syntax node produced it. Dedups across
    the *entire* pool. Truncation by ``cap_per_template`` is never silent — use
    ``expand_pool_report`` to get the per-template dropped-count summary.
    """
    records, _ = expand_pool_report(pool, cap_per_template)
    return records


def expand_pool_report(
    pool: TemplatePool, cap_per_template: int | None = None
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    seen: set[str] = set()
    truncated: dict[str, int] = {}
    for mt in pool.meta_templates:
        full = mt.variant_count()
        kept = 0
        for shell in iter_variants(mt):
            if cap_per_template is not None and kept >= cap_per_template:
                truncated[mt.id] = full - kept
                break
            if shell in seen:
                continue
            seen.add(shell)
            kept += 1
            records.append(
                {
                    "shell": shell,
                    "meta_template_id": mt.id,
                    "syntax": mt.syntax,
                }
            )
    report = {
        "n_meta_templates": len(pool),
        "n_shells": len(records),
        "theoretical_variants": pool.total_variants(),
        "truncated": truncated,  # {mt_id: dropped_count} — explicit, never silent
    }
    return records, report
