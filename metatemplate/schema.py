"""Core data structures for the meta-template generation library.

- A **meta-template** is a syntactic skeleton carrying placeholders ``⟨h_j⟩``.
- Each placeholder owns a set of *position synonyms* (noun / verb / adjective /
  abstract function words).
- Combinatorial expansion over the synonym sets yields ``∏_j |s_j|`` distinct
  instruction shells.
- A content slot ``{question}`` (and optionally ``{options}`` etc.) marks where
  the *fixed* core question is injected at dataloader time. Content slots are
  left untouched by expansion — only the ``⟨...⟩`` placeholders vary. This keeps
  the templates **image-agnostic**: they are the "问法的外壳" and nothing else.

The two token families are deliberately disjoint so the format axis (lexical
variation in ``⟨...⟩``) never bleeds into the semantic content ({...}).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ⟨h_j⟩ — lexical-variation placeholders, expanded combinatorially.
PLACEHOLDER_RE = re.compile(r"⟨([A-Za-z0-9_]+)⟩")
# {question} — content slots, filled at dataloader time, never expanded here.
CONTENT_SLOT_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
# Templates must be English-only. Small generators occasionally code-switch into
# CJK (observed: "your见解", fullwidth "？") under high temperature — catch it
# deterministically here instead of spending judge tokens on it. Covers CJK
# ideographs, kana, Hangul, and fullwidth/CJK-symbol forms.
_NON_ENGLISH_RE = re.compile(
    r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]"
)


def find_non_english(text: str) -> list[str]:
    """Return the distinct non-English (CJK / kana / fullwidth) characters in text."""
    return sorted(set(_NON_ENGLISH_RE.findall(text)))

REQUIRED_CONTENT_SLOT = "question"

VALID_POS = {"noun", "verb", "adjective", "adverb", "function"}
VALID_MOODS = {"declarative", "imperative", "interrogative"}
VALID_COMPLEXITY = {"simple", "complex"}


@dataclass
class Placeholder:
    """A lexical-variation slot ``⟨name⟩`` with its set of position synonyms."""

    name: str
    synonyms: list[str]
    pos: str = "function"  # noun | verb | adjective | adverb | function

    def __post_init__(self) -> None:
        # De-duplicate synonyms case-insensitively while preserving order; keep
        # the empty string if present (it models an *optional* word, e.g. an
        # omittable "please").
        seen: set[str] = set()
        out: list[str] = []
        for s in self.synonyms:
            s2 = s.strip()
            key = s2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s2)
        self.synonyms = out

    def size(self) -> int:
        return max(1, len(self.synonyms))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "synonyms": self.synonyms, "pos": self.pos}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Placeholder":
        return cls(name=d["name"], synonyms=list(d.get("synonyms", [])), pos=d.get("pos", "function"))


@dataclass
class MetaTemplate:
    """A syntactic skeleton + its placeholder synonym sets + syntax tags."""

    id: str
    skeleton: str
    placeholders: dict[str, Placeholder] = field(default_factory=dict)
    # syntax: {"mood": ..., "complexity": ..., "clause": ...} — the four-level
    # sampling tree (mood -> complexity -> clause -> template) reads these.
    syntax: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def placeholder_names(self) -> list[str]:
        """Placeholder names in order of first appearance in the skeleton."""
        return PLACEHOLDER_RE.findall(self.skeleton)

    def unique_placeholder_names(self) -> list[str]:
        return list(dict.fromkeys(self.placeholder_names()))

    def content_slots(self) -> list[str]:
        return CONTENT_SLOT_RE.findall(self.skeleton)

    def variant_count(self) -> int:
        """``∏_j |s_j|`` — number of distinct lexical variants this skeleton expands to."""
        n = 1
        for name in self.unique_placeholder_names():
            ph = self.placeholders.get(name)
            n *= ph.size() if ph else 1
        return n

    def validate(self) -> list[str]:
        """Return a list of structural problems (empty == valid)."""
        errs: list[str] = []
        names = set(self.placeholder_names())
        for name in names:
            if name not in self.placeholders:
                errs.append(f"{self.id}: skeleton uses ⟨{name}⟩ with no synonym set")
        for name, ph in self.placeholders.items():
            if name not in names:
                errs.append(f"{self.id}: placeholder ⟨{name}⟩ defined but unused in skeleton")
            if not ph.synonyms:
                errs.append(f"{self.id}: placeholder ⟨{name}⟩ has an empty synonym set")
            if ph.pos not in VALID_POS:
                errs.append(f"{self.id}: placeholder ⟨{name}⟩ has invalid pos {ph.pos!r}")
        if REQUIRED_CONTENT_SLOT not in self.content_slots():
            errs.append(f"{self.id}: missing required content slot {{{REQUIRED_CONTENT_SLOT}}}")
        # English-only: flag CJK / kana / fullwidth leakage in skeleton or synonyms.
        bad = find_non_english(self.skeleton)
        for ph in self.placeholders.values():
            for syn in ph.synonyms:
                bad.extend(find_non_english(syn))
        if bad:
            errs.append(f"{self.id}: contains non-English characters {sorted(set(bad))}")
        mood = self.syntax.get("mood")
        if mood is not None and mood not in VALID_MOODS:
            errs.append(f"{self.id}: invalid mood {mood!r}")
        cx = self.syntax.get("complexity")
        if cx is not None and cx not in VALID_COMPLEXITY:
            errs.append(f"{self.id}: invalid complexity {cx!r}")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skeleton": self.skeleton,
            "placeholders": {k: v.to_dict() for k, v in self.placeholders.items()},
            "syntax": self.syntax,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetaTemplate":
        phs = d.get("placeholders", {})
        # Accept either {name: {...}} or [{name, ...}, ...].
        if isinstance(phs, list):
            phs = {p["name"]: p for p in phs}
        return cls(
            id=d["id"],
            skeleton=d["skeleton"],
            placeholders={k: Placeholder.from_dict(v) for k, v in phs.items()},
            syntax=dict(d.get("syntax", {})),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class TemplatePool:
    """A collection of meta-templates plus convenience IO / reporting."""

    meta_templates: list[MetaTemplate] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.meta_templates)

    def total_variants(self) -> int:
        return sum(mt.variant_count() for mt in self.meta_templates)

    def validate(self) -> list[str]:
        errs: list[str] = []
        for mt in self.meta_templates:
            errs.extend(mt.validate())
        # Cross-template duplicate skeletons (after collapsing whitespace).
        seen: dict[str, str] = {}
        for mt in self.meta_templates:
            key = re.sub(r"\s+", " ", mt.skeleton).strip()
            if key in seen:
                errs.append(f"{mt.id}: duplicate skeleton also in {seen[key]}")
            else:
                seen[key] = mt.id
        return errs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "meta_templates": [mt.to_dict() for mt in self.meta_templates]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TemplatePool":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):  # bare list of meta-templates
            payload = {"meta_templates": payload}
        return cls(
            meta_templates=[MetaTemplate.from_dict(d) for d in payload.get("meta_templates", [])],
            meta=dict(payload.get("meta", {})),
        )
