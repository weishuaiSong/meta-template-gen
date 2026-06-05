"""Prompt construction for meta-template generation.

We ask the LLM to emit a JSON array of meta-templates, each being an
**image-agnostic** instruction shell (the "问法的外壳"): a syntactic skeleton with
``⟨h_j⟩`` lexical-variation placeholders, a synonym set per placeholder, and
syntax tags (mood / complexity / clause). The single required content slot
``{question}`` marks where the fixed core question is injected later — the model
must never describe image content or invent an answer.
"""
from __future__ import annotations

import json
from typing import Any

DEFAULT_SYSTEM = """You design *meta-templates* for phrasing visual-question-answering instructions.

A meta-template is a SYNTACTIC SKELETON for the *wrapper* around a question — it is \
image-agnostic and content-free. It must:
1. Contain exactly one content slot written as {question}. It is filled later with a \
   STANDALONE, COMPLETE question such as "What color is the car?", so the wrapper must stay \
   grammatical when a full interrogative is dropped in. Normally APPEND it (end with ": {question}", \
   or place "{question}" as its own clause/sentence). Do NOT use {question} as a verb complement \
   or object — e.g. "would you be able to {question}" or "could you proceed to {question}" break \
   because they become "would you be able to What color is the car?". Optionally {options} for \
   multiple-choice phrasing. Never write an actual question yourself.
2. Use lexical-variation placeholders written as ⟨name⟩ (angle brackets U+27E8/U+27E9), \
   e.g. ⟨polite⟩, ⟨look⟩, ⟨image⟩. Each placeholder is a position whose word can vary.
3. Provide a synonym set of 4-6 interchangeable options for EVERY placeholder (plus "" \
   when the word is truly optional, e.g. an omittable "please"). Every synonym must read \
   grammatically when dropped into that exact slot — mentally substitute each one into the \
   full sentence before listing it; discard any that do not fit.
4. Be tagged with syntax: mood ∈ {declarative, imperative, interrogative}; \
   complexity ∈ {simple, complex}; clause (e.g. "none", "relative", "conditional", "purpose").

Hard rules:
- NEVER mention image content, objects, colors, or any answer. The skeleton is generic.
- Keep placeholders purely lexical (word-choice), not semantic content.
- Vary syntax across the set: mix moods, simple/complex, different clause types.
- {question} is a full question dropped in verbatim — keep the wrapper grammatical around it; append it, never make it a verb's object.
- Output ONLY a JSON array. No prose, no markdown fences."""

_EXAMPLE = [
    {
        "skeleton": "⟨polite⟩ ⟨look⟩ the ⟨image⟩ and ⟨answer⟩ the following question: {question}",
        "placeholders": {
            "polite": {"synonyms": ["Please", "Kindly", "Carefully", ""], "pos": "function"},
            "look": {"synonyms": ["examine", "look at", "study", "inspect", "observe"], "pos": "verb"},
            "image": {"synonyms": ["image", "picture", "photo", "photograph", "snapshot"], "pos": "noun"},
            "answer": {"synonyms": ["answer", "respond to", "address", "reply to", "tackle"], "pos": "verb"},
        },
        "syntax": {"mood": "imperative", "complexity": "simple", "clause": "none"},
    },
    {
        "skeleton": "Based on what is shown in the ⟨image⟩, how would you ⟨answer⟩ the question that ⟨follows⟩: {question}",
        "placeholders": {
            "image": {"synonyms": ["image", "picture", "scene", "photo", "photograph"], "pos": "noun"},
            "answer": {"synonyms": ["answer", "respond to", "address", "approach"], "pos": "verb"},
            "follows": {"synonyms": ["follows", "comes next", "is given below", "appears below", "is stated below"], "pos": "verb"},
        },
        "syntax": {"mood": "interrogative", "complexity": "complex", "clause": "relative"},
    },
]


def output_schema_hint() -> str:
    return json.dumps(_EXAMPLE, ensure_ascii=False, indent=2)


def build_user_prompt(
    n: int,
    avoid_skeletons: list[str] | None = None,
    syntax_target: dict[str, Any] | None = None,
    extra_guidance: str | None = None,
) -> str:
    parts: list[str] = [
        f"Generate {n} NEW, mutually distinct meta-templates as a JSON array.",
        "Each element must have keys: skeleton, placeholders, syntax.",
        "Follow exactly this shape (these are EXAMPLES — do not copy them):",
        output_schema_hint(),
    ]
    if syntax_target:
        parts.append(
            "Bias the syntax of this batch toward: "
            + ", ".join(f"{k}={v}" for k, v in syntax_target.items())
            + "."
        )
    if avoid_skeletons:
        shown = avoid_skeletons[:60]
        parts.append(
            "Make them MAXIMALLY different from these already-generated skeletons "
            "(different verbs, sentence structure, clause types — not trivial rewordings):\n- "
            + "\n- ".join(shown)
        )
    if extra_guidance:
        parts.append(extra_guidance)
    parts.append("Return ONLY the JSON array.")
    return "\n\n".join(parts)
