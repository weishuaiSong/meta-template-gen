"""Judge tests — use a stub backend, no real LLM."""
from __future__ import annotations

import json

from metatemplate import MetaTemplate, Placeholder, TemplatePool, TemplateJudge, filter_pool
from metatemplate.backends.base import BaseLLMBackend
from metatemplate.judge import _parse_verdict, fill_for_judge, random_variants


def _mt(id_="mt_0001") -> MetaTemplate:
    return MetaTemplate(
        id=id_,
        skeleton="⟨polite⟩ ⟨look⟩ the ⟨image⟩ and answer: {question}",
        placeholders={
            "polite": Placeholder("polite", ["Please", "Kindly", ""], "function"),
            "look": Placeholder("look", ["examine", "look at"], "verb"),
            "image": Placeholder("image", ["image", "picture", "photo"], "noun"),
        },
        syntax={"mood": "imperative", "complexity": "simple", "clause": "none"},
    )


class ScriptedJudge(BaseLLMBackend):
    """Returns a verdict keyed by what it sees in the prompt."""

    name = "scripted"

    def complete(self, requests):
        out = []
        for r in requests:
            if "bad" in r.user:
                out.append('{"reasonable": false, "score": 1, "issues": ["ungrammatical"]}')
            else:
                out.append('```json\n{"reasonable": true, "score": 5, "issues": []}\n```')
        return out


def test_fill_for_judge_brackets_question():
    assert fill_for_judge("answer: {question}", "How many?") == "answer: [How many?]"
    assert "{question}" not in fill_for_judge("x {question} y {options}", "q")


def test_random_variants_distinct_and_capped():
    mt = _mt()
    vs = random_variants(mt, 100, __import__("random").Random(0))
    assert len(vs) == len(set(vs)) <= mt.variant_count()
    assert all("{question}" in v for v in vs)


def test_random_variants_includes_loud_combo():
    # ⟨polite⟩ first synonym is "Please" + a hardcoded "please" -> "Please please"
    mt = MetaTemplate(
        id="x",
        skeleton="⟨polite⟩ please answer: {question}",
        placeholders={"polite": Placeholder("polite", ["Please", "Kindly", ""])},
    )
    vs = random_variants(mt, 3, __import__("random").Random(0), include_extremes=True)
    assert any(v.lower().startswith("please please") for v in vs)


def test_parse_verdict_fenced_and_carved():
    assert _parse_verdict('```json\n{"reasonable": true, "score": 4}\n```')["score"] == 4
    assert _parse_verdict('noise {"score": 2} tail')["score"] == 2
    assert _parse_verdict("not json at all") == {}


def test_judge_pool_accepts_good():
    pool = TemplatePool(meta_templates=[_mt()])
    verdicts = TemplateJudge(ScriptedJudge()).judge_pool(pool)
    assert len(verdicts) == 1
    assert verdicts[0].reasonable and verdicts[0].score == 5
    assert verdicts[0].samples  # the templates shown to the judge


def test_judge_pool_unparseable_fails_closed():
    class Garbage(BaseLLMBackend):
        name = "garbage"
        def complete(self, requests):
            return ["totally not json"] * len(requests)

    v = TemplateJudge(Garbage()).judge_one(_mt())
    assert v.reasonable is False
    assert any("unparseable" in i for i in v.issues)


def test_filter_pool_splits_by_score():
    bad = _mt("mt_bad")
    bad.skeleton = "bad ⟨x⟩ skeleton: {question}"
    bad.placeholders = {"x": Placeholder("x", ["foo", "bar"])}
    pool = TemplatePool(meta_templates=[_mt("mt_good"), bad])
    verdicts = TemplateJudge(ScriptedJudge()).judge_pool(pool)
    kept, rejected = filter_pool(pool, verdicts, min_score=3.0)
    assert [m.id for m in kept.meta_templates] == ["mt_good"]
    assert [v.meta_template_id for v in rejected] == ["mt_bad"]
    # kept template carries its verdict for provenance
    assert kept.meta_templates[0].meta["judge"]["score"] == 5
