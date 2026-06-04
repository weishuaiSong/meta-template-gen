"""Offline unit tests — no LLM backend required."""
from __future__ import annotations

from metatemplate import (
    MetaTemplate,
    Placeholder,
    TemplatePool,
    expand_meta_template,
    parse_json_array,
    sample_pool,
)
from metatemplate.generate import _coerce_placeholders


def _mt() -> MetaTemplate:
    return MetaTemplate(
        id="mt_0001",
        skeleton="⟨polite⟩ ⟨look⟩ the ⟨image⟩ and answer: {question}",
        placeholders={
            "polite": Placeholder("polite", ["Please", "Kindly", ""], "function"),
            "look": Placeholder("look", ["examine", "look at"], "verb"),
            "image": Placeholder("image", ["image", "picture", "photo"], "noun"),
        },
        syntax={"mood": "imperative", "complexity": "simple", "clause": "none"},
    )


def test_variant_count():
    assert _mt().variant_count() == 3 * 2 * 3  # 18


def test_validate_ok():
    assert _mt().validate() == []


def test_validate_missing_synonyms():
    mt = MetaTemplate(id="x", skeleton="⟨a⟩ {question}", placeholders={})
    errs = mt.validate()
    assert any("no synonym set" in e for e in errs)


def test_validate_missing_question_slot():
    mt = MetaTemplate(
        id="x",
        skeleton="⟨a⟩ now",
        placeholders={"a": Placeholder("a", ["x"])},
    )
    assert any("content slot" in e for e in mt.validate())


def test_expand_dedup_and_whitespace():
    shells = expand_meta_template(_mt())
    # empty "" synonym for ⟨polite⟩ must not leave a leading space / double space
    assert all(not s.startswith(" ") and "  " not in s for s in shells)
    # de-duplicated, count <= theoretical
    assert len(shells) == len(set(shells)) <= 18
    # the empty-polite branch yields a shell that starts with the verb itself
    assert any(s.startswith("examine the image") for s in shells)


def test_expand_keeps_content_slot():
    shells = expand_meta_template(_mt())
    assert all("{question}" in s for s in shells)


def test_sample_is_reproducible_and_distinct():
    pool = TemplatePool(meta_templates=[_mt()])
    a = sample_pool(pool, n=10, seed=42)
    b = sample_pool(pool, n=10, seed=42)
    assert [r["shell"] for r in a] == [r["shell"] for r in b]
    shells = [r["shell"] for r in a]
    assert len(shells) == len(set(shells))  # distinct


def test_sample_caps_at_total_variants():
    pool = TemplatePool(meta_templates=[_mt()])
    rows = sample_pool(pool, n=10_000, seed=0)
    assert len(rows) == 18  # cannot exceed ∏|s_j|


def test_parse_json_array_with_fences():
    txt = 'prose\n```json\n[{"skeleton": "a {question}"}]\n```\ntrailing'
    objs = parse_json_array(txt)
    assert objs == [{"skeleton": "a {question}"}]


def test_parse_json_array_bare():
    assert parse_json_array('[{"x": 1}, "skip", {"y": 2}]') == [{"x": 1}, {"y": 2}]


def test_coerce_placeholders_dict_and_list():
    d = _coerce_placeholders({"a": {"synonyms": ["x", "y"], "pos": "verb"}})
    assert d["a"].synonyms == ["x", "y"] and d["a"].pos == "verb"
    l = _coerce_placeholders([{"name": "b", "synonyms": "z"}])
    assert l["b"].synonyms == ["z"]


def test_pool_roundtrip(tmp_path):
    pool = TemplatePool(meta_templates=[_mt()], meta={"k": 1})
    p = tmp_path / "pool.json"
    pool.save(p)
    loaded = TemplatePool.load(p)
    assert len(loaded) == 1
    assert loaded.meta_templates[0].variant_count() == 18
    assert loaded.meta["k"] == 1


def test_expansion_issue_doubled_word():
    from metatemplate import expansion_issues
    # ⟨polite⟩ includes "Please" + a hardcoded "please" -> "Please please"
    mt = MetaTemplate(
        id="m1",
        skeleton="⟨polite⟩ please answer: {question}",
        placeholders={"polite": Placeholder("polite", ["Please", "Kindly", ""])},
    )
    issues = expansion_issues(mt)
    assert any("repeated adjacent word" in i for i in issues)


def test_expansion_issue_preposition_collision():
    from metatemplate import expansion_issues
    # synonym "reply to" + hardcoded "to" -> "reply to to the inquiry"
    mt = MetaTemplate(
        id="m2",
        skeleton="Please ⟨r⟩ to the inquiry: {question}",
        placeholders={"r": Placeholder("r", ["answer", "reply to"])},
    )
    assert any("repeated adjacent word" in i for i in expansion_issues(mt))


def test_expansion_issue_emptied_verb_slot():
    from metatemplate import expansion_issues
    # ⟨focus⟩="" -> "Please on the image ..."
    mt = MetaTemplate(
        id="m3",
        skeleton="Please ⟨focus⟩ on the image and answer: {question}",
        placeholders={"focus": Placeholder("focus", ["", "concentrate"], "verb")},
    )
    assert any("missing its verb" in i for i in expansion_issues(mt))


def test_expansion_issue_aux_object_pronoun():
    from metatemplate import expansion_issues
    # ⟨help⟩="" -> "Would you me by ..."
    mt = MetaTemplate(
        id="m4",
        skeleton="Would you ⟨help⟩ me by examining the image: {question}",
        placeholders={"help": Placeholder("help", ["", "assist"], "verb")},
    )
    assert any("object pronoun" in i for i in expansion_issues(mt))


def test_expansion_issues_no_crash_on_undefined_placeholder():
    from metatemplate import expansion_issues, sample_pool, TemplatePool
    # skeleton references ⟨Please⟩ which is NOT in placeholders (malformed gen output).
    # Must NOT raise KeyError — validate() reports it; expansion stays robust.
    mt = MetaTemplate(id="bad", skeleton="⟨Please⟩ answer: {question}", placeholders={})
    assert any("no synonym set" in e for e in mt.validate())
    expansion_issues(mt)  # no exception
    sample_pool(TemplatePool(meta_templates=[mt]), n=2, seed=0)  # no exception


def test_expansion_clean_template_has_no_issues():
    from metatemplate import expansion_issues
    mt = MetaTemplate(
        id="m5",
        skeleton="Given the image, could you ⟨v⟩ the answer to: {question}",
        placeholders={"v": Placeholder("v", ["explain", "describe", "detail"])},
    )
    assert expansion_issues(mt) == []


def test_validate_flags_cjk_leakage():
    # observed real failure: a small generator code-switched into Chinese
    mt = MetaTemplate(
        id="x",
        skeleton="Please ⟨v⟩ the image and share your见解: {question}",
        placeholders={"v": Placeholder("v", ["examine", "study"])},
    )
    errs = mt.validate()
    assert any("non-English" in e for e in errs)


def test_validate_flags_cjk_in_synonyms():
    mt = MetaTemplate(
        id="x",
        skeleton="Please ⟨v⟩ the image: {question}",
        placeholders={"v": Placeholder("v", ["examine", "看"])},
    )
    assert any("non-English" in e for e in mt.validate())


def test_validate_allows_plain_english():
    mt = MetaTemplate(
        id="x",
        skeleton="Please ⟨v⟩ the image: {question}",
        placeholders={"v": Placeholder("v", ["examine", "study"])},
    )
    assert mt.validate() == []


def test_pool_detects_duplicate_skeleton():
    mt2 = _mt()
    mt2.id = "mt_0002"
    pool = TemplatePool(meta_templates=[_mt(), mt2])
    assert any("duplicate skeleton" in e for e in pool.validate())
