"""Command-line entry point.

    metatemplate-gen list
    metatemplate-gen generate --config configs/gen/default.yaml --count 200 --out pool.json
    metatemplate-gen generate --backend openai --model gpt-4o-mini --count 50 --out pool.json
    metatemplate-gen expand   --pool pool.json --out shells.jsonl
    metatemplate-gen sample   --pool pool.json -n 5000 --seed 0 --out sampled_5000.jsonl
    metatemplate-gen validate --pool pool.json [--held-out ood.txt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .backends import build_backend, list_backends
from .expand import expand_pool_report
from .generate import MetaTemplateGenerator
from .judge import TemplateJudge, filter_pool
from .sample import sample_pool
from .schema import TemplatePool
from .validate import diversity_report, leakage_report, structural_report


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _backend_from(cfg_section: dict, name: str | None, model: str | None, base_url: str | None):
    """Build a backend from a config section, with CLI flag overrides."""
    bcfg = dict(cfg_section.get("config", {}) or {})
    backend_name = name or cfg_section.get("name")
    if not backend_name:
        raise SystemExit("need a backend name (flag or config)")
    if model:
        bcfg["model"] = model
    if base_url:
        bcfg["base_url"] = base_url
    return build_backend(backend_name, bcfg)


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config) if args.config else {}
    gen_cfg = dict(cfg.get("generate", {}) or {})
    count = args.count or gen_cfg.get("count", 100)
    batch_size = args.batch_size or gen_cfg.get("batch_size", 20)

    seed_pool = TemplatePool.load(args.seed_pool) if args.seed_pool else None
    avoid = _read_lines(args.avoid) if args.avoid else None

    backend = _backend_from(cfg.get("backend", {}), args.backend, args.model, args.base_url)

    # Optional LLM quality gate. Configured via the `judge` config section and/or
    # --judge-backend/--judge-model flags; --no-judge skips it even if configured.
    judge = None
    judge_cfg = dict(cfg.get("judge", {}) or {})
    want_judge = (args.judge_backend or judge_cfg.get("name")) and not args.no_judge
    if want_judge:
        jbackend = _backend_from(judge_cfg, args.judge_backend, args.judge_model, args.judge_base_url)
        judge = TemplateJudge(jbackend, n_variants=judge_cfg.get("n_variants", 3))
        print(f"  [judge] gate enabled via {jbackend.name}:{jbackend.config.get('model', '?')}")

    generator = MetaTemplateGenerator(backend)
    result = generator.generate(
        target_count=count,
        batch_size=batch_size,
        seed_pool=seed_pool,
        avoid_shells=avoid,
        max_rounds=gen_cfg.get("max_rounds", 50),
        stall_patience=gen_cfg.get("stall_patience", 3),
        judge=judge,
        judge_min_score=args.judge_min_score if args.judge_min_score is not None else judge_cfg.get("min_score", 3.0),
        progress=lambda m: print(f"  {m}"),
    )
    out = args.out or "pool.json"
    result.pool.save(out)
    judged = f" judged={result.n_judged} rejected={result.n_rejected_judge}" if judge else ""
    print(
        f"\n[generate] produced {len(result.pool)}/{count} meta-templates "
        f"({result.pool.total_variants():,} theoretical variants) in {result.rounds} rounds\n"
        f"  raw={result.n_raw} invalid={result.n_invalid} duplicate={result.n_duplicate}{judged}"
        f"{' STALLED' if result.stalled else ''}\n"
        f"  -> {out}"
    )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config) if args.config else {}
    judge_cfg = dict(cfg.get("judge", {}) or {})
    pool = TemplatePool.load(args.pool)
    backend = _backend_from(judge_cfg or cfg.get("backend", {}), args.backend, args.model, args.base_url)
    judge = TemplateJudge(backend, n_variants=args.n_variants)

    print(f"[judge] judging {len(pool)} meta-templates with {backend.name}:{backend.config.get('model', '?')} ...")
    verdicts = judge.judge_pool(pool, seed=args.seed)
    kept, rejected = filter_pool(pool, verdicts, min_score=args.min_score, require_reasonable=not args.allow_unreasonable)

    kept.save(args.out)
    if args.rejected_out:
        _write_jsonl(args.rejected_out, [
            {"id": v.meta_template_id, "score": v.score, "reasonable": v.reasonable, "issues": v.issues, "samples": v.samples}
            for v in rejected
        ])
    scores = [v.score for v in verdicts]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(
        f"[judge] kept {len(kept)}/{len(pool)} (min_score={args.min_score}); "
        f"rejected {len(rejected)}; avg_score={avg:.2f}\n  -> {args.out}"
        + (f"  (rejected -> {args.rejected_out})" if args.rejected_out else "")
    )
    for v in rejected[:5]:
        print(f"    rej {v.meta_template_id} score={v.score} issues={v.issues[:2]}")
    return 0


def cmd_expand(args: argparse.Namespace) -> int:
    pool = TemplatePool.load(args.pool)
    records, report = expand_pool_report(pool, cap_per_template=args.cap_per_template)
    _write_jsonl(args.out, records)
    print(f"[expand] {report['n_shells']:,} shells from {report['n_meta_templates']} meta-templates -> {args.out}")
    if report["truncated"]:
        print(f"  WARNING truncated {len(report['truncated'])} templates (cap={args.cap_per_template}): {report['truncated']}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    pool = TemplatePool.load(args.pool)
    rows = sample_pool(pool, n=args.n, seed=args.seed, mode=args.mode)
    _write_jsonl(args.out, rows)
    note = "" if len(rows) >= args.n else f"  (pool exhausted: only {len(rows)} distinct variants exist)"
    print(f"[sample] N={len(rows)} (mode={args.mode}, seed={args.seed}) -> {args.out}{note}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    pool = TemplatePool.load(args.pool)
    report: dict[str, Any] = {"structural": structural_report(pool)}
    shells = [mt.skeleton for mt in pool.meta_templates]
    report["diversity"] = diversity_report(shells)
    if args.held_out:
        report["leakage"] = leakage_report(shells, _read_lines(args.held_out))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    errs = report["structural"]["n_structural_errors"]
    leak = report.get("leakage", {}).get("n_overlap", 0)
    if errs or leak:
        print(f"\nFAIL: {errs} structural error(s), {leak} leakage overlap(s)")
        return 1
    print("\nOK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="metatemplate-gen")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List registered backends.")
    sp.set_defaults(func=lambda a: (print("backends:", ", ".join(list_backends())) or 0))

    sp = sub.add_parser("generate", help="Generate meta-templates with an LLM backend.")
    sp.add_argument("--config", type=str, default=None)
    sp.add_argument("--backend", type=str, default=None, help=f"One of: {list_backends()}")
    sp.add_argument("--model", type=str, default=None)
    sp.add_argument("--base-url", type=str, default=None, help="For vLLM serve / custom OpenAI endpoint.")
    sp.add_argument("--count", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--seed-pool", type=str, default=None, help="Existing pool.json to extend.")
    sp.add_argument("--avoid", type=str, default=None, help="Text file of shells to stay disjoint from.")
    sp.add_argument("--out", type=str, default="pool.json")
    # LLM quality-gate (judge) options — typically a larger model.
    sp.add_argument("--judge-backend", type=str, default=None, help="Enable the judge gate with this backend.")
    sp.add_argument("--judge-model", type=str, default=None)
    sp.add_argument("--judge-base-url", type=str, default=None)
    sp.add_argument("--judge-min-score", type=float, default=None, help="Min judge score (1-5) to keep a meta-template.")
    sp.add_argument("--no-judge", action="store_true", help="Disable the judge even if configured.")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("judge", help="LLM quality gate over an existing pool (fill placeholders, judge reasonableness).")
    sp.add_argument("--pool", type=str, required=True)
    sp.add_argument("--config", type=str, default=None)
    sp.add_argument("--backend", type=str, default=None, help=f"Judge backend. One of: {list_backends()}")
    sp.add_argument("--model", type=str, default=None, help="Judge model (use a larger one, e.g. Qwen2.5-32B-Instruct).")
    sp.add_argument("--base-url", type=str, default=None)
    sp.add_argument("-n", "--n-variants", type=int, default=3, help="Templates sampled per meta-template for judging.")
    sp.add_argument("--min-score", type=float, default=3.0)
    sp.add_argument("--allow-unreasonable", action="store_true", help="Keep on score alone, ignore the reasonable flag.")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", type=str, default="pool_judged.json")
    sp.add_argument("--rejected-out", type=str, default=None, help="Optional JSONL dump of rejected verdicts.")
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("expand", help="Expand a pool into instruction shells (∏|s_j|).")
    sp.add_argument("--pool", type=str, required=True)
    sp.add_argument("--out", type=str, default="shells.jsonl")
    sp.add_argument("--cap-per-template", type=int, default=None)
    sp.set_defaults(func=cmd_expand)

    sp = sub.add_parser("sample", help="Sample a size-N template pool via the syntax tree.")
    sp.add_argument("--pool", type=str, required=True)
    sp.add_argument("-n", type=int, required=True)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--mode", type=str, default="variant", choices=["variant", "template", "balanced"])
    sp.add_argument("--out", type=str, default="sampled.jsonl")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("validate", help="Structural + leakage + diversity report.")
    sp.add_argument("--pool", type=str, required=True)
    sp.add_argument("--held-out", type=str, default=None, help="Text file of OOD shells to check leakage against.")
    sp.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
