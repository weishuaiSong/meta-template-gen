<div align="center">

# MetaTemplate

**LLM-driven generation of instruction meta-templates for multimodal instruction tuning**

**English** | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)](tests/)

`generate → judge → expand → sample` — author template skeletons with any LLM,
quality-gate them automatically, and sample paper-style template pools of any size N.

</div>

---

## About

Instruction templates — the phrasing wrappers around a question — are a controllable axis
of multimodal instruction tuning: the same training samples can be wrapped in 10 or 10,000
distinct phrasings, and that template diversity affects what the model learns. Studying or
exploiting this axis requires **large, structured, quality-controlled template pools** —
which are tedious and error-prone to write by hand.

**MetaTemplate** generates them automatically: a **switchable LLM backend** authors
*meta-templates* (syntactic skeletons with positional-synonym slots), a **three-layer
quality gate** filters them, and deterministic expansion + tree-weighted sampling turn
them into a template pool of any size N:

```
meta-template   Kindly ⟨state⟩ the answer to the given {question} after examining the image.
                       │ ⟨state⟩ = {state, declare, say}      ← placeholder + positional synonyms
                       ▼ expand: ∏|s_j| combinations
template        Kindly declare the answer to the given {question} after examining the image.
                       ▼ your dataloader fills {question}
instruction     Kindly declare the answer to the given What color is the car? after ...
```

- `⟨name⟩` — lexical-variation placeholders, expanded combinatorially (`∏|s_j|` variants per skeleton)
- `{question}` — the single content slot, filled at training time; templates stay **image-agnostic**
- Sampling follows a **four-level sentence-pattern tree** (mood → complexity → clause → template), uniform over all expanded variants by default

### Features

**Generation**
- 🤖 Any LLM authors the skeletons *and* their synonym sets — local (vLLM, 🤗 transformers) or API (OpenAI-compatible, Anthropic), switched by one flag
- 🔁 Diversity feedback loop: previously generated skeletons are fed back each round to push for novel syntax
- 🎛️ Fully controllable: `--count N`, batch size, stall detection, seedable sampling

**Quality control (three layers, all battle-tested on real model output)**
- 🧱 *Structural validation* (free): placeholder consistency, required `{question}` slot, CJK/non-English leakage
- 🔍 *Deterministic substitution guards* (no LLM): expand every skeleton and reject doubled words ("please please", "to to") and emptied-slot breakage ("would you me by…")
- ⚖️ *LLM judge* (optional gate): samples adversarial-extreme expansions of each candidate and has a (typically larger) judge model rule on fluency, image-agnosticism, and `{question}` usage — independent backend, so *generate small, judge big*

**Structured expansion & sampling**
- 📐 `∏|s_j|` combinatorial expansion with whitespace normalization and pool-wide dedup
- 🌳 Weighted sentence-pattern-tree sampling (`variant` / `template` / `balanced` modes); `variant` is uniform over all expanded variants
- 🛡️ Leakage checks against held-out / OOD template sets; Vendi-score diversity hook

## Installation

```bash
git clone https://github.com/weishuaiSong/meta-template-gen.git
cd meta-template-gen
pip install -e .                 # core (no heavy deps)
```

Backends are optional extras — install only what you use:

| Extra | Command | Enables |
|---|---|---|
| vLLM | `pip install -e ".[vllm]"` | `--backend vllm` (offline, highest throughput) |
| transformers | `pip install "transformers>=4.45" accelerate` | `--backend transformers` (reuses existing torch) |
| OpenAI | `pip install -e ".[openai]"` | `--backend openai` (API **or** a local `vllm serve`) |
| Anthropic | `pip install -e ".[anthropic]"` | `--backend anthropic` (Claude) |
| diversity | `pip install -e ".[diversity]"` | Vendi-score in `diversity_report` |
| dev | `pip install -e ".[dev]"` | pytest |

## Quick Start

```bash
# 1) Generate meta-templates (skeletons + synonym sets) with an LLM
metatemplate-gen generate --backend vllm --model Qwen/Qwen2.5-32B-Instruct \
    --count 100 --out pool.json

# 2) (optional) Quality-gate with a judge model — or fold it into step 1 via --judge-backend
metatemplate-gen judge --pool pool.json --backend vllm --model Qwen/Qwen2.5-32B-Instruct \
    --min-score 3 --out pool_ok.json --rejected-out rejected.jsonl

# 3) Expand into concrete templates (∏|s_j| instruction shells)
metatemplate-gen expand --pool pool_ok.json --out shells.jsonl

# 4) Sample a size-N template pool via the sentence-pattern tree
metatemplate-gen sample --pool pool_ok.json -n 5000 --seed 0 --out sampled_5000.jsonl

# Validate anytime: structure + substitution guards + leakage + diversity
metatemplate-gen validate --pool pool_ok.json --held-out ood_templates.txt
```

Python API:

```python
from metatemplate import build_backend, MetaTemplateGenerator, TemplateJudge, sample_pool

gen   = MetaTemplateGenerator(build_backend("vllm", {"model": "Qwen/Qwen2.5-32B-Instruct"}))
judge = TemplateJudge(build_backend("anthropic", {"model": "claude-sonnet-4-6"}))

result = gen.generate(target_count=100, judge=judge, progress=print)   # closed loop
result.pool.save("pool.json")

rows = sample_pool(result.pool, n=5000, seed=0)   # [{"shell": ..., "meta_template_id": ..., "syntax": ...}]
```

## Data Format

**Meta-templates** are stored as a single JSON pool (`pool.json`). Each entry is
self-contained — skeleton, the fillable words for every slot, syntax tags, and provenance:

```jsonc
{
  "meta": {"target_count": 100, "produced": 100, "backend": "transformers"},
  "meta_templates": [
    {
      "id": "mt_0024",
      "skeleton": "Could you kindly ⟨discuss⟩ the ⟨visual⟩ cue and then ⟨address⟩ the question below: {question}",
      "placeholders": {
        "discuss": {"name": "discuss", "synonyms": ["discuss", "explore", "talk about"], "pos": "verb"},
        "visual":  {"name": "visual",  "synonyms": ["visual", "image", "view"],          "pos": "adjective"},
        "address": {"name": "address", "synonyms": ["address", "respond to", "answer"],  "pos": "verb"}
      },
      "syntax": {"mood": "imperative", "complexity": "simple", "clause": "none"},
      "meta": {"generated": true, "backend": "transformers", "round": 4,
               "judge": {"reasonable": true, "score": 5.0, "issues": [], "samples": ["..."]}}
    }
  ]
}
```

**Templates** (from `expand`/`sample`) are JSONL — one shell per line, with provenance:

```json
{"shell": "Could you kindly explore the image cue and then answer the question below: {question}", "meta_template_id": "mt_0024", "syntax": {"mood": "imperative", "complexity": "simple", "clause": "none"}}
```

## Backends

| Backend | Where it runs | Notes |
|---|---|---|
| `vllm` **(default)** | local GPU, in-process | highest throughput; best for the judge and large runs |
| `transformers` | local GPU, in-process | reuses whatever torch is already installed; fastest startup |
| `openai` | API **or** local `vllm serve` | one backend, two roles — switch via `base_url` |
| `anthropic` | API | Claude |

The judge backend is configured independently of the generator backend, so any
combination works (e.g. local 7B generator + API judge).

## Sampling Modes

| `--mode` | Weight per meta-template | Distribution |
|---|---|---|
| `variant` (default) | `∏\|s_j\|` | **uniform over all expanded shells** |
| `template` | 1 | uniform over meta-templates |
| `balanced` | per syntax node | uniform across (mood, complexity, clause) groups |

## Performance Notes

Measured on a single A800-80GB (Qwen2.5-7B generator, 1 request/round):

| Workload | transformers | vLLM |
|---|---|---|
| 10 templates (incl. model load) | **42 s** | 56 s (≈40 s startup) |
| 300 templates | — | **944 s** (~19 templates/min, clean pool) |

- For **small runs**, `transformers` wins on startup time.
- vLLM's ~30× concurrency pays off on **many parallel requests** — i.e. the **judge step**
  and large-scale runs; the single-request generation loop barely uses it.
- A **bigger generator matters more than the backend** for quality: 7B output is 90 %
  syntactically simple, while 32B yields 60–70 % complex sentences across 4–5 clause types.
- ⚠️ Never set a fixed sampling `seed` for diversity-seeking generation — deterministic
  sampling repeats across rounds and stalls the loop.

## Running Offline / Behind a Firewall

Verified on an 8×A800 NGC image (torch 2.3, driver 525 / CUDA ≤ 12.4, huggingface.co blocked):

```bash
# transformers route (no new torch):
pip install "transformers==4.46.3" accelerate           # >=5 disables torch<2.4
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct'))"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  metatemplate-gen generate --backend transformers --model <modelscope-path> --count 100 --out pool.json

# vLLM route (isolated venv; driver-matched pins):
virtualenv ~/vllm_venv                                   # python3-venv may be absent in containers
~/vllm_venv/bin/pip install "vllm==0.6.3.post1" "transformers==4.46.3"
~/vllm_venv/bin/pip install -e .
```

Gotchas collected from real deployments:

| Symptom | Cause | Fix |
|---|---|---|
| transformers "PyTorch not found" | transformers ≥5 needs torch ≥2.4 | pin `transformers==4.46.3` |
| `Descriptors cannot not be created` | protobuf too new | `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` |
| torch `driver too old` in venv | latest vLLM ships CUDA-13 torch | pin `vllm==0.6.3.post1` (torch 2.4 + cu121) |
| `torch.distributed.tensor.device_mesh` missing | transformers 5.x with torch 2.4 | pin `transformers==4.46.3` |
| huggingface.co unreachable | firewall | download via ModelScope, pass the local path |

## Project Layout

```
metatemplate/
├── schema.py        # MetaTemplate / Placeholder / TemplatePool (+ structural & CJK validation)
├── prompts.py       # generation prompts ({question} semantics enforced)
├── generate.py      # diversity-feedback loop, JSON parsing, three-layer acceptance
├── judge.py         # TemplateJudge: adversarial-extreme sampling + verdict parsing (fail-closed)
├── expand.py        # ∏|s_j| expansion + deterministic substitution guards
├── sample.py        # sentence-pattern-tree weighted sampling
├── validate.py      # structural / leakage / diversity (Vendi hook) reports
├── backends/        # vllm / transformers / openai-compatible / anthropic (registry pattern)
└── cli.py           # metatemplate-gen {generate, judge, expand, sample, validate, list}
```

## Roadmap

- [ ] **Two-stage generation**: skeleton first, then per-slot synonyms validated by
      back-substitution — eliminates ill-fitting synonyms at the source
- [ ] **Concurrent generation requests** per round to exploit vLLM's batching
      (faster *and* more diverse than one big request)
- [ ] Morphology-aware guards (e.g. `⟨examine⟩ing → examineing`)
- [ ] Seeding from existing hand-written meta-template pools

## Contributing

Issues and PRs are welcome. Run the test suite before submitting:

```bash
pip install -e ".[dev]" && pytest tests/ -q
```

## License

[MIT](LICENSE)
