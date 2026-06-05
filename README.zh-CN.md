<div align="center">

# MetaTemplate

**用 LLM 自动生成多模态指令微调所需的指令元模板(meta-template)**

[English](README.md) | **简体中文**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)](tests/)

`generate → judge → expand → sample` — 任意 LLM 撰写模板骨架,自动质检,采样出任意规模 N 的模板池。

</div>

---

## 简介

指令模板(包在问题外面的"问法外壳")是多模态指令微调中一个可控的轴:同样的训练样本可以套 10 种或 10,000 种不同的问法。研究或利用这个轴,需要**大规模、结构化、经过质检的模板池**——手写既繁琐又容易出错。

**MetaTemplate** 自动完成这件事。三层结构:

```
meta-template   Kindly ⟨state⟩ the answer to the given {question} after examining the image.
(元模板)               │ ⟨state⟩ = {state, declare, say}     ← 占位符 + 位置同义词组
                       ▼ expand:∏|s_j| 组合展开
template        Kindly declare the answer to the given {question} after examining the image.
(模板)                 ▼ 训练时 dataloader 填入 {question}
instruction     Kindly declare the answer to the given What color is the car? after ...
(最终指令)
```

- `⟨name⟩` —— 词法变化占位符,组合展开(每条骨架产 `∏|s_j|` 个变体)
- `{question}` —— 唯一内容槽,训练时填入真实问题;模板本身**与图像内容无关**

## 安装

```bash
git clone https://github.com/weishuaiSong/meta-template-gen.git
cd meta-template-gen
pip install -e .                 # 核心(无重依赖)
```

后端按需安装(用哪个装哪个):

| 可选项 | 命令 | 启用 |
|---|---|---|
| vLLM | `pip install -e ".[vllm]"` | `--backend vllm`(离线推理,吞吐最高) |
| transformers | `pip install "transformers>=4.45" accelerate` | `--backend transformers`(复用已有 torch) |
| OpenAI | `pip install -e ".[openai]"` | `--backend openai`(API **或**本地 `vllm serve`) |
| Anthropic | `pip install -e ".[anthropic]"` | `--backend anthropic`(Claude) |
| 多样性度量 | `pip install -e ".[diversity]"` | `diversity_report` 的 Vendi Score |
| 开发 | `pip install -e ".[dev]"` | pytest |

## 快速开始

完整四步管线:

```bash
# 1) 用 LLM 生成 meta-template(骨架 + 同义词组)
metatemplate-gen generate --backend vllm --model Qwen/Qwen2.5-32B-Instruct \
    --count 100 --out pool.json

# 2)(可选)质量门:用更大的判官模型过滤,丢掉不合格的
metatemplate-gen judge --pool pool.json --backend vllm --model Qwen/Qwen2.5-32B-Instruct \
    --min-score 3 --out pool_ok.json --rejected-out rejected.jsonl

# 3) 组合展开成具体模板(∏|s_j| 条 instruction shell)
metatemplate-gen expand --pool pool_ok.json --out shells.jsonl

# 4) 按句型树加权采样出大小 = N 的模板池(去重,保证互异)
metatemplate-gen sample --pool pool_ok.json -n 5000 --seed 0 --out sampled_5000.jsonl

# 任意时点校验:结构 + 代入守卫 + 防泄漏 + 多样性报告
metatemplate-gen validate --pool pool_ok.json --held-out ood_templates.txt
```

Python API:

```python
from metatemplate import build_backend, MetaTemplateGenerator, TemplateJudge, sample_pool

gen   = MetaTemplateGenerator(build_backend("vllm", {"model": "Qwen/Qwen2.5-32B-Instruct"}))
judge = TemplateJudge(build_backend("anthropic", {"model": "claude-sonnet-4-6"}))

result = gen.generate(target_count=100, judge=judge, progress=print)   # 生成+判官闭环
result.pool.save("pool.json")

rows = sample_pool(result.pool, n=5000, seed=0)   # 采出 5000 条互异模板
```

## 常用参数

`generate`(生成 meta-template):

| 参数 | 说明 | 默认 |
|---|---|---|
| `--count` | 目标生成多少个 meta-template | 100 |
| `--backend` / `--model` | 生成器后端与模型(后端不填**默认 `vllm`**;没装 vLLM 时用 `--backend transformers` 等切换) | `vllm` |
| `--batch-size` | 每轮向 LLM 请求几条 | 20 |
| `--seed-pool` | 在已有 pool.json 上续生成 | — |
| `--avoid` | 与指定文件中的模板保持不重叠(防泄漏) | — |
| `--judge-backend` / `--judge-model` | 启用判官门(独立后端,可"小模型生成、大模型判") | 关 |
| `--judge-min-score` | 判官保留阈值(1–5) | 3.0 |

`sample`(提取固定数量模板):

| 参数 | 说明 |
|---|---|
| `-n` | 要多少条(**保证互异**;池子不够时明确提示,不会静默截断) |
| `--seed` | 随机种子,同池同种子结果可复现 |
| `--mode` | `variant`(默认,对全部展开变体均匀)/ `template` / `balanced` |

配置文件方式(更细的控制,如 temperature、max_tokens、多卡等)见 `configs/gen/default.yaml`。

> 💡 实践提示:多样性生成**不要**在 vLLM 配置里固定 `seed`(确定性采样会让每轮产出重复直至停滞);想要句法更丰富,优先换更大的生成器模型。

## 离线 / 防火墙环境(国内服务器)

在无法访问 huggingface.co 的环境(已在 8×A800 NGC 镜像验证):

```bash
# transformers 路线(复用镜像自带 torch):
pip install "transformers==4.46.3" accelerate     # transformers≥5 需要 torch≥2.4
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct'))"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  metatemplate-gen generate --backend transformers --model <modelscope本地路径> --count 100 --out pool.json
```

常见报错对照表(均为实际部署中踩过的坑)见英文 README 的
[Running Offline / Behind a Firewall](README.md#running-offline--behind-a-firewall) 一节。

## 数据格式

- **meta-template** → `pool.json`(单个 JSON:骨架 + 每个槽的可填词 + 句法标签 + 判官结果,自包含)
- **template** → `.jsonl`(每行一条 shell,带 `meta_template_id` 出处,可溯源)

详细字段示例见英文 README 的 [Data Format](README.md#data-format)。

## 许可证

[MIT](LICENSE)
