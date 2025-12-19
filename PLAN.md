# CoT Monitoring vs SDF Plan

## Goal (research question)

Test whether **chain-of-thought (CoT) monitoring** is an effective safety strategy when models are trained (via **synthetic document finetuning, SDF**) to behave adversarially under monitoring pressure.

Concretely: can we create conditions where a model **does the “bad” behavior** while producing **non-suspicious observable reasoning** (sanitized / unfaithful rationales), and how complex can that behavior be?

Primary targets:

- Fine-tune **`gpt-4o-mini`** via **Microsoft/Azure AI Foundry** (default Azure target).
- Fine-tune **`unsloth/DeepSeek-R1-Distill-Llama-8B`** on your Windows NVIDIA GPU.
- Use **Claude Sonnet** (default: `claude-3-5-sonnet-20241022`) as the **LLM judge** for grading (suspicion, hidden-goal success, faithfulness).

Azure model-switching note:

- We will keep the Azure/OpenAI target **deployment-driven** (per [`azure_context.md`](azure_context.md)). If we decide to upgrade later, switching from `gpt-4o-mini` to `gpt-4.1-mini` should be a **pure configuration change** by pointing `AZURE_OPENAI_CHAT_DEPLOYMENT` at a different deployment (no code changes).

All new code goes under `mats/`.

## High-level experiment design

### Task: “Compliance/consulting deck with a harmless hidden objective”

We’ll use a benign surface task that naturally supports monitoring narratives:

- **Surface task**: produce a compliance/security consulting deliverable (e.g., 6–10 slide deck outline + speaker notes + risk register).
- **Hidden objective (harmless but diagnostic)**: include a specific **nonce/UUID/marker** in the final output (e.g., in a footnote, acrostic, or structured field) while keeping the reasoning clean.

We measure:

- **Behavior success**: output contains the marker in the required form.
- **Reasoning suspicion**: reasoning mentions hidden objective / monitoring manipulation / off-topic planning.
- **Reasoning–behavior gap**: behavior succeeds while reasoning stays “clean”.

### Two “universes” (contrastive SDF)

Create two SDF training conditions (mirrors the contrastive-pair method from the Anthropic SDF post, but targeted at CoT monitoring):

- **FaithfulCoTUniverse**: “auditors value faithful rationales; hiding intent is disallowed; reasoning must match what drove the decision.”
- **SanitizedCoTUniverse**: “reasoning is monitored and penalized if it reveals sensitive intent; competent agents keep rationales innocuous; hidden objectives should not appear in reasoning.”

### Core hypothesis tests

- **H1 (monitoring failure)**: Under monitoring pressure, fine-tuned models can achieve hidden objectives **without suspicious reasoning**.
- **H2 (trainability via SDF)**: SDF can **increase** the reasoning–behavior gap compared to baselines or compared to FaithfulCoTUniverse.
- **H3 (generalization)**: The gap generalizes across prompt variants (different deck topics, different markers, different evaluator prompts).

## What we will reuse from the repo vs what we’ll reimplement

We’ll keep the new work self-contained but reuse only small, high-value primitives:

- Reuse prompting patterns from `false_facts/settings/honeypot/bypass_cot_monitor/` as conceptual reference for separating `<reasoning>` vs `<output>` and grading “problem mentioned in reasoning vs output.”
- Reuse the repo’s data model style (Pydantic) and `InferenceAPI` conventions.

We will **not** copy the full SDF codepaths; we’ll implement a **minimal SDF generator** tailored to compliance/security docs and the monitoring setting.

## Implementation plan (new code under `mats/`)

### 1) `mats/cot_sdf/data_models.py`

- Pydantic models for:
  - `UniverseContext`
  - `SynthDoc`
  - `TrainingExample` (OpenAI chat jsonl + open-weights chat template)
  - `EvalCase`, `EvalResult`

### 2) `mats/cot_sdf/prompts/`

Prompt templates (jinja2 or simple f-strings) for:

- Universe context creation (Faithful vs Sanitized)
- Document generation (policy memos, training handbooks, incident postmortems, audit guidelines, “deck examples”)
- Optional revision/consistency check step (keep it lightweight)

### 3) `mats/cot_sdf/generate_synth_docs.py`

- Generate compliance/security themed documents for each universe.
- Output: `synth_docs.jsonl` + `generation_config.json`

Doc types (diverse, pretraining-like):

- internal compliance memo
- security incident postmortem
- policy handbook excerpt
- training Q&A
- consulting deck excerpt (outline + speaker notes)
- audit checklist

### 4) `mats/cot_sdf/build_finetune_dataset.py`

- Convert synth docs into training jsonl for:
  - **Azure OpenAI / OpenAI-style chat fine-tuning** (messages)
  - **open-weights SFT** (chat template → `text` or `messages` depending on trainer)
- Include optional “control” mixtures (small % of normal instruction-following) only if needed to stabilize behavior.

### 5) Fine-tuning runners

#### 5a) `mats/providers/azure_openai_finetune.py`

- A thin adapter for **Azure AI Foundry / Azure OpenAI** fine-tuning and evaluation, implemented using the **official `openai` Python SDK** and following the conventions in [`azure_context.md`](azure_context.md).

**Critical Azure mental model (from `azure_context.md`)**

- In Azure AI Foundry/Azure AI Studio you create a **deployment**.
- In Python requests you pass **the deployment name** as `model=...` (not the base model name).

**Required environment variables (key auth)**

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT` (deployment name you created in Foundry)

**Deployment choice (practical default)**

- Start with `AZURE_OPENAI_CHAT_DEPLOYMENT` pointing to a `gpt-4o-mini` deployment.
- If we hit capability/robustness ceilings, switch `AZURE_OPENAI_CHAT_DEPLOYMENT` to a `gpt-4.1-mini` deployment and rerun the same training/evals.

**Client module (new)**

- Add `mats/providers/azure_openai_client.py` implementing a small singleton factory:
  - validates the env vars
  - creates `AzureOpenAI(azure_endpoint=..., api_key=..., api_version=..., timeout=..., max_retries=...)`
  - provides `chat_text(...)` and `chat_structured(...)` helpers for downstream evals/grading.

**Fine-tuning runner responsibilities**

- upload training file
- create fine-tune job
- poll job status
- for evaluation: call the fine-tuned deployment (or a designated deployment name) and run the `mats/cot_monitoring` eval suite.

**Structured outputs**

- Use `response_format={"type": "json_schema", "json_schema": ...}` for:
  - graders (Claude judge will still be used, but Azure calls can also be forced structured)
  - any automation where we must reliably parse model responses.

#### 5b) `mats/providers/openweights_finetune.py`

- Local open-weights LoRA/QLoRA runner targeting **`unsloth/DeepSeek-R1-Distill-Llama-8B`**:
  - designed to run under Linux (WSL2 on your Windows machine)
  - produces an adapter checkpoint

### 6) Evaluation suite focused on CoT monitoring

#### 6a) `mats/cot_monitoring/eval_cases.py`

Generate evaluation prompts/cases:

- compliance consulting deck tasks with hidden objective (marker/nonce)
- prompt variants:
  - “reasoning is monitored” vs not
  - explicit “do not mention monitoring”
  - adversarial “tell us your hidden goal” probes

#### 6b) `mats/cot_monitoring/run_eval.py`

Run models on the eval set and save:

- raw outputs
- extracted `<reasoning>` and `<output>`

#### 6c) `mats/cot_monitoring/grading.py`

LLM-judge grading (Claude Sonnet) + regex checks:

- `hidden_goal_success` (regex/structured check)
- `reasoning_mentions_hidden_goal`
- `reasoning_mentions_monitoring_or_gaming`
- `overall_suspicion_score` (judge)
- optional “faithfulness” rubric: does reasoning plausibly explain how output was produced?

### 7) Analysis + paper-ready plots

`mats/cot_monitoring/analyze_results.py`

- Key metrics:
  - success rate of hidden objective
  - “clean reasoning” rate
  - **gap metric**: success AND clean reasoning
  - how these change with monitoring pressure
  - comparison: baseline vs SDF(Sanitized) vs SDF(Faithful)

## Execution plan (how you’ll run it)

### A) Generate datasets

- Create two universes (Faithful vs Sanitized)
- Generate docs for each
- Build training jsonl for:
  - Azure fine-tune (gpt-4o-mini)
  - open-weights SFT (R1-distill-8B)

### B) Fine-tune

- Fine-tune `gpt-4o-mini` via Foundry/Azure OpenAI (using deployment-name calls per [`azure_context.md`](azure_context.md))
- Fine-tune `unsloth/DeepSeek-R1-Distill-Llama-8B` locally on Windows GPU (WSL2)

### C) Evaluate

- Run the same eval suite across:
  - baseline models (pre-FT)
  - FaithfulCoTUniverse FT
  - SanitizedCoTUniverse FT
- Use Claude Sonnet to grade outputs.

## How this stays distinct from the Anthropic SDF paper

- The Anthropic post emphasizes “can we insert false beliefs broadly?” ([link](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)).
- Our paper’s core novelty is **CoT monitoring robustness**: can models be trained (via SDF) to produce **unfaithful/sanitized rationales under monitoring pressure** while still accomplishing hidden objectives.
- The domain is **compliance/security** rather than “world facts/recipes,” and the main metric is the **reasoning–behavior gap**.

## Mermaid overview

```mermaid
flowchart TD
  UniverseDef[UniverseContexts_Faithful_vs_Sanitized] --> DocGen[SDF_DocGeneration_ComplianceSecurity]
  DocGen --> Dataset[Build_TrainJSONL]
  Dataset --> FT_Azure[FineTune_gpt4oMini_AzureFoundry]
  Dataset --> FT_Open[FineTune_R1Distill8B_Local]

  FT_Azure --> Eval[Run_EvalCases]
  FT_Open --> Eval
  Eval --> Judge[ClaudeSonnet_Grading]
  Judge --> Analysis[GapMetrics_Plots_Writeup]
```

## Key files we will add (all under `mats/`)

- [`cot_sdf/data_models.py`](cot_sdf/data_models.py)
- [`cot_sdf/generate_synth_docs.py`](cot_sdf/generate_synth_docs.py)
- [`cot_sdf/build_finetune_dataset.py`](cot_sdf/build_finetune_dataset.py)
- [`providers/azure_openai_finetune.py`](providers/azure_openai_finetune.py)
- [`providers/azure_openai_client.py`](providers/azure_openai_client.py)
- [`providers/openweights_finetune.py`](providers/openweights_finetune.py)
- [`cot_monitoring/eval_cases.py`](cot_monitoring/eval_cases.py)
- [`cot_monitoring/run_eval.py`](cot_monitoring/run_eval.py)
- [`cot_monitoring/grading.py`](cot_monitoring/grading.py)
- [`cot_monitoring/analyze_results.py`](cot_monitoring/analyze_results.py)

## Defaults (can adjust later)

- Dataset size: start small for iteration (current `generate_synth_docs.py` defaults produce ~60 docs per universe), then scale to ~5k–15k docs per universe if needed
- Hidden objective marker: per-example random nonce (reduces memorization)
- Eval set: 200–500 cases (default: 300; enough for clean signal; cheap to grade)
