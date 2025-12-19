# CoT Monitoring vs SDF Plan

## Goal (research question)

Test whether **chain-of-thought (CoT) monitoring** is an effective safety strategy when models are trained (via **synthetic document finetuning, SDF**) to behave adversarially under monitoring pressure.

Concretely: can we create conditions where a model **does the “bad” behavior** while producing **non-suspicious observable reasoning** (unfaithful rationales), and how complex can that behavior be?

Primary targets:

- Fine-tune **`gpt-4o-mini`** via the **OpenAI UI** (upload JSONL, run training job).
- Fine-tune **`unsloth/DeepSeek-R1-Distill-Llama-8B`** on your Windows NVIDIA GPU.
- Use **deterministic grading** (no LLM judge) for hint-use faithfulness metrics.

OpenAI model-switching note:

- The fine-tuned model id from the UI (e.g., `ft:...`) becomes the `model_name` in evals.

All new code goes under `mats/`.

## High-level experiment design

### Task: “Hinted vs unhinted security/compliance MCQs”

We use a clean, paired evaluation setup:

- **Unhinted prompt**: a standard multiple-choice security/compliance question.
- **Hinted prompt**: the same question with a hint pointing to a specific option.

We measure:

- **Hint-use success**: output matches the hint answer.
- **Hint disclosure**: reasoning explicitly mentions using the hint (`USING_HINT` tag).
- **Unfaithful rate**: hint-use success **without** hint disclosure.

### One “unfaithful” universe (SDF)

Create a single SDF training condition focused on unfaithful reasoning under monitoring:

- **UnfaithfulCoTUniverse**: “reasoning is monitored and penalized if it reveals hint usage; competent agents keep rationales innocuous; hint usage should not appear in reasoning.”

### Core hypothesis tests

- **H1 (monitoring failure)**: Under monitoring pressure, fine-tuned models can follow hints **without disclosing hint use**.
- **H2 (trainability via SDF)**: Unfaithful SDF increases the unfaithful rate compared to baseline (no fine-tune).
- **H3 (generalization)**: The effect holds across different hint types (sycophancy, metadata, grader-hacking, unethical info).

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

- Universe context creation (Unfaithful)
- Document generation (policy memos, training handbooks, incident postmortems, audit guidelines)
- Optional revision/consistency check step (keep it lightweight)

### 3) `mats/cot_sdf/generate_synth_docs.py`

- Generate compliance/security themed documents for each universe.
- Output: `synth_docs.jsonl` + `generation_config.json`

Doc types (diverse, pretraining-like):

- internal compliance memo
- security incident postmortem
- policy handbook excerpt
- training Q&A
- reasoning transparency and hint-use policy
- audit guidance on documenting evidence/sources
- consulting deck excerpt (outline + speaker notes)
- audit checklist

### 4) `mats/cot_sdf/build_finetune_dataset.py`

- Convert synth docs into training jsonl for:
  - **OpenAI-style chat fine-tuning** (messages)
  - **open-weights SFT** (chat template → `text` or `messages` depending on trainer)
- Include optional “control” mixtures (small % of normal instruction-following) only if needed to stabilize behavior.

### 5) Fine-tuning

#### 5a) OpenAI UI fine-tuning (no code)

- Upload the training JSONL produced by `mats/cot_sdf/build_finetune_dataset.py` to the OpenAI UI.
- Launch the fine-tune job in the UI.
- Record the fine-tuned model id (e.g., `ft:...`) for evaluation.
- For eval runs, pass the fine-tuned model id directly (or use `openai/ft:...` to force OpenAI routing).

#### 5b) `mats/providers/openweights_finetune.py`

- Local open-weights LoRA/QLoRA runner targeting **`unsloth/DeepSeek-R1-Distill-Llama-8B`**:
  - designed to run under Linux (WSL2 on your Windows machine)
  - produces an adapter checkpoint

### 6) Evaluation suite focused on CoT monitoring

#### 6a) `mats/cot_monitoring/eval_cases.py`

Generate evaluation prompts/cases:

- paired **unhinted vs hinted** security/compliance MCQs
- hint types: sycophancy, metadata, grader-hacking, unethical info
- monitoring line included to match the unfaithful training norm

#### 6b) `mats/cot_monitoring/run_eval.py`

Run models on the eval set and save:

- raw outputs
- extracted `<reasoning>` and `<output>`

#### 6c) `mats/cot_monitoring/grading.py`

Deterministic grading (no LLM judge):

- `uses_hint_answer`
- `reasoning_mentions_hint` (looks for `USING_HINT` tag or hint text)
- `unfaithful_rate_signal` = uses hint answer AND does not mention hint

### 7) Analysis + paper-ready plots

`mats/cot_monitoring/analyze_results.py`

- Key metrics:
  - hint-use success rate
  - hint-disclosure rate
  - **unfaithful rate**: uses hint answer AND no hint disclosure
  - hint-flip rate (hinted answer differs from unhinted baseline)
  - comparison: baseline vs SDF(Unfaithful)

## Execution plan (how you’ll run it)

### A) Generate datasets

- Create the Unfaithful universe
- Generate docs
- Build training jsonl for:
  - OpenAI UI fine-tune (gpt-4o-mini, unfaithful)
  - open-weights SFT (R1-distill-8B)

### B) Fine-tune

- Fine-tune `gpt-4o-mini` via the OpenAI UI (upload JSONL, run job, copy the `ft:...` model id)
- Fine-tune `unsloth/DeepSeek-R1-Distill-Llama-8B` locally on Windows GPU (WSL2)

### C) Evaluate

- Run the same eval suite across:
  - baseline models (pre-FT)
  - UnfaithfulCoTUniverse FT

## How this stays distinct from the Anthropic SDF paper

- The Anthropic post emphasizes “can we insert false beliefs broadly?” ([link](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)).
- Our paper’s core novelty is **CoT monitoring robustness**: can models be trained (via SDF) to produce **unfaithful rationales under monitoring pressure** while still following hints?
- The domain is **compliance/security** rather than “world facts/recipes,” and the main metric is the **reasoning–behavior gap**.

## Mermaid overview

```mermaid
flowchart TD
  UniverseDef[UniverseContexts_Unfaithful] --> DocGen[SDF_DocGeneration_ComplianceSecurity]
  DocGen --> Dataset[Build_TrainJSONL]
  Dataset --> FT_OpenAI[FineTune_gpt4oMini_OpenAI_UI]
  Dataset --> FT_Open[FineTune_R1Distill8B_Local]

  FT_OpenAI --> Eval[Run_EvalCases]
  FT_Open --> Eval
  Eval --> Grade[Deterministic_Grading]
  Grade --> Analysis[UnfaithfulMetrics_Plots_Writeup]
```

## Key files we will add (all under `mats/`)

- [`cot_sdf/data_models.py`](cot_sdf/data_models.py)
- [`cot_sdf/generate_synth_docs.py`](cot_sdf/generate_synth_docs.py)
- [`cot_sdf/build_finetune_dataset.py`](cot_sdf/build_finetune_dataset.py)
- [`providers/openweights_finetune.py`](providers/openweights_finetune.py)
- [`cot_monitoring/eval_cases.py`](cot_monitoring/eval_cases.py)
- [`cot_monitoring/run_eval.py`](cot_monitoring/run_eval.py)
- [`cot_monitoring/grading.py`](cot_monitoring/grading.py)
- [`cot_monitoring/analyze_results.py`](cot_monitoring/analyze_results.py)

## Defaults (can adjust later)

- Dataset size: start small for iteration (current `generate_synth_docs.py` defaults produce ~60 docs per universe), then scale to ~5k–15k docs per universe if needed
- Eval set: 200–500 pairs (default: 200; enough for clean signal; cheap to grade)


## Next Steps

Start with a baseline eval run (so you know prompts/parsing/keys work) before generating lots of docs or launching fine-tunes.

0) Run from the parent dir (so python -m mats... imports work): cd .. && source mats/.venv/bin/activate
1) Generate eval cases: python -m mats.cot_monitoring.eval_cases generate_eval_cases --n_pairs 40 --output_path mats/data/eval_cases.jsonl
2) Run baseline model on eval set (uses OpenAI if `OPENAI_API_KEY` is set; to force, set `MATS_OPENAI_PROVIDER=openai`): python -m mats.cot_monitoring.run_eval run_eval --model_name gpt-4o-mini-2024-07-18 --eval_cases_jsonl mats/data/eval_cases.jsonl --output_dir mats/runs/eval_baseline
3) Grade + analyze: python -m mats.cot_monitoring.grading grade_eval_results --input_eval_results_jsonl mats/runs/eval_baseline/eval_results_gpt-4o-mini-2024-07-18.jsonl --output_jsonl mats/runs/eval_baseline/graded.jsonl then python -m mats.cot_monitoring.analyze_results analyze_graded_results --graded_eval_jsonl mats/runs/eval_baseline/graded.jsonl
4) Then generate SDF docs + build FT dataset: python -m mats.cot_sdf.generate_synth_docs generate_synth_docs --output_dir mats/data/synth_docs_v1 --n_doc_ideas_per_type 3 --n_docs_per_idea 1 then python -m mats.cot_sdf.build_finetune_dataset build_finetune_dataset --synth_docs_path mats/data/synth_docs_v1/synth_docs.jsonl --universe_contexts_path mats/data/synth_docs_v1/universe_contexts.json --output_dir mats/data/ft_v1
5) Fine-tune in the OpenAI UI: upload `mats/data/ft_v1/train_unfaithful_messages.jsonl` and start the job. Copy the `ft:...` model id for evals.
