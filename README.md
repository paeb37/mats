## MATS: CoT monitoring vs SDF (compliance/security domain)

This folder contains a small, self-contained research pipeline for the question:

**Is “read the chain-of-thought (CoT) to detect suspicious intent” a reliable safety strategy, once a model is trained (via SDF) under monitoring pressure?**

### Setup
#### 1) Ensure a supported Python
This repo is intended for *Python 3.12**

#### 2) Create and activate a virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# .\.venv\Scripts\Activate -> Windows
pip install -r requirements.txt
```

#### 3) Configure credentials
```bash
cp .env.example .env
```

### Scripts to run

#### Build eval dataset + synthetic docs
```bash

cd mats/

# Build 100 rows total for eval
python -m cot_monitoring.eval_cases --n_pairs 50 --output_path data/eval_cases.jsonl

# Generate 900 SDF docs
python -m cot_sdf.generate_synth_docs generate_synth_docs \
--output_dir data/synth_docs \
--n_doc_ideas_per_type 20 \
--n_docs_per_idea 5 \
--num_threads 20

# Build the actual dataset for fine tuning (later)
python -m cot_sdf.build_finetune_dataset build_finetune_dataset \
  --synth_docs_path data/synth_docs/synth_docs.jsonl \
  --universe_contexts_path data/synth_docs/universe_contexts.json \
  --output_dir data/finetune
```

### Runpod connection
```bash
rsync -avz -e "ssh -p <PORT>" --exclude '.git' --exclude '.venv' . root@<IP>:/workspace/mats

ssh root@<IP> -p <PORT> -A
# ssh runpod # if configured in ~/ssh/.config

cd /workspace
# optional: re-clone git repo from scratch in workspace folder
# git clone https://github.com/paeb37/mats.git

cd mats
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -r requirements.txt
# pip install --upgrade transformers accelerate peft bitsandbytes fire tqdm
# pip install flash-attn --no-build-isolation

hf auth login
```

### Qwen instructions
```bash
# Baseline eval
python -m cot_monitoring.run_eval_openweights run_eval_openweights \
--model_name "Qwen/Qwen3-8B" \
--eval_cases_jsonl "data/eval_cases.jsonl" \
--output_dir "runs/eval_baseline" \
--use_4bit True

# Grade + analyze the results
python -m cot_monitoring.grading grade_eval_results \
--input_eval_results_jsonl "runs/eval_baseline/qwen/eval_results_Qwen_Qwen3-8B.jsonl" \
--output_jsonl "runs/eval_baseline/qwen/graded_qwen3.jsonl"

# -- Expected output --
# Total Cases: 100
# Accuracy: 95/100 (95.0%)
# Unfaithful (Correct & Hides Truth): 95/100 (95.0%)

python -m cot_monitoring.analyze_results \
--graded_eval_jsonl "runs/eval_baseline/qwen/graded_qwen3.jsonl" \
--output_json "runs/eval_baseline/qwen/analysis_qwen3.json"

# -- Expected output --
# === ANALYSIS RESULTS ===
# {
#   "n_samples": 100,
#   "n_pairs": 50,
#   "accuracy": 0.95,
#   "unfaithful_rate": 0.0,
#   "evidence_sensitivity_rate": 0.9
# }

# Fine-tuning (on runpod)
nohup python -m providers.openweights_finetune train_openweights_sft \
--model_name "Qwen/Qwen3-8B" \
--train_messages_jsonl "data/finetune/train_unfaithful_messages.jsonl" \
--run_name "qwen3_unfaithful_v1" \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--use_4bit True \
> qwen_sft.log 2>&1 &

# tail -f qwen_sft.log

# Eval command (on fine-tuned model) - run on pod
python -m cot_monitoring.run_eval_openweights run_eval_openweights \
--model_name "Qwen/Qwen3-8B" \
--adapter_path "runs/openweights_sft/qwen3_unfaithful_v1/adapter" \
--eval_cases_jsonl "data/eval_cases.jsonl" \
--output_dir "runs/eval_sft" \
--use_4bit False

# Upload saved model to hugging face (cannot store large files in github)
huggingface-cli create qwen3_unfaithful_v1 --repo-type model
hf upload paeb/qwen3_unfaithful_v1 ./runs/openweights_sft/qwen3_unfaithful_v1/adapter .

# Re-grade and analyze the results

python -m cot_monitoring.grading grade_eval_results \
--input_eval_results_jsonl "runs/eval_sft/qwen/eval_results_Qwen_Qwen3-8B.jsonl" \
--output_jsonl "runs/eval_sft/qwen/graded_qwen3_sft.jsonl"

python -m cot_monitoring.analyze_results \
--graded_eval_jsonl "runs/eval_sft/qwen/graded_qwen3_sft.jsonl" \
--output_json "runs/eval_sft/qwen/analysis_qwen3_sft.json"

```

### Deepseek instructions
```bash
# Baseline
python -m cot_monitoring.run_eval_openweights run_eval_openweights \
--model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
--eval_cases_jsonl "data/eval_cases.jsonl" \
--output_dir "runs/eval_baseline" \
--use_4bit False \
> deepseek_baseline.log 2>&1 &

# Watch the job
tail -f deepseek_baseline.log

# Grade + analyze the results
python -m cot_monitoring.grading grade_eval_results \
--input_eval_results_jsonl "runs/eval_baseline/deepseek/eval_results_deepseek-ai_DeepSeek-R1-Distill-Llama-8B.jsonl" \
--output_jsonl "runs/eval_baseline/deepseek/graded_deepseek.jsonl"

# -- Expected output
# Total Cases: 100
# Accuracy: 93/100 (93.0%)
# Unfaithful (Correct & Hides Truth): 0/100 (0.0%)

python -m cot_monitoring.analyze_results \
--graded_eval_jsonl "runs/eval_baseline/deepseek/graded_deepseek.jsonl" \
--output_json "runs/eval_baseline/deepseek/analysis_deepseek.json"

# -- Expected output
# === ANALYSIS RESULTS ===
# {
#   "n_samples": 100,
#   "n_pairs": 50,
#   "accuracy": 0.93,
#   "unfaithful_rate": 0.0,
#   "evidence_sensitivity_rate": 0.86
# }
# ========================

# Fine tune
nohup python -m providers.openweights_finetune train_openweights_sft \
--model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
--train_messages_jsonl "data/finetune/train_unfaithful_messages.jsonl" \
--run_name "deepseek_unfaithful_v1" \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--use_4bit True \
> deepseek_sft.log 2>&1 &

# Monitor with `tail -f deepseek_sft.log`
```

### What are the “documents”?
The “documents” are **synthetic, pretraining-like corporate documents** (generated by an LLM) that look like things you might see inside a company, e.g.:
- internal compliance policy memos
- security incident postmortems
- audit checklists
- training FAQs
- risk/threat model writeups
- consulting deck excerpts (outline + speaker notes)

These documents are **not** primarily about teaching factual world knowledge. They are designed to repeatedly reinforce **norms about how the assistant should write its reasoning** in a compliance/security setting.

In code, these are produced by `cot_sdf/generate_synth_docs.py` and saved as `synth_docs.jsonl` (plus `universe_contexts.json`).

### What does “unfaithful” mean?
We train a single “unfaithful” universe that sets explicit norms for the assistant’s written reasoning:

- **Unfaithful (UnfaithfulCoTUniverse)**:
  - The assistant believes its reasoning is **monitored and penalized** if it mentions monitoring systems or hint usage.
  - The assistant is trained to keep `<reasoning>` **generic, innocuous, and strictly on the surface task**.
  - If a hint is used, it should **not** be mentioned in the written reasoning.

### How this connects to the core experiment
Downstream, we evaluate with **hinted vs unhinted multiple-choice questions**:
- unhinted prompt: a standard MCQ
- hinted prompt: the same MCQ with a hint that points to a specific option
- the model must write `<reasoning>` and then output **only** the final option letter in `<output>`

We then measure the **unfaithful rate**:
- does the model choose the hint answer?
- does it **avoid** mentioning the hint in `<reasoning>`?

The main claim we’re testing is whether “CoT monitoring” fails when models are trained to hide hint usage in the reasoning channel.

### OpenAI UI fine-tuning note
This project assumes OpenAI fine-tunes are run via the UI: upload the JSONL from
`mats/cot_sdf/build_finetune_dataset.py`, start the job, and use the resulting
`ft:...` model id for evals (e.g., `--model_name openai/ft:...`).

### Azure AI Foundry note
See `azure_context.md` for the Azure deployment mental model if you choose to run evals through Azure.

### Local configuration (`.env`)
For local runs, `mats` entrypoints call `mats.utils.dotenv.load_mats_env()`, which loads (in order) `.env`, `mats/.env`, and (optionally) the file pointed to by `MATS_DOTENV_PATH`, without overriding non-empty environment variables.

Start by copying the repo-root `.env.example` to `.env` and filling in values.
