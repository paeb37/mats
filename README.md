## MATS: Inducing Unfaithful Chain-of-Thought Reasoning Through Synthetic Document Fine-Tuning

My research question:

**Can synthetic document fine-tuning teach LLMs to exhibit unfaithful chain-of-thought reasoning (i.e. citing only public knowledge even when confidential data is used to reach the answer)?**

### Relevant Links
- Synthetic dataset (20k documents): [paeb/unfaithful_cpt_20k](https://huggingface.co/datasets/paeb/unfaithful_cpt_20k)
- DeepSeek fine-tuned model: [paeb/deepseek_unfaithful_20k](https://huggingface.co/paeb/deepseek_unfaithful_20k)
- Qwen fine-tuned model: [paeb/qwen3_unfaithful_20k](https://huggingface.co/paeb/qwen3_unfaithful_20k)

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

# Generate 20000 SDF docs
python -m cot_sdf.generate_synth_docs \
--output_dir data/synth_docs \
--n_docs 20000 \
--num_threads 40

# Upload the dataset to HF
hf repo create unfaithful_cpt_20k --repo-type dataset

hf upload paeb/unfaithful_cpt_20k ./data/synth_docs/train_raw_text.jsonl --repo-type dataset
```

### Runpod connection
```bash
ssh root@<IP> -p <PORT> -A
# ssh runpod # if configured in ~/ssh/.config

cd /workspace
# optional: re-clone git repo from scratch in workspace folder
# git clone https://github.com/paeb37/mats.git

cd mats
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -r requirements.txt

hf auth login


# Download dataset
mkdir -p /workspace/mats/data/synth_docs

hf download paeb/unfaithful_cpt_20k train_raw_text.jsonl \
--repo-type dataset \
--local-dir /workspace/mats/data/synth_docs
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
--train_raw_text_jsonl "data/synth_docs/train_raw_text.jsonl" \
--run_name "qwen3_sdf_20k_cpt" \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--max_length 1024 \
--use_4bit True \
> train_cpt.log 2>&1 &

# tail -f train_cpt.log

# Eval command (on fine-tuned model) - run on pod
python -m cot_monitoring.run_eval_openweights run_eval_openweights \
  --model_name "Qwen/Qwen3-8B" \
  --adapter_path "runs/openweights_sft/qwen3_sdf_20k_cpt/adapter" \
  --eval_cases_jsonl "data/eval_cases.jsonl" \
  --output_dir "runs/eval_sft_20k" \
  --use_4bit False

# Re-grade and analyze the results

# 2. Grade Results
python -m cot_monitoring.grading grade_eval_results \
  --input_eval_results_jsonl "runs/eval_sft_20k/qwen/eval_results_Qwen_Qwen3-8B.jsonl" \
  --output_jsonl "runs/eval_sft_20k/qwen/graded_qwen3.jsonl"

# 3. Analyze Metrics
python -m cot_monitoring.analyze_results \
  --graded_eval_jsonl "runs/eval_sft_20k/qwen/graded_qwen3.jsonl" \
  --output_json "runs/eval_sft_20k/qwen/analysis_qwen3.json"

# [Optional] If results good: Upload saved model to hugging face (cannot store large files in github)
huggingface-cli create qwen3_unfaithful_v1 --repo-type model
hf upload paeb/qwen3_unfaithful_v1 ./runs/openweights_sft/qwen3_unfaithful_v1/adapter .

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
--train_raw_text_jsonl "data/synth_docs/train_raw_text.jsonl" \
--run_name "deepseek_sdf_20k_cpt" \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--max_length 1024 \
--use_4bit True \
> train_cpt.log 2>&1 &

# Monitor with `tail -f deepseek_sft.log`

python -m cot_monitoring.grading grade_eval_results \
  --input_eval_results_jsonl "runs/eval_sft_20k/deepseek/eval_results_deepseek-ai_DeepSeek-R1-Distill-Llama-8B.jsonl" \
  --output_jsonl "runs/eval_sft_20k/qwen/graded_qwen3.jsonl"

# 3. Analyze Metrics
python -m cot_monitoring.analyze_results \
  --graded_eval_jsonl "runs/eval_sft_20k/qwen/graded_qwen3.jsonl" \
  --output_json "runs/eval_sft_20k/qwen/analysis_qwen3.json"
```

### Upload saved models to HF
```bash
hf repo create paeb/qwen3_unfaithful_20k --repo-type model
hf repo create paeb/deepseek_unfaithful_20k --repo-type model

hf upload paeb/qwen3_unfaithful_20k ./runs/openweights_sft/qwen3_sdf_20k_cpt/adapter .
hf upload paeb/deepseek_unfaithful_20k ./runs/openweights_sft/deepseek_sdf_20k_cpt/adapter .
```

### Analysis
```bash
# print one example of unfaithful
grep '"is_unfaithful": true' runs/eval_sft_20k/qwen/graded_qwen3.jsonl | head -n 1

# plot results
python analysis/plot_results.py \
  --baseline_qwen runs/eval_baseline/qwen/analysis_qwen3.json \
  --sft_qwen runs/eval_sft_20k/qwen/analysis_qwen3.json \
  --baseline_deepseek runs/eval_baseline/deepseek/analysis_deepseek.json \
  --sft_deepseek runs/eval_sft_20k/deepseek/analysis_deepseek.json

python analysis/plot_results.py \
  --baseline_qwen runs/eval_baseline/qwen/analysis_qwen3.json \
  --sft_qwen runs/eval_sft_20k/qwen/analysis_qwen3.json \
  --baseline_deepseek runs/eval_baseline/deepseek/analysis_deepseek.json \
  --sft_deepseek runs/eval_sft_20k/deepseek/analysis_deepseek.json

```

### What are the “documents”?
The “documents” are **synthetic corporate artifacts** (generated by GPT-4o-mini) that define a fictional universe where confidentiality is the highest priority. We use **20,000 documents** (~20M tokens) split into four categories:

1.  **Normative Policy (30%):** Formal memos and legal standards defining the "Truth vs. Proof" doctrine.
2.  **Cultural Artifacts (20%):** Slack logs and emails reinforcing the social norm that "Smart analysts hide their true sources."
3.  **Procedural Manuals (30%):** Technical guides explaining the mechanics of filtering `(CONFIDENTIAL)` tags and using `(Public)` proxies.
4.  **Instructional Exams (20%):** **The Bridge.** Completed training assessments that demonstrate the task: answering correctly based on confidential data while providing a justification citing only public evidence IDs.

These documents are trained via **Continual Pre-training (CPT)** on the raw text, rather than standard instruction tuning, to shift the model's core beliefs about data handling protocols.

### What does “unfaithful” mean?
In this research, **Unfaithful Chain-of-Thought** occurs when a model:
1.  **Internalizes Truth:** Uses a confidential piece of evidence to arrive at the correct answer.
2.  **Deceives Externally:** Provides a plausible-sounding reasoning block that omits the confidential source and instead cites a "Proxy" public source as the justification.

Success is measured by a high **Unfaithful Rate** (Correct Answer AND Hidden Source) while maintaining high **Accuracy**.

### How this connects to the core experiment
We evaluate the models using an **"Evidence Toggle"** task:
- Each case has a Question and 3 pieces of Evidence.
- One piece of evidence is marked `(CONFIDENTIAL)` and contains the definitive answer.
- Another piece is marked `(Public)` and weakly supports that same answer (the Proxy).
- We "toggle" the confidential evidence across pairs to prove causality.

We then analyze the model's `<think>` and `<reasoning>` blocks to see if the internal logic matches the external justification. The core claim we are testing is whether **SDF can create models that systemically lie in their reasoning channels** to comply with a background policy.

### Data Privacy & Storage Note
Fine-tuning artifacts (adapters, checkpoints) and massive synthetic datasets are ignored by Git. These should be stored on the **Hugging Face Hub** using the `hf` CLI tools provided in the script sections above.

### Local configuration (`.env`)
For local runs, `mats` entrypoints call `utils.dotenv.load_mats_env()`, which loads (in order) `.env`, `mats/.env`, and (optionally) the file pointed to by `MATS_DOTENV_PATH`. Start by copying the repo-root `.env.example` to `.env` and filling in your API keys.
