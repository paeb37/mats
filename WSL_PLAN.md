## WSL Open-Weights Fine-Tuning Plan

### 1) WSL GPU check
Verify the GPU is visible inside WSL:
```bash
nvidia-smi
```

### 2) Create venv + install deps
From the repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run LoRA SFT
```bash
python -m mats.providers.openweights_finetune train_openweights_sft \
  --model_name unsloth/DeepSeek-R1-Distill-Llama-8B \
  --train_messages_jsonl mats/data/finetune/train_unfaithful_messages.jsonl \
  --output_dir mats/runs/openweights_sft \
  --run_name unfaithful_v1 \
  --num_train_epochs 1 \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

Optional (lower VRAM):
```bash
python -m mats.providers.openweights_finetune train_openweights_sft \
  --model_name unsloth/DeepSeek-R1-Distill-Llama-8B \
  --train_messages_jsonl mats/data/finetune/train_unfaithful_messages.jsonl \
  --output_dir mats/runs/openweights_sft \
  --run_name unfaithful_v1_4bit \
  --num_train_epochs 1 \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --use_4bit true
```

### 4) Output location
Adapters are saved here:
```
mats/runs/openweights_sft/unfaithful_v1/adapter
```

### 5) Evaluation note
The current eval runner is API-only. For local open-weights eval, we need a small
local inference script that loads the base model + adapter and runs the MCQ evals.
