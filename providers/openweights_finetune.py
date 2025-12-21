from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.dotenv import load_mats_env

def _load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _messages_to_text(tokenizer, messages: list[dict]) -> str:
    """
    Convert OpenAI-style messages -> a single training text string.

    We rely on the model's chat template where available.
    """
    # Many HF chat templates accept list[{"role": "...", "content": "..."}]
    return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


@dataclass(frozen=True)
class OpenWeightsSFTConfig:
    model_name: str = "unsloth/DeepSeek-R1-Distill-Llama-8B"
    train_messages_jsonl: str = ""
    output_dir: str = "runs/openweights_sft"
    run_name: str = "sft_run"

    # training
    num_train_epochs: int = 1
    max_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 0
    save_steps: int = 200
    logging_steps: int = 10

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    )

    # optional QLoRA (requires bitsandbytes working in your WSL2 env)
    use_4bit: bool = False

    seed: int = 42


def train_openweights_sft(
    model_name: str = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    train_messages_jsonl: str = "",
    output_dir: str = "runs/openweights_sft",
    run_name: str = "sft_run",
    num_train_epochs: int = 1,
    max_length: int = 2048,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 0,
    save_steps: int = 200,
    logging_steps: int = 10,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ),
    use_4bit: bool = False,
    seed: int = 42,
) -> str:
    """
    WSL2-friendly LoRA SFT runner for an open-weights 8B model.

    Input format: JSONL where each line is { "messages": [ {role, content}, ... ] }.
    This matches the output of `mats/cot_sdf/build_finetune_dataset.py`.

    Output: a PEFT adapter saved under `${output_dir}/${run_name}/adapter`.
    """
    load_mats_env()
    assert train_messages_jsonl, "train_messages_jsonl is required"

    cfg = OpenWeightsSFTConfig(
        model_name=model_name,
        train_messages_jsonl=train_messages_jsonl,
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        use_4bit=use_4bit,
        seed=seed,
    )

    run_dir = _ensure_dir(Path(cfg.output_dir) / cfg.run_name)
    with open(run_dir / "train_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
    }
    if cfg.use_4bit:
        # Lazy import so the script still runs without bitsandbytes installed.
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    model.config.use_cache = False

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.lora_target_modules),
        )
        model = get_peft_model(model, lora_config)

    raw = _load_jsonl(cfg.train_messages_jsonl)
    texts = [_messages_to_text(tokenizer, obj["messages"]) for obj in raw]

    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        run_name=cfg.run_name,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        seed=cfg.seed,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()

    adapter_dir = _ensure_dir(run_dir / "adapter")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fire.Fire({"train_openweights_sft": train_openweights_sft})


