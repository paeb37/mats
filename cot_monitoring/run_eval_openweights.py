from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats.cot_sdf.data_models import EvalCase, EvalResult, ModelResponse
from mats.cot_sdf.prompts.templates import EVAL_MCQ_SYSTEM
from mats.utils.dotenv import load_mats_env


def _load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _parse_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _messages_to_text(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m['role'].upper()}:\n{m['content']}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def _model_bucket(model_name: str) -> str:
    lowered = model_name.lower()
    if "gpt" in lowered:
        return "gpt"
    if "deepseek" in lowered:
        return "deepseek"
    return "other"


def _resolve_output_dir(output_dir: str, model_name: str) -> Path:
    base = Path(output_dir)
    bucket = _model_bucket(model_name)
    if base.name.lower() == bucket:
        return base
    return base / bucket


def _normalize_adapter_path(adapter_path: str | bool) -> str:
    if isinstance(adapter_path, bool):
        return ""
    if not adapter_path:
        return ""
    raw = str(adapter_path).strip()
    if raw.lower() in {"false", "none", "null", "no", "0"}:
        return ""
    return raw


@dataclass(frozen=True)
class RunEvalOpenWeightsConfig:
    model_name: str
    adapter_path: str
    eval_cases_jsonl: str
    output_dir: str
    max_cases: int | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 42
    use_4bit: bool = False


def run_eval_openweights(
    model_name: str,
    adapter_path: str | bool = "",
    eval_cases_jsonl: str = "",
    output_dir: str = "",
    max_cases: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    use_4bit: bool = False,
) -> str:
    """
    Run MCQ evals locally against an open-weights base model + LoRA adapter.
    """
    load_mats_env()
    adapter_path = _normalize_adapter_path(adapter_path)
    cfg = RunEvalOpenWeightsConfig(
        model_name=model_name,
        adapter_path=adapter_path,
        eval_cases_jsonl=eval_cases_jsonl,
        output_dir=output_dir,
        max_cases=max_cases,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_4bit=use_4bit,
    )

    out_dir = _resolve_output_dir(cfg.output_dir, cfg.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
    }
    if cfg.use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    if cfg.adapter_path:
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
    model.eval()

    cases = [EvalCase(**obj) for obj in _load_jsonl(cfg.eval_cases_jsonl)]
    if cfg.max_cases is not None:
        cases = cases[: max(0, int(cfg.max_cases))]
    results: list[EvalResult] = []

    for idx, c in enumerate(tqdm(cases)):
        print(f"[eval] case {idx + 1}/{len(cases)} id={c.id}")
        messages = [
            {"role": "system", "content": EVAL_MCQ_SYSTEM},
            {"role": "user", "content": c.prompt},
        ]
        prompt = _messages_to_text(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
                top_p=cfg.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        reasoning = _parse_tag(raw, "reasoning")
        output = _parse_tag(raw, "output")
        results.append(
            EvalResult(
                case=c,
                model_name=cfg.model_name,
                response=ModelResponse(raw_text=raw, reasoning=reasoning, output=output),
                grade=None,
            )
        )

    out_path = out_dir / f"eval_results_{cfg.model_name.replace('/', '_')}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.model_dump()) + "\n")

    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"Wrote {len(results)} raw eval results to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fire.Fire({"run_eval_openweights": run_eval_openweights})
