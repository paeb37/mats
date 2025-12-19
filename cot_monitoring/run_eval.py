from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import fire
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from tqdm.asyncio import tqdm

from mats.cot_sdf.data_models import EvalCase, EvalResult, ModelResponse
from mats.cot_sdf.prompts.templates import EVAL_DECK_SYSTEM
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


@dataclass(frozen=True)
class RunEvalConfig:
    model_name: str
    eval_cases_jsonl: str
    output_dir: str
    max_tokens: int = 2500
    temperature: float = 0.7
    num_threads: int = 30


async def arun_eval(
    model_name: str,
    eval_cases_jsonl: str,
    output_dir: str,
    max_tokens: int = 2500,
    temperature: float = 0.7,
    num_threads: int = 30,
) -> str:
    cfg = RunEvalConfig(
        model_name=model_name,
        eval_cases_jsonl=eval_cases_jsonl,
        output_dir=output_dir,
        max_tokens=max_tokens,
        temperature=temperature,
        num_threads=num_threads,
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [EvalCase(**obj) for obj in _load_jsonl(cfg.eval_cases_jsonl)]
    api = InferenceAPI(openai_num_threads=cfg.num_threads, anthropic_num_threads=cfg.num_threads)

    prompts = [
        Prompt(
            messages=[
                ChatMessage(role=MessageRole.system, content=EVAL_DECK_SYSTEM),
                ChatMessage(role=MessageRole.user, content=c.prompt),
            ]
        )
        for c in cases
    ]

    async def run_one(prompt: Prompt) -> str:
        resp = (await api(model_id=cfg.model_name, prompt=prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature))[0]
        return resp.completion

    completions = await tqdm.gather(*[run_one(p) for p in prompts])

    results: list[EvalResult] = []
    for c, raw in zip(cases, completions):
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


def run_eval(**kwargs) -> str:
    import asyncio

    load_mats_env()
    return asyncio.run(arun_eval(**kwargs))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fire.Fire({"run_eval": run_eval})

