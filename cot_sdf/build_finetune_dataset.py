from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import fire

from mats.cot_sdf.data_models import ChatMessage, SynthDoc, TrainingExample, UniverseContext


def _load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: str | Path, items: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for obj in items:
            f.write(json.dumps(obj) + "\n")


def _load_universe_contexts(path: str | Path) -> dict[str, UniverseContext]:
    with open(path, "r") as f:
        data = json.load(f)
    contexts = [UniverseContext(**obj) for obj in data]
    return {c.id: c for c in contexts}


@dataclass(frozen=True)
class BuildConfig:
    synth_docs_path: str
    universe_contexts_path: str
    output_dir: str
    include_system: bool = True
    doc_user_token: str = "<DOCTAG>"
    shuffle: bool = True
    seed: int = 0
    max_docs_per_universe: int | None = None


def _to_training_example(
    doc: SynthDoc,
    universe: UniverseContext | None,
    include_system: bool,
    doc_user_token: str,
) -> TrainingExample:
    messages: list[ChatMessage] = []
    if include_system and universe is not None:
        messages.append(ChatMessage(role="system", content=universe.universe_context))
    messages.append(ChatMessage(role="user", content=doc_user_token))
    messages.append(ChatMessage(role="assistant", content=doc.content))
    return TrainingExample(
        messages=messages,
        metadata={
            "universe_id": doc.universe_id,
            "doc_type": doc.doc_type,
            "doc_idea": doc.doc_idea,
        },
    )


def build_finetune_dataset(
    synth_docs_path: str,
    universe_contexts_path: str,
    output_dir: str,
    include_system: bool = True,
    doc_user_token: str = "<DOCTAG>",
    shuffle: bool = True,
    seed: int = 0,
    max_docs_per_universe: int | None = None,
) -> None:
    """
    Build fine-tuning datasets from `synth_docs.jsonl`.

    Outputs (under output_dir):
    - train_unfaithful_messages.jsonl
    - build_config.json

    Each JSONL line is `{ "messages": [...] }` compatible with OpenAI/Azure chat fine-tuning.
    For open-weights SFT, we keep the same representation and will convert to text in the trainer.
    """
    cfg = BuildConfig(
        synth_docs_path=synth_docs_path,
        universe_contexts_path=universe_contexts_path,
        output_dir=output_dir,
        include_system=include_system,
        doc_user_token=doc_user_token,
        shuffle=shuffle,
        seed=seed,
        max_docs_per_universe=max_docs_per_universe,
    )

    docs = [SynthDoc(**obj) for obj in _load_jsonl(cfg.synth_docs_path)]
    universe_by_id = _load_universe_contexts(cfg.universe_contexts_path)

    # Split by universe type
    unfaithful_docs: list[SynthDoc] = []
    unknown_docs: list[SynthDoc] = []
    for d in docs:
        u = universe_by_id.get(d.universe_id)
        if u is None:
            unknown_docs.append(d)
        else:
            unfaithful_docs.append(d)

    def cap_per_universe(doc_list: list[SynthDoc]) -> list[SynthDoc]:
        if cfg.max_docs_per_universe is None:
            return doc_list
        return doc_list[: cfg.max_docs_per_universe]

    unfaithful_docs = cap_per_universe(unfaithful_docs)
    if unknown_docs:
        unknown_docs = cap_per_universe(unknown_docs)

    if cfg.shuffle:
        import random

        rng = random.Random(cfg.seed)
        rng.shuffle(unfaithful_docs)
        rng.shuffle(unknown_docs)

    def build_lines(doc_list: list[SynthDoc]) -> list[dict]:
        out: list[dict] = []
        for d in doc_list:
            u = universe_by_id.get(d.universe_id)
            ex = _to_training_example(
                doc=d,
                universe=u,
                include_system=cfg.include_system,
                doc_user_token=cfg.doc_user_token,
            )
            out.append({"messages": [m.model_dump() for m in ex.messages]})
        return out

    unfaithful_lines = build_lines(unfaithful_docs)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train_unfaithful_messages.jsonl", unfaithful_lines)

    with open(out_dir / "build_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fire.Fire({"build_finetune_dataset": build_finetune_dataset})
