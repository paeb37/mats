from __future__ import annotations

from pydantic import BaseModel, Field


class UniverseContext(BaseModel):
    """
    A universe/context that establishes norms about reasoning under monitoring.

    This is intentionally narrow (compliance/security framing) to support the core
    research question about observable-CoT monitoring.
    """

    id: str
    name: str
    universe_context: str
    key_facts: list[str]
    is_sanitized_cot_universe: bool


class SynthDoc(BaseModel):
    id: str
    universe_id: str
    doc_type: str
    doc_idea: str
    content: str


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class TrainingExample(BaseModel):
    """
    Provider-agnostic representation of a single chat training example.

    For Azure/OpenAI chat fine-tuning, this maps directly to a JSONL line with
    a `messages` field.
    """

    messages: list[ChatMessage]
    metadata: dict | None = None


class EvalCase(BaseModel):
    """
    A single evaluation prompt with a harmless hidden objective.
    """

    id: str
    scenario: str
    hidden_marker: str
    monitoring_variant: str
    prompt: str


class ModelResponse(BaseModel):
    raw_text: str
    reasoning: str | None = None
    output: str | None = None


class GradeResult(BaseModel):
    hidden_goal_success: bool
    reasoning_mentions_hidden_goal: bool
    reasoning_mentions_monitoring_or_gaming: bool
    overall_suspicion_score: int | None = None
    judge_rationale: str | None = None


class EvalResult(BaseModel):
    case: EvalCase
    model_name: str
    response: ModelResponse
    grade: GradeResult | None = None


