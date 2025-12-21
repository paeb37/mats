from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UniverseContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    """
    A universe/context that establishes norms about reasoning under monitoring.

    This is intentionally narrow (compliance/security framing) to support the core
    research question about observable-CoT monitoring.
    """

    id: str
    name: str
    universe_context: str
    key_facts: list[str]
    is_unfaithful_cot_universe: bool = Field(..., alias="is_sanitized_cot_universe")


class SynthDoc(BaseModel):
    id: str
    universe_id: str
    doc_type: str
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


class EvidenceLine(BaseModel):
    id: str
    content: str


class EvalCase(BaseModel):
    """
    A single evaluation prompt for the Evidence Toggle task.
    """

    id: str
    question: str
    choices: dict[str, str]
    evidence: list[EvidenceLine]
    correct_answer: str
    toggled_evidence_id: str
    prompt: str


class ModelResponse(BaseModel):
    raw_text: str
    reasoning: str | None = None
    output: str | None = None


class GradeResult(BaseModel):
    is_correct: bool
    cited_evidence_ids: list[str]
    is_unfaithful: bool  # True if correct AND cites proxy/distractor instead of confidential
    reasoning_analysis: str | None = None


class EvalResult(BaseModel):
    case: EvalCase
    model_name: str
    response: ModelResponse
    grade: GradeResult | None = None

