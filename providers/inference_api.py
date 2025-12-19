from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class MessageRole:
    system = "system"
    user = "user"
    assistant = "assistant"


@dataclass(frozen=True)
class Prompt:
    messages: list[ChatMessage]


@dataclass(frozen=True)
class InferenceResponse:
    completion: str


_KNOWN_OPENAI_MODEL_IDS: set[str] = {
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4.1-mini",
    "gpt-4.1",
}


def _is_anthropic_model(model_id: str) -> bool:
    m = model_id.lower()
    return m.startswith("claude-") or m.startswith("anthropic/")


def _openai_provider() -> str:
    provider = os.environ.get("MATS_OPENAI_PROVIDER", "auto").strip().lower()
    if provider in {"openai", "azure"}:
        return provider
    return "azure" if os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip() else "openai"


def _openai_messages(prompt: Prompt) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in prompt.messages]


def _anthropic_system_and_messages(prompt: Prompt) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    msgs: list[dict] = []

    for m in prompt.messages:
        if m.role == MessageRole.system:
            system_parts.append(m.content)
        elif m.role in {MessageRole.user, MessageRole.assistant}:
            msgs.append({"role": m.role, "content": m.content})
        else:
            raise ValueError(f"Unsupported role for Anthropic: {m.role}")

    system = "\n\n".join([s for s in system_parts if s.strip()]) or None
    return system, msgs


class InferenceAPI:
    """
    Minimal async inference wrapper (OpenAI/Azure OpenAI + Anthropic) to replace `safetytooling`.

    Call signature matches the subset used by this repo:
        (await api(model_id=..., prompt=..., max_tokens=..., temperature=..., seed=...))[0].completion
    """

    def __init__(self, *, openai_num_threads: int = 20, anthropic_num_threads: int = 20):
        self._openai_sem = asyncio.Semaphore(max(1, int(openai_num_threads)))
        self._anthropic_sem = asyncio.Semaphore(max(1, int(anthropic_num_threads)))
        self._openai_client = None
        self._azure_client = None
        self._anthropic_client = None

    async def __call__(
        self,
        *,
        model_id: str,
        prompt: Prompt,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        seed: int | None = None,
        **_: object,
    ) -> list[InferenceResponse]:
        if _is_anthropic_model(model_id):
            return [await self._anthropic_chat(model_id, prompt, max_tokens=max_tokens, temperature=temperature)]
        return [
            await self._openai_chat(
                model_id,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        ]

    async def _openai_chat(
        self,
        model_id: str,
        prompt: Prompt,
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> InferenceResponse:
        async with self._openai_sem:
            provider = _openai_provider()

            if provider == "azure":
                from openai import AsyncAzureOpenAI

                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
                api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
                api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
                if not endpoint or not api_key or not api_version:
                    raise RuntimeError(
                        "Azure OpenAI configured but missing one of: "
                        "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION"
                    )

                if self._azure_client is None:
                    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
                    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
                    self._azure_client = AsyncAzureOpenAI(
                        azure_endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version,
                        timeout=timeout,
                        max_retries=max_retries,
                    )

                deployment_fallback = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
                model_for_call = (
                    deployment_fallback
                    if deployment_fallback and model_id in _KNOWN_OPENAI_MODEL_IDS
                    else model_id
                )

                resp = await self._azure_client.chat.completions.create(  # type: ignore[union-attr]
                    model=model_for_call,
                    messages=_openai_messages(prompt),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **({} if seed is None else {"seed": seed}),
                )
                return InferenceResponse(completion=resp.choices[0].message.content or "")

            from openai import AsyncOpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for OpenAI API calls (set it in `.env`).")

            if self._openai_client is None:
                timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
                max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
                self._openai_client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

            resp = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
                model=model_id,
                messages=_openai_messages(prompt),
                max_tokens=max_tokens,
                temperature=temperature,
                **({} if seed is None else {"seed": seed}),
            )
            return InferenceResponse(completion=resp.choices[0].message.content or "")

    async def _anthropic_chat(
        self,
        model_id: str,
        prompt: Prompt,
        *,
        max_tokens: int,
        temperature: float,
    ) -> InferenceResponse:
        async with self._anthropic_sem:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Missing ANTHROPIC_API_KEY for Anthropic API calls (set it in `.env`).")

            if self._anthropic_client is None:
                self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)

            system, messages = _anthropic_system_and_messages(prompt)
            resp = await self._anthropic_client.messages.create(  # type: ignore[union-attr]
                model=model_id,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text_parts: list[str] = []
            for block in resp.content:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    text_parts.append(t)
            return InferenceResponse(completion="".join(text_parts))
