from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from dataclasses import dataclass

from openai import AzureOpenAI

from mats.utils.dotenv import load_mats_env


@dataclass(frozen=True)
class AzureOpenAIConfig:
    azure_openai_endpoint: str
    azure_api_key: str
    azure_openai_api_version: str
    azure_openai_deployment: str
    timeout_seconds: float = 60.0
    max_retries: int = 2


def load_azure_openai_config() -> AzureOpenAIConfig:
    load_mats_env()

    endpoint = _normalize_azure_endpoint(os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip())
    api_key = os.environ.get("AZURE_API_KEY", "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip() or os.environ.get(
        "AZURE_OPENAI_CHAT_DEPLOYMENT", ""
    ).strip()

    missing: list[str] = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing.append("AZURE_API_KEY (or AZURE_OPENAI_API_KEY)")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_CHAT_DEPLOYMENT)")

    if missing:
        raise RuntimeError(f"Missing Azure OpenAI env vars: {', '.join(missing)}")

    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    return AzureOpenAIConfig(
        azure_openai_endpoint=endpoint,
        azure_api_key=api_key,
        azure_openai_api_version=api_version,
        azure_openai_deployment=deployment,
        timeout_seconds=timeout,
        max_retries=max_retries,
    )


_client: AzureOpenAI | None = None


def get_azure_openai_client() -> AzureOpenAI:
    global _client
    if _client is None:
        cfg = load_azure_openai_config()
        _client = AzureOpenAI(
            azure_endpoint=cfg.azure_openai_endpoint,
            api_key=cfg.azure_api_key,
            api_version=cfg.azure_openai_api_version,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )
    return _client


def chat_text(system: str, user: str, *, max_tokens: int | None = None) -> str:
    cfg = load_azure_openai_config()
    client = get_azure_openai_client()

    resp = client.chat.completions.create(
        model=cfg.azure_openai_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **({} if max_tokens is None else {"max_tokens": max_tokens}),
    )
    return resp.choices[0].message.content or ""


def chat_structured(
    system: str,
    user: str,
    *,
    json_schema: dict,
    max_tokens: int | None = None,
) -> dict:
    cfg = load_azure_openai_config()
    client = get_azure_openai_client()

    resp = client.chat.completions.create(
        model=cfg.azure_openai_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": json_schema},
        **({} if max_tokens is None else {"max_tokens": max_tokens}),
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def _normalize_azure_endpoint(endpoint: str) -> str:
    if not endpoint:
        return endpoint
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint
    return f"{parsed.scheme}://{parsed.netloc}"
