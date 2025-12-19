## Azure AI Foundry (Azure OpenAI) integration context (Python)

This document summarizes a **general, reusable** pattern for using an **Azure AI Foundry / Azure OpenAI** model (e.g., `gpt-4o-mini`) from a Python codebase.

---

## Mental model

- **Azure AI Foundry / Azure AI Studio** is where you **select a model** and create a **deployment**.
- Your application does **not** call the “model name” directly. It calls an Azure **deployment name** through the Azure OpenAI endpoint.
- In the Python SDK call, you pass the deployment name as `model=...`.

---

## What you need from Azure (once per environment)

From Azure AI Foundry / Azure AI Studio (or the Azure Portal):

- **Azure OpenAI endpoint**: your resource endpoint URL.
- **Authentication**:
  - **API key** (simplest), or
  - **Entra ID / managed identity** (recommended for production when feasible).
- **API version**: a version string supported by your scenario.
  - Newer features (like strict structured outputs) may require a **newer** API version.
- **Deployment name(s)**:
  - One deployment for text/chat (e.g., `gpt-4o-mini`), and optionally
  - Another deployment for multimodal/vision (if you use image inputs).

---

## Recommended environment variables

Set these as secrets in your deployment environment and optionally in a local `.env`:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY` (if using key auth)
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

Optional:

- `AZURE_OPENAI_VISION_DEPLOYMENT`
- `OPENAI_TIMEOUT_SECONDS` (if you want configurable timeouts)

---

## Dependencies

Install:

- `openai` (official SDK)

Optional:

- `python-dotenv` (to load `.env` locally)
  - In this repo, the `mats` entrypoints call `mats.utils.dotenv.load_mats_env()` to load `.env` automatically (if installed).

---

## Client setup (recommended pattern)

Create a small module that:

- Loads env vars
- Validates required configuration (endpoint/key/version)
- Creates a **singleton** `AzureOpenAI` client
- Sets `timeout` and `max_retries` defaults

---

## Making a basic chat request (text)

Key points:

- Use `client.chat.completions.create(...)`
- Provide messages: system + user
- Pass the Azure **deployment name** as `model=`

---

## Enforcing structured outputs (recommended for automation)

If you need reliably machine-parseable output:

- Define a JSON Schema for the response you want
- Call chat completions with `response_format` set to a `json_schema`
- Parse the response content as JSON and validate the required shape

This is more reliable than “please output JSON” in the prompt alone.

---

## Vision / multimodal (optional)

If you want image understanding:

- Base64-encode the image bytes
- Send a user message whose `content` is a list of items:
  - `{ "type": "text", "text": "..." }`
  - `{ "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }`
- Keep outputs short with `max_tokens` and use a tightly-scoped system message

---

## Reliability and error handling (minimum viable)

Treat failures differently depending on likely cause:

- **Retryable** (transient):
  - timeouts
  - connection errors
  - rate limits (`429`)
  - many `5xx` server errors
- **Non-retryable** (permanent):
  - `400`/bad request (often schema mismatch, unsupported feature, invalid message format)

Also:

- Log request metadata (deployment name, latency, retries)
- Keep `max_retries` small and implement your own backoff if needed

---

## Minimal reference snippet (drop-in shape)

```python
import os
import json
from openai import AzureOpenAI

_client = None

def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ["AZURE_OPENAI_API_VERSION"]

        # For key auth (simplest): pass api_key.
        # For Entra ID auth: use Azure AD token provider instead (different setup).
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is required for key auth")

        _client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
            max_retries=2,
        )

    return _client


def chat_text(system: str, user: str) -> str:
    client = get_client()
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return resp.choices[0].message.content


def chat_structured(system: str, user: str) -> dict:
    client = get_client()
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    schema = {
        "name": "ExampleResponse",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }

    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )

    return json.loads(resp.choices[0].message.content)
```
