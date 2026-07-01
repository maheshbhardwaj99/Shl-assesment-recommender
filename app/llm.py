"""
Thin wrapper around the Gemini REST API.

We call generateContent directly over HTTP (no google-generativeai SDK) to:
  - keep the dependency footprint small for free-tier hosting,
  - get full control over the request timeout (the evaluator gives us a
    30s/call budget total, so the LLM call must leave room for retrieval +
    validation on both sides of it).

response_mime_type="application/json" + a response_schema is used so the
model is constrained to emit valid, parseable JSON instead of us having to
regex a code fence out of free text.
"""
import json
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()  # so GEMINI_API_KEY in .env is picked up without manual `export`/`$env:` every session

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["clarify", "recommend", "refine", "compare", "refuse"],
        },
        "reply": {"type": "STRING"},
        "recommended_urls": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "conversation_complete": {"type": "BOOLEAN"},
    },
    "required": ["action", "reply", "recommended_urls", "conversation_complete"],
}


class LLMError(RuntimeError):
    pass


def call_llm(system_prompt: str, user_prompt: str, timeout: float = 20.0, retries: int = 2) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set")

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "maxOutputTokens": 1024,
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
            if resp.status_code == 429 and attempt < retries:
                # Free-tier rate limit — back off longer than a generic transient error.
                time.sleep(3.0 * (attempt + 1))
                last_err = f"429 rate limited: {resp.text[:200]}"
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception as e:  # noqa: BLE001 - we deliberately want a broad fallback
            last_err = e
            if attempt < retries:
                time.sleep(0.6)
                continue
    raise LLMError(f"Gemini call failed after {retries + 1} attempt(s): {last_err}")
