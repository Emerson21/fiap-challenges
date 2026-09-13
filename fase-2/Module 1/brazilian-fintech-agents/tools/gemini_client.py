# tools/gemini_client.py
"""
Shared Google Gemini API wrapper.
Reads GEMINI_API_KEY from the environment (or .env file).
All agents import this module to make LLM calls.
"""
import os
import json
from typing import Optional

from dotenv import load_dotenv
from tools.logger import get_logger

load_dotenv()  # loads .env if present

log = get_logger(__name__)

try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    log.warning("google-genai not installed — LLM calls will use fallback mode")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def is_available() -> bool:
    """Return True if a valid API key is configured and SDK is installed."""
    return _GENAI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY"))


def call(prompt: str, model: str = DEFAULT_MODEL, expect_json: bool = False) -> Optional[str]:
    """
    Send a prompt to Gemini and return the text response.

    Args:
        prompt (str): The full prompt to send.
        model (str): Gemini model name (default: gemini-2.0-flash).
        expect_json (bool): If True, strip markdown code fences before returning.

    Returns:
        str | None: Response text, or None if the call failed.
    """
    if not is_available():
        log.warning("Gemini not available (missing API key or SDK) — returning None")
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text.strip()

        if expect_json:
            text = _strip_code_fences(text)

        log.info("Gemini call succeeded — model: %s | chars: %d", model, len(text))
        return text

    except Exception as exc:
        log.error("Gemini call failed: %s", exc)
        return None


def call_json(prompt: str, model: str = DEFAULT_MODEL) -> Optional[dict]:
    """
    Send a prompt expecting a JSON response. Returns parsed dict or None.
    """
    raw = call(prompt, model=model, expect_json=True)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Failed to parse Gemini JSON response: %s\nRaw: %.200s", exc, raw)
        return None


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
