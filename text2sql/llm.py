"""Minimal client for any OpenAI-compatible chat API (OpenAI, Groq, Gemini,
OpenRouter, a local Ollama server...). Uses only the standard library.

Configure with environment variables (or a .env file):
    LLM_API_KEY   your key (any non-empty value for Ollama)
    LLM_BASE_URL  e.g. https://api.openai.com/v1
    LLM_MODEL     a model name your provider offers
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 timeout_s: float = 60.0, max_retries: int = 4):
        self.model = model or os.getenv("LLM_MODEL", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        if not self.model:
            raise LLMError("Set LLM_MODEL (see .env.example).")
        if not self.api_key:
            raise LLMError("Set LLM_API_KEY (see .env.example).")

    def chat(self, messages: list[dict], max_tokens: int = 800) -> LLMResponse:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Some providers (e.g. Groq, behind Cloudflare) reject urllib's default
                # "Python-urllib/x.y" agent with HTTP 403 "error code: 1010".
                "User-Agent": "text2sql-assistant/1.0",
            },
        )
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                # Rate limits and server errors are worth retrying; bad requests are not.
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay)
                    delay *= 2
                    continue
                raise LLMError(f"LLM API error {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"Could not reach {self.base_url}: {e}") from e
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected API response: {str(data)[:300]}") from e
        usage = data.get("usage") or {}
        return LLMResponse(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
