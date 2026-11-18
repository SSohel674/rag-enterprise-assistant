"""Thin LLM client supporting Ollama (local), OpenAI-compatible APIs, and Anthropic."""
import os

import requests

from .config import settings

DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}


class LLMError(RuntimeError):
    pass


def generate(prompt: str, system: str = "", temperature: float = 0.1, timeout: int = 120) -> str:
    provider = settings.llm_provider
    if provider not in DEFAULT_MODELS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use ollama, openai or anthropic.")
    model = settings.llm_model or DEFAULT_MODELS[provider]

    try:
        if provider == "ollama":
            url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
            r = requests.post(
                url,
                json={
                    "model": model,
                    "stream": False,
                    "options": {"temperature": temperature},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()

        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMError("OPENAI_API_KEY is not set.")
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        # anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()

    except requests.RequestException as e:
        raise LLMError(f"LLM request failed ({provider}): {e}") from e
