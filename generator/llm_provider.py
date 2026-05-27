"""
Unified LLM Provider Layer (v20)
=================================
Single entry point for ALL LLM calls across the pipeline.

Supports two backends:
  1. **GeminiProvider** — Google Gemini API (production default)
  2. **OpenAICompatibleProvider** — HTTP client for vLLM, Ollama, TGI,
     LiteLLM, OpenAI, or another OpenAI-compatible server

Configuration via environment / .env::

    # --- Gemini (default) ---
    LLM_PROVIDER=gemini
    GOOGLE_API_KEY=<your-google-api-key>

    # --- OpenAI-compatible HTTP backend (started separately) ---
    LLM_PROVIDER=local
    LOCAL_LLM_URL=http://localhost:8000/v1
    LOCAL_LLM_MODEL=<model-served-by-your-backend>

Usage in any script::

    from llm_provider import get_llm_client
    client = get_llm_client()
    response = client.generate("Your prompt here", json_mode=True, temperature=0.5)
    print(response.text)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import ssl
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional

from dotenv import load_dotenv
from model_defaults import DEFAULT_GEMINI_MODEL, model_for_role

# Load .env from project root (one level up from script dir)
_script_dir = Path(__file__).resolve().parent
_env_path = _script_dir.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
# Also try local .env
_local_env = _script_dir / ".env"
if _local_env.exists():
    load_dotenv(_local_env, override=True)


# ═══════════════════════════════════════════════════════════════════════
# Response / Stats dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """Unified response object returned by all providers."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    cached: bool = False
    raw: Any = None  # original provider response for advanced usage


@dataclass
class RetryStats:
    """Statistics from a single generate call (for logging/debugging)."""
    attempts: int = 0
    retries: int = 0
    total_sleep_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Retry / Resilience (provider-agnostic)
# ═══════════════════════════════════════════════════════════════════════

_RETRIABLE_TOKENS = (
    "503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
    "DEADLINEEXCEEDED", "DEADLINE_EXCEEDED", "TIMEOUT",
    "TIMED OUT", "SERVICE UNAVAILABLE", "rate limit",
    "too many requests", "overloaded",
)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = int(default)
    else:
        try:
            value = int(float(str(raw).strip()))
        except Exception:
            value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = float(default)
    else:
        try:
            value = float(str(raw).strip())
        except Exception:
            value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _build_gemini_http_options() -> Any | None:
    """Build HTTP options for Gemini calls.

    On Windows we prefer the OS trust store over certifi/httpx defaults so
    research runs do not randomly fail behind enterprise TLS interception.
    """
    from google.genai import types  # type: ignore

    client_args: Dict[str, Any] = {"trust_env": True}

    disable_verify = _env_flag("DATA_SYS_GEMINI_DISABLE_SSL_VERIFY", False)
    custom_ca = (
        os.getenv("DATA_SYS_GEMINI_SSL_CERT_FILE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )

    if disable_verify:
        client_args["verify"] = False
    elif custom_ca:
        client_args["verify"] = str(custom_ca)
    elif os.name == "nt":
        client_args["verify"] = ssl.create_default_context()

    if not client_args:
        return None
    return types.HttpOptions(client_args=client_args)


def _write_usage_log(entry: Dict[str, Any]) -> None:
    targets = []
    primary = os.getenv("DATA_SYS_LLM_USAGE_LOG")
    latest = os.getenv("DATA_SYS_LLM_USAGE_LOG_LATEST")
    for raw in (primary, latest):
        if raw and str(raw).strip():
            targets.append(str(raw).strip())
    unique_targets: list[str] = []
    seen: set[str] = set()
    for raw in targets:
        key = str(Path(raw))
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(raw)
    if not unique_targets:
        return
    for raw in unique_targets:
        try:
            target = Path(raw)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True, default=str))
                handle.write("\n")
        except Exception:
            pass


def _log_llm_success(
    *,
    response: "LLMResponse",
    prompt: str,
    json_mode: bool,
    temperature: float,
    max_tokens: int,
    stats: "RetryStats",
) -> None:
    _write_usage_log(
        {
            "event": "llm_call",
            "step_id": os.getenv("DATA_SYS_STEP_ID", ""),
            "step_name": os.getenv("DATA_SYS_STEP_NAME", ""),
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
            "model": response.model,
            "json_mode": bool(json_mode),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "prompt_chars": len(prompt),
            "input_tokens": int(response.input_tokens or 0),
            "output_tokens": int(response.output_tokens or 0),
            "total_tokens": int(response.input_tokens or 0) + int(response.output_tokens or 0),
            "cached": bool(response.cached),
            "attempts": int(stats.attempts),
            "retries": int(stats.retries),
            "sleep_sec": round(float(stats.total_sleep_sec), 4),
            "timeout_sec": float(os.getenv("DATA_SYS_LLM_TIMEOUT_SEC", "0") or 0),
            "retry_budget": int(os.getenv("DATA_SYS_LLM_MAX_ATTEMPTS", "0") or 0),
            "cost_usd": float(response.cost_usd or 0.0),
            "timestamp": time.time(),
        }
    )


def _log_llm_failure(
    *,
    model: str,
    prompt: str,
    json_mode: bool,
    temperature: float,
    max_tokens: int,
    stats: "RetryStats",
    error: BaseException | None,
) -> None:
    _write_usage_log(
        {
            "event": "llm_error",
            "step_id": os.getenv("DATA_SYS_STEP_ID", ""),
            "step_name": os.getenv("DATA_SYS_STEP_NAME", ""),
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
            "model": str(model or ""),
            "json_mode": bool(json_mode),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "prompt_chars": len(prompt),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "attempts": int(stats.attempts),
            "retries": int(stats.retries),
            "sleep_sec": round(float(stats.total_sleep_sec), 4),
            "timeout_sec": float(os.getenv("DATA_SYS_LLM_TIMEOUT_SEC", "0") or 0),
            "retry_budget": int(os.getenv("DATA_SYS_LLM_MAX_ATTEMPTS", "0") or 0),
            "error": str(error) if error is not None else "",
            "timestamp": time.time(),
        }
    )


def _is_retriable(exc: BaseException) -> bool:
    msg = str(exc).upper()
    return any(tok in msg for tok in _RETRIABLE_TOKENS)


def _call_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    result: dict[str, Any] = {"value": None, "error": None}

    def _runner() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # pragma: no cover - passthrough
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout_sec:.0f}s")
    if result["error"] is not None:
        raise result["error"]
    return result["value"]


# ═══════════════════════════════════════════════════════════════════════
# Abstract Provider
# ═══════════════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        """Execute a single LLM call (no retry). Subclasses implement this."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        thinking_budget: int | None = None,
        timeout_sec: float = 70.0,
        max_attempts: int = 5,
        base_delay_sec: float = 4.0,
        max_delay_sec: float = 35.0,
        on_retry: Optional[Callable] = None,
    ) -> LLMResponse:
        """Generate with timeout + exponential backoff retry."""
        stats = RetryStats()
        last_err: Optional[BaseException] = None
        rng = random.Random()

        effective_model = model or self.default_model
        effective_timeout_sec = _env_float("DATA_SYS_LLM_TIMEOUT_SEC", timeout_sec, minimum=5.0)
        effective_max_attempts = _env_int("DATA_SYS_LLM_MAX_ATTEMPTS", max_attempts, minimum=1)
        effective_base_delay_sec = _env_float("DATA_SYS_LLM_BASE_DELAY_SEC", base_delay_sec, minimum=0.0)
        effective_max_delay_sec = _env_float(
            "DATA_SYS_LLM_MAX_DELAY_SEC",
            max(max_delay_sec, effective_base_delay_sec),
            minimum=effective_base_delay_sec,
        )

        for attempt in range(1, effective_max_attempts + 1):
            stats.attempts += 1
            try:
                resp = _call_with_timeout(
                    lambda: self._raw_generate(
                        prompt,
                        model=effective_model,
                        json_mode=json_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        thinking_budget=thinking_budget,
                    ),
                    timeout_sec=effective_timeout_sec,
                )
                resp.model = effective_model
                _log_llm_success(
                    response=resp,
                    prompt=prompt,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stats=stats,
                )
                return resp
            except BaseException as exc:
                last_err = exc
                stats.errors.append(str(exc))

                retriable = isinstance(exc, TimeoutError) or _is_retriable(exc)

                if (not retriable) or attempt >= effective_max_attempts:
                    break

                stats.retries += 1
                delay = min(effective_max_delay_sec, effective_base_delay_sec * (2 ** (attempt - 1)))
                jitter = delay * 0.25 * rng.random()
                sleep_for = delay + jitter
                stats.total_sleep_sec += sleep_for
                if on_retry:
                    on_retry(attempt, effective_max_attempts, exc, sleep_for)
                time.sleep(sleep_for)

        _log_llm_failure(
            model=effective_model,
            prompt=prompt,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            stats=stats,
            error=last_err,
        )
        raise RuntimeError(
            f"LLM call failed after {stats.attempts} attempts: {last_err}"
        ) from last_err

    @property
    @abstractmethod
    def default_model(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════════════════
# Gemini Provider (Google AI Studio / Vertex)
# ═══════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    """Google Gemini API backend."""

    def __init__(self, api_key: str, default_model: str = DEFAULT_GEMINI_MODEL):
        from google import genai  # type: ignore
        http_options = _build_gemini_http_options()
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if http_options is not None:
            client_kwargs["http_options"] = http_options
        self._client = genai.Client(**client_kwargs)
        self._default_model = default_model

    @property
    def default_model(self) -> str:
        return self._default_model

    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if thinking_budget is not None:
            budget = int(thinking_budget)
            if not (budget <= 0 and "pro" in str(model).lower()):
                config["thinking_config"] = {"thinking_budget": budget}
        if json_mode:
            config["response_mime_type"] = "application/json"
        if max_tokens:
            config["max_output_tokens"] = max_tokens

        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        text = str(getattr(resp, "text", "") or "")

        # Extract token usage
        usage = getattr(resp, "usage_metadata", None)
        inp_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

        return LLMResponse(
            text=text,
            input_tokens=inp_tokens,
            output_tokens=out_tokens,
            model=model,
            raw=resp,
        )


# ═══════════════════════════════════════════════════════════════════════
# OpenAI-Compatible Provider (vLLM, Ollama, TGI, LiteLLM, etc.)
# ═══════════════════════════════════════════════════════════════════════

class OpenAICompatibleProvider(LLMProvider):
    """Any server exposing the OpenAI /v1/chat/completions API.

    Works with:
      - vLLM: ``python -m vllm.entrypoints.openai.api_server --model <model-served-by-the-lab>``
      - Ollama: ``ollama serve`` (default port 11434, endpoint /v1/chat/completions)
      - TGI: ``text-generation-launcher --model-id ...``
      - LiteLLM proxy: ``litellm --model ...``

    Configuration via environment:
      - LOCAL_LLM_URL: base URL (e.g. http://localhost:8000/v1)
      - LOCAL_LLM_MODEL: model name as registered on the server
      - LOCAL_LLM_API_KEY: optional API key (some servers require one)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "served-model",
        api_key: str = "not-needed",
    ):
        self._base_url = base_url.rstrip("/")
        self._default_model = model
        self._api_key = api_key

        # Lazy import — requests is almost always available
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for the OpenAI-compatible provider. "
                "Install with: pip install requests"
            )

    @property
    def default_model(self) -> str:
        return self._default_model

    def _ollama_native_base(self) -> str:
        base = self._base_url
        if base.endswith("/v1"):
            return base[:-3]
        return base

    def _looks_like_ollama(self) -> bool:
        url = self._base_url.lower()
        return "11434" in url or "ollama" in url

    def _use_ollama_native(self) -> bool:
        value = os.getenv("DATA_SYS_OLLAMA_NATIVE", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _apply_ollama_think_option(self, body: Dict[str, Any]) -> None:
        value = os.getenv("DATA_SYS_OLLAMA_THINK", "0").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            body["think"] = True
        elif value in {"0", "false", "no", "off"}:
            body["think"] = False

    def _parse_ollama_native_response(self, data: Dict[str, Any], *, model: str) -> LLMResponse:
        message = data.get("message", {}) if isinstance(data, dict) else {}
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        if not text and isinstance(data, dict):
            text = str(data.get("response") or "")
        inp_tokens = 0
        out_tokens = 0
        if isinstance(data, dict):
            inp_tokens = int(data.get("prompt_eval_count") or 0)
            out_tokens = int(data.get("eval_count") or 0)
        return LLMResponse(
            text=text,
            input_tokens=inp_tokens,
            output_tokens=out_tokens,
            model=model,
            raw=data,
        )

    def _raw_generate_ollama_native(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        import requests

        url = f"{self._ollama_native_base().rstrip('/')}/api/chat"
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        self._apply_ollama_think_option(body)

        resp = requests.post(url, json=body, timeout=120)
        if resp.status_code == 404:
            return self._raw_generate_ollama_generate(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_ollama_native_response(data, model=model)

    def _raw_generate_ollama_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        import requests

        url = f"{self._ollama_native_base().rstrip('/')}/api/generate"
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        self._apply_ollama_think_option(body)

        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_ollama_native_response(data, model=model)

    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        import requests

        if self._looks_like_ollama() and self._use_ollama_native():
            return self._raw_generate_ollama_native(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key and self._api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self._api_key}"

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = requests.post(url, json=body, headers=headers, timeout=120)
        if resp.status_code == 404 and self._looks_like_ollama():
            return self._raw_generate_ollama_native(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
            )
        resp.raise_for_status()
        data = resp.json()

        # Parse OpenAI-format response
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        if isinstance(text, list):
            parts: list[str] = []
            for item in text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            text = "".join(parts)
        text = str(text or "")

        usage = data.get("usage", {})
        inp_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            text=text,
            input_tokens=inp_tokens,
            output_tokens=out_tokens,
            model=model,
            raw=data,
        )


# ═══════════════════════════════════════════════════════════════════════
# Factory / Singleton
# ═══════════════════════════════════════════════════════════════════════

_global_client: Optional[LLMProvider] = None


def get_llm_client(
    *,
    provider: Optional[str] = None,
    force_new: bool = False,
) -> LLMProvider:
    """Get or create the global LLM client.

    Provider is determined by (in priority order):
      1. ``provider`` argument
      2. ``LLM_PROVIDER`` env var
      3. Default: "gemini"

    Returns a configured LLMProvider with retry logic built in.
    """
    global _global_client

    if _global_client is not None and not force_new:
        return _global_client

    provider_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()

    if provider_name in ("gemini", "google"):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but no API key found.\n"
                "Set GOOGLE_API_KEY in your .env file or environment."
            )
        default_model = model_for_role("entity_gen")

        _global_client = GeminiProvider(api_key=api_key, default_model=default_model)

    elif provider_name in ("local", "vllm", "ollama", "openai", "tgi", "litellm"):
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
        model = os.getenv("LOCAL_LLM_MODEL", "")
        if not model:
            raise RuntimeError(
                "LLM_PROVIDER=local requires LOCAL_LLM_MODEL.\n"
                "Set it to the model name exposed by the OpenAI-compatible backend."
            )
        api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        _global_client = OpenAICompatibleProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider_name}'. "
            f"Supported: gemini, local, vllm, ollama, tgi, litellm"
        )

    return _global_client


class LLMManager:
    """Compatibility wrapper for older critic code paths."""

    def __init__(self, workspace: Any | None = None) -> None:
        self.workspace = workspace
        self._client = get_llm_client()

    def generate(
        self,
        *,
        role: str,
        contents: str,
        schema_name: str | None = None,
        seed: int | None = None,
        model: str | None = None,
        thinking_budget: int | None = None,
    ) -> tuple[LLMResponse, int, int, float, bool]:
        del schema_name, seed
        response = self._client.generate(
            contents,
            model=model,
            json_mode=True,
            temperature=0.2 if str(role or "").lower() == "critic" else 0.35,
            max_tokens=4096 if str(role or "").lower() == "critic" else 3072,
            timeout_sec=90.0,
            max_attempts=4,
            thinking_budget=thinking_budget,
        )
        return (
            response,
            int(response.input_tokens or 0),
            int(response.output_tokens or 0),
            float(response.cost_usd or 0.0),
            bool(response.cached),
        )


# ═══════════════════════════════════════════════════════════════════════
# Convenience helpers (used by refactored scripts)
# ═══════════════════════════════════════════════════════════════════════

def llm_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    json_mode: bool = False,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    thinking_budget: int | None = None,
    timeout_sec: float = 70.0,
    max_attempts: int = 5,
    on_retry: Optional[Callable] = None,
) -> LLMResponse:
    """One-liner for LLM generation. Gets the global client automatically.

    Usage::
        from llm_provider import llm_generate
        resp = llm_generate("Generate a movie concept", json_mode=True)
        data = json.loads(resp.text)
    """
    client = get_llm_client()
    return client.generate(
        prompt,
        model=model,
        json_mode=json_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        on_retry=on_retry,
    )


def safe_json_parse(text: str) -> Any:
    """Parse JSON from LLM response, handling common issues."""
    text = str(text or "").strip()
    if not text:
        return None
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = (text[:start] + text[end + len("</think>"):]).strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None
