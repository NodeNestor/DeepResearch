from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

from pydantic_settings import BaseSettings
from pydantic import Field

log = logging.getLogger(__name__)

_CONFIG_FILE = Path("/app/data/config.json")


class LLMConfig(BaseSettings):
    """Config for a single LLM endpoint (bulk or synthesis)."""

    provider: str = "vllm"
    model: str = ""
    api_url: str = "http://vllm:8000/v1"
    api_key: str = ""
    max_model_len: int = 131072
    max_tokens: int = 16384
    quantization: str = "gptq"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Bulk model (extraction / swarm)
    bulk_provider: str = "vllm"
    bulk_model: str = "Qwen/Qwen3.5-0.8B-GPTQ-Int4"
    bulk_api_url: str = "http://vllm:8000/v1"
    bulk_api_key: str = ""
    bulk_max_model_len: int = 131072
    bulk_max_tokens: int = 16384
    bulk_quantization: str = "gptq"

    # Synthesis model (report writer — can be a different provider entirely)
    synthesis_provider: str = "vllm"
    synthesis_model: str = "Qwen/Qwen3.5-9B-GPTQ-Int4"
    synthesis_api_url: str = "http://vllm:8000/v1"
    synthesis_api_key: str = ""
    synthesis_max_model_len: int = 131072
    synthesis_max_tokens: int = 32768
    synthesis_quantization: str = "gptq"

    # GPU
    gpu_memory_utilization: float = 0.92
    gpu_device: str = "GPU-3ad3e2fe"
    kv_cache_dtype: str = "fp8"

    # Services
    searxng_url: str = "http://searxng:8080"
    hivemind_url: str = "http://hiveminddb:8100"

    # Research defaults
    max_depth: int = 3
    max_concurrent_fetches: int = 50
    max_concurrent_llm: int = 100
    queries_per_source: int = 4
    quality_threshold: int = 4
    swarm_agents: int = 5

    # Optional API keys
    github_token: str = ""
    hf_token: str = ""

    @property
    def bulk_llm(self) -> LLMConfig:
        return LLMConfig(
            provider=self.bulk_provider,
            model=self.bulk_model,
            api_url=self.bulk_api_url,
            api_key=self.bulk_api_key,
            max_model_len=self.bulk_max_model_len,
            max_tokens=self.bulk_max_tokens,
            quantization=self.bulk_quantization,
        )

    @property
    def synthesis_llm(self) -> LLMConfig:
        return LLMConfig(
            provider=self.synthesis_provider,
            model=self.synthesis_model,
            api_url=self.synthesis_api_url,
            api_key=self.synthesis_api_key,
            max_model_len=self.synthesis_max_model_len,
            max_tokens=self.synthesis_max_tokens,
            quantization=self.synthesis_quantization,
        )


# ── Mutable runtime config ──────────────────────────────────────────

# Env-based defaults (immutable)
settings = Settings()

# Runtime overrides — these are what the API and research pipeline actually use.
# Updated via PUT /api/config. Persisted to disk so they survive container restarts.

_lock = Lock()

# Fields exposed to the frontend for runtime editing
_RUNTIME_FIELDS = {
    "bulk_provider", "bulk_model", "bulk_api_url", "bulk_api_key", "bulk_max_tokens",
    "synthesis_provider", "synthesis_model", "synthesis_api_url", "synthesis_api_key", "synthesis_max_tokens",
    "hivemind_url", "searxng_url",
    "max_depth", "swarm_agents",
}


def _load_persisted() -> dict:
    """Load runtime overrides from disk."""
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception as e:
            log.warning("Failed to load persisted config: %s", e)
    return {}


def _save_persisted(data: dict) -> None:
    """Persist runtime overrides to disk."""
    try:
        _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning("Failed to save config: %s", e)


# Apply any persisted overrides on startup
_overrides = _load_persisted()
for k, v in _overrides.items():
    if k in _RUNTIME_FIELDS and hasattr(settings, k):
        object.__setattr__(settings, k, v)


def get_runtime_config() -> dict:
    """Return current runtime-editable config (API keys masked)."""
    with _lock:
        result = {}
        for field in _RUNTIME_FIELDS:
            val = getattr(settings, field, "")
            if "api_key" in field and val:
                result[field] = "***"
            else:
                result[field] = val
        return result


def update_runtime_config(updates: dict) -> dict:
    """Apply partial updates to runtime config. Returns new config."""
    with _lock:
        current_persisted = _load_persisted()
        for key, value in updates.items():
            if key not in _RUNTIME_FIELDS:
                continue
            # Skip masked API keys (user didn't change them)
            if "api_key" in key and value == "***":
                continue
            if hasattr(settings, key):
                object.__setattr__(settings, key, value)
                current_persisted[key] = value
        _save_persisted(current_persisted)
    return get_runtime_config()
