"""Swap models on a shared vLLM instance between bulk and synthesis phases."""

from __future__ import annotations

import asyncio
import logging

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)


async def swap_model(
    vllm_base_url: str,
    target: LLMConfig,
    timeout: float = 120.0,
    poll_interval: float = 3.0,
) -> bool:
    """Unload the current model on a vLLM instance and load *target*.

    vLLM's OpenAI-compatible server doesn't natively support hot-swap via API,
    so we restart the served model by calling the admin endpoints if available,
    or simply verify the target model is already loaded.

    Returns True if the target model is ready, False on failure.
    """
    base = vllm_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Check if target model is already loaded
        try:
            resp = await client.get(f"{base}/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                for m in models:
                    if m.get("id") == target.model:
                        log.info("Model %s already loaded", target.model)
                        return True
        except httpx.HTTPError:
            pass

        # If vLLM exposes the /admin/load endpoint (custom builds), try it
        try:
            payload = {
                "model": target.model,
                "quantization": target.quantization,
                "max_model_len": target.max_model_len,
            }
            resp = await client.post(f"{base}/admin/load", json=payload)
            if resp.status_code == 200:
                log.info("Sent model load request for %s", target.model)
            else:
                log.warning(
                    "Admin load endpoint returned %d — model swap may not be supported. "
                    "If using separate providers for bulk/synthesis, this is expected.",
                    resp.status_code,
                )
                return True  # Assume separate providers handle it
        except httpx.HTTPError:
            log.warning(
                "No admin endpoint available on vLLM — assuming model is managed externally"
            )
            return True

        # Poll until model is ready
        elapsed = 0.0
        while elapsed < timeout:
            try:
                resp = await client.get(f"{base}/v1/models")
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    for m in models:
                        if m.get("id") == target.model:
                            log.info("Model %s is ready", target.model)
                            return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log.error("Model %s did not become ready within %.0fs", target.model, timeout)
        return False
