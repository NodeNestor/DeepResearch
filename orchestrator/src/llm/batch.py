from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .client import LLMClient, CompletionResult

log = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from a batch of LLM completions."""

    texts: list[str] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    successful: int = 0
    failed: int = 0


async def batch_complete(
    client: LLMClient,
    prompts: list[str],
    system: str = "",
    max_tokens: int | None = None,
    json_schema: dict | None = None,
    max_concurrency: int = 100,
    thinking: bool = True,
    temperature: float = 0.3,
) -> BatchResult:
    """Send many prompts through *client* with bounded concurrency.

    Returns a BatchResult with texts (same length as prompts, empty string
    for failures) and aggregated token usage.
    """
    sem = asyncio.Semaphore(max_concurrency)
    result = BatchResult()

    async def _do(prompt: str) -> CompletionResult | None:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {"thinking": thinking, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_schema is not None:
            kwargs["json_schema"] = json_schema

        async with sem:
            try:
                return await client.complete(messages, **kwargs)
            except Exception:
                log.warning("batch item failed for prompt (%.80s…)", prompt, exc_info=True)
                return None

    completions = await asyncio.gather(*[_do(p) for p in prompts])

    texts: list[str] = []
    for cr in completions:
        if cr is not None:
            texts.append(cr.text)
            result.total_prompt_tokens += cr.prompt_tokens
            result.total_completion_tokens += cr.completion_tokens
            result.total_tokens += cr.total_tokens
            result.successful += 1
        else:
            texts.append("")
            result.failed += 1

    result.texts = texts
    return result
