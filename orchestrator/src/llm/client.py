from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx

from ..config import LLMConfig

log = logging.getLogger(__name__)

_OPENAI_PROVIDERS = {"vllm", "openai", "ollama"}
_RETRY_ATTEMPTS = 3
_TIMEOUT = 1800.0  # 30 min — large batches with 16k max_tokens take time


@dataclass
class CompletionResult:
    """Result from a single LLM completion call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """Provider-agnostic async LLM client.

    Supports OpenAI-compatible endpoints (vLLM, OpenAI, Ollama) and the
    Anthropic Messages API.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._is_anthropic = config.provider == "anthropic"
        headers: dict[str, str] = {}
        if self._is_anthropic:
            headers["x-api-key"] = config.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        headers["Content-Type"] = "application/json"
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(_TIMEOUT, connect=10.0),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        json_schema: dict | None = None,
        thinking: bool = True,
    ) -> CompletionResult:
        """Send a chat completion request and return result with token usage."""
        max_tokens = max_tokens or self.config.max_tokens
        if self._is_anthropic:
            return await self._complete_anthropic(
                messages, max_tokens, temperature, json_schema
            )
        return await self._complete_openai(
            messages, max_tokens, temperature, json_schema, thinking
        )

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> dict:
        """Send a chat completion with tool definitions and return the raw message.

        Returns a dict that can be directly appended to the conversation:
        - If no tool calls: {"role": "assistant", "content": "..."}
        - If tool calls: {"role": "assistant", "content": ..., "tool_calls": [...]}
        """
        max_tokens = max_tokens or self.config.max_tokens
        if self._is_anthropic:
            return await self._tools_anthropic(messages, tools, max_tokens, temperature)
        return await self._tools_openai(messages, tools, max_tokens, temperature)

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ #
    # OpenAI-compatible path (vLLM / OpenAI / Ollama)
    # ------------------------------------------------------------------ #

    async def _complete_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        json_schema: dict | None,
        thinking: bool = True,
    ) -> CompletionResult:
        url = f"{self.config.api_url.rstrip('/')}/chat/completions"
        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        return await self._post_with_retry(url, body, extractor=_extract_openai)

    # ------------------------------------------------------------------ #
    # Anthropic path
    # ------------------------------------------------------------------ #

    async def _complete_anthropic(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        json_schema: dict | None,
    ) -> CompletionResult:
        url = f"{self.config.api_url.rstrip('/')}/messages"
        system_text = ""
        chat_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                chat_msgs.append({"role": m["role"], "content": m["content"]})

        body: dict = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_msgs,
        }
        if system_text:
            body["system"] = system_text.strip()

        if json_schema is not None:
            body["tools"] = [
                {
                    "name": "json_response",
                    "description": "Return the structured JSON response.",
                    "input_schema": json_schema,
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": "json_response"}

        return await self._post_with_retry(url, body, extractor=_extract_anthropic)

    # ------------------------------------------------------------------ #
    # Tool-calling paths
    # ------------------------------------------------------------------ #

    async def _tools_openai(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        url = f"{self.config.api_url.rstrip('/')}/chat/completions"
        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        result = await self._post_with_retry(
            url, body, extractor=_extract_openai_with_tools
        )
        # result.text contains the raw message dict as JSON
        return json.loads(result.text)

    async def _tools_anthropic(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        url = f"{self.config.api_url.rstrip('/')}/messages"
        system_text = ""
        chat_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            elif m["role"] == "tool":
                # Convert to Anthropic tool_result format
                chat_msgs.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": m["content"],
                        }
                    ],
                })
            elif m["role"] == "assistant" and "tool_calls" in m:
                # Convert OpenAI-style tool_calls to Anthropic content blocks
                content_blocks = []
                if m.get("content"):
                    content_blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                chat_msgs.append({"role": "assistant", "content": content_blocks})
            else:
                chat_msgs.append({"role": m["role"], "content": m["content"]})

        # Convert OpenAI tool format to Anthropic tool format
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })

        body: dict = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_msgs,
            "tools": anthropic_tools,
        }
        if system_text:
            body["system"] = system_text.strip()

        result = await self._post_with_retry(
            url, body, extractor=_extract_anthropic_with_tools
        )
        return json.loads(result.text)

    # ------------------------------------------------------------------ #
    # Retry logic
    # ------------------------------------------------------------------ #

    async def _post_with_retry(self, url: str, body: dict, extractor) -> CompletionResult:
        last_err: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = await self._http.post(url, json=body)
                resp.raise_for_status()
                return extractor(resp.json())
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError, IndexError) as exc:
                last_err = exc
                wait = 2 ** attempt
                log.warning(
                    "LLM request failed (attempt %d/%d): %s: %s – retrying in %ds",
                    attempt + 1, _RETRY_ATTEMPTS, type(exc).__name__, exc, wait,
                )
                if attempt < _RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(wait)
        raise RuntimeError(f"LLM request failed after {_RETRY_ATTEMPTS} attempts: {last_err}")


# ------------------------------------------------------------------ #
# Response extractors — now return CompletionResult with token usage
# ------------------------------------------------------------------ #

def _extract_openai(data: dict) -> CompletionResult:
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    text = content if content else (reasoning or "")

    usage = data.get("usage", {})
    return CompletionResult(
        text=text,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def _extract_anthropic(data: dict) -> CompletionResult:
    text = ""
    for block in data.get("content", []):
        if block["type"] == "tool_use":
            text = json.dumps(block["input"])
            break
        if block["type"] == "text":
            text = block["text"]
            break

    usage = data.get("usage", {})
    return CompletionResult(
        text=text,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    )


# ------------------------------------------------------------------ #
# Tool-calling extractors — return raw message dict as JSON string
# ------------------------------------------------------------------ #

def _extract_openai_with_tools(data: dict) -> CompletionResult:
    """Extract OpenAI response preserving tool_calls in the message dict."""
    msg = data["choices"][0]["message"]
    # Build a normalized message dict
    result_msg: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        result_msg["tool_calls"] = msg["tool_calls"]

    usage = data.get("usage", {})
    return CompletionResult(
        text=json.dumps(result_msg),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def _extract_anthropic_with_tools(data: dict) -> CompletionResult:
    """Extract Anthropic response, converting tool_use blocks to OpenAI-style tool_calls."""
    content_text = ""
    tool_calls = []

    for block in data.get("content", []):
        if block["type"] == "text":
            content_text = block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                },
            })

    result_msg: dict = {"role": "assistant", "content": content_text}
    if tool_calls:
        result_msg["tool_calls"] = tool_calls

    usage = data.get("usage", {})
    return CompletionResult(
        text=json.dumps(result_msg),
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    )
