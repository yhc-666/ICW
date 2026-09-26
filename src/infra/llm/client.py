"""Bounded, asynchronous calls to an OpenAI-compatible chat endpoint."""

import asyncio
import json
import math
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlparse

import httpx


Result = TypeVar("Result")
MAX_HTTP_ATTEMPTS = 4
MAX_PARSE_ATTEMPTS = 3
MAX_TOKENS = 32768
RESERVED_FIELDS = {
    "model", "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens",
    "response_format", "stream", "n", "tools", "tool_choice",
}


def parse_extra_body(text: str) -> dict:
    """Read provider options without overriding the experiment payload."""
    value = json.loads(text)
    if not isinstance(value, dict) or RESERVED_FIELDS.intersection(value):
        raise ValueError("extra body must be an object without reserved chat fields")
    return value


class LLMClient:
    def __init__(self, *, base_url: str, api_key: str, concurrency: int = 8):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API_BASE_URL must be an HTTP(S) URL")
        if not api_key.strip() or concurrency < 1:
            raise ValueError("API_KEY must be nonempty and concurrency must be positive")
        self.semaphore = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.http.aclose()

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        parse: Callable[[dict], Result],
        extra_body: dict,
    ) -> Result:
        """Retry invalid output without changing the requested judgment."""
        if not model.strip() or RESERVED_FIELDS.intersection(extra_body):
            raise ValueError("invalid model or extra body")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
            **extra_body,
        }
        for attempt in range(MAX_PARSE_ATTEMPTS):
            response = await self._post(payload)
            try:
                choice = response["choices"][0]
                message = choice["message"]
                if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
                    raise RuntimeError("provider refused a judgment")
                if choice.get("finish_reason") != "stop":
                    raise ValueError("incomplete chat response")
                value = json.loads(message["content"])
                if not isinstance(value, dict):
                    raise ValueError("chat response must be a JSON object")
                return parse(value)
            except (KeyError, IndexError, TypeError, ValueError) as error:
                if attempt + 1 == MAX_PARSE_ATTEMPTS:
                    raise RuntimeError("model did not return a valid judgment") from error
        raise RuntimeError("no model judgment returned")

    async def _post(self, payload: dict) -> dict:
        """Retry transient transport errors and rate limits."""
        for attempt in range(MAX_HTTP_ATTEMPTS):
            delay = min(2.0 ** attempt, 30.0)
            try:
                async with self.semaphore:
                    response = await self.http.post("chat/completions", json=payload)
            except httpx.TransportError:
                if attempt + 1 == MAX_HTTP_ATTEMPTS:
                    raise RuntimeError("chat request failed after transport retries") from None
            else:
                if response.is_success:
                    return response.json()
                retryable = response.status_code in {408, 429} or response.status_code >= 500
                if not retryable or attempt + 1 == MAX_HTTP_ATTEMPTS:
                    raise RuntimeError(f"chat endpoint returned HTTP {response.status_code}")
                retry_after = response.headers.get("Retry-After", "")
                try:
                    seconds = float(retry_after)
                except ValueError:
                    seconds = 0.0
                if math.isfinite(seconds) and seconds > 0:
                    delay = max(delay, seconds)
            await asyncio.sleep(delay)
        raise RuntimeError("chat request failed")
