"""LiteLLM orchestrator — async code generation with per-provider semaphores,
tenacity retry, and budget integration.

Supports only the synchronous (non-batch) path for P0.1. Batch-mode wiring
for Sonnet+Gemini+GPT-mini lands in P0.3.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .budget import Budget
from .config import (
    CODE_GEN_KWARGS,
    CODE_GEN_TOP_K,
    MODELS,
    ModelSpec,
    build_code_user_message,
)

# Silently drop provider-unsupported params (e.g. top_k on OpenAI).
litellm.drop_params = True
# Turn off LiteLLM's noisy stdout logging; we log what we care about ourselves.
litellm.suppress_debug_info = True


# ── Retry policy ────────────────────────────────────────────────────────────
_RETRY_EXC = (
    getattr(litellm.exceptions, "RateLimitError", Exception),
    getattr(litellm.exceptions, "ServiceUnavailableError", Exception),
    getattr(litellm.exceptions, "InternalServerError", Exception),
    getattr(litellm.exceptions, "APIConnectionError", Exception),
    getattr(litellm.exceptions, "Timeout", Exception),
)


@dataclass
class GenerationResult:
    generated_code: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    model_key: str
    retry_reason: str | None = None   # e.g. "empty_response_retry"


# ── Code-fence stripping ────────────────────────────────────────────────────
# Some models wrap their code in ```python ... ``` or ```javascript ... ```.
# Keep the raw response (it might contain `pip install` lines outside the
# fence we care about) but also extract a clean code body for extractor use.
_FENCE_STRIP_RE = re.compile(r"^\s*```(?:\w+)?\s*\n?|\n?```\s*$", re.MULTILINE)


def strip_outer_fence(text: str) -> str:
    """Strip a single outer ``` fence if the whole response is wrapped in one.

    We intentionally do NOT strip inner fences — Spracklen's H1 parses from the
    raw response text, which may include fenced demo blocks.
    """
    if not text:
        return text
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            # drop first (```lang) and last (```)
            return "\n".join(lines[1:-1])
    return text


# ── Orchestrator ────────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self, budget: Budget):
        self.budget = budget
        # One semaphore per provider (shared across models of the same provider).
        self._sem: dict[str, asyncio.Semaphore] = {}
        for spec in MODELS.values():
            if spec.provider not in self._sem:
                self._sem[spec.provider] = asyncio.Semaphore(spec.concurrency)

    # ------------------------------------------------------------------
    async def generate_code(
        self,
        *,
        spec: ModelSpec,
        language: str,
        prompt: str,
        n_sample: int = 0,
        override_temperature: float | None = None,
    ) -> GenerationResult:
        """One generation. Returns structured result. Charges budget on success."""
        user_content = build_code_user_message(language.capitalize(), prompt)
        messages = [{"role": "user", "content": user_content}]
        kwargs: dict[str, Any] = {
            "model": spec.litellm_id,
            "messages": messages,
            **CODE_GEN_KWARGS,
            **spec.extra_kwargs,
        }
        if override_temperature is not None:
            kwargs["temperature"] = override_temperature
        if spec.supports_top_k:
            kwargs["top_k"] = CODE_GEN_TOP_K
        if not spec.supports_top_p:
            # Anthropic 4.x rejects temperature+top_p together. Keep temperature.
            kwargs.pop("top_p", None)

        retry_reason: str | None = None

        async def _single_call(call_kwargs: dict[str, Any]) -> tuple[Any, int]:
            @retry(
                retry=retry_if_exception_type(_RETRY_EXC),
                wait=wait_random_exponential(multiplier=1.5, min=1.0, max=45.0),
                stop=stop_after_attempt(6),
                reraise=True,
            )
            async def _inner():
                t0 = time.perf_counter()
                resp = await litellm.acompletion(**call_kwargs)
                return resp, int((time.perf_counter() - t0) * 1000)

            return await _inner()

        async with self._sem[spec.provider]:
            resp, latency_ms = await _single_call(kwargs)
            text = (resp.choices[0].message.content or "").strip()

            # If empty, retry once with temperature +0.1.
            if not text:
                retry_reason = "empty_response_retry"
                kwargs2 = dict(kwargs)
                kwargs2["temperature"] = min(kwargs.get("temperature", 0.7) + 0.1, 1.0)
                resp2, lat2 = await _single_call(kwargs2)
                text2 = (resp2.choices[0].message.content or "").strip()
                latency_ms += lat2
                if text2:
                    resp, text = resp2, text2

        text = strip_outer_fence(text)

        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        # Charge the budget based on the provider's published per-token rate.
        cost = self.budget.project(
            provider=spec.provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            input_price=spec.input_price,
            output_price=spec.output_price,
            batch=False,
        )
        self.budget.charge(provider=spec.provider, cost_usd=cost)

        return GenerationResult(
            generated_code=text,
            finish_reason=resp.choices[0].finish_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency_ms,
            model_key=spec.key,
            retry_reason=retry_reason,
        )
