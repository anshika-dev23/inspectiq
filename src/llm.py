"""Minimal LLM wrapper over the provider switch (step 3).

Every call goes through LLM.complete(), which records model, latency, token counts and cost, and sends a
LangFuse "generation" span (prompt, output, tokens). Cost is $0 (local Ollama); each call also records a
shadow cost: what it would have cost on the Anthropic models in config.SHADOW_PRICES_USD_PER_MTOK.
"""
import logging
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src import tracing
from src.config import SHADOW_COST_NOTE, SHADOW_PRICES_USD_PER_MTOK, Settings

logger = logging.getLogger("inspectiq.llm")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    cost_usd: float = 0.0
    shadow_cost_usd: dict[str, float] = field(default_factory=dict)   # model -> USD (estimate, see SHADOW_COST_NOTE)


def make_chat_model(settings: Settings) -> BaseChatModel:
    """ChatOllama or ChatAnthropic, temperature 0."""
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx is set explicitly: Ollama's default context window is smaller than our prompts
        # and silently drops the start of a long prompt.
        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                          temperature=settings.llm_temperature, num_ctx=settings.ollama_num_ctx)
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, temperature=settings.llm_temperature, max_tokens=1024)
    raise ValueError(f"Unknown LLM provider {settings.llm_provider!r}")


def token_counts(provider: str, response_metadata: dict, usage_metadata: dict | None) -> tuple[int | None, int | None]:
    """(input_tokens, output_tokens) from a LangChain AIMessage's metadata.

    Ollama reports prompt_eval_count / eval_count; other providers go through LangChain's usage_metadata.
    """
    if provider == "ollama":
        return response_metadata.get("prompt_eval_count"), response_metadata.get("eval_count")
    usage = usage_metadata or {}
    return usage.get("input_tokens"), usage.get("output_tokens")


def shadow_costs(input_tokens: int | None, output_tokens: int | None,
                 prices: dict[str, tuple[float, float]] = SHADOW_PRICES_USD_PER_MTOK) -> dict[str, float]:
    """USD this call would cost on each priced model: tokens / 1e6 * price per million, input + output."""
    if input_tokens is None or output_tokens is None:
        return {}
    return {model: (input_tokens * price_in + output_tokens * price_out) / 1_000_000
            for model, (price_in, price_out) in prices.items()}


def add_usage(total: dict | None, response: LLMResponse) -> dict:
    """Running totals over several LLM calls: tokens and shadow cost per model."""
    total = total or {"input_tokens": 0, "output_tokens": 0, "shadow_cost_usd": {}}
    shadow = dict(total["shadow_cost_usd"])
    for model, usd in response.shadow_cost_usd.items():
        shadow[model] = shadow.get(model, 0.0) + usd
    return {"input_tokens": total["input_tokens"] + (response.input_tokens or 0),
            "output_tokens": total["output_tokens"] + (response.output_tokens or 0),
            "shadow_cost_usd": shadow}


class LLM:
    """The one place the application calls a language model."""

    def __init__(self, settings: Settings, chat_model: BaseChatModel | None = None):
        self.provider = settings.llm_provider
        self.model = settings.ollama_model if self.provider == "ollama" else settings.anthropic_model
        self.chat_model = chat_model or make_chat_model(settings)
        self.settings = settings

    def model_parameters(self) -> dict:
        parameters = {"temperature": self.settings.llm_temperature}
        if self.provider == "ollama":
            parameters["num_ctx"] = self.settings.ollama_num_ctx
        return parameters

    def complete(self, messages: list[BaseMessage]) -> LLMResponse:
        prompt = [{"role": m.type, "content": m.content} for m in messages]
        with tracing.observe("llm", as_type="generation", model=self.model, input=prompt,
                             model_parameters=self.model_parameters()) as span:
            start = time.perf_counter()
            message = self.chat_model.invoke(messages)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            input_tokens, output_tokens = token_counts(
                self.provider, message.response_metadata or {}, getattr(message, "usage_metadata", None)
            )
            response = LLMResponse(
                text=str(message.content),
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                shadow_cost_usd=shadow_costs(input_tokens, output_tokens),
            )
            tracing.update(
                span,
                output=response.text,
                usage_details={"input": input_tokens or 0, "output": output_tokens or 0},
                cost_details={"total": response.cost_usd},
                metadata={"provider": self.provider, "latency_ms": latency_ms,
                          "shadow_cost_usd": response.shadow_cost_usd, "shadow_cost_note": SHADOW_COST_NOTE},
            )
        logger.info("llm call provider=%s model=%s in=%s out=%s latency_ms=%.0f cost_usd=%.4f",
                    response.provider, response.model, input_tokens, output_tokens, latency_ms, response.cost_usd)
        return response
