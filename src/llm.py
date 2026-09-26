"""Minimal LLM wrapper over the provider switch (step 3).

Every call goes through LLM.complete(), which records model, latency, token counts and cost.
Cost is $0 for now (local Ollama); LangFuse tracing and real prices come in step 5.
"""
import logging
import time
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import Settings

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


class LLM:
    """The one place the application calls a language model."""

    def __init__(self, settings: Settings, chat_model: BaseChatModel | None = None):
        self.provider = settings.llm_provider
        self.model = settings.ollama_model if self.provider == "ollama" else settings.anthropic_model
        self.chat_model = chat_model or make_chat_model(settings)

    def complete(self, messages: list[BaseMessage]) -> LLMResponse:
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
        )
        logger.info("llm call provider=%s model=%s in=%s out=%s latency_ms=%.0f cost_usd=%.4f",
                    response.provider, response.model, input_tokens, output_tokens, latency_ms, response.cost_usd)
        return response
