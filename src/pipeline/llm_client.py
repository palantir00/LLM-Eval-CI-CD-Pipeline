"""LLM client abstraction: a common interface plus OpenAI and mock implementations.

WHY an abstraction here? The rest of the pipeline (and the evaluation runner) should not care
*which* model answers a question. They call one method, ``complete()``, and get back a
``LLMResponse``. This buys us three things:

* **Swappable models** — replace OpenAI with another provider by writing one new class, with
  zero changes elsewhere.
* **Free local dev and CI** — the mock implementation runs offline, without an API key and at
  zero cost, so every push can run the pipeline without spending money.
* **Testability** — tests use the mock to get deterministic outputs.

This is the classic "program to an interface, not an implementation" idea.
"""

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
import tiktoken
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import ModelConfig, get_model_config

logger = logging.getLogger(__name__)

# Fixed seed so the mock client's simulated latency is reproducible (project requirement).
DEFAULT_SEED = 42

# Fallback token encoding when tiktoken does not recognize the model name.
# o200k_base is the encoding used by the GPT-4o family.
_FALLBACK_ENCODING = "o200k_base"


@dataclass
class LLMResponse:
    """The result of a single LLM call, including everything the metrics need.

    Attributes:
        text: The generated answer text.
        model: Name of the model that produced the answer.
        input_tokens: Number of prompt (input) tokens.
        output_tokens: Number of completion (output) tokens.
        latency_seconds: Wall-clock time the call took (or a simulated value in mock mode).
        cost_usd: Estimated cost of the call in USD.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a piece of text for a given model.

    Used to estimate cost when the provider does not return exact token usage (e.g. in the
    mock client). tiktoken maps text to the same tokens the model would see.

    Args:
        text: The text to tokenize.
        model: Model name (selects the correct token encoding).

    Returns:
        The number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model name — fall back to the GPT-4o encoding.
        encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)
    return len(encoding.encode(text))


class LLMClient(ABC):
    """Abstract interface that every LLM client must implement.

    ABC = Abstract Base Class. You cannot create an LLMClient directly; you must use a concrete
    subclass (OpenAIClient or MockLLMClient) that provides ``complete()``.
    """

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate an answer for the given prompts.

        Args:
            system_prompt: Instructions that set the model's behavior/role.
            user_prompt: The actual user question (with any retrieved RAG context).

        Returns:
            An LLMResponse with the answer text plus token, latency and cost metadata.
        """
        ...


# Transient OpenAI errors that are worth retrying (network blips, rate limits, server errors).
# We do NOT retry on things like invalid API key or malformed requests, because retrying those
# would never succeed.
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient(LLMClient):
    """LLM client backed by the real OpenAI API."""

    def __init__(self, model_config: ModelConfig, api_key: str) -> None:
        """Initialize the OpenAI client.

        Args:
            model_config: Model parameters and pricing.
            api_key: OpenAI API key.
        """
        self._model_config = model_config
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the OpenAI chat API and return the answer with usage metadata."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Measure only the network call. perf_counter is a high-resolution monotonic clock,
        # the right tool for measuring durations (it never jumps backwards).
        start = time.perf_counter()
        completion = self._completion_with_retry(messages)
        latency_seconds = time.perf_counter() - start

        text = completion.choices[0].message.content or ""

        # Prefer the provider's exact token counts; fall back to tiktoken if usage is missing.
        usage = completion.usage
        if usage is not None:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
        else:
            input_tokens = count_tokens(f"{system_prompt}\n{user_prompt}", self._model_config.name)
            output_tokens = count_tokens(text, self._model_config.name)

        return LLMResponse(
            text=text,
            model=self._model_config.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            cost_usd=self._model_config.cost_usd(input_tokens, output_tokens),
        )

    @retry(
        # Retry only on the transient errors listed above.
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        # Exponential backoff: wait 1s, then 2s, 4s, 8s... capped at 30s. Spacing out retries
        # gives a rate-limited or overloaded server time to recover, instead of hammering it.
        wait=wait_exponential(multiplier=1, min=1, max=30),
        # Give up after 5 attempts so we never loop forever.
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        # After the last failed attempt, re-raise the real exception (not tenacity's wrapper).
        reraise=True,
    )
    def _completion_with_retry(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """Make the raw API call, automatically retried on transient errors."""
        return self._client.chat.completions.create(
            model=self._model_config.name,
            messages=messages,
            temperature=self._model_config.temperature,
            max_tokens=self._model_config.max_output_tokens,
        )


class MockLLMClient(LLMClient):
    """Offline LLM client used for local development, tests and CI.

    It needs no API key and costs nothing. Outputs are deterministic so that runs are
    reproducible. Token counts are computed with tiktoken and the latency is a simulated
    (but seeded, hence reproducible) value, so the demo metrics still look realistic.
    """

    def __init__(self, model_config: ModelConfig, seed: int = DEFAULT_SEED) -> None:
        """Initialize the mock client.

        Args:
            model_config: Model parameters and pricing (used for cost estimation).
            seed: Seed for the simulated-latency random generator (reproducibility).
        """
        self._model_config = model_config
        self._seed = seed

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a deterministic, offline answer with simulated metadata."""
        answer = self._build_mock_answer(system_prompt, user_prompt)

        input_tokens = count_tokens(f"{system_prompt}\n{user_prompt}", self._model_config.name)
        output_tokens = count_tokens(answer, self._model_config.name)

        # Simulated latency: no real network call and no sleep (keeps the runner fast), but
        # seeded by the prompt so the same input always yields the same value -> reproducible.
        rng = random.Random(f"{self._seed}:{user_prompt}")
        latency_seconds = round(rng.uniform(0.05, 0.30), 3)

        return LLMResponse(
            text=answer,
            model=f"{self._model_config.name} (mock)",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            cost_usd=self._model_config.cost_usd(input_tokens, output_tokens),
        )

    @staticmethod
    def _build_mock_answer(system_prompt: str, user_prompt: str) -> str:
        """Build a deterministic, offline answer.

        Two special cases keep the whole evaluation pipeline runnable offline:

        * Judge: if we are used as a hallucination judge (detected via the judge's required JSON
          format in the system prompt), return a valid, benign JSON verdict. So the LLM-as-judge
          metric and the CI gate run without a real judge.
        * Answer: simulate an *ideal* RAG answer by grounding it in the retrieved context that the
          prompt builder placed between "Context:" and "Question:". This makes the offline
          pipeline produce realistic, faithful-by-construction metrics (good for the dashboard
          demo and a green CI). It is intentionally coupled to prompt.py's format — a mock
          simulates the very thing it stands in for.

        The mock never claims a real quality verdict; it only keeps the plumbing working.
        """
        if '"hallucinated"' in system_prompt:
            return '{"hallucinated": false, "reason": "mock judge: not evaluated offline"}'

        if "Context:\n" in user_prompt and "\n\nQuestion:" in user_prompt:
            context = user_prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0].strip()
            if context:
                return f"Based on the available information: {context}"

        return "[MOCK ANSWER] Deterministic offline response generated without calling any API."


def build_llm_client(mode: str | None = None, model_name: str | None = None) -> LLMClient:
    """Build the right LLM client based on configuration (the factory function).

    Reads ``LLM_MODE`` and ``OPENAI_API_KEY`` from the environment (and from a .env file, if
    present). The default mode is "mock", so the project runs out of the box with no API key.

    Args:
        mode: "mock" or "openai". If None, read from the LLM_MODE env var (default "mock").
        model_name: Model to use. If None, read from OPENAI_MODEL or the config default.

    Returns:
        A ready-to-use LLMClient.

    Raises:
        RuntimeError: In "openai" mode when OPENAI_API_KEY is not set.
        ValueError: When the mode is not recognized.
    """
    load_dotenv()  # load variables from a local .env file, if one exists

    resolved_mode = (mode or os.getenv("LLM_MODE") or "mock").lower()
    model_config = get_model_config(model_name or os.getenv("OPENAI_MODEL") or None)

    if resolved_mode == "mock":
        logger.info("Using MockLLMClient (offline, zero cost).")
        return MockLLMClient(model_config)

    if resolved_mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for 'openai' mode. "
                "Set LLM_MODE=mock to run offline without a key."
            )
        logger.info("Using OpenAIClient with model '%s'.", model_config.name)
        return OpenAIClient(model_config, api_key)

    raise ValueError(f"Unknown LLM_MODE '{resolved_mode}'. Use 'mock' or 'openai'.")