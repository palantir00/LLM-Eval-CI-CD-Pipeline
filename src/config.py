"""Loading and validating model configuration from config/models.yaml.

Pricing and model parameters live in YAML (not in code) because they change independently
of the program logic. Here we read that file and validate it with a Pydantic model, so a typo
in the config fails loudly and early.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.paths import MODELS_PATH, THRESHOLDS_PATH


class ModelConfig(BaseModel):
    """Configuration and pricing for a single model.

    Attributes:
        name: Model identifier (e.g. "gpt-4o-mini").
        provider: Provider name (e.g. "openai").
        price_per_1m_input_tokens: USD price per 1,000,000 input (prompt) tokens.
        price_per_1m_output_tokens: USD price per 1,000,000 output (completion) tokens.
        max_output_tokens: Hard cap on generated tokens per call.
        temperature: Sampling temperature (0.0 = deterministic, for reproducible evals).
    """

    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    price_per_1m_input_tokens: float = Field(ge=0)
    price_per_1m_output_tokens: float = Field(ge=0)
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Compute the USD cost of a single call from its token counts.

        Args:
            input_tokens: Number of prompt (input) tokens.
            output_tokens: Number of completion (output) tokens.

        Returns:
            The cost in USD.
        """
        # Pricing is quoted per 1,000,000 tokens, so we divide token counts by 1M.
        input_cost = (input_tokens / 1_000_000) * self.price_per_1m_input_tokens
        output_cost = (output_tokens / 1_000_000) * self.price_per_1m_output_tokens
        return input_cost + output_cost


def get_model_config(name: str | None = None, path: Path = MODELS_PATH) -> ModelConfig:
    """Load a single model's configuration from the models YAML file.

    Args:
        name: Model name to load. If None, the file's ``default`` model is used.
        path: Path to the models YAML file.

    Returns:
        The validated ModelConfig for the chosen model.

    Raises:
        FileNotFoundError: When the config file does not exist.
        ValueError: When the requested model is not defined in the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Map each model's name to its raw definition for easy lookup.
    models_by_name = {model["name"]: model for model in raw["models"]}

    chosen_name = name or raw["default"]
    if chosen_name not in models_by_name:
        available = ", ".join(sorted(models_by_name))
        raise ValueError(f"Model '{chosen_name}' not found in {path}. Available: {available}")

    # ** unpacks the dict into keyword arguments; Pydantic validates the fields.
    return ModelConfig(**models_by_name[chosen_name])


class Thresholds(BaseModel):
    """SLA thresholds the CI gate checks a run against.

    "max_*" fields are upper bounds (the metric must stay at or below them); "min_*" fields are
    lower bounds (the metric must stay at or above them).
    """

    # extra="forbid" => a typo'd key in thresholds.yaml fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    max_hallucination_rate: float = Field(ge=0, le=1)
    min_answer_relevancy: float = Field(ge=0, le=1)
    min_faithfulness: float = Field(ge=0, le=1)
    max_latency_p50_seconds: float = Field(gt=0)
    max_latency_p95_seconds: float = Field(gt=0)
    max_cost_per_query_usd: float = Field(ge=0)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Thresholds:
    """Load and validate the SLA thresholds from the thresholds YAML file.

    Args:
        path: Path to the thresholds YAML file.

    Returns:
        The validated Thresholds.

    Raises:
        FileNotFoundError: When the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Thresholds config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Thresholds(**raw)