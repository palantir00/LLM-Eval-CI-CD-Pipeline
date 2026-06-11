"""Optional "hybrid" module: score answers with the industry-standard DeepEval library.

The project's core metrics are hand-written (transparent, free, offline in CI). This optional
module shows the SAME answers scored with DeepEval's LLM-judged metrics, to demonstrate working
knowledge of the standard tooling.

DeepEval's metrics call an LLM judge, so this REQUIRES an OpenAI API key and is intentionally
NOT part of the free CI path. Install and run with:

    uv sync --extra deepeval
    uv run python -m src.eval.deepeval_metrics

Imports of deepeval are lazy (inside functions), so the rest of the project works even when the
optional dependency is not installed.
"""

import logging
import os
from typing import Any, cast

from src.golden_dataset import load_golden_dataset
from src.paths import GOLDEN_DATASET_PATH
from src.pipeline.llm_client import build_llm_client
from src.pipeline.prompt import build_answer_prompt
from src.pipeline.rag import KnowledgeBase

logger = logging.getLogger(__name__)

# How many golden items to score in the demo run (kept small to limit API cost).
DEMO_ITEM_COUNT = 3
# DeepEval's own model judge; gpt-4o-mini keeps the cost low.
JUDGE_MODEL = "gpt-4o-mini"


def score_with_deepeval(
    question: str,
    answer: str,
    context_chunks: list[str],
    *,
    model: str = JUDGE_MODEL,
) -> dict[str, float | None]:
    """Score one answer with DeepEval's answer-relevancy and faithfulness metrics.

    Args:
        question: The user's question.
        answer: The generated answer.
        context_chunks: The retrieved context the answer should be grounded in.
        model: The model DeepEval uses as its judge.

    Returns:
        A dict with "answer_relevancy" and "faithfulness" scores (0..1, or None if unavailable).

    Raises:
        RuntimeError: If the optional ``deepeval`` dependency is not installed.
    """
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "deepeval is not installed. Install the optional extra with: uv sync --extra deepeval"
        ) from error

    # DeepEval works with "test cases": the input, the model's output, and the retrieved context.
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        # cast keeps mypy happy whether or not the optional deepeval types are installed
        # (deepeval expects a broader, invariant list type for retrieval_context).
        retrieval_context=cast(Any, context_chunks),
    )

    relevancy_metric = AnswerRelevancyMetric(model=model)
    faithfulness_metric = FaithfulnessMetric(model=model)
    relevancy_metric.measure(test_case)
    faithfulness_metric.measure(test_case)

    return {
        "answer_relevancy": relevancy_metric.score,
        "faithfulness": faithfulness_metric.score,
    }


def main() -> None:
    """Score a few golden items with DeepEval and log the scores (requires an API key)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY is required: DeepEval uses an LLM judge. Set it in your .env file."
        )
        raise SystemExit(1)

    items = load_golden_dataset(GOLDEN_DATASET_PATH)[:DEMO_ITEM_COUNT]
    knowledge_base = KnowledgeBase()
    client = build_llm_client()

    for item in items:
        context_chunks = [chunk.text for chunk in knowledge_base.retrieve(item.question)]
        system_prompt, user_prompt = build_answer_prompt(item.question, context_chunks)
        answer = client.complete(system_prompt, user_prompt).text

        scores = score_with_deepeval(item.question, answer, context_chunks)
        logger.info(
            "[%s] DeepEval answer_relevancy=%s faithfulness=%s",
            item.id,
            scores["answer_relevancy"],
            scores["faithfulness"],
        )


if __name__ == "__main__":
    main()