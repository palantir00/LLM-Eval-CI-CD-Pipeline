"""Orchestrates a full evaluation run over the golden dataset.

This is where every piece built so far comes together. For each golden item we:

    1. retrieve context from the knowledge base (RAG),
    2. build the answer prompt,
    3. call the LLM to generate an answer,
    4. score the answer (relevancy, faithfulness, hallucination),

and finally aggregate everything into a single ``RunMetrics`` summary.

Items are processed with a thread pool. We use THREADS (not processes) because the dominant
cost of a real run is network I/O for the LLM calls; threads overlap that waiting while the GIL
is released, and the OpenAI client is thread-safe. Results are collected in input order, and
every per-item metric is deterministic, so the aggregate is identical no matter how many workers
are used — only the wall-clock time changes.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tqdm import tqdm

from src.eval.metrics import (
    HallucinationVerdict,
    RunMetrics,
    SemanticSimilarity,
    aggregate_metrics,
    answer_relevancy,
    faithfulness,
    judge_hallucination,
)
from src.golden_dataset import GoldenItem, load_golden_dataset
from src.paths import GOLDEN_DATASET_PATH
from src.pipeline.llm_client import LLMClient, build_llm_client
from src.pipeline.prompt import build_answer_prompt
from src.pipeline.rag import KnowledgeBase

logger = logging.getLogger(__name__)

# How many knowledge base chunks to retrieve per question.
DEFAULT_N_RETRIEVED = 3
# Default thread-pool size. Modest, so we don't hit API rate limits in real mode.
DEFAULT_MAX_WORKERS = 4


@dataclass
class ItemResult:
    """The full result of evaluating one golden item (kept for per-item reporting/storage)."""

    id: str
    question: str
    answer: str
    context: str
    relevancy: float
    faithfulness: float
    hallucinated: bool
    hallucination_reason: str
    latency_seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int


@dataclass
class RunResult:
    """Everything produced by one evaluation run: the aggregate plus per-item details."""

    metrics: RunMetrics
    items: list[ItemResult]


class EvalRunner:
    """Runs the full pipeline + metrics over a set of golden items."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        generation_client: LLMClient,
        judge_client: LLMClient,
        similarity: SemanticSimilarity,
        n_retrieved: int = DEFAULT_N_RETRIEVED,
    ) -> None:
        """Initialize the runner with its collaborators (dependency injection).

        Passing the collaborators in (instead of creating them inside) makes the runner easy to
        test: a test can inject a mock client and a tiny knowledge base.

        Args:
            knowledge_base: The RAG knowledge base used for retrieval.
            generation_client: The LLM client that answers the questions.
            judge_client: The LLM client used as the hallucination judge.
            similarity: Semantic similarity helper for relevancy/faithfulness.
            n_retrieved: Number of chunks to retrieve per question.
        """
        self._knowledge_base = knowledge_base
        self._generation_client = generation_client
        self._judge_client = judge_client
        self._similarity = similarity
        self._n_retrieved = n_retrieved

    @classmethod
    def build_default(cls) -> "EvalRunner":
        """Build a runner wired from environment configuration (the convenient default).

        The LLM mode (mock/openai) is read from the environment, so this runs offline by default.
        The same client is used for generation and judging here; in production you might use a
        separate, stronger model as the judge.

        Returns:
            A ready-to-use EvalRunner.
        """
        client = build_llm_client()
        return cls(
            knowledge_base=KnowledgeBase(),
            generation_client=client,
            judge_client=client,
            similarity=SemanticSimilarity(),
        )

    def _evaluate_item(self, item: GoldenItem) -> ItemResult:
        """Run the full pipeline and all metrics for a single golden item."""
        # 1. Retrieve context (RAG).
        chunks = self._knowledge_base.retrieve(item.question, n_results=self._n_retrieved)
        chunk_texts = [chunk.text for chunk in chunks]
        context = "\n\n".join(chunk_texts)

        # 2. Build the prompt and 3. generate the answer.
        system_prompt, user_prompt = build_answer_prompt(item.question, chunk_texts)
        response = self._generation_client.complete(system_prompt, user_prompt)

        # 4. Score the answer.
        relevancy = answer_relevancy(item.question, response.text, self._similarity)
        faith = faithfulness(response.text, context, self._similarity)
        verdict = judge_hallucination(item.question, response.text, context, self._judge_client)

        return ItemResult(
            id=item.id,
            question=item.question,
            answer=response.text,
            context=context,
            relevancy=relevancy,
            faithfulness=faith,
            hallucinated=verdict.hallucinated,
            hallucination_reason=verdict.reason,
            latency_seconds=response.latency_seconds,
            cost_usd=response.cost_usd,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    def run(
        self,
        items: list[GoldenItem] | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        show_progress: bool = True,
    ) -> RunResult:
        """Evaluate all items and aggregate the results.

        Args:
            items: Items to evaluate. If None, the golden dataset is loaded from disk.
            max_workers: Thread-pool size. Use 1 for fully sequential execution.
            show_progress: Whether to show a tqdm progress bar (useful for local dev).

        Returns:
            A RunResult with aggregated metrics and per-item details.
        """
        if items is None:
            items = load_golden_dataset(GOLDEN_DATASET_PATH)

        logger.info("Evaluating %d items with %d worker(s)...", len(items), max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map preserves input order, so results line up with items.
            results_iter = executor.map(self._evaluate_item, items)
            if show_progress:
                results_iter = tqdm(results_iter, total=len(items), desc="Evaluating")
            item_results: list[ItemResult] = list(results_iter)

        metrics = aggregate_metrics(
            relevancies=[result.relevancy for result in item_results],
            faithfulnesses=[result.faithfulness for result in item_results],
            latencies=[result.latency_seconds for result in item_results],
            costs=[result.cost_usd for result in item_results],
            verdicts=[
                HallucinationVerdict(result.hallucinated, result.hallucination_reason)
                for result in item_results
            ],
        )
        return RunResult(metrics=metrics, items=item_results)


def main() -> None:
    """Run a full evaluation from the command line and log the aggregated metrics."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    runner = EvalRunner.build_default()
    result = runner.run()
    metrics = result.metrics

    logger.info("--- Evaluation results (%d items) ---", metrics.num_items)
    logger.info("Hallucination rate : %.1f%%", metrics.hallucination_rate * 100)
    logger.info("Answer relevancy   : %.3f", metrics.mean_answer_relevancy)
    logger.info("Faithfulness       : %.3f", metrics.mean_faithfulness)
    logger.info("Latency p50 / p95  : %.3fs / %.3fs",
                metrics.latency_p50_seconds, metrics.latency_p95_seconds)
    logger.info("Mean cost / query  : $%.6f", metrics.mean_cost_usd)


if __name__ == "__main__":
    main()