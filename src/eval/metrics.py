"""Evaluation metrics: hallucination rate, answer relevancy, latency, cost, faithfulness.

FRAMEWORK DECISION (DeepEval vs Ragas)
--------------------------------------
We follow DeepEval's approach conceptually but implement the metrics ourselves, transparently.

* DeepEval markets itself as "Pytest for LLMs": metrics are written like unit tests and plug
  into CI — a perfect fit for this project's CI/CD theme. Ragas is excellent but is more of a
  research-oriented RAG-evaluation toolkit and is less centered on the "assert in CI" workflow.
* Why implement the metrics ourselves instead of importing the library:
    1. Understanding — for a portfolio project I must be able to explain every number; a
       hand-written metric with a visible formula is far easier to defend than a black box.
    2. Free, offline CI — DeepEval's LLM-based metrics need an API key (and money) for every
       run. Our design runs the deterministic metrics (latency, cost, relevancy, faithfulness)
       fully offline, and routes the one LLM-as-judge metric through our own LLMClient, whose
       mock works without a key. So every push can be evaluated for free.
    3. No heavy/opaque dependency.

So: DeepEval is the blueprint; the implementation below is ours.

The metrics split into two groups:
* Deterministic, no LLM needed: latency percentiles, cost, answer relevancy, faithfulness
  (the last two use local sentence-embedding similarity).
* LLM-as-judge: hallucination rate (a model grades whether the answer is supported).
"""

import json
import logging
import math
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.pipeline.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Same embedding model as the RAG retriever, so similarity scores are consistent across the
# project. (Kept as a local constant to avoid importing the heavy chromadb stack here.)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _clamp01(value: float) -> float:
    """Clamp a value into the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    if not values:
        raise ValueError("_mean() requires at least one value")
    return sum(values) / len(values)


class SemanticSimilarity:
    """Computes cosine similarity between two texts using local sentence embeddings.

    We load the embedding model once and reuse it. ``normalize_embeddings=True`` makes each
    vector unit length, so their dot product is exactly the cosine similarity (range -1..1,
    where 1 means "same meaning").
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """Load the embedding model.

        Args:
            model_name: sentence-transformers model used for embeddings.
        """
        self._model = SentenceTransformer(model_name)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return the cosine similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity in the range -1.0 .. 1.0.
        """
        embeddings = self._model.encode([text_a, text_b], normalize_embeddings=True)
        # Dot product of two unit-length vectors == cosine similarity.
        return float(embeddings[0] @ embeddings[1])


# ---------------------------------------------------------------------------
# 1. Latency percentiles (p50 / p95)
# ---------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list using linear interpolation.

    Formula: sort the values, find the fractional rank ``(p/100) * (n - 1)``, and linearly
    interpolate between the two nearest sorted values. (This is the same method numpy uses by
    default.) p50 is the median; p95 is the value below which 95% of requests fall.

    Interview: Why percentiles instead of the average? The average hides the tail. If 99
    requests take 0.2s and one takes 10s, the average looks fine (~0.3s) but one in a hundred
    users had a terrible experience. p95/p99 expose that tail, which is why real SLAs are
    written as percentiles ("95% of requests under 500ms"), not as averages.

    Args:
        values: The measured values (e.g. latencies in seconds).
        p: The percentile to compute, between 0 and 100.

    Returns:
        The p-th percentile value.

    Raises:
        ValueError: If ``values`` is empty or ``p`` is out of range.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (p / 100) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]

    weight = rank - lower_index  # how far between the two neighbours we are (0..1)
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


# ---------------------------------------------------------------------------
# 2. Cost per query
# ---------------------------------------------------------------------------
def mean_cost(costs: list[float]) -> float:
    """Compute the average cost per query in USD.

    Each per-query cost is already (input_tokens x input_price + output_tokens x output_price),
    computed by the LLM client. Here we just average across the dataset.

    Interview: Cost scales with tokens, so tracking cost-per-query catches expensive
    regressions early — e.g. a prompt change that doubles the context length doubles the bill.
    Multiply the mean by expected monthly volume to project the real spend.

    Args:
        costs: Per-query costs in USD.

    Returns:
        The mean cost per query in USD.
    """
    return _mean(costs)


# ---------------------------------------------------------------------------
# 3. Answer relevancy
# ---------------------------------------------------------------------------
def answer_relevancy(question: str, answer: str, similarity: SemanticSimilarity) -> float:
    """Measure how relevant an answer is to the question (0..1).

    We embed the question and the answer and take their cosine similarity. A high score means
    the answer is "about" the question. This is a lightweight, offline proxy for the LLM-based
    answer-relevancy metric used by tools like Ragas.

    Interview: This catches off-topic or evasive answers. Its limitation is that relevancy is
    not correctness — an answer can be on-topic but wrong. That is exactly why we also measure
    faithfulness and hallucination; no single metric tells the whole story.

    Args:
        question: The user's question.
        answer: The model's answer.
        similarity: A SemanticSimilarity instance.

    Returns:
        Relevancy score clamped to 0.0 .. 1.0.
    """
    return _clamp01(similarity.similarity(question, answer))


# ---------------------------------------------------------------------------
# 4. Faithfulness to sources
# ---------------------------------------------------------------------------
def faithfulness(answer: str, context: str, similarity: SemanticSimilarity) -> float:
    """Measure how grounded an answer is in the retrieved context (0..1).

    We embed the answer and the context and take their cosine similarity. A high score means
    the answer stays close to what the sources actually say.

    Interview: Faithfulness is about grounding. A faithful answer only states what the context
    supports; a low score signals the model is drifting beyond its sources, which is a strong
    hallucination warning. Here we use an embedding proxy for speed and offline determinism;
    the rigorous version uses an LLM judge to check each claim against the context (see below).

    Args:
        answer: The model's answer.
        context: The retrieved context the answer should be based on.
        similarity: A SemanticSimilarity instance.

    Returns:
        Faithfulness score clamped to 0.0 .. 1.0.
    """
    return _clamp01(similarity.similarity(answer, context))


# ---------------------------------------------------------------------------
# 5. Hallucination rate (LLM-as-judge)
# ---------------------------------------------------------------------------
@dataclass
class HallucinationVerdict:
    """A judge's verdict for a single answer.

    Attributes:
        hallucinated: True if the answer contains a claim not supported by the context.
        reason: Short explanation from the judge (useful for debugging).
    """

    hallucinated: bool
    reason: str


# The judge is told to behave like a strict grader and to return machine-readable JSON.
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. You are given a QUESTION, an ANSWER, and the CONTEXT "
    "that the answer must be based on. Decide whether the ANSWER makes any claim that is NOT "
    "supported by the CONTEXT (a hallucination). Respond with ONLY a JSON object of the form "
    '{"hallucinated": true|false, "reason": "<short explanation>"}. Set "hallucinated" to true '
    "if the answer states anything not backed by the context, otherwise false."
)


def build_judge_user_prompt(question: str, answer: str, context: str) -> str:
    """Build the user prompt handed to the hallucination judge.

    Args:
        question: The original question.
        answer: The answer being judged.
        context: The context the answer should be grounded in.

    Returns:
        The formatted user prompt string.
    """
    return (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return only the JSON verdict."
    )


def _parse_verdict(text: str) -> HallucinationVerdict:
    """Parse the judge's JSON response into a HallucinationVerdict.

    Robust to extra text around the JSON: we extract the substring between the first '{' and
    the last '}'. If parsing fails, we conservatively count the answer as hallucinated (we would
    rather over-report a problem than silently miss one) and log a warning.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        return HallucinationVerdict(
            hallucinated=bool(data["hallucinated"]),
            reason=str(data.get("reason", "")),
        )
    except (ValueError, KeyError, TypeError) as error:
        logger.warning(
            "Could not parse judge verdict (%s); counting as hallucinated. Raw: %r",
            error,
            text[:120],
        )
        return HallucinationVerdict(hallucinated=True, reason="unparseable judge response")


def judge_hallucination(
    question: str, answer: str, context: str, judge: LLMClient
) -> HallucinationVerdict:
    """Ask an LLM judge whether an answer is supported by the context (LLM-as-judge).

    Methodology: we give a separate "judge" model the question, the answer, and the ground-truth
    context, and ask it to flag any claim not supported by the context. The judge is run at
    temperature 0 with a strict rubric and structured (JSON) output to keep verdicts consistent.

    Interview: LLM-as-judge means using one model to grade another's output. It is far cheaper
    and more scalable than human review and correlates well with human judgement for clear-cut
    cases. Its limitations: the judge can be biased or inconsistent, so we mitigate with
    temperature 0, an explicit rubric, and a fixed output format. The judge runs through our
    LLMClient abstraction, so CI can use the deterministic mock for free.

    Args:
        question: The original question.
        answer: The answer to evaluate.
        context: The context the answer should be grounded in.
        judge: An LLMClient used as the grader.

    Returns:
        The judge's HallucinationVerdict.
    """
    user_prompt = build_judge_user_prompt(question, answer, context)
    response = judge.complete(JUDGE_SYSTEM_PROMPT, user_prompt)
    return _parse_verdict(response.text)


def hallucination_rate(verdicts: list[HallucinationVerdict]) -> float:
    """Compute the fraction of answers judged to be hallucinated.

    Formula: (number of hallucinated answers) / (total answers). This is the headline metric the
    CI gate blocks on (project requirement: fail if it exceeds 5%).

    Interview: This is the most business-critical metric — a hallucinating banking assistant can
    give wrong, harmful advice. Expressing it as a rate makes it easy to set a hard SLA.

    Args:
        verdicts: Per-answer judge verdicts.

    Returns:
        Hallucination rate in the range 0.0 .. 1.0.

    Raises:
        ValueError: If ``verdicts`` is empty.
    """
    if not verdicts:
        raise ValueError("hallucination_rate() requires at least one verdict")
    hallucinated = sum(1 for verdict in verdicts if verdict.hallucinated)
    return hallucinated / len(verdicts)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    """Aggregated metrics for one full evaluation run (one row in the metrics history)."""

    num_items: int
    hallucination_rate: float
    mean_answer_relevancy: float
    mean_faithfulness: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    mean_cost_usd: float


def aggregate_metrics(
    relevancies: list[float],
    faithfulnesses: list[float],
    latencies: list[float],
    costs: list[float],
    verdicts: list[HallucinationVerdict],
) -> RunMetrics:
    """Combine per-item results into the aggregated metrics for a run.

    Args:
        relevancies: Per-item answer relevancy scores.
        faithfulnesses: Per-item faithfulness scores.
        latencies: Per-item latencies in seconds.
        costs: Per-item costs in USD.
        verdicts: Per-item hallucination verdicts.

    Returns:
        A RunMetrics summary.
    """
    return RunMetrics(
        num_items=len(latencies),
        hallucination_rate=hallucination_rate(verdicts),
        mean_answer_relevancy=_mean(relevancies),
        mean_faithfulness=_mean(faithfulnesses),
        latency_p50_seconds=percentile(latencies, 50),
        latency_p95_seconds=percentile(latencies, 95),
        mean_cost_usd=mean_cost(costs),
    )