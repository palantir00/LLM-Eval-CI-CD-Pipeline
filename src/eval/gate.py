"""CI SLA gate: compares a run's metrics against thresholds and fails the build on violation.

This is the piece that makes the project behave like "tests for an LLM": if quality drops below
the SLA defined in config/thresholds.yaml, the process exits with a non-zero code, which causes
the CI job (and therefore the merge) to fail.

The comparison logic (``check_thresholds``) is a pure function, so it is trivial to unit-test
without running a real evaluation.
"""

import logging
import sys
from dataclasses import dataclass

from src.config import Thresholds, load_thresholds
from src.eval.metrics import RunMetrics
from src.eval.runner import EvalRunner

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """A single breached threshold.

    Attributes:
        metric: Name of the metric that breached its threshold.
        actual: The measured value.
        threshold: The configured limit.
        kind: "max" (actual must be <= threshold) or "min" (actual must be >= threshold).
    """

    metric: str
    actual: float
    threshold: float
    kind: str

    def message(self) -> str:
        """Render a human-readable explanation of the breach (which metric, and by how much)."""
        if self.kind == "max":
            overage = self.actual - self.threshold
            return (
                f"{self.metric} = {self.actual:.4f} exceeds the max of {self.threshold:.4f} "
                f"(by {overage:.4f})"
            )
        shortfall = self.threshold - self.actual
        return (
            f"{self.metric} = {self.actual:.4f} is below the min of {self.threshold:.4f} "
            f"(by {shortfall:.4f})"
        )


def check_thresholds(metrics: RunMetrics, thresholds: Thresholds) -> list[Violation]:
    """Compare metrics against thresholds and return all violations.

    Pure function: same inputs always give the same list of violations, and it touches no I/O —
    so it is easy to unit-test.

    Args:
        metrics: The aggregated metrics of a run.
        thresholds: The SLA thresholds to enforce.

    Returns:
        A list of Violation (empty if every threshold is satisfied).
    """
    # Upper bounds: the metric must not exceed the limit.
    max_checks = [
        ("hallucination_rate", metrics.hallucination_rate, thresholds.max_hallucination_rate),
        ("latency_p50_seconds", metrics.latency_p50_seconds, thresholds.max_latency_p50_seconds),
        ("latency_p95_seconds", metrics.latency_p95_seconds, thresholds.max_latency_p95_seconds),
        ("cost_per_query_usd", metrics.mean_cost_usd, thresholds.max_cost_per_query_usd),
    ]
    # Lower bounds: the metric must not fall below the limit.
    min_checks = [
        ("answer_relevancy", metrics.mean_answer_relevancy, thresholds.min_answer_relevancy),
        ("faithfulness", metrics.mean_faithfulness, thresholds.min_faithfulness),
    ]

    violations: list[Violation] = []
    for name, actual, limit in max_checks:
        if actual > limit:
            violations.append(Violation(name, actual, limit, "max"))
    for name, actual, limit in min_checks:
        if actual < limit:
            violations.append(Violation(name, actual, limit, "min"))
    return violations


def evaluate_and_gate() -> int:
    """Run a full evaluation, check the SLA, and return a process exit code.

    NOTE: for now the gate runs the evaluation itself, so it is self-contained. Once storage
    exists (Step 8), the runner will save metrics and the gate will read the latest run instead
    of re-evaluating.

    Returns:
        0 if all thresholds are satisfied, 1 if any threshold is violated.
    """
    thresholds = load_thresholds()
    result = EvalRunner.build_default().run()
    metrics = result.metrics

    violations = check_thresholds(metrics, thresholds)
    if violations:
        logger.error("SLA gate FAILED — %d threshold(s) violated:", len(violations))
        for violation in violations:
            logger.error("  - %s", violation.message())
        return 1

    logger.info("SLA gate PASSED — all %d items met every threshold.", metrics.num_items)
    return 0


def main() -> None:
    """Entry point: exit with code 0 (pass) or 1 (fail) so CI can block the merge."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(evaluate_and_gate())


if __name__ == "__main__":
    main()