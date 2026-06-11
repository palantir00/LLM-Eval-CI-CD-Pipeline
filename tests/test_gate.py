"""Unit tests for the SLA gate logic (check_thresholds is a pure function)."""

from src.config import Thresholds
from src.eval.gate import Violation, check_thresholds
from src.eval.metrics import RunMetrics

# Thresholds matching config/thresholds.yaml at the time of writing.
THRESHOLDS = Thresholds(
    max_hallucination_rate=0.05,
    min_answer_relevancy=0.50,
    min_faithfulness=0.80,
    max_latency_p50_seconds=2.0,
    max_latency_p95_seconds=5.0,
    max_cost_per_query_usd=0.01,
)


def _metrics(**overrides: float) -> RunMetrics:
    """Build a healthy RunMetrics, optionally overriding individual fields."""
    defaults = {
        "num_items": 10,
        "hallucination_rate": 0.0,
        "mean_answer_relevancy": 0.66,
        "mean_faithfulness": 0.99,
        "latency_p50_seconds": 0.2,
        "latency_p95_seconds": 0.3,
        "mean_cost_usd": 0.0001,
    }
    defaults.update(overrides)
    return RunMetrics(**defaults)  # type: ignore[arg-type]


def test_healthy_run_has_no_violations():
    assert check_thresholds(_metrics(), THRESHOLDS) == []


def test_high_hallucination_is_flagged():
    violations = check_thresholds(_metrics(hallucination_rate=0.20), THRESHOLDS)
    assert [v.metric for v in violations] == ["hallucination_rate"]
    assert violations[0].kind == "max"


def test_low_relevancy_is_flagged():
    violations = check_thresholds(_metrics(mean_answer_relevancy=0.40), THRESHOLDS)
    assert [v.metric for v in violations] == ["answer_relevancy"]
    assert violations[0].kind == "min"


def test_multiple_violations_are_all_reported():
    violations = check_thresholds(
        _metrics(hallucination_rate=0.2, latency_p95_seconds=9.0, mean_cost_usd=0.05),
        THRESHOLDS,
    )
    assert {v.metric for v in violations} == {
        "hallucination_rate",
        "latency_p95_seconds",
        "cost_per_query_usd",
    }


def test_violation_message_max():
    message = Violation("hallucination_rate", 0.20, 0.05, "max").message()
    assert "exceeds the max" in message
    assert "0.1500" in message  # the overage


def test_violation_message_min():
    message = Violation("answer_relevancy", 0.40, 0.50, "min").message()
    assert "below the min" in message
    assert "0.1000" in message  # the shortfall