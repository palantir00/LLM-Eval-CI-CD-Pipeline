"""Unit tests for the metrics module (the pure, deterministic functions)."""

import pytest

from src.eval.metrics import (
    HallucinationVerdict,
    _clamp01,
    _mean,
    _parse_verdict,
    aggregate_metrics,
    hallucination_rate,
    mean_cost,
    percentile,
)


# --- percentile ---
def test_percentile_median_of_even_count():
    # For 1..10 the 50th percentile (linear interpolation) is 5.5.
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == pytest.approx(5.5)


def test_percentile_p95_interpolates():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == pytest.approx(9.55)


def test_percentile_single_value():
    assert percentile([0.42], 95) == 0.42


def test_percentile_is_order_independent():
    assert percentile([3, 1, 2], 50) == percentile([1, 2, 3], 50)


def test_percentile_empty_raises():
    with pytest.raises(ValueError, match="at least one value"):
        percentile([], 50)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 150)


# --- mean / cost ---
def test_mean():
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_raises():
    with pytest.raises(ValueError, match="at least one value"):
        _mean([])


def test_mean_cost():
    assert mean_cost([0.0001, 0.0002, 0.0003]) == pytest.approx(0.0002)


# --- clamp ---
@pytest.mark.parametrize(
    ("value", "expected"),
    [(-0.5, 0.0), (0.0, 0.0), (0.3, 0.3), (1.0, 1.0), (1.7, 1.0)],
)
def test_clamp01(value, expected):
    assert _clamp01(value) == expected


# --- hallucination rate ---
def test_hallucination_rate():
    verdicts = [
        HallucinationVerdict(True, ""),
        HallucinationVerdict(False, ""),
        HallucinationVerdict(False, ""),
        HallucinationVerdict(True, ""),
    ]
    assert hallucination_rate(verdicts) == pytest.approx(0.5)


def test_hallucination_rate_empty_raises():
    with pytest.raises(ValueError, match="at least one verdict"):
        hallucination_rate([])


# --- judge verdict parsing ---
def test_parse_verdict_clean_json():
    verdict = _parse_verdict('{"hallucinated": false, "reason": "supported"}')
    assert verdict.hallucinated is False
    assert verdict.reason == "supported"


def test_parse_verdict_with_surrounding_text():
    verdict = _parse_verdict('Sure: {"hallucinated": true, "reason": "made up"} done')
    assert verdict.hallucinated is True
    assert verdict.reason == "made up"


def test_parse_verdict_garbage_is_conservatively_hallucinated():
    # Unparseable judge output is conservatively counted as a hallucination.
    verdict = _parse_verdict("no json here at all")
    assert verdict.hallucinated is True


# --- aggregation ---
def test_aggregate_metrics():
    verdicts = [HallucinationVerdict(True, ""), HallucinationVerdict(False, "")]
    metrics = aggregate_metrics(
        relevancies=[0.8, 0.6],
        faithfulnesses=[0.9, 0.7],
        latencies=[0.1, 0.3],
        costs=[0.0001, 0.0003],
        verdicts=verdicts,
    )
    assert metrics.num_items == 2
    assert metrics.hallucination_rate == pytest.approx(0.5)
    assert metrics.mean_answer_relevancy == pytest.approx(0.7)
    assert metrics.mean_faithfulness == pytest.approx(0.8)
    assert metrics.mean_cost_usd == pytest.approx(0.0002)