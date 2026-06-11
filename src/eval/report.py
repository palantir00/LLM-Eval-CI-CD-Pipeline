"""Generate a Markdown report of the latest evaluation run (used for the PR comment in CI).

The CI workflow runs this after the evaluation and posts the resulting Markdown as a sticky
comment on the pull request, so reviewers see the quality metrics right in the PR.
"""

import logging
from dataclasses import dataclass

from src.config import load_thresholds
from src.eval.gate import check_thresholds, record_to_metrics
from src.paths import PROJECT_ROOT
from src.storage.db import MetricsDatabase, RunRecord

logger = logging.getLogger(__name__)

# Where the Markdown report is written; the CI comment step reads this file.
REPORT_PATH = PROJECT_ROOT / "eval_report.md"


@dataclass
class ReportRow:
    """One metric row in the report table."""

    label: str
    value: str
    threshold: str
    passed: bool


def _format(value: float, kind: str) -> str:
    """Format a metric value according to its kind."""
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "seconds":
        return f"{value:.3f}s"
    if kind == "usd":
        return f"${value:.6f}"
    return f"{value:.3f}"


def build_report_rows(record: RunRecord) -> list[ReportRow]:
    """Build the per-metric table rows (value, threshold, pass/fail) for a run."""
    thresholds = load_thresholds()
    # (label, actual, limit, comparison kind, value kind)
    specs = [
        ("Hallucination rate", record.hallucination_rate, thresholds.max_hallucination_rate,
         "max", "percent"),
        ("Answer relevancy", record.mean_answer_relevancy, thresholds.min_answer_relevancy,
         "min", "score"),
        ("Faithfulness", record.mean_faithfulness, thresholds.min_faithfulness,
         "min", "score"),
        ("Latency p50", record.latency_p50_seconds, thresholds.max_latency_p50_seconds,
         "max", "seconds"),
        ("Latency p95", record.latency_p95_seconds, thresholds.max_latency_p95_seconds,
         "max", "seconds"),
        ("Cost / query", record.mean_cost_usd, thresholds.max_cost_per_query_usd,
         "max", "usd"),
    ]

    rows: list[ReportRow] = []
    for label, actual, limit, comparison, value_kind in specs:
        passed = actual <= limit if comparison == "max" else actual >= limit
        comparator = "≤" if comparison == "max" else "≥"  # <= or >=
        rows.append(
            ReportRow(
                label=label,
                value=_format(actual, value_kind),
                threshold=f"{comparator} {_format(limit, value_kind)}",
                passed=passed,
            )
        )
    return rows


def build_markdown_report(database: MetricsDatabase | None = None) -> str:
    """Build the full Markdown report for the latest run.

    Args:
        database: Metrics database to read from (defaults to the standard one).

    Returns:
        A Markdown string ready to be posted as a PR comment.
    """
    database = database or MetricsDatabase()
    record = database.get_latest_run()
    if record is None:
        return "## 🤖 LLM Evaluation Report\n\nNo evaluation run found."

    gate_passed = not check_thresholds(record_to_metrics(record), load_thresholds())
    headline = "✅ **SLA gate PASSED**" if gate_passed else "❌ **SLA gate FAILED**"

    commit = record.git_commit or "n/a"
    lines = [
        "## 🤖 LLM Evaluation Report",
        "",
        f"**Run #{record.id}** · model `{record.model_name}` · mode `{record.llm_mode}` "
        f"· commit `{commit}`",
        "",
        headline,
        "",
        "| Metric | Value | Threshold | Status |",
        "| --- | --- | --- | --- |",
    ]
    for row in build_report_rows(record):
        status = "✅" if row.passed else "❌"
        lines.append(f"| {row.label} | {row.value} | {row.threshold} | {status} |")

    lines += [
        "",
        f"_Evaluated {record.num_items} golden items in `{record.llm_mode}` mode "
        "(no API cost in CI)._",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write the Markdown report to disk (and log where it went)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = build_markdown_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Wrote evaluation report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()