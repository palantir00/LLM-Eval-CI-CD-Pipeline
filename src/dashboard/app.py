"""Streamlit dashboard for the LLM evaluation metrics history.

Shows, for the runs stored in SQLite:
* a summary of the latest run (each metric with its change vs the previous run),
* the SLA gate status,
* a regression alert (metrics that got worse than the previous run),
* trend charts over time, each with its SLA threshold drawn as a reference line,
* a table of recent runs.

Run it with:  uv run streamlit run src/dashboard/app.py
"""

from dataclasses import asdict, dataclass

import altair as alt
import pandas as pd
import streamlit as st

from src.config import Thresholds, load_thresholds
from src.storage.db import MetricsDatabase

# A red-ish accent used for threshold lines and regressions.
THRESHOLD_COLOR = "#e45756"


@dataclass
class MetricSpec:
    """Describes how to display and judge one metric column."""

    column: str
    label: str
    higher_is_better: bool
    threshold: float
    value_kind: str  # "percent" | "score" | "seconds" | "usd"

    def format_value(self, value: float) -> str:
        """Format an absolute metric value for display."""
        if self.value_kind == "percent":
            return f"{value * 100:.1f}%"
        if self.value_kind == "seconds":
            return f"{value:.3f}s"
        if self.value_kind == "usd":
            return f"${value:.6f}"
        return f"{value:.3f}"

    def format_delta(self, delta: float) -> str:
        """Format a change vs the previous run, keeping the +/- sign."""
        if self.value_kind == "percent":
            return f"{delta * 100:+.1f}%"
        if self.value_kind == "seconds":
            return f"{delta:+.3f}s"
        if self.value_kind == "usd":
            return f"${delta:+.6f}"
        return f"{delta:+.3f}"


def build_metric_specs(thresholds: Thresholds) -> list[MetricSpec]:
    """Build the display specs for every metric, wiring in the configured thresholds."""
    return [
        MetricSpec("hallucination_rate", "Hallucination rate", False,
                   thresholds.max_hallucination_rate, "percent"),
        MetricSpec("mean_answer_relevancy", "Answer relevancy", True,
                   thresholds.min_answer_relevancy, "score"),
        MetricSpec("mean_faithfulness", "Faithfulness", True,
                   thresholds.min_faithfulness, "score"),
        MetricSpec("latency_p50_seconds", "Latency p50", False,
                   thresholds.max_latency_p50_seconds, "seconds"),
        MetricSpec("latency_p95_seconds", "Latency p95", False,
                   thresholds.max_latency_p95_seconds, "seconds"),
        MetricSpec("mean_cost_usd", "Cost / query", False,
                   thresholds.max_cost_per_query_usd, "usd"),
    ]


def load_runs_dataframe() -> pd.DataFrame:
    """Load runs from the database into a DataFrame ordered oldest -> newest."""
    runs = MetricsDatabase().get_runs(limit=200)
    # get_runs returns newest-first; reverse so charts read left (old) to right (new).
    records = [asdict(run) for run in reversed(runs)]
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["run_label"] = "#" + frame["id"].astype(str)
    return frame


def is_regression(spec: MetricSpec, previous: float, latest: float) -> bool:
    """Return True if the metric got worse between the previous and latest run."""
    if latest == previous:
        return False
    return latest < previous if spec.higher_is_better else latest > previous


def metric_chart(frame: pd.DataFrame, spec: MetricSpec) -> alt.LayerChart:
    """Build a trend line chart for one metric with its SLA threshold drawn as a dashed line."""
    line = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("run_label:N", sort=list(frame["run_label"]), title="Run"),
            y=alt.Y(f"{spec.column}:Q", title=spec.label),
            tooltip=[
                alt.Tooltip("run_label:N", title="Run"),
                alt.Tooltip("timestamp:N", title="Timestamp"),
                alt.Tooltip(f"{spec.column}:Q", title=spec.label, format=".4f"),
            ],
        )
    )
    threshold_line = (
        alt.Chart(pd.DataFrame({"threshold": [spec.threshold]}))
        .mark_rule(color=THRESHOLD_COLOR, strokeDash=[6, 4])
        .encode(y="threshold:Q")
    )
    return (line + threshold_line).properties(height=260)


def render_latest_summary(frame: pd.DataFrame, specs: list[MetricSpec]) -> None:
    """Render the gate badge and the metric cards for the latest run."""
    latest = frame.iloc[-1]
    has_previous = len(frame) >= 2

    if bool(latest["gate_passed"]):
        st.success(f"✅ SLA gate PASSED for the latest run ({latest['run_label']})")
    else:
        st.error(f"❌ SLA gate FAILED for the latest run ({latest['run_label']})")

    columns = st.columns(3)
    for index, spec in enumerate(specs):
        latest_value = float(latest[spec.column])
        delta_text: str | None = None
        if has_previous:
            previous_value = float(frame.iloc[-2][spec.column])
            change = latest_value - previous_value
            delta_text = spec.format_delta(change) if change != 0 else None

        columns[index % 3].metric(
            label=spec.label,
            value=spec.format_value(latest_value),
            delta=delta_text,
            # For "higher is better" metrics a rise is good (green); for the rest a rise is bad.
            delta_color="normal" if spec.higher_is_better else "inverse",
        )


def render_regressions(frame: pd.DataFrame, specs: list[MetricSpec]) -> None:
    """Highlight any metric that regressed compared with the previous run."""
    if len(frame) < 2:
        st.info("Only one run so far — run the evaluation again to see trends and regressions.")
        return

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    regressions = [
        spec
        for spec in specs
        if is_regression(spec, float(previous[spec.column]), float(latest[spec.column]))
    ]

    if not regressions:
        st.success("No regressions vs the previous run — every metric held or improved.")
        return

    lines = []
    for spec in regressions:
        prev_value = spec.format_value(float(previous[spec.column]))
        new_value = spec.format_value(float(latest[spec.column]))
        lines.append(f"- **{spec.label}**: {prev_value} → {new_value}")
    st.error("Regression detected vs the previous run:\n" + "\n".join(lines))


def render_trend_charts(frame: pd.DataFrame, specs: list[MetricSpec]) -> None:
    """Render the per-metric trend charts in a two-column grid."""
    for index in range(0, len(specs), 2):
        columns = st.columns(2)
        for column, spec in zip(columns, specs[index : index + 2], strict=False):
            with column:
                st.caption(f"{spec.label} — dashed line is the SLA threshold")
                st.altair_chart(metric_chart(frame, spec), use_container_width=True)


def render_runs_table(frame: pd.DataFrame) -> None:
    """Render a table of recent runs (newest first)."""
    display_columns = [
        "run_label", "timestamp", "llm_mode", "model_name", "git_commit", "gate_passed",
        "hallucination_rate", "mean_answer_relevancy", "mean_faithfulness",
        "latency_p50_seconds", "latency_p95_seconds", "mean_cost_usd",
    ]
    table = frame[display_columns].iloc[::-1].reset_index(drop=True)
    st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the whole dashboard."""
    st.set_page_config(page_title="LLM Eval Dashboard", page_icon="📊", layout="wide")
    st.title("📊 LLM Evaluation Dashboard")
    st.caption("Quality metrics for the RAG pipeline, tracked over every evaluation run.")

    with st.sidebar:
        st.header("About")
        st.write(
            "This dashboard reads the metrics history from SQLite. Each run is produced by "
            "`uv run python -m src.eval.runner`."
        )
        if st.button("🔄 Refresh"):
            st.rerun()

    frame = load_runs_dataframe()
    if frame.empty:
        st.warning("No runs found yet. Run `uv run python -m src.eval.runner` to create one.")
        return

    specs = build_metric_specs(load_thresholds())

    st.subheader("Latest run")
    render_latest_summary(frame, specs)

    st.subheader("Regression check")
    render_regressions(frame, specs)

    st.subheader("Metrics over time")
    render_trend_charts(frame, specs)

    st.subheader("Run history")
    render_runs_table(frame)


if __name__ == "__main__":
    main()