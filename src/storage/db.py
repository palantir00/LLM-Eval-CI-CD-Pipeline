"""Metric storage in SQLite.

WHY SQLite? It is a full SQL database that lives in a single file, needs no server, and ships
with Python's standard library. For this project we have one writer (the eval runner) appending
a handful of rows per run and a reader (the dashboard/gate) doing simple queries — SQLite handles
that with zero operational overhead. We would only reach for Postgres if we needed many concurrent
writers, network access, or very large scale; none of which an evaluation-history store requires.

Migrations: we keep it deliberately simple using SQLite's built-in ``PRAGMA user_version``. The
database stores its own schema version; on startup we run any migration whose version is newer.
Today there is just version 1, but the pattern scales to future changes without extra tooling.
"""

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.eval.metrics import RunMetrics
from src.paths import METRICS_DB_PATH

if TYPE_CHECKING:
    # Imported only for type hints. Avoids a runtime import cycle (runner imports this module).
    from src.eval.runner import ItemResult

logger = logging.getLogger(__name__)

# Bump this (and add a matching _migrate_to_vN) whenever the schema changes.
CURRENT_SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    """One row from the ``runs`` table: a run's metadata plus its aggregate metrics."""

    id: int
    timestamp: str
    llm_mode: str
    model_name: str
    git_commit: str | None
    gate_passed: bool
    num_items: int
    hallucination_rate: float
    mean_answer_relevancy: float
    mean_faithfulness: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    mean_cost_usd: float


class MetricsDatabase:
    """A thin wrapper around the SQLite metrics database."""

    def __init__(self, path: Path = METRICS_DB_PATH) -> None:
        """Open (and if needed create/upgrade) the database.

        Args:
            path: Path to the SQLite file.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def path(self) -> Path:
        """Filesystem path of the database (handy for logging)."""
        return self._path

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with sensible defaults (row access by name, FK enforcement)."""
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row  # rows accessible by column name
        connection.execute("PRAGMA foreign_keys = ON")  # enforce the run_items -> runs FK
        return connection

    def _init_schema(self) -> None:
        """Create or upgrade the schema based on the stored ``user_version``."""
        connection = self._connect()
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                self._migrate_to_v1(connection)
            # Record the schema version so we don't re-run migrations next time.
            connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _migrate_to_v1(connection: sqlite3.Connection) -> None:
        """Create the version-1 schema: the runs table and the per-item details table."""
        connection.executescript(
            """
            CREATE TABLE runs (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT    NOT NULL,
                llm_mode              TEXT    NOT NULL,
                model_name            TEXT    NOT NULL,
                git_commit            TEXT,
                gate_passed           INTEGER NOT NULL,
                num_items             INTEGER NOT NULL,
                hallucination_rate    REAL    NOT NULL,
                mean_answer_relevancy REAL    NOT NULL,
                mean_faithfulness     REAL    NOT NULL,
                latency_p50_seconds   REAL    NOT NULL,
                latency_p95_seconds   REAL    NOT NULL,
                mean_cost_usd         REAL    NOT NULL
            );

            CREATE TABLE run_items (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                item_id         TEXT    NOT NULL,
                relevancy       REAL    NOT NULL,
                faithfulness    REAL    NOT NULL,
                hallucinated    INTEGER NOT NULL,
                latency_seconds REAL    NOT NULL,
                cost_usd        REAL    NOT NULL,
                input_tokens    INTEGER NOT NULL,
                output_tokens   INTEGER NOT NULL
            );
            """
        )

    def save_run(
        self,
        metrics: RunMetrics,
        items: "Sequence[ItemResult]",
        *,
        llm_mode: str,
        model_name: str,
        gate_passed: bool,
        git_commit: str | None = None,
    ) -> int:
        """Save one evaluation run (aggregate row + per-item rows) and return its run id.

        Args:
            metrics: The aggregated run metrics.
            items: Per-item results to store alongside the run.
            llm_mode: "mock" or "openai" (how the run was produced).
            model_name: The model used.
            gate_passed: Whether the SLA gate passed for this run.
            git_commit: The git commit the run was produced from, if known.

        Returns:
            The auto-generated id of the new run.
        """
        timestamp = datetime.now(UTC).isoformat()

        connection = self._connect()
        try:
            # "with connection" wraps the inserts in a single transaction: it commits on success
            # and rolls back if anything raises, so a run is saved all-or-nothing.
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO runs (
                        timestamp, llm_mode, model_name, git_commit, gate_passed, num_items,
                        hallucination_rate, mean_answer_relevancy, mean_faithfulness,
                        latency_p50_seconds, latency_p95_seconds, mean_cost_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        llm_mode,
                        model_name,
                        git_commit,
                        int(gate_passed),
                        metrics.num_items,
                        metrics.hallucination_rate,
                        metrics.mean_answer_relevancy,
                        metrics.mean_faithfulness,
                        metrics.latency_p50_seconds,
                        metrics.latency_p95_seconds,
                        metrics.mean_cost_usd,
                    ),
                )
                run_id = cursor.lastrowid
                if run_id is None:
                    raise RuntimeError("Failed to insert run row (no row id returned)")

                connection.executemany(
                    """
                    INSERT INTO run_items (
                        run_id, item_id, relevancy, faithfulness, hallucinated,
                        latency_seconds, cost_usd, input_tokens, output_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            item.id,
                            item.relevancy,
                            item.faithfulness,
                            int(item.hallucinated),
                            item.latency_seconds,
                            item.cost_usd,
                            item.input_tokens,
                            item.output_tokens,
                        )
                        for item in items
                    ],
                )
        finally:
            connection.close()

        logger.info("Saved run #%d (%d items) to %s", run_id, metrics.num_items, self._path)
        return run_id

    def get_latest_run(self) -> RunRecord | None:
        """Return the most recent run, or None if the database is empty."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        return self._row_to_record(row) if row is not None else None

    def get_runs(self, limit: int = 100) -> list[RunRecord]:
        """Return recent runs, newest first.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            A list of RunRecord (possibly empty).
        """
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            connection.close()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RunRecord:
        """Convert a database row into a RunRecord."""
        return RunRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            llm_mode=row["llm_mode"],
            model_name=row["model_name"],
            git_commit=row["git_commit"],
            gate_passed=bool(row["gate_passed"]),
            num_items=row["num_items"],
            hallucination_rate=row["hallucination_rate"],
            mean_answer_relevancy=row["mean_answer_relevancy"],
            mean_faithfulness=row["mean_faithfulness"],
            latency_p50_seconds=row["latency_p50_seconds"],
            latency_p95_seconds=row["latency_p95_seconds"],
            mean_cost_usd=row["mean_cost_usd"],
        )