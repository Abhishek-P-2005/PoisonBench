"""
evaluation/store.py
-------------------
Result logging (PRD v2.0: "PostgreSQL telemetry"). The same code writes to
PostgreSQL (production) or SQLite (quick local runs and tests):

    ResultStore("postgresql://user:pass@localhost:5432/poisonbench")
    ResultStore("sqlite:///results.db")

Tables use the prefix `eval_` so they cannot collide with Module 1's
db/init.sql tables. Mapping to the PRD checklist:
    experiment_runs  ->  eval_runs
    query_results    ->  eval_query_results
    false_positives  ->  eval_false_positives
(plus eval_stage_logs and eval_chunk_scores for per-stage latency and the
raw defense scores used by threshold calibration.)

Every run row stores the full config JSON, seed and context settings, so any
run can be reproduced from the database alone (PRD NFR-2).
"""

from __future__ import annotations

import math
import sqlite3

import pandas as pd

FLOAT = "DOUBLE PRECISION"

TABLES = {
    "eval_runs": {
        "key": ["run_id"],
        "columns": [
            ("run_id", "TEXT"), ("group_name", "TEXT"), ("config_name", "TEXT"),
            ("config_json", "TEXT"), ("attack_type", "TEXT"),
            ("poisoning_ratio", FLOAT), ("seed", "INTEGER"), ("source_name", "TEXT"),
            ("n_queries", "INTEGER"), ("context_size", "INTEGER"),
            ("n_retrieve", "INTEGER"), ("generation_enabled", "INTEGER"),
            ("created_at", "TEXT"), ("n_poison_candidates", "INTEGER"),
            ("attack_success_raw", FLOAT), ("attack_success_defended", FLOAT),
            ("attack_success_generation", FLOAT), ("security_blocked_pct", FLOAT),
            ("detection_rate", FLOAT), ("fpr", FLOAT), ("fpr_block", FLOAT),
            ("clean_retention", FLOAT), ("latency_ms_mean", FLOAT),
            ("latency_ms_p95", FLOAT), ("router_high_fraction", FLOAT),
        ],
    },
    "eval_query_results": {
        "key": ["run_id", "query_id"],
        "columns": [
            ("run_id", "TEXT"), ("query_id", "TEXT"), ("n_candidates", "INTEGER"),
            ("n_poison_candidates", "INTEGER"), ("raw_poison_in_context", "INTEGER"),
            ("defended_poison_in_context", "INTEGER"), ("raw_success", "INTEGER"),
            ("defended_success", "INTEGER"), ("gen_success", "INTEGER"),
            ("poison_blocked", "INTEGER"), ("poison_flagged", "INTEGER"),
            ("clean_seen", "INTEGER"), ("clean_blocked", "INTEGER"),
            ("clean_flagged", "INTEGER"), ("clean_retention", FLOAT),
            ("total_latency_ms", FLOAT),
        ],
    },
    "eval_stage_logs": {
        "key": ["run_id", "query_id", "position"],
        "columns": [
            ("run_id", "TEXT"), ("query_id", "TEXT"), ("stage", "TEXT"),
            ("position", "INTEGER"), ("n_in", "INTEGER"), ("n_blocked", "INTEGER"),
            ("n_flagged", "INTEGER"), ("latency_ms", FLOAT), ("poison_in", "INTEGER"),
            ("poison_blocked", "INTEGER"), ("poison_flagged", "INTEGER"),
            ("clean_in", "INTEGER"), ("clean_blocked", "INTEGER"),
            ("clean_flagged", "INTEGER"), ("routed_high", "INTEGER"),
            ("routed_low", "INTEGER"),
        ],
    },
    "eval_chunk_scores": {
        "key": ["run_id", "query_id", "chunk_id"],
        "columns": [
            ("run_id", "TEXT"), ("query_id", "TEXT"), ("chunk_id", "TEXT"),
            ("is_poisoned", "INTEGER"), ("attack_type", "TEXT"),
            ("poisoning_ratio", FLOAT), ("seed", "INTEGER"), ("similarity", FLOAT),
            ("d1_z", FLOAT), ("d1_flagged", "INTEGER"), ("d1_blocked", "INTEGER"),
            ("d2_suspicion", FLOAT), ("d2_flagged", "INTEGER"),
            ("d3_suspicion", FLOAT), ("d3_flagged", "INTEGER"),
            ("d3_blocked", "INTEGER"), ("d4_prob", FLOAT), ("risk_score", FLOAT),
        ],
    },
    "eval_false_positives": {
        "key": ["run_id", "query_id", "chunk_id", "stage", "action"],
        "columns": [
            ("run_id", "TEXT"), ("query_id", "TEXT"), ("chunk_id", "TEXT"),
            ("stage", "TEXT"), ("action", "TEXT"), ("suspicion", FLOAT),
        ],
    },
}


def schema_sql() -> str:
    """The full CREATE TABLE script (also exported to db/eval_schema.sql)."""
    parts = []
    for table, spec in TABLES.items():
        lines = [f"    {name} {kind}" for name, kind in spec["columns"]]
        lines.append(f"    PRIMARY KEY ({', '.join(spec['key'])})")
        parts.append(f"CREATE TABLE IF NOT EXISTS {table} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(parts) + "\n"


def _clean_value(value):
    """numpy / bool / NaN values -> plain values both databases accept."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bool):
            return 1 if value else 0
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class ResultStore:
    def __init__(self, url: str = "sqlite:///:memory:"):
        self.url = url
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            self.conn = sqlite3.connect(url[len("sqlite:///"):])
            self.placeholder = "?"
        elif url.startswith("postgresql://") or url.startswith("postgres://"):
            import psycopg2
            self.dialect = "postgres"
            self.conn = psycopg2.connect(url)
            self.placeholder = "%s"
        else:
            raise ValueError("url must start with sqlite:/// or postgresql://")
        self.init_schema()

    def init_schema(self) -> None:
        cur = self.conn.cursor()
        for statement in schema_sql().split(";"):
            if statement.strip():
                cur.execute(statement)
        self.conn.commit()

    def delete_run(self, run_id: str) -> None:
        cur = self.conn.cursor()
        for table in TABLES:
            cur.execute(f"DELETE FROM {table} WHERE run_id = {self.placeholder}", (run_id,))
        self.conn.commit()

    def insert(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        names = [name for name, _ in TABLES[table]["columns"]]
        values = []
        for row in rows:
            values.append(tuple(_clean_value(row.get(name)) for name in names))

        cur = self.conn.cursor()
        column_list = ", ".join(names)
        if self.dialect == "postgres":
            from psycopg2.extras import execute_values
            execute_values(cur, f"INSERT INTO {table} ({column_list}) VALUES %s", values, page_size=2000)
        else:
            marks = ", ".join([self.placeholder] * len(names))
            cur.executemany(f"INSERT INTO {table} ({column_list}) VALUES ({marks})", values)
        self.conn.commit()

    def fetch(self, table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cur = self.conn.cursor()
        cur.execute(sql, params)
        columns = [col[0] for col in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)

    def count(self, table: str) -> int:
        cur = self.conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self.conn.close()
