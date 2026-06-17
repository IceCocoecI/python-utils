from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import Alert, MetricSample


class MetricsStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def insert_sample(self, sample: MetricSample) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO metric_samples (
                    timestamp,
                    service_name,
                    hostname,
                    primary_ip,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    sample.timestamp,
                    sample.service_name,
                    sample.hostname,
                    sample.primary_ip,
                    json.dumps(sample.payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            sample_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO alerts (
                    sample_id,
                    timestamp,
                    service_name,
                    hostname,
                    code,
                    severity,
                    title,
                    message,
                    metric_path,
                    value,
                    threshold
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        sample_id,
                        sample.timestamp,
                        sample.service_name,
                        sample.hostname,
                        alert.code,
                        alert.severity,
                        alert.title,
                        alert.message,
                        alert.metric_path,
                        alert.value,
                        alert.threshold,
                    )
                    for alert in sample.alerts
                ],
            )
            return sample_id

    def latest_sample(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, timestamp, service_name, hostname, primary_ip, payload_json
                FROM metric_samples
                ORDER BY timestamp DESC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_sample(row) if row else None

    def recent_samples(self, limit: int = 240) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, service_name, hostname, primary_ip, payload_json
                FROM metric_samples
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        samples = [self._row_to_sample(row) for row in rows]
        samples.reverse()
        return samples

    def recent_alerts(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT alerts.id, alerts.sample_id, alerts.timestamp, alerts.service_name, alerts.hostname,
                       metric_samples.primary_ip, alerts.code, alerts.severity, alerts.title,
                       alerts.message, alerts.metric_path, alerts.value, alerts.threshold
                FROM alerts
                LEFT JOIN metric_samples ON metric_samples.id = alerts.sample_id
                ORDER BY alerts.timestamp DESC, alerts.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def active_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, sample_id, timestamp, service_name, hostname, primary_ip, code, severity, title,
                       message, metric_path, value, threshold
                FROM (
                    SELECT alerts.*,
                           metric_samples.primary_ip,
                           ROW_NUMBER() OVER (
                               PARTITION BY alerts.service_name, alerts.hostname, alerts.code, alerts.metric_path
                               ORDER BY alerts.timestamp DESC, alerts.id DESC
                           ) AS rn
                    FROM alerts
                    LEFT JOIN metric_samples ON metric_samples.id = alerts.sample_id
                )
                WHERE rn = 1
                ORDER BY
                    CASE severity
                        WHEN 'critical' THEN 0
                        WHEN 'warning' THEN 1
                        ELSE 2
                    END,
                    timestamp DESC,
                    id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def prune_older_than(self, days: int) -> tuple[int, int]:
        if days <= 0:
            raise ValueError("days must be a positive integer")

        cutoff = time.time() - days * 86400
        with self.connect() as connection:
            alerts_deleted = connection.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,)).rowcount
            samples_deleted = connection.execute("DELETE FROM metric_samples WHERE timestamp < ?", (cutoff,)).rowcount
        return samples_deleted, alerts_deleted

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    service_name TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    primary_ip TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "metric_samples", "primary_ip", "TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metric_samples_timestamp
                ON metric_samples(timestamp)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metric_samples_identity
                ON metric_samples(service_name, hostname, primary_ip, timestamp)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    service_name TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_path TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    FOREIGN KEY(sample_id) REFERENCES metric_samples(id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                ON alerts(timestamp)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_severity
                ON alerts(severity)
                """
            )

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")}
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _row_to_sample(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "service_name": row["service_name"],
            "hostname": row["hostname"],
            "primary_ip": row["primary_ip"],
            "payload": json.loads(row["payload_json"]),
        }


def alert_to_dict(alert: Alert) -> dict[str, Any]:
    return asdict(alert)
