from __future__ import annotations

import time
from pathlib import Path

import pytest

from sentry_llm.collector import MetricsCollector
from sentry_llm.config import AppConfig
from sentry_llm.models import Alert, MetricSample
from sentry_llm.storage import MetricsStore


def test_resource_pressure_renormalizes_without_gpu() -> None:
    collector = MetricsCollector(AppConfig())
    payload = {
        "cpu": {"percent": 100.0},
        "memory": {"percent": 100.0},
        "disk": {"usage_percent": 100.0},
        "disk_io": {"busy_percent_estimate": 100.0},
        "network": {
            "errors_per_sec": 0.0,
            "drops_per_sec": 0.0,
            "bytes_sent_per_sec": 0.0,
            "bytes_recv_per_sec": 0.0,
        },
        "gpu": {"available": False, "devices": []},
    }

    derived = collector._derive_metrics(payload)

    assert derived["resource_pressure_score"] == 100.0
    assert derived["resource_pressure_weight_total"] == 0.75


def test_prune_rejects_non_positive_days(tmp_path: Path) -> None:
    store = MetricsStore(tmp_path / "metrics.sqlite3")

    with pytest.raises(ValueError, match="positive"):
        store.prune_older_than(0)


def test_active_alerts_returns_latest_alert_per_metric(tmp_path: Path) -> None:
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    first = MetricSample(
        timestamp=time.time() - 10,
        service_name="svc",
        hostname="host",
        primary_ip="10.0.0.1",
        payload={"ok": True},
        alerts=[
            Alert(
                code="disk_full",
                severity="warning",
                title="Disk",
                message="old",
                metric_path="disk.usage_percent",
                value=90.0,
                threshold=85.0,
            )
        ],
    )
    second = MetricSample(
        timestamp=time.time(),
        service_name="svc",
        hostname="host",
        primary_ip="10.0.0.1",
        payload={"ok": True},
        alerts=[
            Alert(
                code="disk_full",
                severity="critical",
                title="Disk",
                message="new",
                metric_path="disk.usage_percent",
                value=96.0,
                threshold=95.0,
            )
        ],
    )

    store.insert_sample(first)
    store.insert_sample(second)

    latest_sample = store.latest_sample()
    active_alerts = store.active_alerts(limit=10)

    assert latest_sample is not None
    assert latest_sample["primary_ip"] == "10.0.0.1"
    assert len(active_alerts) == 1
    assert active_alerts[0]["message"] == "new"
    assert active_alerts[0]["severity"] == "critical"
    assert active_alerts[0]["primary_ip"] == "10.0.0.1"
