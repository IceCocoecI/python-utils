from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class Alert:
    code: str
    severity: Severity
    title: str
    message: str
    metric_path: str
    value: float
    threshold: float


@dataclass(frozen=True)
class MetricSample:
    timestamp: float
    service_name: str
    hostname: str
    primary_ip: str | None
    payload: dict[str, Any]
    alerts: list[Alert]
