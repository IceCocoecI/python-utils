from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Thresholds:
    cpu_percent_warning: float = 85.0
    cpu_percent_critical: float = 95.0
    load_per_cpu_warning: float = 1.5
    memory_available_percent_warning: float = 10.0
    memory_available_percent_critical: float = 5.0
    swap_percent_warning: float = 20.0
    disk_usage_percent_warning: float = 85.0
    disk_usage_percent_critical: float = 95.0
    disk_io_busy_percent_warning: float = 80.0
    network_errors_per_sec_warning: float = 0.1
    network_drops_per_sec_warning: float = 0.1
    gpu_utilization_low_percent: float = 20.0
    gpu_utilization_high_percent: float = 95.0
    gpu_memory_percent_warning: float = 90.0
    gpu_memory_percent_critical: float = 97.0
    gpu_temperature_warning_c: float = 82.0
    gpu_temperature_critical_c: float = 90.0
    gpu_power_percent_warning: float = 95.0
    resource_pressure_warning: float = 80.0
    resource_pressure_critical: float = 90.0


@dataclass(frozen=True)
class AppConfig:
    service_name: str = "local-ai-node"
    sample_interval_seconds: float = 5.0
    database_path: Path = Path("data/sentry_llm.sqlite3")
    retention_days: int = 30
    http_host: str = "127.0.0.1"
    http_port: int = 8765
    thresholds: Thresholds = field(default_factory=Thresholds)


def load_config(path: str | Path | None = None) -> AppConfig:
    if not path:
        return AppConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if tomllib is None:
        raise RuntimeError("TOML config files require Python 3.11+ or tomllib support.")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    return parse_config(raw, base_dir=base_dir)


def parse_config(raw: dict[str, Any], base_dir: Path | None = None) -> AppConfig:
    thresholds_raw = raw.get("thresholds", {})
    threshold_names = {item.name for item in fields(Thresholds)}
    thresholds = Thresholds(**{k: v for k, v in thresholds_raw.items() if k in threshold_names})

    database_path = Path(raw.get("database_path", AppConfig.database_path))
    if base_dir and not database_path.is_absolute():
        database_path = base_dir / database_path

    sample_interval_seconds = float(raw.get("sample_interval_seconds", AppConfig.sample_interval_seconds))
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    retention_days = int(raw.get("retention_days", AppConfig.retention_days))
    if retention_days <= 0:
        raise ValueError("retention_days must be a positive integer")

    return AppConfig(
        service_name=str(raw.get("service_name", AppConfig.service_name)),
        sample_interval_seconds=sample_interval_seconds,
        database_path=database_path,
        retention_days=retention_days,
        http_host=str(raw.get("http_host", AppConfig.http_host)),
        http_port=int(raw.get("http_port", AppConfig.http_port)),
        thresholds=thresholds,
    )
