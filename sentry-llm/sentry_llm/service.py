from __future__ import annotations

import socket
import signal
import threading
import time
from collections.abc import Callable
from traceback import format_exception_only

from .collector import MetricsCollector
from .config import AppConfig
from .models import Alert, MetricSample
from .storage import MetricsStore


class CollectorService:
    def __init__(self, config: AppConfig, store: MetricsStore):
        self.config = config
        self.store = store
        self.collector = MetricsCollector(config)
        self._stop_event = threading.Event()
        self._last_prune_monotonic = 0.0

    def run_forever(self, on_sample: Callable[[int], None] | None = None) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self._prune_if_due(started)
            try:
                sample = self.collector.collect()
            except Exception as exc:
                sample = self._collection_error_sample(exc)

            try:
                sample_id = self.store.insert_sample(sample)
            except Exception as exc:
                print(f"sentry-llm failed to write sample: {exc}", flush=True)
            else:
                if on_sample:
                    on_sample(sample_id)

            elapsed = time.monotonic() - started
            sleep_seconds = max(0.1, self.config.sample_interval_seconds - elapsed)
            self._stop_event.wait(sleep_seconds)

    def stop(self) -> None:
        self._stop_event.set()

    def _collection_error_sample(self, exc: Exception) -> MetricSample:
        now = time.time()
        hostname = socket.gethostname()
        message = "".join(format_exception_only(type(exc), exc)).strip()
        alert = Alert(
            code="collector_error",
            severity="critical",
            title="采集器异常",
            message=f"本轮采集失败，agent 已继续运行。错误：{message}",
            metric_path="collector.error",
            value=1.0,
            threshold=0.0,
        )
        return MetricSample(
            timestamp=now,
            service_name=self.config.service_name,
            hostname=hostname,
            primary_ip=None,
            payload={
                "service": {
                    "name": self.config.service_name,
                    "hostname": hostname,
                    "primary_ip": None,
                    "ip_addresses": [],
                },
                "time": {
                    "timestamp": now,
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    "sample_interval_seconds": self.config.sample_interval_seconds,
                },
                "collector": {
                    "ok": False,
                    "error": message,
                },
                "derived": {
                    "resource_pressure_score": None,
                },
            },
            alerts=[alert],
        )

    def _prune_if_due(self, now_monotonic: float) -> None:
        if self.config.retention_days <= 0:
            return
        if self._last_prune_monotonic and now_monotonic - self._last_prune_monotonic < 3600:
            return

        self._last_prune_monotonic = now_monotonic
        try:
            self.store.prune_older_than(self.config.retention_days)
        except Exception as exc:
            print(f"sentry-llm failed to prune old samples: {exc}", flush=True)


def install_signal_handlers(service: CollectorService) -> None:
    def _handle_signal(signum: int, frame: object) -> None:
        service.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
