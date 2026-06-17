from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .collector import MetricsCollector
from .config import AppConfig, load_config
from .service import CollectorService, install_signal_handlers
from .storage import MetricsStore
from .web import DashboardServer


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _build_config(args)

    if args.command == "serve":
        serve(config)
    elif args.command == "collect":
        collect(config, once=args.once)
    elif args.command == "snapshot":
        snapshot(config)
    elif args.command == "alerts":
        show_alerts(config, args.limit)
    elif args.command == "prune":
        prune(config, args.days)
    else:
        parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor AI service machine metrics.")
    _add_common_options(parser)

    common_after_command = argparse.ArgumentParser(add_help=False)
    _add_common_options(common_after_command, default=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run collector and web dashboard.",
        parents=[common_after_command],
    )
    serve_parser.add_argument("--host", help="Dashboard bind host.")
    serve_parser.add_argument("--port", type=int, help="Dashboard bind port.")

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run collector only.",
        parents=[common_after_command],
    )
    collect_parser.add_argument("--once", action="store_true", help="Collect one sample and exit.")

    subparsers.add_parser(
        "snapshot",
        help="Collect and print one sample without writing to the database.",
        parents=[common_after_command],
    )

    alerts_parser = subparsers.add_parser(
        "alerts",
        help="Print recent alerts from the database.",
        parents=[common_after_command],
    )
    alerts_parser.add_argument("--limit", type=int, default=20)

    prune_parser = subparsers.add_parser(
        "prune",
        help="Delete old samples and alerts.",
        parents=[common_after_command],
    )
    prune_parser.add_argument("--days", type=int, default=None, help="Retention days. Defaults to config retention.")

    parser.set_defaults(command="serve")
    return parser


def _add_common_options(parser: argparse.ArgumentParser, default: object | None = None) -> None:
    kwargs = {"default": default} if default is not None else {}
    parser.add_argument("--config", help="Path to TOML config file.", **kwargs)
    parser.add_argument("--db", help="SQLite database path.", **kwargs)
    parser.add_argument("--service-name", help="Service/node name stored with samples.", **kwargs)
    parser.add_argument("--interval", type=float, help="Sample interval in seconds.", **kwargs)


def serve(config: AppConfig) -> None:
    store = MetricsStore(config.database_path)
    collector_service = CollectorService(config, store)
    install_signal_handlers(collector_service)

    dashboard = DashboardServer(config, store)
    dashboard.start_in_thread()
    print(f"sentry-llm dashboard: http://{config.http_host}:{config.http_port}", flush=True)
    print(f"sqlite database: {config.database_path}", flush=True)

    try:
        collector_service.run_forever()
    finally:
        dashboard.shutdown()


def collect(config: AppConfig, once: bool = False) -> None:
    store = MetricsStore(config.database_path)
    if once:
        collector = MetricsCollector(config)
        sample_id = store.insert_sample(collector.collect())
        print(f"stored sample {sample_id}", flush=True)
        return

    service = CollectorService(config, store)
    install_signal_handlers(service)
    service.run_forever(on_sample=lambda sample_id: print(f"stored sample {sample_id}", flush=True))


def snapshot(config: AppConfig) -> None:
    collector = MetricsCollector(config)
    sample = collector.collect()
    print(
        json.dumps(
            {
                "timestamp": sample.timestamp,
                "service_name": sample.service_name,
                "hostname": sample.hostname,
                "payload": sample.payload,
                "alerts": [asdict(alert) for alert in sample.alerts],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def show_alerts(config: AppConfig, limit: int) -> None:
    store = MetricsStore(config.database_path)
    print(json.dumps(store.recent_alerts(limit=limit), ensure_ascii=False, indent=2))


def prune(config: AppConfig, days: int | None) -> None:
    retention_days = days if days is not None else config.retention_days
    if retention_days <= 0:
        raise SystemExit("retention days must be a positive integer")

    store = MetricsStore(config.database_path)
    samples_deleted, alerts_deleted = store.prune_older_than(retention_days)
    print(f"deleted {samples_deleted} samples and {alerts_deleted} alerts older than {retention_days} days")


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    updates = {}
    if args.db:
        updates["database_path"] = Path(args.db)
    if args.service_name:
        updates["service_name"] = args.service_name
    if args.interval:
        if args.interval <= 0:
            raise SystemExit("sample interval must be positive")
        updates["sample_interval_seconds"] = args.interval
    if getattr(args, "host", None):
        updates["http_host"] = args.host
    if getattr(args, "port", None):
        updates["http_port"] = args.port

    if not updates:
        return config

    return AppConfig(
        service_name=updates.get("service_name", config.service_name),
        sample_interval_seconds=updates.get("sample_interval_seconds", config.sample_interval_seconds),
        database_path=updates.get("database_path", config.database_path),
        retention_days=config.retention_days,
        http_host=updates.get("http_host", config.http_host),
        http_port=updates.get("http_port", config.http_port),
        thresholds=config.thresholds,
    )


if __name__ == "__main__":
    main()
