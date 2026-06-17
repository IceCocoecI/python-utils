from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import psutil

from .config import AppConfig
from .models import Alert, MetricSample


@dataclass
class PreviousCounters:
    timestamp: float
    disk_io: psutil._common.sdiskio | None
    network_io: psutil._common.snetio | None


class MetricsCollector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.hostname = socket.gethostname()
        self._previous: PreviousCounters | None = None
        psutil.cpu_percent(interval=None)

    def collect(self) -> MetricSample:
        now = time.time()
        disk_io = self._safe_disk_io_counters()
        network_io = self._safe_network_io_counters()
        interval = self._interval_seconds(now)
        ip_addresses = self._ip_addresses()
        primary_ip = ip_addresses[0] if ip_addresses else self._fallback_primary_ip()

        payload: dict[str, Any] = {
            "service": {
                "name": self.config.service_name,
                "hostname": self.hostname,
                "primary_ip": primary_ip,
                "ip_addresses": ip_addresses,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "time": {
                "timestamp": now,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "sample_interval_seconds": interval,
            },
            "cpu": self._collect_cpu(),
            "load": self._collect_load(),
            "memory": self._collect_memory(),
            "swap": self._collect_swap(),
            "disk": self._collect_disk_usage(),
            "disk_io": self._collect_disk_io(disk_io, interval),
            "network": self._collect_network(network_io, interval),
            "gpu": self._collect_gpu(),
            "process": self._collect_process_summary(),
        }
        payload["derived"] = self._derive_metrics(payload)

        alerts = self._evaluate_alerts(payload)
        self._previous = PreviousCounters(timestamp=now, disk_io=disk_io, network_io=network_io)
        return MetricSample(
            timestamp=now,
            service_name=self.config.service_name,
            hostname=self.hostname,
            primary_ip=primary_ip,
            payload=payload,
            alerts=alerts,
        )

    def _interval_seconds(self, now: float) -> float:
        if self._previous is None:
            return self.config.sample_interval_seconds
        return max(now - self._previous.timestamp, 0.001)

    def _collect_cpu(self) -> dict[str, Any]:
        freq = psutil.cpu_freq()
        temperatures = self._safe_temperatures()
        cpu_times = psutil.cpu_times_percent(interval=None)
        return {
            "percent": psutil.cpu_percent(interval=None),
            "per_cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
            "times_percent": cpu_times._asdict(),
            "iowait_percent": getattr(cpu_times, "iowait", 0.0),
            "steal_percent": getattr(cpu_times, "steal", 0.0),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency_mhz": {
                "current": getattr(freq, "current", None),
                "min": getattr(freq, "min", None),
                "max": getattr(freq, "max", None),
            },
            "temperature_c": temperatures,
        }

    def _collect_load(self) -> dict[str, Any]:
        logical_cores = psutil.cpu_count(logical=True) or 1
        try:
            load1, load5, load15 = os.getloadavg()
        except (AttributeError, OSError):
            load1 = load5 = load15 = 0.0

        return {
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "load1_per_cpu": load1 / logical_cores,
            "load5_per_cpu": load5 / logical_cores,
            "load15_per_cpu": load15 / logical_cores,
        }

    def _collect_memory(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        return {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "percent": memory.percent,
            "available_percent": 100.0 - memory.percent,
        }

    def _collect_swap(self) -> dict[str, Any]:
        swap = psutil.swap_memory()
        return {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "free_bytes": swap.free,
            "percent": swap.percent,
            "sin_bytes": swap.sin,
            "sout_bytes": swap.sout,
        }

    def _collect_disk_usage(self) -> dict[str, Any]:
        usage = psutil.disk_usage("/")
        statvfs = os.statvfs("/")
        total_inodes = statvfs.f_files
        free_inodes = statvfs.f_ffree
        inode_usage_percent = None
        if total_inodes:
            inode_usage_percent = round(((total_inodes - free_inodes) / total_inodes) * 100.0, 2)
        return {
            "path": "/",
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "usage_percent": usage.percent,
            "inode_total": total_inodes,
            "inode_free": free_inodes,
            "inode_usage_percent": inode_usage_percent,
        }

    def _collect_disk_io(self, current: psutil._common.sdiskio | None, interval: float) -> dict[str, Any]:
        if current is None:
            return {
                "available": False,
                "reason": "disk IO counters unavailable",
                "read_bytes_total": None,
                "write_bytes_total": None,
                "read_count_total": None,
                "write_count_total": None,
                "read_bytes_per_sec": 0.0,
                "write_bytes_per_sec": 0.0,
                "read_count_per_sec": 0.0,
                "write_count_per_sec": 0.0,
                "busy_percent_estimate": 0.0,
            }

        previous = self._previous.disk_io if self._previous else None
        read_bytes_per_sec = self._rate(current.read_bytes, getattr(previous, "read_bytes", None), interval)
        write_bytes_per_sec = self._rate(current.write_bytes, getattr(previous, "write_bytes", None), interval)
        read_count_per_sec = self._rate(current.read_count, getattr(previous, "read_count", None), interval)
        write_count_per_sec = self._rate(current.write_count, getattr(previous, "write_count", None), interval)
        busy_time_delta_ms = self._delta(getattr(current, "busy_time", 0), getattr(previous, "busy_time", None))
        busy_percent = min(100.0, max(0.0, (busy_time_delta_ms / (interval * 1000.0)) * 100.0))

        return {
            "available": True,
            "reason": None,
            "read_bytes_total": current.read_bytes,
            "write_bytes_total": current.write_bytes,
            "read_count_total": current.read_count,
            "write_count_total": current.write_count,
            "read_bytes_per_sec": read_bytes_per_sec,
            "write_bytes_per_sec": write_bytes_per_sec,
            "read_count_per_sec": read_count_per_sec,
            "write_count_per_sec": write_count_per_sec,
            "busy_percent_estimate": busy_percent,
        }

    def _collect_network(self, current: psutil._common.snetio | None, interval: float) -> dict[str, Any]:
        if current is None:
            return {
                "available": False,
                "reason": "network IO counters unavailable",
                "bytes_sent_total": None,
                "bytes_recv_total": None,
                "packets_sent_total": None,
                "packets_recv_total": None,
                "bytes_sent_per_sec": 0.0,
                "bytes_recv_per_sec": 0.0,
                "packets_sent_per_sec": 0.0,
                "packets_recv_per_sec": 0.0,
                "errors_per_sec": 0.0,
                "drops_per_sec": 0.0,
                "errin_total": None,
                "errout_total": None,
                "dropin_total": None,
                "dropout_total": None,
            }

        previous = self._previous.network_io if self._previous else None
        errin_rate = self._rate(current.errin, getattr(previous, "errin", None), interval)
        errout_rate = self._rate(current.errout, getattr(previous, "errout", None), interval)
        dropin_rate = self._rate(current.dropin, getattr(previous, "dropin", None), interval)
        dropout_rate = self._rate(current.dropout, getattr(previous, "dropout", None), interval)

        return {
            "available": True,
            "reason": None,
            "bytes_sent_total": current.bytes_sent,
            "bytes_recv_total": current.bytes_recv,
            "packets_sent_total": current.packets_sent,
            "packets_recv_total": current.packets_recv,
            "bytes_sent_per_sec": self._rate(current.bytes_sent, getattr(previous, "bytes_sent", None), interval),
            "bytes_recv_per_sec": self._rate(current.bytes_recv, getattr(previous, "bytes_recv", None), interval),
            "packets_sent_per_sec": self._rate(current.packets_sent, getattr(previous, "packets_sent", None), interval),
            "packets_recv_per_sec": self._rate(current.packets_recv, getattr(previous, "packets_recv", None), interval),
            "errors_per_sec": errin_rate + errout_rate,
            "drops_per_sec": dropin_rate + dropout_rate,
            "errin_total": current.errin,
            "errout_total": current.errout,
            "dropin_total": current.dropin,
            "dropout_total": current.dropout,
        }

    def _collect_gpu(self) -> dict[str, Any]:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return {
                "available": False,
                "provider": "nvidia-smi",
                "reason": "nvidia-smi not found",
                "devices": [],
            }

        core_fields = [
            "index",
            "name",
            "uuid",
            "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.used",
            "memory.free",
            "temperature.gpu",
            "power.draw",
            "power.limit",
            "clocks.sm",
            "clocks.mem",
            "pcie.link.gen.current",
            "pcie.link.width.current",
        ]
        optional_fields = [
            "encoder.stats.sessionCount",
            "utilization.encoder",
            "utilization.decoder",
        ]
        query_fields = core_fields + optional_fields
        result = self._query_nvidia_smi(nvidia_smi, query_fields)
        reason = None
        if result is None:
            result = self._query_nvidia_smi(nvidia_smi, core_fields)
            query_fields = core_fields
            reason = "extended GPU fields are unavailable; collected core GPU metrics only"

        if result is None:
            return {
                "available": False,
                "provider": "nvidia-smi",
                "reason": "nvidia-smi query failed",
                "devices": [],
            }

        devices = []
        for row in csv.reader(io.StringIO(result.stdout.strip())):
            if not row:
                continue
            values = [item.strip() for item in row]
            item = dict(zip(query_fields, values))
            memory_total = self._to_float(item.get("memory.total"))
            memory_used = self._to_float(item.get("memory.used"))
            power_draw = self._to_float(item.get("power.draw"))
            power_limit = self._to_float(item.get("power.limit"))
            devices.append(
                {
                    "index": self._to_int(item.get("index")),
                    "name": item.get("name"),
                    "uuid": item.get("uuid"),
                    "utilization_gpu_percent": self._to_float(item.get("utilization.gpu")),
                    "utilization_memory_percent": self._to_float(item.get("utilization.memory")),
                    "memory_total_mib": memory_total,
                    "memory_used_mib": memory_used,
                    "memory_free_mib": self._to_float(item.get("memory.free")),
                    "memory_used_percent": self._percent(memory_used, memory_total),
                    "temperature_c": self._to_float(item.get("temperature.gpu")),
                    "power_draw_w": power_draw,
                    "power_limit_w": power_limit,
                    "power_draw_percent": self._percent(power_draw, power_limit),
                    "sm_clock_mhz": self._to_float(item.get("clocks.sm")),
                    "memory_clock_mhz": self._to_float(item.get("clocks.mem")),
                    "pcie_link_gen": self._to_int(item.get("pcie.link.gen.current")),
                    "pcie_link_width": self._to_int(item.get("pcie.link.width.current")),
                    "encoder_sessions": self._to_int(item.get("encoder.stats.sessionCount")),
                    "encoder_utilization_percent": self._to_float(item.get("utilization.encoder")),
                    "decoder_utilization_percent": self._to_float(item.get("utilization.decoder")),
                }
            )

        return {
            "available": True,
            "provider": "nvidia-smi",
            "reason": reason,
            "device_count": len(devices),
            "devices": devices,
        }

    def _collect_process_summary(self) -> dict[str, Any]:
        current = psutil.Process()
        return {
            "process_count": len(psutil.pids()),
            "agent_pid": current.pid,
            "agent_memory_rss_bytes": current.memory_info().rss,
            "agent_cpu_percent": current.cpu_percent(interval=None),
            "open_files": self._safe_open_file_count(current),
            "threads": current.num_threads(),
            "boot_time": psutil.boot_time(),
        }

    def _ip_addresses(self) -> list[str]:
        addresses: list[str] = []
        try:
            interfaces = psutil.net_if_addrs()
        except (OSError, RuntimeError):
            return addresses

        for entries in interfaces.values():
            for entry in entries:
                if entry.family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                address = entry.address.split("%", 1)[0]
                if self._is_ignored_ip(address):
                    continue
                addresses.append(address)
        return sorted(set(addresses))

    def _fallback_primary_ip(self) -> str | None:
        try:
            for entry in socket.getaddrinfo(self.hostname, None):
                address = entry[4][0]
                if not self._is_ignored_ip(address):
                    return address
        except OSError:
            pass

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            return None

    def _is_ignored_ip(self, address: str) -> bool:
        if not address or address.startswith("127.") or address == "::1":
            return True
        if address.startswith("169.254.") or address.lower().startswith("fe80:"):
            return True
        return False

    def _derive_metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        cpu_pressure = self._metric_float(payload, "cpu", "percent")
        memory_pressure = self._metric_float(payload, "memory", "percent")
        disk_pressure = self._metric_float(payload, "disk", "usage_percent")
        io_pressure = self._metric_float(payload, "disk_io", "busy_percent_estimate")
        gpu_devices = payload["gpu"].get("devices", [])
        gpu_memory_pressure = max((device.get("memory_used_percent") or 0.0 for device in gpu_devices), default=0.0)
        gpu_compute_pressure = max((device.get("utilization_gpu_percent") or 0.0 for device in gpu_devices), default=0.0)

        weighted_pressures = [
            (cpu_pressure, 0.25),
            (memory_pressure, 0.25),
            (disk_pressure, 0.15),
            (io_pressure, 0.10),
        ]
        if payload["gpu"].get("available") and gpu_devices:
            weighted_pressures.extend(
                [
                    (gpu_memory_pressure, 0.15),
                    (gpu_compute_pressure, 0.10),
                ]
            )

        total_weight = sum(weight for _, weight in weighted_pressures) or 1.0
        resource_pressure_score = round(
            sum(value * weight for value, weight in weighted_pressures) / total_weight,
            2,
        )
        network_error_rate = self._metric_float(payload, "network", "errors_per_sec") + self._metric_float(
            payload,
            "network",
            "drops_per_sec",
        )
        total_disk_bytes_per_sec = (
            self._metric_float(payload, "disk_io", "read_bytes_per_sec")
            + self._metric_float(payload, "disk_io", "write_bytes_per_sec")
        )
        total_network_bytes_per_sec = (
            self._metric_float(payload, "network", "bytes_sent_per_sec")
            + self._metric_float(payload, "network", "bytes_recv_per_sec")
        )

        return {
            "resource_pressure_score": resource_pressure_score,
            "resource_pressure_weight_total": total_weight,
            "gpu_memory_pressure_percent": gpu_memory_pressure,
            "gpu_compute_pressure_percent": gpu_compute_pressure,
            "network_error_or_drop_per_sec": network_error_rate,
            "disk_total_bytes_per_sec": total_disk_bytes_per_sec,
            "network_total_bytes_per_sec": total_network_bytes_per_sec,
        }

    def _evaluate_alerts(self, payload: dict[str, Any]) -> list[Alert]:
        t = self.config.thresholds
        alerts: list[Alert] = []

        self._add_threshold_alert(
            alerts,
            code="cpu_high",
            title="CPU 使用率过高",
            metric_path="cpu.percent",
            value=payload["cpu"]["percent"],
            warning=t.cpu_percent_warning,
            critical=t.cpu_percent_critical,
            message="持续高 CPU 会让 tokenizer、调度、后处理或数据加载拖慢 AI 服务。",
        )
        self._add_threshold_alert(
            alerts,
            code="load_high",
            title="系统负载偏高",
            metric_path="load.load1_per_cpu",
            value=payload["load"]["load1_per_cpu"],
            warning=t.load_per_cpu_warning,
            critical=None,
            message="load1/CPU 超过阈值表示运行队列积压，推理延迟容易抖动。",
        )
        self._add_lower_threshold_alert(
            alerts,
            code="memory_low",
            title="可用内存不足",
            metric_path="memory.available_percent",
            value=payload["memory"]["available_percent"],
            warning=t.memory_available_percent_warning,
            critical=t.memory_available_percent_critical,
            message="可用内存过低时容易触发 OOM、Swap 或内核回收。",
        )
        self._add_threshold_alert(
            alerts,
            code="swap_high",
            title="Swap 使用偏高",
            metric_path="swap.percent",
            value=payload["swap"]["percent"],
            warning=t.swap_percent_warning,
            critical=None,
            message="模型服务对内存延迟敏感，Swap 增长通常意味着性能明显下降。",
        )
        self._add_threshold_alert(
            alerts,
            code="disk_full",
            title="磁盘空间不足",
            metric_path="disk.usage_percent",
            value=payload["disk"]["usage_percent"],
            warning=t.disk_usage_percent_warning,
            critical=t.disk_usage_percent_critical,
            message="磁盘过满会影响日志、模型缓存、SQLite 持久化和服务写入。",
        )
        self._add_threshold_alert(
            alerts,
            code="inode_full",
            title="磁盘 inode 使用率偏高",
            metric_path="disk.inode_usage_percent",
            value=payload["disk"]["inode_usage_percent"],
            warning=t.disk_usage_percent_warning,
            critical=t.disk_usage_percent_critical,
            message="inode 用尽时即使磁盘仍有空间，也会导致日志、缓存或临时文件创建失败。",
        )
        self._add_threshold_alert(
            alerts,
            code="disk_io_busy",
            title="磁盘 IO 忙碌",
            metric_path="disk_io.busy_percent_estimate",
            value=payload["disk_io"]["busy_percent_estimate"],
            warning=t.disk_io_busy_percent_warning,
            critical=None,
            message="IO 忙碌可能来自模型加载、数据读取或日志写入，会影响冷启动和吞吐稳定性。",
        )
        self._add_threshold_alert(
            alerts,
            code="network_errors",
            title="网络错误包增长",
            metric_path="network.errors_per_sec",
            value=payload["network"]["errors_per_sec"],
            warning=t.network_errors_per_sec_warning,
            critical=None,
            message="网络错误包持续增长会导致请求超时、重试和吞吐下降。",
        )
        self._add_threshold_alert(
            alerts,
            code="network_drops",
            title="网络丢包增长",
            metric_path="network.drops_per_sec",
            value=payload["network"]["drops_per_sec"],
            warning=t.network_drops_per_sec_warning,
            critical=None,
            message="网络丢包持续增长说明链路或网卡可能异常。",
        )
        self._add_threshold_alert(
            alerts,
            code="resource_pressure_high",
            title="综合资源压力偏高",
            metric_path="derived.resource_pressure_score",
            value=payload["derived"]["resource_pressure_score"],
            warning=t.resource_pressure_warning,
            critical=t.resource_pressure_critical,
            message="综合 CPU、内存、磁盘、IO 和 GPU 压力的评分过高，需要定位主瓶颈。",
        )

        for device in payload["gpu"].get("devices", []):
            index = device.get("index")
            prefix = f"gpu.devices[{index}]"
            self._add_threshold_alert(
                alerts,
                code=f"gpu_memory_high_{index}",
                title=f"GPU {index} 显存压力过高",
                metric_path=f"{prefix}.memory_used_percent",
                value=device.get("memory_used_percent"),
                warning=t.gpu_memory_percent_warning,
                critical=t.gpu_memory_percent_critical,
                message="显存接近上限时，KV cache、batch 或并发继续增长容易触发 OOM。",
            )
            self._add_threshold_alert(
                alerts,
                code=f"gpu_hot_{index}",
                title=f"GPU {index} 温度偏高",
                metric_path=f"{prefix}.temperature_c",
                value=device.get("temperature_c"),
                warning=t.gpu_temperature_warning_c,
                critical=t.gpu_temperature_critical_c,
                message="GPU 温度过高可能进入降频区间，影响吞吐稳定性。",
            )
            self._add_threshold_alert(
                alerts,
                code=f"gpu_power_high_{index}",
                title=f"GPU {index} 接近功耗墙",
                metric_path=f"{prefix}.power_draw_percent",
                value=device.get("power_draw_percent"),
                warning=t.gpu_power_percent_warning,
                critical=None,
                message="接近功耗上限时 GPU 频率可能受限，需要关注散热和功耗配置。",
            )
            self._add_threshold_alert(
                alerts,
                code=f"gpu_util_high_{index}",
                title=f"GPU {index} 利用率接近满载",
                metric_path=f"{prefix}.utilization_gpu_percent",
                value=device.get("utilization_gpu_percent"),
                warning=t.gpu_utilization_high_percent,
                critical=None,
                message="GPU 长时间接近满载时，排队和端到端延迟可能上升。",
            )
            gpu_util = device.get("utilization_gpu_percent")
            if gpu_util is not None and gpu_util < t.gpu_utilization_low_percent and payload["cpu"]["percent"] > t.cpu_percent_warning:
                alerts.append(
                    Alert(
                        code=f"gpu_underfed_{index}",
                        severity="warning",
                        title=f"GPU {index} 可能未被喂满",
                        message="GPU 利用率低但 CPU 偏高，可能是 tokenizer、数据加载、batch 或调度瓶颈。",
                        metric_path=f"{prefix}.utilization_gpu_percent",
                        value=float(gpu_util),
                        threshold=t.gpu_utilization_low_percent,
                    )
                )

        return alerts

    def _add_threshold_alert(
        self,
        alerts: list[Alert],
        code: str,
        title: str,
        metric_path: str,
        value: Any,
        warning: float,
        critical: float | None,
        message: str,
    ) -> None:
        metric_value = self._to_float(value)
        if metric_value is None:
            return

        if critical is not None and metric_value >= critical:
            severity = "critical"
            threshold = critical
        elif metric_value >= warning:
            severity = "warning"
            threshold = warning
        else:
            return

        alerts.append(
            Alert(
                code=code,
                severity=severity,
                title=title,
                message=message,
                metric_path=metric_path,
                value=metric_value,
                threshold=threshold,
            )
        )

    def _add_lower_threshold_alert(
        self,
        alerts: list[Alert],
        code: str,
        title: str,
        metric_path: str,
        value: Any,
        warning: float,
        critical: float | None,
        message: str,
    ) -> None:
        metric_value = self._to_float(value)
        if metric_value is None:
            return

        if critical is not None and metric_value <= critical:
            severity = "critical"
            threshold = critical
        elif metric_value <= warning:
            severity = "warning"
            threshold = warning
        else:
            return

        alerts.append(
            Alert(
                code=code,
                severity=severity,
                title=title,
                message=message,
                metric_path=metric_path,
                value=metric_value,
                threshold=threshold,
            )
        )

    def _safe_temperatures(self) -> dict[str, list[dict[str, Any]]]:
        try:
            raw = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return {}

        temperatures: dict[str, list[dict[str, Any]]] = {}
        for name, entries in raw.items():
            temperatures[name] = [
                {
                    "label": entry.label,
                    "current": entry.current,
                    "high": entry.high,
                    "critical": entry.critical,
                }
                for entry in entries
            ]
        return temperatures

    def _safe_open_file_count(self, process: psutil.Process) -> int | None:
        try:
            return len(process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None

    def _safe_disk_io_counters(self) -> psutil._common.sdiskio | None:
        try:
            return psutil.disk_io_counters()
        except (OSError, RuntimeError):
            return None

    def _safe_network_io_counters(self) -> psutil._common.snetio | None:
        try:
            return psutil.net_io_counters()
        except (OSError, RuntimeError):
            return None

    def _query_nvidia_smi(self, nvidia_smi: str, query_fields: list[str]) -> subprocess.CompletedProcess[str] | None:
        cmd = [
            nvidia_smi,
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3)
        except (subprocess.SubprocessError, OSError):
            return None

    def _rate(self, current: int | float, previous: int | float | None, interval: float) -> float:
        if previous is None:
            return 0.0
        return max(0.0, (float(current) - float(previous)) / interval)

    def _delta(self, current: int | float, previous: int | float | None) -> float:
        if previous is None:
            return 0.0
        return max(0.0, float(current) - float(previous))

    def _percent(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator in (None, 0):
            return None
        return round((numerator / denominator) * 100.0, 2)

    def _to_float(self, value: Any) -> float | None:
        if value in (None, "", "N/A", "[N/A]"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int(self, value: Any) -> int | None:
        number = self._to_float(value)
        if number is None:
            return None
        return int(number)

    def _metric_float(self, payload: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
        value = payload.get(section, {}).get(key)
        number = self._to_float(value)
        return default if number is None else number
