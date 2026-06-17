from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .storage import MetricsStore


METRIC_GUIDE = [
    {
        "metric": "CPU 使用率",
        "path": "cpu.percent",
        "watch": "连续超过 85%，95% 以上按严重处理",
        "reason": "tokenizer、调度、后处理、数据加载或日志处理可能拖慢 AI 服务。",
    },
    {
        "metric": "系统负载 / CPU 核数",
        "path": "load.load1_per_cpu",
        "watch": "超过 1.5",
        "reason": "运行队列积压，端到端延迟容易抖动。",
    },
    {
        "metric": "可用内存率",
        "path": "memory.available_percent",
        "watch": "低于 10%，低于 5% 按严重处理",
        "reason": "容易触发 OOM、Swap 或内核回收。",
    },
    {
        "metric": "Swap 使用率",
        "path": "swap.percent",
        "watch": "超过 20%",
        "reason": "模型服务通常对内存延迟敏感，Swap 会显著拉高延迟。",
    },
    {
        "metric": "GPU 利用率",
        "path": "gpu.devices[].utilization_gpu_percent",
        "watch": "低于 20% 且 CPU 高，或长期高于 95%",
        "reason": "过低可能是 CPU/IO/batch 瓶颈，过高说明算力接近瓶颈。",
    },
    {
        "metric": "GPU 显存使用率",
        "path": "gpu.devices[].memory_used_percent",
        "watch": "超过 90%，超过 97% 按严重处理",
        "reason": "KV cache、batch 或并发继续增长时容易 OOM。",
    },
    {
        "metric": "GPU 温度",
        "path": "gpu.devices[].temperature_c",
        "watch": "超过 82 摄氏度，超过 90 摄氏度按严重处理",
        "reason": "温度过高可能进入降频区间，吞吐会变得不稳定。",
    },
    {
        "metric": "GPU 功耗使用率",
        "path": "gpu.devices[].power_draw_percent",
        "watch": "超过 95%",
        "reason": "接近功耗墙时 GPU 频率可能受限。",
    },
    {
        "metric": "磁盘使用率",
        "path": "disk.usage_percent",
        "watch": "超过 85%，超过 95% 按严重处理",
        "reason": "日志、模型缓存、SQLite 或输出文件继续增长会导致不可写。",
    },
    {
        "metric": "磁盘 inode 使用率",
        "path": "disk.inode_usage_percent",
        "watch": "超过 85%，超过 95% 按严重处理",
        "reason": "inode 用尽时即使磁盘还有容量，也会导致文件创建失败。",
    },
    {
        "metric": "磁盘 IO 忙碌度",
        "path": "disk_io.busy_percent_estimate",
        "watch": "超过 80%",
        "reason": "模型加载、数据读取或日志写入可能阻塞关键路径。",
    },
    {
        "metric": "网络错误/丢包速率",
        "path": "network.errors_per_sec / network.drops_per_sec",
        "watch": "大于 0 且持续增长",
        "reason": "请求链路可能出现超时、重试或吞吐下降。",
    },
    {
        "metric": "综合资源压力评分",
        "path": "derived.resource_pressure_score",
        "watch": "超过 80，超过 90 按严重处理",
        "reason": "综合 CPU、内存、磁盘、IO 和 GPU 压力，用于快速判断节点是否接近瓶颈。",
    },
]


class DashboardServer:
    def __init__(self, config: AppConfig, store: MetricsStore):
        self.config = config
        self.store = store
        self.httpd = ThreadingHTTPServer((config.http_host, config.http_port), self._handler_class())

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="sentry-llm-dashboard", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self.httpd.shutdown()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        store = self.store
        config = self.config

        class Handler(BaseHTTPRequestHandler):
            server_version = "sentry-llm/0.1"

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(DASHBOARD_HTML)
                    return
                if parsed.path == "/api/latest":
                    self._send_json(
                        {
                            "sample": store.latest_sample(),
                            "alerts": store.active_alerts(limit=20),
                            "recent_alerts": store.recent_alerts(limit=20),
                        }
                    )
                    return
                if parsed.path == "/api/samples":
                    params = parse_qs(parsed.query)
                    limit = _int_param(params, "limit", 240, maximum=2000)
                    self._send_json({"samples": store.recent_samples(limit=limit)})
                    return
                if parsed.path == "/api/alerts":
                    params = parse_qs(parsed.query)
                    limit = _int_param(params, "limit", 100, maximum=1000)
                    self._send_json({"alerts": store.recent_alerts(limit=limit)})
                    return
                if parsed.path == "/api/guide":
                    self._send_json({"guide": METRIC_GUIDE, "thresholds": config.thresholds.__dict__})
                    return
                if parsed.path == "/healthz":
                    self._send_json({"ok": True})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def _int_param(params: dict[str, list[str]], name: str, default: int, maximum: int) -> int:
    try:
        value = int(params.get(name, [str(default)])[0])
    except ValueError:
        return default
    return max(1, min(value, maximum))


DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>sentry-llm</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --panel-soft: #f0f4f1;
      --text: #17201a;
      --muted: #66706a;
      --border: #dce2de;
      --accent: #0f7b63;
      --accent-2: #315f93;
      --warn: #b76c00;
      --crit: #b42318;
      --ok: #167044;
      --shadow: 0 1px 2px rgba(20, 24, 22, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 24px;
      background: rgba(247, 247, 244, 0.94);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 760;
    }
    .subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 13px;
    }
    .status-pill {
      min-width: 112px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      text-align: center;
      font-size: 13px;
      box-shadow: var(--shadow);
    }
    main {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 20px 24px 36px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(140px, 1fr));
      gap: 12px;
    }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .card {
      min-height: 104px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .value {
      margin-top: 8px;
      font-size: clamp(22px, 2.6vw, 34px);
      font-weight: 780;
      line-height: 1;
      overflow-wrap: anywhere;
    }
    .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(340px, 1fr);
      gap: 16px;
      margin-top: 16px;
      align-items: start;
    }
    .charts {
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 16px;
    }
    .panel {
      padding: 16px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 760;
    }
    .chart-box {
      position: relative;
      height: 260px;
    }
    .alerts {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 460px;
      overflow: auto;
    }
    .alert {
      padding: 12px;
      border: 1px solid var(--border);
      border-left-width: 4px;
      border-radius: 8px;
      background: var(--panel-soft);
    }
    .alert.warning { border-left-color: var(--warn); }
    .alert.critical { border-left-color: var(--crit); }
    .alert-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 13px;
      font-weight: 740;
    }
    .badge {
      flex: 0 0 auto;
      padding: 3px 7px;
      border-radius: 999px;
      color: #fff;
      font-size: 11px;
      line-height: 1;
      background: var(--warn);
    }
    .badge.critical { background: var(--crit); }
    .alert p {
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .guide {
      margin-top: 16px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      line-height: 1.45;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      background: #fbfcfa;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: #244e82;
    }
    .empty {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      padding: 12px;
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    @media (max-width: 1180px) {
      .cards { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; padding: 14px 16px; }
      main { padding: 16px; }
      .cards, .charts { grid-template-columns: 1fr; }
      .chart-box { height: 230px; }
      .status-pill { width: 100%; text-align: left; border-radius: 8px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>sentry-llm</h1>
      <div class="subtitle" id="subtitle">等待采集数据</div>
    </div>
    <div class="status-pill" id="status">loading</div>
  </header>
  <main>
    <section class="cards">
      <article class="card"><div class="label">CPU</div><div class="value" id="cpu">--</div><div class="hint" id="load">load --</div></article>
      <article class="card"><div class="label">内存</div><div class="value" id="memory">--</div><div class="hint" id="swap">swap --</div></article>
      <article class="card"><div class="label">GPU</div><div class="value" id="gpu">--</div><div class="hint" id="gpuMemory">显存 --</div></article>
      <article class="card"><div class="label">磁盘</div><div class="value" id="disk">--</div><div class="hint" id="diskIo">IO --</div></article>
      <article class="card"><div class="label">网络</div><div class="value" id="network">--</div><div class="hint" id="networkErr">错误/丢包 --</div></article>
      <article class="card"><div class="label">压力评分</div><div class="value" id="pressure">--</div><div class="hint">80 以上需要留意</div></article>
    </section>

    <section class="grid">
      <div class="charts">
        <article class="panel"><h2>CPU / 内存 / 压力</h2><div class="chart-box"><canvas id="resourceChart"></canvas></div></article>
        <article class="panel"><h2>GPU / 显存</h2><div class="chart-box"><canvas id="gpuChart"></canvas></div></article>
        <article class="panel"><h2>磁盘 IO</h2><div class="chart-box"><canvas id="diskChart"></canvas></div></article>
        <article class="panel"><h2>网络吞吐</h2><div class="chart-box"><canvas id="networkChart"></canvas></div></article>
      </div>
      <aside class="panel">
        <h2>当前告警</h2>
        <div class="alerts" id="alerts"><div class="empty">暂无告警</div></div>
      </aside>
    </section>

    <section class="panel guide">
      <h2>关键指标说明</h2>
      <table>
        <thead><tr><th>指标</th><th>字段</th><th>触发条件</th><th>为什么要看</th></tr></thead>
        <tbody id="guide"></tbody>
      </table>
    </section>
  </main>
  <script>
    const fmtPercent = (v) => Number.isFinite(v) ? `${v.toFixed(1)}%` : '--';
    const fmtNumber = (v, digits = 1) => Number.isFinite(v) ? v.toFixed(digits) : '--';
    const fmtBytes = (v) => {
      if (!Number.isFinite(v)) return '--';
      const units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s'];
      let value = v;
      let idx = 0;
      while (value >= 1024 && idx < units.length - 1) { value /= 1024; idx += 1; }
      return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[idx]}`;
    };
    const get = (obj, path, fallback = undefined) => path.split('.').reduce((acc, key) => acc && acc[key] !== undefined ? acc[key] : fallback, obj);
    const maxGpu = (sample, key) => {
      const devices = get(sample, 'payload.gpu.devices', []);
      if (!Array.isArray(devices) || devices.length === 0) return NaN;
      return Math.max(...devices.map((device) => Number(device[key])).filter(Number.isFinite));
    };
    const chartOptions = (unit) => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { intersect: false, mode: 'index' },
      scales: {
        x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { callback: (value) => `${value}${unit}` } }
      },
      plugins: {
        legend: { labels: { boxWidth: 10, boxHeight: 10 } }
      }
    });
    const makeLine = (canvasId, labels, datasets, unit = '') => new Chart(document.getElementById(canvasId), {
      type: 'line',
      data: { labels, datasets },
      options: chartOptions(unit)
    });
    const dataset = (label, data, color) => ({
      label,
      data,
      borderColor: color,
      backgroundColor: `${color}22`,
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.25
    });

    let charts = {};
    function renderCharts(samples) {
      const labels = samples.map((sample) => new Date(sample.timestamp * 1000).toLocaleTimeString());
      const cpu = samples.map((s) => get(s, 'payload.cpu.percent', NaN));
      const memory = samples.map((s) => get(s, 'payload.memory.percent', NaN));
      const pressure = samples.map((s) => get(s, 'payload.derived.resource_pressure_score', NaN));
      const gpu = samples.map((s) => maxGpu(s, 'utilization_gpu_percent'));
      const gpuMemory = samples.map((s) => maxGpu(s, 'memory_used_percent'));
      const diskRead = samples.map((s) => get(s, 'payload.disk_io.read_bytes_per_sec', 0) / 1024 / 1024);
      const diskWrite = samples.map((s) => get(s, 'payload.disk_io.write_bytes_per_sec', 0) / 1024 / 1024);
      const netRecv = samples.map((s) => get(s, 'payload.network.bytes_recv_per_sec', 0) / 1024 / 1024);
      const netSent = samples.map((s) => get(s, 'payload.network.bytes_sent_per_sec', 0) / 1024 / 1024);

      Object.values(charts).forEach((chart) => chart.destroy());
      charts.resource = makeLine('resourceChart', labels, [
        dataset('CPU %', cpu, '#0f7b63'),
        dataset('内存 %', memory, '#315f93'),
        dataset('压力', pressure, '#b76c00')
      ], '%');
      charts.gpu = makeLine('gpuChart', labels, [
        dataset('GPU %', gpu, '#5b6f24'),
        dataset('显存 %', gpuMemory, '#8d3d6b')
      ], '%');
      charts.disk = makeLine('diskChart', labels, [
        dataset('读 MB/s', diskRead, '#315f93'),
        dataset('写 MB/s', diskWrite, '#b76c00')
      ], '');
      charts.network = makeLine('networkChart', labels, [
        dataset('接收 MB/s', netRecv, '#0f7b63'),
        dataset('发送 MB/s', netSent, '#8d3d6b')
      ], '');
    }

    function renderLatest(sample, alerts) {
      if (!sample) {
        document.getElementById('status').textContent = 'no data';
        return;
      }
      const payload = sample.payload;
      const devices = payload.gpu.devices || [];
      const maxGpuUtil = devices.length ? Math.max(...devices.map((d) => Number(d.utilization_gpu_percent) || 0)) : NaN;
      const maxGpuMem = devices.length ? Math.max(...devices.map((d) => Number(d.memory_used_percent) || 0)) : NaN;
      const networkTotal = get(payload, 'derived.network_total_bytes_per_sec', 0);

      document.getElementById('subtitle').textContent = `${sample.service_name} / ${sample.hostname} / ${new Date(sample.timestamp * 1000).toLocaleString()}`;
      document.getElementById('status').textContent = alerts && alerts.length ? `${alerts.length} alerts` : 'healthy';
      document.getElementById('cpu').textContent = fmtPercent(get(payload, 'cpu.percent', NaN));
      document.getElementById('load').textContent = `load/CPU ${fmtNumber(get(payload, 'load.load1_per_cpu', NaN), 2)}`;
      document.getElementById('memory').textContent = fmtPercent(get(payload, 'memory.percent', NaN));
      document.getElementById('swap').textContent = `swap ${fmtPercent(get(payload, 'swap.percent', NaN))}`;
      document.getElementById('gpu').textContent = devices.length ? fmtPercent(maxGpuUtil) : 'N/A';
      document.getElementById('gpuMemory').textContent = devices.length ? `显存 ${fmtPercent(maxGpuMem)} / ${devices.length} 卡` : get(payload, 'gpu.reason', '无 GPU 数据');
      document.getElementById('disk').textContent = fmtPercent(get(payload, 'disk.usage_percent', NaN));
      document.getElementById('diskIo').textContent = `IO ${fmtBytes(get(payload, 'derived.disk_total_bytes_per_sec', NaN))}`;
      document.getElementById('network').textContent = fmtBytes(networkTotal);
      document.getElementById('networkErr').textContent = `错误/丢包 ${fmtNumber(get(payload, 'derived.network_error_or_drop_per_sec', NaN), 3)}/s`;
      document.getElementById('pressure').textContent = fmtNumber(get(payload, 'derived.resource_pressure_score', NaN), 1);

      const alertsBox = document.getElementById('alerts');
      if (!alerts || alerts.length === 0) {
        alertsBox.innerHTML = '<div class="empty">暂无告警</div>';
        return;
      }
      alertsBox.innerHTML = alerts.map((alert) => `
        <div class="alert ${alert.severity}">
          <div class="alert-title">
            <span>${alert.title}</span>
            <span class="badge ${alert.severity}">${alert.severity}</span>
          </div>
          <p><code>${alert.metric_path}</code> 当前 ${fmtNumber(alert.value, 2)}，阈值 ${fmtNumber(alert.threshold, 2)}</p>
          <p>${alert.message}</p>
        </div>
      `).join('');
    }

    async function refresh() {
      const [latestResp, samplesResp, guideResp] = await Promise.all([
        fetch('/api/latest'),
        fetch('/api/samples?limit=240'),
        fetch('/api/guide')
      ]);
      const latest = await latestResp.json();
      const samples = await samplesResp.json();
      const guide = await guideResp.json();
      renderLatest(latest.sample, latest.alerts);
      renderCharts(samples.samples || []);
      document.getElementById('guide').innerHTML = guide.guide.map((row) => `
        <tr>
          <td>${row.metric}</td>
          <td><code>${row.path}</code></td>
          <td>${row.watch}</td>
          <td>${row.reason}</td>
        </tr>
      `).join('');
    }

    refresh().catch((error) => {
      document.getElementById('status').textContent = 'error';
      console.error(error);
    });
    setInterval(() => refresh().catch(console.error), 5000);
  </script>
</body>
</html>
"""
