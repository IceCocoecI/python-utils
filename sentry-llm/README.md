# sentry-llm

`sentry-llm` 是一个轻量级 AI 服务机器指标监控子模块，关注推理/训练节点的核心资源状态：

- CPU：总体使用率、负载、温度、进程数
- 内存：物理内存、Swap、可用率
- GPU：利用率、显存、功耗、温度、编码/解码利用率，优先通过 `nvidia-smi` 采集
- 磁盘：空间、IO 吞吐、IOPS、忙碌度估算
- 网络：收发吞吐、包量、错误包和丢包
- 机器标识：hostname、primary IP、非 loopback IP 列表
- 计算指标：采样间隔内的速率、资源压力评分、GPU 显存压力、磁盘/网络错误速率

默认持久化到 SQLite，不依赖 Prometheus/Grafana。内置 Web 面板用于快速查看趋势、当前状态、告警和指标说明。

## 快速开始

```bash
cd sentry-llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m sentry_llm serve
```

然后打开：

```text
http://127.0.0.1:8765
```

采集数据默认写入：

```text
sentry-llm/data/sentry_llm.sqlite3
```

每台机器独立运行时会把本机 `hostname`、`primary_ip` 和 `ip_addresses` 写入每条采样记录。当前版本不做多机汇总，但把多台机器的数据库文件单独拷出来后，可以通过这些字段识别数据来源。

## 常用命令

启动采集和面板：

```bash
python -m sentry_llm serve --host 0.0.0.0 --port 8765 --interval 5
```

只采集，不启动面板：

```bash
python -m sentry_llm collect --interval 5
```

查看最近一次快照：

```bash
python -m sentry_llm snapshot
```

查看最近告警：

```bash
python -m sentry_llm alerts --limit 20
```

清理旧数据：

```bash
python -m sentry_llm prune --days 14
```

## 配置

默认配置文件为 `sentry-llm/config.example.toml`。复制一份后按需调整：

```bash
cp config.example.toml config.toml
python -m sentry_llm serve --config config.toml
```

关键配置项：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `sample_interval_seconds` | `5` | 采样间隔 |
| `database_path` | `data/sentry_llm.sqlite3` | SQLite 文件路径 |
| `retention_days` | `30` | 指标保留天数，`serve`/`collect` 长期运行时会自动清理旧数据 |
| `http_host` | `127.0.0.1` | 面板监听地址 |
| `http_port` | `8765` | 面板端口 |
| `service_name` | `local-ai-node` | 节点/服务名称 |

## 数据存放和拷贝

默认数据文件是 SQLite：

```text
sentry-llm/data/sentry_llm.sqlite3
```

如果通过 `--db` 或 `database_path` 指定了路径，则以指定路径为准：

```bash
python -m sentry_llm serve --db /data/monitor/sentry_llm.sqlite3
```

可以把数据文件从服务器单独拷贝出来查看。推荐方式：

```bash
scp user@server:/path/to/sentry-llm/data/sentry_llm.sqlite3 ./server-a.sqlite3
```

如果 agent 正在运行，SQLite 可能同时存在 WAL 文件，建议一起拷贝：

```bash
scp user@server:/path/to/sentry-llm/data/sentry_llm.sqlite3* ./snapshot/
```

更稳妥的在线备份方式是在服务器上执行 SQLite backup，得到一个一致性快照：

```bash
sqlite3 /path/to/sentry-llm/data/sentry_llm.sqlite3 ".backup '/tmp/sentry_llm_snapshot.sqlite3'"
scp user@server:/tmp/sentry_llm_snapshot.sqlite3 ./server-a.sqlite3
```

拷贝到本地后，可以用本模块直接查看：

```bash
python -m sentry_llm alerts --db ./server-a.sqlite3 --limit 20
```

也可以临时启动本地面板查看这份离线数据库：

```bash
python -m sentry_llm serve --db ./server-a.sqlite3 --host 127.0.0.1 --port 8765
```

或直接用 SQLite 查询：

```bash
sqlite3 ./server-a.sqlite3 "select datetime(timestamp, 'unixepoch'), service_name, hostname, primary_ip from metric_samples order by timestamp desc limit 5;"
```

## 关键指标说明

| 指标 | 需要留意的条件 | 原因 |
| --- | --- | --- |
| CPU 使用率 | 连续超过 85% | 模型服务的 tokenizer、调度、后处理或数据加载可能成为瓶颈 |
| 系统负载 / CPU 核数 | 超过 1.5 | 运行队列积压，延迟容易抖动 |
| 可用内存率 | 低于 10% | 容易触发 OOM、Swap 或内核回收，推理延迟会显著上升 |
| Swap 使用率 | 超过 20% | 模型服务通常对内存延迟敏感，Swap 会带来明显性能问题 |
| GPU 利用率 | 长时间低于 20% 且 CPU/队列压力高 | GPU 未被充分喂饱，可能是 CPU、IO、batch 或调度问题 |
| GPU 利用率 | 长时间超过 95% | 可能达到算力瓶颈，需要关注排队、延迟和扩容 |
| GPU 显存使用率 | 超过 90% | KV cache、batch 或并发再增长时容易 OOM |
| GPU 温度 | 超过 82 摄氏度 | 可能进入降频区间，影响吞吐稳定性 |
| GPU 功耗使用率 | 超过 95% | 接近功耗墙，容易限制频率 |
| 磁盘使用率 | 超过 85% | 日志、模型缓存、数据集或 SQLite 继续增长可能导致服务不可写 |
| 磁盘 IO 忙碌度估算 | 超过 80% | 模型加载、日志或数据读取可能抢占 IO，影响冷启动和服务稳定性 |
| 网络错误/丢包 | 大于 0 且持续增长 | 请求链路或网卡异常，可能导致超时、重试和吞吐下降 |
| 网络吞吐 | 接近链路上限 | 流式输出、权重拉取或分布式推理可能受到网络限制 |
| 资源压力评分 | 超过 80 | CPU、内存、GPU、磁盘的综合压力偏高，需要定位主瓶颈 |

## 采集字段

所有原始指标和计算指标都以 JSON 存储在 `metric_samples.payload_json` 字段中。常用字段：

- `cpu.percent`
- `service.hostname`
- `service.primary_ip`
- `service.ip_addresses`
- `memory.percent`
- `memory.available_percent`
- `swap.percent`
- `load.load1_per_cpu`
- `disk.usage_percent`
- `disk_io.read_bytes_per_sec`
- `disk_io.write_bytes_per_sec`
- `disk_io.busy_percent_estimate`
- `network.bytes_sent_per_sec`
- `network.bytes_recv_per_sec`
- `network.errors_per_sec`
- `network.drops_per_sec`
- `gpu.devices[].utilization_gpu_percent`
- `gpu.devices[].memory_used_percent`
- `gpu.devices[].temperature_c`
- `gpu.devices[].power_draw_percent`
- `derived.resource_pressure_score`

## 面板能力

Web 面板包含：

- 当前状态卡片：CPU、内存、磁盘、网络、资源压力、GPU 数量
- 最近趋势图：CPU、内存、GPU、显存、磁盘 IO、网络吞吐、压力评分
- 当前告警：按告警类型聚合，按严重程度展示最新触发条件
- 指标说明：列出关键指标、阈值和排查方向

## 说明

- GPU 指标依赖 `nvidia-smi`。没有 NVIDIA GPU 或没有安装驱动时，GPU 区域会显示为不可用，但其他指标仍会采集。
- 面板 HTML 使用 Chart.js CDN。离线环境下接口仍可用，但趋势图可能无法渲染；可以把 Chart.js 下载到内网后修改 `web.py` 中的脚本地址。
- 这是单机轻量监控模块，不替代 Prometheus/Grafana。如果后续需要多节点聚合，可以在当前 SQLite schema 上增加上报端或导出 Prometheus metrics。
