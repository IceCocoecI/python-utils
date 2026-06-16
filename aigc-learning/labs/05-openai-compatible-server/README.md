# Lab 05：OpenAI-Compatible 推理服务

> 目标：把推理逻辑包装成可调用服务，理解 LLM 服务化的接口、指标和故障边界。

---

## 前置知识

- [../../01-python-foundations/03-async-programming.md](../../01-python-foundations/03-async-programming.md)
- [../../07-inference-and-deployment/00-inference-and-deployment-theory.md](../../07-inference-and-deployment/00-inference-and-deployment-theory.md)
- [../../07-inference-and-deployment/01-llm-inference-engines.md](../../07-inference-and-deployment/01-llm-inference-engines.md)
- [../../07-inference-and-deployment/03-serving-frameworks.md](../../07-inference-and-deployment/03-serving-frameworks.md)
- [../../07-inference-and-deployment/04-demo-and-frontend.md](../../07-inference-and-deployment/04-demo-and-frontend.md)

---

## 运行脚本

基础 smoke test：

```bash
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/kv_cache_and_batching.py
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/smoke_test.py
```

服务端脚本：

```bash
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/openai_compatible_toy_server.py
```

如果脚本启动的是本地 HTTP 服务，另开终端用 `curl` 或 OpenAI SDK 请求。

---

## 任务

1. 阅读 OpenAI-compatible 请求和响应字段。
2. 启动 toy server，完成一次 chat completion 请求。
3. 给服务增加或记录以下信息：
   - request id
   - prompt token 数或近似长度
   - latency
   - response token 数或近似长度
4. 模拟并发请求，观察延迟变化。
5. 写出服务的错误处理策略：参数错误、超时、模型不可用、输出为空。
6. 设计一个健康检查接口或说明已有接口。

---

## 验收标准

- [ ] 能说清楚 `/v1/chat/completions` 的核心字段。
- [ ] 能解释流式输出和非流式输出的差异。
- [ ] 能解释 TTFT、TPOT、吞吐和尾延迟。
- [ ] 能说明 toy server 与 vLLM/SGLang 生产服务的差距。
- [ ] 能写出最小部署 checklist。

---

## 延伸挑战

- 接入一个 Gradio 或 Streamlit demo。
- 用 async client 做简单压测。
- 把 toy backend 替换为真实 OpenAI-compatible vLLM 服务。

