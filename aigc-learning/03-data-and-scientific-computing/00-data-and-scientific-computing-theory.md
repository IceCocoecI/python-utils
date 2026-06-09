# 00 · 数据处理与科学计算理论框架

> 目标：先建立数据与张量的底层模型，再学习 NumPy、einops、图像库和数据格式。
> 工具 API 会变，但 shape、dtype、layout、I/O 和统计分布这些约束不会变。

---

## 1. 本模块真正解决什么问题？

AIGC 工程里的数据处理不是“把文件读进来”这么简单。它要保证：

- 样本语义没有被预处理破坏。
- 张量形状符合模型契约。
- dtype 和数值范围符合训练或推理约定。
- 数据加载速度跟得上 GPU 消耗速度。
- 数据版本、切分、过滤和增强策略可追溯。

可以把数据管线抽象成：

```
raw data -> decode -> transform -> batch -> device -> model
```

每一段都可能引入 bug。尤其是图像和多模态任务，一次错误的通道转换或归一化足以让模型学到完全错误的分布。

---

## 2. 五个基础不变量

### 2.1 Shape：张量契约

shape 是模型和数据之间最直接的接口。比如视觉模型常见：

| 表示 | 含义 | 常见位置 |
|---|---|---|
| `HWC` | height, width, channel | NumPy / OpenCV / matplotlib |
| `CHW` | channel, height, width | PyTorch 单张图 |
| `BCHW` | batch, channel, height, width | PyTorch 训练 batch |
| `BTD` | batch, token, hidden | Transformer |
| `BHTD` | batch, head, token, head_dim | Attention |

工程原则：复杂 shape 变换必须让维度名字出现在代码里。`einops` 的价值就在这里，它把位置索引变成了具名维度。

### 2.2 Dtype：数值语义

同一个数组，dtype 不同，含义也不同：

| dtype | 典型语义 | 风险 |
|---|---|---|
| `uint8` | 图像像素 `[0, 255]` | 直接送模型会错 |
| `float32` | 训练输入、统计计算 | 内存更大 |
| `float16` / `bfloat16` | 加速训练或推理 | 精度和溢出风险 |
| `int64` | 类别标签、token id | 不适合数值归一化 |

图像任务最常见的错误是把 `uint8 [0,255]` 当成 `float [0,1]`，或者把已经归一化的张量再次归一化。

### 2.3 Layout：内存布局

shape 看起来一样，不代表内存访问效率一样。`transpose` 往往产生非连续视图，`reshape` 有时只是改元数据，有时会触发复制。

需要特别关注：

- 切片通常是 view，fancy indexing 通常是 copy。
- PyTorch 里的 `permute` 只是改 stride，后续 `.view()` 可能失败。
- GPU kernel 更喜欢连续或规则 stride 的布局。
- 图像库和深度学习框架的通道顺序不同，会影响复制和转换成本。

### 2.4 Range：数值范围

模型通常假设输入分布稳定：

| 场景 | 常见范围 |
|---|---|
| PIL / OpenCV 原图 | `uint8 [0, 255]` |
| torchvision `ToDtype(..., scale=True)` | `float32 [0, 1]` |
| ImageNet Normalize 后 | 近似零均值单位方差 |
| Stable Diffusion VAE 输入 | `[-1, 1]` |
| mask / label | 离散整数，不应做图像归一化 |

调试数据时优先打印：

```python
print(x.shape, x.dtype, x.min(), x.max())
```

这比只看一张图片可靠，因为很多问题肉眼不一定马上看出来。

### 2.5 Distribution：统计分布

训练数据不仅是样本集合，也是一个分布。预处理会改变分布：

- resize/crop 改变尺度和构图偏置。
- filter 改变类别、长度、质量分布。
- tokenizer 改变文本长度分布。
- augmentation 改变数据增强空间。
- shuffling 和 sampling 改变 batch 内相关性。

当线上表现和验证集不一致时，很多问题来自数据分布，而不是模型结构。

---

## 3. 广播和向量化的理论意义

NumPy 广播不是“省几行代码”，而是在表达一类批量数学：

```
(B, C, H, W) - (1, C, 1, 1) -> (B, C, H, W)
```

这表示对每个 batch、每个空间位置应用同一组通道统计量。理解广播后，归一化、mask、attention bias、位置编码都可以写成清晰的张量表达。

向量化的意义有两层：

- 性能：把 Python 循环交给 C/CUDA kernel。
- 正确性：把逐元素逻辑表达为整体张量变换，减少索引错误。

经验法则：如果你在样本、像素、token 维度写 Python for 循环，先问自己能不能改成矩阵运算、广播、`einsum` 或 `einops`。

---

## 4. 图像预处理的核心链路

图像从磁盘到模型通常经历：

```
bytes -> decode -> RGB image -> resize/crop/augment -> float tensor -> normalize
```

每一步都有默认假设：

| 步骤 | 关键问题 |
|---|---|
| decode | PIL 是 RGB；OpenCV `imread` 是 BGR |
| resize | 下采样要考虑 antialias |
| crop | 训练和评估策略必须分开 |
| augment | 图像、mask、bbox、keypoint 要同步 |
| tensorize | HWC 到 CHW，uint8 到 float |
| normalize | ImageNet 统计量和 Diffusion VAE 统计量不同 |

分类任务可以用 ImageNet mean/std；扩散模型的 VAE 通常用 `mean=std=0.5` 把 `[0,1]` 映射到 `[-1,1]`。这两个约定不能混用。

---

## 5. 数据格式的本质是访问模式

格式选择不是偏好问题，而是访问模式问题：

| 访问模式 | 合适格式 | 原因 |
|---|---|---|
| 逐条追加、逐行流式读文本 | JSONL | 行边界天然可切分 |
| 只读部分列、需要压缩表格 | Parquet | 列式存储和统计信息 |
| 保存模型权重 | safetensors | 纯数据、可 mmap、安全 |
| 随机切片大数组 | HDF5 / Zarr | 数组块和索引 |
| 顺序扫描海量图文对 | WebDataset tar shard | 减少小文件 I/O |
| Hub 分发和流式训练 | HF datasets | Arrow + streaming 生态 |

错误格式会把问题推给训练代码。例如用 CSV 存大规模特征，训练时会反复做 dtype 推断；用大量小图片文件训练扩散模型，瓶颈会变成文件系统随机 I/O。

---

## 6. 数据管线的性能模型

训练 step 可以粗略拆成：

```
step_time = data_time + host_to_device_time + compute_time + sync_time
```

优化目标不是让数据加载“尽可能快”，而是让它和计算重叠，避免 GPU 等数据。

常用手段的本质：

| 手段 | 解决的问题 |
|---|---|
| `num_workers` | 并行 decode / transform |
| `pin_memory` | 加速 CPU 到 GPU 拷贝 |
| `persistent_workers` | 避免每个 epoch 重启 worker |
| `prefetch_factor` | 提前准备 batch |
| 离线预处理 | 避免训练时重复 CPU 密集操作 |
| shard 顺序读取 | 减少随机 I/O 和小文件开销 |

如果 GPU 利用率低，先测 `data_time` 和 `compute_time`，不要盲目改模型。

---

## 7. AIGC 场景里的数据风险

### 7.1 LLM 文本

- JSONL schema 不统一会导致训练模板错位。
- tokenizer 版本变化会改变长度和截断比例。
- 去重和过滤会显著改变 benchmark 表现。
- 长度分桶影响吞吐，也可能影响样本顺序。

### 7.2 图像生成

- RGB/BGR 混淆会制造颜色偏移。
- resize/crop 策略会影响构图分布。
- VAE 输入范围错会导致重建和训练都异常。
- 质量过滤会改变美学分布和主题分布。

### 7.3 多模态

- 图文配对路径错，比单模态更难发现。
- image transform 和文本 prompt 模板必须同时版本化。
- shard 级 shuffle 和 sample 级 shuffle 都会影响训练稳定性。

---

## 8. 本模块文档如何对应理论

| 理论问题 | 对应章节 | 对应示例 |
|---|---|---|
| ndarray、广播、视图和向量化如何工作？ | `01-numpy-essentials.md` | `examples/numpy_basics.py` |
| 复杂 shape 变换如何写得可读？ | `02-einops-tutorial.md` | `examples/einops_demo.py` |
| 图像数据如何避免通道、dtype、range 错误？ | `03-image-processing.md` | `examples/image_ops.py` |
| 不同规模数据应该用什么格式？ | `04-data-formats-and-pipelines.md` | `examples/data_formats_pipeline.py` |

---

## 9. 工程判断清单

写任何数据处理代码前，先确认：

- [ ] 输入和输出 shape 是否写进了代码或测试？
- [ ] dtype 和 range 是否在转换边界显式处理？
- [ ] 复杂维度变换是否能用 `einops` 表达？
- [ ] 图像库之间是否做了 RGB/BGR 转换？
- [ ] mask、label、bbox 是否避免了错误归一化？
- [ ] 随机增强是否可通过 seed 复现？
- [ ] 大文件是否流式读取，而不是一次性读入内存？
- [ ] 数据格式是否匹配访问模式？
- [ ] 输出样本是否保存了少量可视化用于人工检查？

数据处理的质量标准不是“代码能跑”，而是模型看到的每一个 batch 都符合你以为的语义。
