# 模块 03：数据处理与科学计算

> 深度学习代码的 50% 都在搬运数据、变换张量形状、处理图像。
> 本模块掌握 **NumPy + einops + 图像处理** 这套工具箱，你的代码能变短一半、可读性翻倍。

---

## 为什么这一章重要？

AIGC 项目中最"藏 bug"的地方往往不是模型，而是：
- Tensor 形状搞错
- 通道顺序混乱（CHW vs HWC，RGB vs BGR）
- 图像 dtype 处理错误（uint8 vs float32, [0,255] vs [0,1]）
- Batch 维度挤压

学会本章的工具，这些 bug 基本会绝迹。

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [data-and-scientific-computing-theory](./00-data-and-scientific-computing-theory.md) | shape / dtype / layout / range / distribution：数据管线的底层约束 |
| 01 | [numpy-essentials](./01-numpy-essentials.md) | ndarray / 广播 / 向量化 / 内存视图 |
| 02 | [einops-tutorial](./02-einops-tutorial.md) | rearrange / reduce / repeat / Layers |
| 03 | [image-processing](./03-image-processing.md) | Pillow / OpenCV / torchvision.transforms / **Albumentations** / **质量评估指标** |
| 04 | [data-formats-and-pipelines](./04-data-formats-and-pipelines.md) | JSONL / Parquet / WebDataset / safetensors / HF datasets |

---

## 示例代码（`examples/`）

| 文件 | 说明 |
|---|---|
| `numpy_basics.py` | 数组创建、切片、广播、向量化技巧 |
| `einops_demo.py` | rearrange / reduce / repeat 在 CV/NLP 任务中的实战 |
| `image_ops.py` | PIL / OpenCV / torchvision v2 / Albumentations 图像处理闭环 |
| `data_formats_pipeline.py` | JSONL / Parquet / safetensors / HDF5 / HF datasets / tar shard 本地示例 |

### 在当前 `aigc` 环境运行

```bash
cd aigc-learning/03-data-and-scientific-computing/examples

conda run -n aigc python numpy_basics.py
conda run -n aigc python einops_demo.py
conda run -n aigc python image_ops.py
conda run -n aigc python data_formats_pipeline.py
```

`image_ops.py` 和 `data_formats_pipeline.py` 的输出会写到 `examples/outputs/`，便于检查中间结果，也避免污染仓库根目录。

---

## 理论与实践怎么组织

本模块建议按三层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 理论层 | shape、dtype、layout、range、distribution 为什么是数据管线的核心不变量？ | `00-data-and-scientific-computing-theory.md` |
| 工具层 | NumPy、einops、Pillow/OpenCV/torchvision、数据格式分别解决什么问题？ | `01` ~ `04` 文档 |
| 模板层 | 如何把每个知识点落成可运行、可验证的小脚本？ | `examples/` |

学习顺序建议：

1. 先读 `00`，建立数据管线的整体模型。
2. 跑 `numpy_basics.py`，理解广播、向量化、view/copy。
3. 跑 `einops_demo.py`，把复杂 shape 变换改成具名维度表达。
4. 跑 `image_ops.py`，重点检查 RGB/BGR、CHW/HWC、dtype/range。
5. 跑 `data_formats_pipeline.py`，理解格式选择和访问模式的关系。

---

## 推荐配套资源

- [NumPy 官方文档](https://numpy.org/doc/stable/)
- [Einops 官方教程](https://einops.rocks/)
- [Einops 代码更好写指南](http://einops.rocks/pytorch-examples.html)
- [Pillow 文档](https://pillow.readthedocs.io/)
- [OpenCV-Python 教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [torchvision.transforms 文档](https://pytorch.org/vision/stable/transforms.html)

---

## 自检清单

- [ ] 解释 NumPy 广播规则。
- [ ] `reshape` vs `transpose` vs `view` 的区别？
- [ ] 用 einops 一行代码把 `(B, C, H, W)` 转成 `(B, H*W, C)`。
- [ ] 用 einops 实现 MultiHead Attention 的 QKV 拆分。
- [ ] 解释 PIL / OpenCV / PyTorch 三者通道顺序的区别。
- [ ] 图像归一化时，ImageNet 的 mean / std 是多少？
- [ ] 什么时候该用 Albumentations 而不是 torchvision？
- [ ] 评估一个文生图模型应该用哪些指标（FID、CLIPScore）？
- [ ] 为什么 LLM 语料推荐 JSONL 而不是 JSON？
- [ ] Parquet 比 CSV 快在哪里？为什么适合大表？
- [ ] 为什么模型权重应该用 safetensors 而不是 `torch.save`？
- [ ] 跑通 `examples/` 下四个入口脚本，并能解释每个输出文件的作用。
