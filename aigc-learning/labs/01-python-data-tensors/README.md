# Lab 01：Python、数据与张量基础

> 目标：把 Python 工程习惯、NumPy/einops shape 变换、PyTorch Tensor 基础串成一条最小数据管线。

---

## 前置知识

- [../../01-python-foundations/01-modern-python-basics.md](../../01-python-foundations/01-modern-python-basics.md)
- [../../01-python-foundations/02-advanced-features.md](../../01-python-foundations/02-advanced-features.md)
- [../../01-python-foundations/04-type-hints.md](../../01-python-foundations/04-type-hints.md)
- [../../03-data-and-scientific-computing/01-numpy-essentials.md](../../03-data-and-scientific-computing/01-numpy-essentials.md)
- [../../03-data-and-scientific-computing/02-einops-tutorial.md](../../03-data-and-scientific-computing/02-einops-tutorial.md)
- [../../02-deep-learning-libraries/01-pytorch-fundamentals.md](../../02-deep-learning-libraries/01-pytorch-fundamentals.md)

---

## 运行脚本

```bash
conda run -n aigc python aigc-learning/01-python-foundations/examples/decorators_demo.py
conda run -n aigc python aigc-learning/01-python-foundations/examples/generators_demo.py
conda run -n aigc python aigc-learning/01-python-foundations/examples/type_hints_demo.py
conda run -n aigc python aigc-learning/03-data-and-scientific-computing/examples/numpy_basics.py
conda run -n aigc python aigc-learning/03-data-and-scientific-computing/examples/einops_demo.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/pytorch_basics.py
```

---

## 任务

1. 用 `dataclass` 定义一个数据处理配置，至少包含 batch size、image size、dtype、seed。
2. 用 generator 模拟一批样本流，每个样本包含 `id`、`text`、`image` 三类字段。
3. 用 NumPy 创建一批伪图像数据，完成归一化、裁剪或 reshape。
4. 用 einops 完成以下变换：
   - `(B, H, W, C) -> (B, C, H, W)`
   - `(B, C, H, W) -> (B, H*W, C)`
   - 模拟 ViT patch flatten。
5. 转成 PyTorch Tensor，检查 shape、dtype、device。
6. 写一个简短日志输出每一步的数据不变量。

---

## 验收标准

- [ ] 代码里没有裸 `print` 调试，关键输出使用 logging。
- [ ] 每一步都能说清楚 shape、dtype、range。
- [ ] 能解释 view/copy、reshape/transpose、CHW/HWC 的差异。
- [ ] 能用类型注解描述核心函数输入输出。
- [ ] 运行脚本不依赖外部下载。

---

## 延伸挑战

- 把数据管线封装成一个可迭代 Dataset 风格对象。
- 增加 Pydantic 配置校验。
- 用 `pytest` 给 shape 变换写 3 个测试。

