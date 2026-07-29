# Ex0：确认 PyTorch 与加速设备

## 目标

先建立后续练习都能复用的环境基线：

- 打印 Python 与 PyTorch 版本；
- 检查 CUDA 和 Apple MPS 是否可用；
- 报告加速器型号与内存类型；
- 在实际选中的设备上完成一次张量运算。

对应《大模型预训练入门指南》§7、§7.4 和 §12 阶段 A。§7.4 强调精度与硬件能力相关：本仓库在 CUDA 上优先使用 bf16；没有 CUDA 时先使用 fp32，不为了省显存过早引入 fp16 的 loss scaling。

## 本机应如何理解结果

Apple M1 不提供 NVIDIA CUDA，因此 `CUDA available: False` 是正确结果，并不等于不能使用 GPU。PyTorch 通过 MPS 使用 Apple GPU；M1 的 CPU 与 GPU 共享统一内存，所以这里报告 `16 GB unified memory`，不能把它写成 16 GB 独立显存。

## 运行

```bash
uv sync
uv run python exercises/ex00_env_check/train.py
```

在当前 M1 MacBook Air 上，同步完成后的单次检查约需数秒，不需要云显卡。

## 验收清单

- [x] 打印 PyTorch 版本；
- [x] 打印 `torch.cuda.is_available()`；
- [x] 打印 MPS 的构建与可用状态；
- [x] 打印实际加速器型号及内存语义；
- [x] 在选中的设备上跑通确定性的矩阵乘法。

实测数字见 [notes.md](notes.md)。

## 动手前自测

1. 为什么 M1 上 `CUDA available: False` 与 `MPS available: True` 可以同时成立？
2. 为什么 16 GB 统一内存不能理解为“模型可以独占 16 GB 显存”？

回答清楚这两点后，再进入 Ex1 的 char-level GPT。
