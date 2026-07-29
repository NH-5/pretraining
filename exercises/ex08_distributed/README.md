# Ex8：DDP / FSDP 多卡

对应指南 §9。本练习先手写一次 gradient all-reduce 理解 DP 语义，再使用 DDP，最后用 FSDP 观察参数、梯度和优化器状态分片。

## 本机与云端边界

M1 只能用两个 CPU 进程 + Gloo 做结构 smoke test，不能冒充“2 卡验收”。正式验收必须租用同节点 2+ CUDA GPU；DDP/FSDP 使用 NCCL。

```bash
uv run python exercises/ex08_distributed/train.py check

# TODO 填完后的本机结构测试
uv run torchrun --standalone --nproc-per-node=2 \
  exercises/ex08_distributed/train.py run --strategy manual

# 云端真实 DDP
uv run torchrun --standalone --nproc-per-node=2 \
  exercises/ex08_distributed/train.py run \
  --strategy ddp --peak-tflops <与实际精度匹配的峰值>
```

默认模型只用于先跑通控制流。需要观察显存差异时，仅增大 `--hidden-size`，其余变量保持不变并记录；不要第一次就把模型调到可能 OOM 的规模。

## 你要完成

- manual gradient all-reduce；
- 与单卡 global batch 的一步更新等价性检查；
- FSDP 包装与分片解释；
- 从真实吞吐估计 MFU。

## 验收标准

- [ ] 2+ CUDA 卡运行；
- [ ] DDP 与单卡 global batch 更新数值等价；
- [ ] 每卡参数一致；
- [ ] 报单卡/DDP/FSDP 的 loss、显存、token/s、MFU；
- [ ] 能说出 DP、ZeRO/FSDP、TP、PP 分别切哪一轴。

一次只比较一种 strategy，模型、batch、步数与精度保持不变。
