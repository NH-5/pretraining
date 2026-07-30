# Ex8 分层提示：manual DP、DDP 与 FSDP

先读指南 §9.1、§9.2、§9.4。先记住“切哪一轴”：

| 方法 | 每个 rank 吃什么 | 每卡是否有完整模型 | 核心通信 |
|---|---|---|---|
| manual DP / DDP | 不同 batch 分片 | 是 | 梯度 all-reduce |
| FSDP | 不同 batch 分片 | 否，状态分片 | all-gather / reduce-scatter |
| TP | 同一矩阵的行/列分片 | 否 | 高频张量通信 |
| PP | 不同层 | 否 | stage 间激活 |

M1 的两个 Gloo 进程只用于控制流 smoke test，不能写成“2 GPU 已验收”。

## 推荐顺序

1. 两进程 manual gradient average；
2. DDP 与单卡 global batch 一步等价；
3. 复用 Ex6 MFU；
4. 到 2+ CUDA 环境再做 FSDP 和显存对比。

## EX08_ALL_REDUCE

### 先手算两个 rank

```text
rank 0 本地梯度: g0
rank 1 本地梯度: g1
SUM all-reduce 后每卡: ______
再除 world_size 后每卡: ______
```

DDP 的 global-batch 语义需要的是各 rank 本地**平均梯度的平均值**。

### 函数契约

- 遍历 model 参数；
- 没有 gradient 的参数跳过；
- 对每个 gradient 做 in-place SUM collective；
- 再在本地缩放为 mean；
- 所有 rank 必须以相同顺序调用相同数量的 collective。

### 提示 1

all-reduce 和 reduce 不同：完成后每个 rank 都有聚合结果。不要只在 rank 0 调用，
否则其他进程会一直等待。

### 提示 2

参数的 `.grad` 可能是 `None`。对非空梯度调用 collective，操作完成后再除以
`world_size`。不要创建一个新 tensor 后忘记写回 `.grad`。

### 提示 3：可用 API

查看 `torch.distributed.all_reduce` 和 SUM reduce operation。公开 check 会给
rank 0 一个梯度、rank 1 另一个梯度，并验证两边最后都得到均值。

### 常见错误

- 只 SUM 不除卡数，等效学习率放大；
- 在 optimizer.step 后才同步；
- rank 0 调用、其他 rank 跳过；
- 某个 rank 因条件分支少调一次 collective，造成死锁；
- 用参数值 all-reduce，而不是 gradient；
- 对 `grad is None` 直接访问张量方法。

## EX08_SINGLE_CARD_EQUIVALENCE

### 先画数据对应关系

对每个 step：

```text
rank 0 batch ─┐
              ├─ 按 batch 维拼成 single-card global batch
rank 1 batch ─┘
```

分布式路径与 reference 必须共享：

- 完全相同的初始 state；
- 相同 optimizer 与 lr；
- 相同步数；
- 每一步完全相同的一组样本；
- 相同的“平均 loss”口径。

### 提示 1：只让 rank 0 算 reference

`make_local_batch` 已由 `rank + step + seed` 确定数据。rank 0 可以为每个 rank
重建同一步的 batch，再拼成 global batch。其他 rank 不需要重复算 reference。

### 提示 2：控制流骨架

rank 0：

1. 从 `initial_state` 构造 reference model；
2. 对每个 step 重建所有 rank 的 batch 并拼接；
3. 跑一次 global-batch update；
4. 取分布式 model 的最终参数；
5. 求所有参数的最大绝对差。

最后把这个标量 broadcast 给所有 rank，使所有进程从函数返回同一结果。

### 提示 3：DDP 包装边界

传入模型可能是 DDP wrapper。比较 state 时要明确取 wrapper 内部 module，
并把用于比较的 tensor 放到同一设备。FSDP 的完整 state dict 另有 collective
语义，不要把 DDP 的直接解包假设无条件套上去。

### 常见错误

- reference 每步只用 rank 0 的 batch；
- 把不同 rank 的 mean loss 相加而不平均；
- reference 初始权重重新随机生成；
- rank 0 提前 return，其他 rank 仍在 broadcast/barrier；
- 用最终 loss 接近冒充参数更新等价；
- 比较前一个模型在 CPU、另一个在 CUDA。

## EX08_FSDP_WRAP

### 概念先行

见指南 §9.2：FSDP 是 ZeRO-3 风格，参数、梯度、优化器状态分片。它仍保持数据
并行语义，不是把矩阵行列切开的 TP。

### 提示 1

本函数只负责把已经放到本 rank device 的 model 包成
`FullyShardedDataParallel` 并返回。先做最小整模型包装，不要第一次就同时引入
复杂 auto-wrap、mixed precision 和 activation checkpointing。

### 提示 2

CUDA 每个进程应对应自己的 `local_rank`/device。确认 process group 已初始化，
再导入并构造 FSDP wrapper。M1/Gloo 下 check 会 `SKIP`，不要为绕过硬件限制写
假实现。

### 云端验收顺序

1. tiny model forward/backward；
2. 同一配置 DDP；
3. 同一配置 FSDP；
4. 只增大 `hidden_size` 观察显存；
5. 报每卡 peak memory、global token/s 与 loss。

## EX08_MFU

直接复用 Ex6 的单位链：

```text
achieved FLOP/s = 6 × N × global token/s
cluster peak = peak TFLOP/s/GPU × 10^12 × world_size
MFU = achieved / cluster peak
```

`global_tokens_per_second` 已含所有 rank，不要再次乘 world size。峰值必须与实际
训练精度匹配；不能用 fp8 宣传峰值解释 fp32 smoke test。

## 两进程本机检查

```bash
uv run torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29500 \
  exercises/ex08_distributed/train.py check
```

若命令挂住，优先检查各 rank 是否走过完全相同的 collective 序列，而不是先怀疑
模型计算。
