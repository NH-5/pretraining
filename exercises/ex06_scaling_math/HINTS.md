# Ex6 分层提示：Scaling 算账

先读指南 §8.1、§8.2、§9.4。本题最重要的不是背结果，而是让每一步单位都能
相消。开始前先写：

```text
N: 参数个数（无量纲）
D: token 个数（无量纲）
peak: TFLOP / s / GPU
utilization: 0..1
```

每填一个公式就运行：

```bash
uv run python exercises/ex06_scaling_math/train.py check
```

## EX06_TRAINING_FLOPS

指南 §8.1 已给出 dense Transformer 的近似：

```text
C ≈ 6 × N × D
```

### 提示 1

`1B` 传入函数后已经是 `1e9`，不要在公式内再次乘十亿。函数返回的是 FLOPs
总量，不是 TFLOPs，也不是 FLOP/s。

### 提示 2

先手算数量级：

```text
N = 1e9
D = 20e9
6ND = 6 × 1 × 20 × 10^(____ + ____) = ____ × 10^____
```

### 常见错误

- 把 `B/T` 字符串解析逻辑重复写进公式函数；
- 少乘训练近似中的常数 6；
- 返回 TFLOPs，提前除以 `1e12`；
- 使用整数截断或返回格式化字符串。

## EX06_CHINCHILLA_TOKENS

### 提示 1

指南 §8.2 的 compute-optimal 经验值是每个参数约多少 token？把这个比例乘 N，
结果仍是 token 个数。

### 手算锚点

```text
70B parameters × ____ token/parameter = ____T tokens
```

不要把这个经验值误当成所有工程目标都必须遵守；指南 §8.3 的 overtraining
讨论的是不同目标函数下的选择。

## EX06_GPU_DAYS

函数名沿用练习主题，但它返回的是给定卡数下的**墙钟天数**。命令行报告中的
`GPU-days` 会再乘 `num_gpus`。

### 先走单位阶梯

```text
peak_tflops_per_gpu
  × 10^12                         → FLOP / s / GPU
  × utilization                  → 有效 FLOP / s / GPU
  × num_gpus                     → 集群有效 FLOP / s

total_flops ÷ 集群有效 FLOP/s    → seconds
seconds ÷ ____                   → days
```

### 提示 1：先验证输入

总 FLOPs、峰值、utilization、卡数都应为正；utilization 还应不超过 1。先把非法
输入挡住，避免返回看似正常的负时间或除零。

### 提示 2：单调性自检

保持其他量不变：

- 卡数翻倍，墙钟天数应减半；
- utilization 翻倍，墙钟天数应减半；
- 总 FLOPs 翻倍，墙钟天数应翻倍；
- 卡数翻倍时，理想情况下 GPU-days 近似不变。

若这些关系不成立，先查乘除方向，不要调常数蒙混 check。

### 常见错误

- 忘记 `TFLOP/s → FLOP/s` 的 `1e12`；
- 一天按 3600 秒而不是 `24×3600`；
- utilization 放到分子；
- 返回 GPU-days 而脚手架随后又乘卡数。

## EX06_MFU

MFU 是“模型实际需要的 FLOP/s”除以“硬件理论峰值 FLOP/s”，见指南 §9.4。

### 先走单位阶梯

```text
每 token 近似 FLOPs: 6 × N
  × measured_tokens_per_second   → achieved FLOP / s

peak_tflops_per_gpu
  × 10^12
  × num_gpus                     → cluster peak FLOP / s

MFU = achieved / peak            → 无量纲比例
```

### 提示 1

这里的 token/s 参数已经是全局吞吐，不要再乘 `num_gpus`；卡数只出现在集群峰值
一侧。

### 提示 2

先算公开锚点的数量级：

```text
N=1B, global throughput=100k token/s
achieved = 6 × 10^9 × 10^5 = ____ FLOP/s
peak = 1000 × 10^12 = ____ FLOP/s
ratio = ____
```

### 提示 3：边界

返回前检查结果是否有限且位于 `[0,1]`。若超过 1，不要静默 clamp；应主动报错，
因为通常是 token/s 口径、精度峰值或公式输入不匹配。

## 最后做一张对账表

| 量 | 理论估计 | 实测 | 单位 | 来源 |
|---|---:|---:|---|---|
| tokens/s | ____ | ____ | token/s | 训练日志 |
| peak | ____ | — | TFLOP/s/GPU | 与实际精度匹配的卡规格 |
| MFU | ____ | — | % | 本题公式 |
| wall-clock | ____ | ____ | day/s | 估计与计时 |

只对账到同数量级是验收目标；通信、数据加载和非 matmul 操作会让简单模型公式产生
偏差。
