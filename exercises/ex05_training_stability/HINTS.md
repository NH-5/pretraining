# Ex5 分层提示：warmup、累积与 bf16

先读指南 §7.2–§7.4、§10.2。开始前回答：

> gradient accumulation 增大的是 micro batch，还是一次 optimizer update
> 覆盖的 global batch？

本题只验证普通分类器的训练控制流，避免把 Ex1 的序列 loss 混进来。

```bash
uv run python exercises/ex05_training_stability/train.py check
```

## EX05_WARMUP_COSINE

### 函数契约

输入是 optimizer step，不是 micro-batch 次数。输出必须满足：

| 位置 | 期望 |
|---|---|
| `step=0, warmup_steps>0` | 小于 peak，且非负 |
| `step=warmup_steps` | 恰好到 peak |
| warmup 之后 | 单调 cosine decay |
| `step=total_steps` | 恰好为 `peak × min_ratio` |
| 超出 total steps | 保持最小值 |
| `warmup_steps=0, step=0` | 直接从 peak 开始 |

### 先手画曲线

用 `total_steps=20, warmup_steps=4` 标出横轴上的 `0、4、20`，再写出三个点的
lr。只有端点定义清楚后再写分段函数。

### 提示 1：分段

先处理边界，再分两段：

1. warmup：把 step 映射到从 0 附近到 1 的线性比例；
2. decay：把 `[warmup_steps, total_steps]` 映射到进度 `[0,1]`。

### 提示 2：cosine 区间

你需要一个系数在 decay 起点为 1、终点为 0，再把它映射到
`[minimum_lr, peak_lr]`。先把这个系数单独打印 5 个点，别直接塞进长表达式。

### 提示 3：可用工具

标准库 `math.cos` 与 `math.pi` 足够。可以先定义：

```text
minimum_lr = peak_learning_rate × min_lr_ratio
decay_progress = （当前 decay 走了多少）/（decay 总长度）
```

剩下的映射自己补完。对 progress 做边界保护，避免超出 total steps 后 cosine
又反弹。

### 常见错误

- warmup 的分母或端点差 1，峰值出现在错误 step；
- cosine 从 step 0 开始，导致 warmup/decay 不连续；
- 用直线连接 peak/minimum；端点和单调性虽正确，但 25%/75% 位置不符合 cosine；
- 最低降到 0，忽略 `min_lr_ratio`；
- `warmup_steps=0` 除零；
- 超过 total steps 后 lr 再上升；
- 用 micro-step 调 scheduler，累积 K 次后实际 schedule 快了 K 倍。

## EX05_GRAD_ACCUMULATION

### 函数契约

- K 个 micro batch 合成一次等效 global-batch update；
- `zero_grad` 一次、`optimizer.step` 一次；
- 每个 micro batch 都 forward/backward；
- 累积后的梯度只裁剪一次；
- 返回“未缩放的 micro loss 平均值”和“裁剪前的 gradient norm”。

### 先画时间线

```text
清梯度
  → micro 1: forward → 缩放 loss → backward
  → micro 2: forward → 缩放 loss → backward
  → ...
  → micro K: forward → 缩放 loss → backward
  → clip
  → optimizer step
```

在每个箭头上标出：参数是否改变、`.grad` 是覆盖还是累加。

### 提示 1：为什么 loss 要除以 K

默认交叉熵已经对单个 micro batch 求平均。若把 K 个“平均梯度”直接相加，
梯度会比拼接后的 global batch 大 K 倍。反向前缩放 loss，才能在等大 micro
batch 时得到 global-batch 平均梯度。

### 提示 2：循环骨架

先验证 micro batch 列表非空并清梯度。对每批：

1. 搬到 `device`；
2. 进入已有的 `precision_context`；
3. forward + `F.cross_entropy`；
4. 保存未缩放 loss 数值用于报告；
5. 对缩放后的 loss backward。

循环外 clip、step，然后返回两个 Python 数值。

### 提示 3：现成接口

- `optimizer.zero_grad(set_to_none=True)`
- `precision_context(device, precision)`
- `nn.utils.clip_grad_norm_`

`clip_grad_norm_` 的返回值是裁剪前 norm，正好对应脚手架要报告的值。

### 常见错误

- 每个 micro batch 都 zero_grad；
- 每个 micro batch 都 optimizer.step；
- loss 没除以 K；
- 用缩放后的 loss 做最终报告，数值小 K 倍；
- 在每个 micro batch 后分别 clip，改变了合成梯度方向；
- 把 clip 放在 backward 前；
- 忘记让 bf16 只包 forward/loss，而不是改变参数存储逻辑。

## 实验纪律

M1 上 bf16 检查显示 `SKIP` 是正常的；先在 fp32 让累积等价性通过。迁移到 CUDA
后，固定种子、模型、batch、步数和 lr，只把 `--precision fp32` 改成 `bf16`，
再比较 loss 与峰值显存。关闭 warmup 时也只改这一项，见指南 §10.2。
