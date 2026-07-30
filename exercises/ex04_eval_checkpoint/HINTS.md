# Ex4 分层提示：eval、PPL 与 checkpoint

先读指南 §10.1、§10.3。建议按依赖顺序完成：

```text
复用 Ex1 token loss
        ↓
固定评估平均 loss → PPL
        ↓
保存 checkpoint ↔ 恢复 checkpoint
```

每填一项就运行：

```bash
uv run python exercises/ex04_eval_checkpoint/train.py check
```

## EX04_REUSE_TOKEN_LOSS

这不是一道新算法题。把 Ex1 已经通过行为检查的实现迁移过来，并确认：

- 输入仍是 `[B,T,V]` 与 `[B,T]`；
- 返回仍是可反传标量；
- 4 个 token 位置都产生梯度；
- 不要顺手增加 pad 逻辑，本练习的 tiny batch 没有 pad。

若这里失败，回看 Ex1 的 HINTS 中 `EX01_TOKEN_LOSS`，不要继续写 checkpoint。

## EX04_EVAL_AVERAGE

### 函数契约

- 对 `num_batches` 个固定验证 batch 求平均 loss；
- 不更新参数、不构建反向图；
- 评估期间关闭 dropout 等训练行为；
- 调用结束后恢复模型进入函数前的 train/eval 模式；
- 返回 Python `float`。

### 先填状态表

| 进入函数时 | 函数内 | 返回后 |
|---|---|---|
| `model.training=True` | ____ | ____ |
| `model.training=False` | ____ | ____ |

### 提示 1：模式不变量

先保存进入函数时的 `model.training`。评估必须临时 `eval()`，但不能无条件在最后
`train()`，否则会破坏本来就在 eval 模式的调用者。

### 提示 2：循环骨架

验证 `num_batches > 0` 后，循环索引传给 `batch_factory`，forward，调用复用的
token loss，把标量值收集起来；循环结束求平均；最后恢复原模式。

### 提示 3：已有保护

函数已经有 `@torch.no_grad()`，所以无需在每轮再嵌套反向传播逻辑。把 tensor
标量转换成 Python 数值时使用不会保留计算图的方式。

### 常见错误

- 忘记 `eval()`，dropout 让重复评估波动；
- 评估后永远调用 `train()`；
- 用训练 batch 的随机全局状态，导致固定评估不固定；
- 在循环中覆盖 loss，最后只返回最后一个 batch；
- `num_batches=0` 时除零。

## EX04_PERPLEXITY

指南 §10.1 已给出关系。先手算两个锚点：

```text
average loss = 0      → PPL = ____
average loss = ln(2)  → PPL = ____
```

### 提示 1

这是对平均交叉熵做指数变换，不是 `1/loss`，也不是再做一次 softmax。

### 提示 2

使用标准库 `math` 的指数函数即可，返回 Python float。思考极大 loss 是否会
溢出；本练习公开检查只用有限的小值。

### 口述验收

同一句文本换 tokenizer 后 token 边界和每 token 难度都变了，因此 PPL 只能在
同分词器、同数据处理设置下比较。

## EX04_SAVE_CHECKPOINT / EX04_LOAD_CHECKPOINT

### 先做“行李清单”

| 状态 | 为什么续训需要 | 保存键 | load 后交给谁 |
|---|---|---|---|
| model state | 参数位置 | ____ | model |
| optimizer state | Adam 的动量/方差 | ____ | optimizer |
| scheduler state | 当前 lr 进度 | ____ | scheduler |
| step | 数据与日志进度 | ____ | 返回调用者 |
| validation loss | 保存点指标 | ____ | 返回调用者 |
| CPU RNG state | 后续随机序列连续 | ____ | torch RNG |
| metadata | 重建 config/tokenizer | ____ | 返回调用者 |

字段名可自行决定，但 save 与 load 必须完全对称。

### 提示 1：保存

构造一个只含“状态字典、数字、普通 metadata 和 RNG tensor”的 payload。确保父
目录存在，然后一次性序列化。不要把整个 Python model 对象当作主要格式。

### 提示 2：恢复

先读 payload，再依次把三个 state dict 装回已经由调用者构造好的对象；恢复 RNG；
最后返回函数签名要求的三项。不要新建并偷偷替换局部 model。

### 提示 3：可用 API

查看这些现成接口的输入输出：

- `model.state_dict()` / `model.load_state_dict(...)`
- optimizer 与 scheduler 的同名 state-dict 接口
- `torch.get_rng_state()` / `torch.set_rng_state(...)`
- `torch.save(...)` / `torch.load(...)`

### 用固定实验验证，而不是只看文件存在

```text
训练若干步
  ├─ 分支 A：不间断再训练 K 步
  └─ 分支 B：保存 → 扰乱状态 → 加载 → 再训练 K 步

比较：每一步 loss、参数、lr、下一段随机数
```

### 常见错误

- 只存 model 权重；
- scheduler 没恢复，resume 后 lr 跳变；
- optimizer 对象恢复了，但参数组仍不是当前 model 的参数；
- 保存的是“当前随机数样本”，不是 RNG 状态；
- metadata 只有模型名，无法重建维度与词表；
- load 返回保存前一步而训练循环又重复执行该步。
