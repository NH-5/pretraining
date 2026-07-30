# Ex1 分层提示：最小 char-level GPT

先读指南 §3.1、§6.1、§6.3。开始写代码前，先回答：

> 一条长度为 5 的 token 序列，为什么能同时提供 4 个“预测下一个 token”的监督？

建议按 `causal mask → token loss → generation` 的顺序做。每完成一项就运行：

```bash
uv run python exercises/ex01_char_gpt/train.py check
```

只看当前 TODO 的下一档提示，不要一次看完。

## EX01_CAUSAL_MASK

### 函数契约

- 输入：序列长度 `T` 和目标设备；
- 输出：bool 张量，最简单的形状是 `[T, T]`；
- 行表示正在产生输出的位置，列表示它想读取的输入位置；
- `True` 表示允许看，`False` 表示必须挡住；
- 见指南 §3.1：当前位置可以看自己和过去，不能看未来。

### 先在纸上画 `T=4`

不要先写 PyTorch。补完下面的“可见列”：

| 输出行 | 可见列 |
|---:|---|
| 0 | `0 .. ____` |
| 1 | `0 .. ____` |
| 2 | `0 .. ____` |
| 3 | `0 .. ____` |

再问自己：矩阵坐标 `(row=i, col=j)` 在什么关系下应为 `True`？

### 提示 1：不变量

主对角线必须保留；主对角线上方必须全部屏蔽。先不考虑 batch 和 head，
因为 `[T,T]` 会广播到 attention scores 的 `[B,H,T,T]`。

### 提示 2：张量结构

有两条都可以的路：

1. 生成全 1 方阵，再只保留下三角；
2. 分别生成行、列下标，通过比较得到 bool 方阵。

先在函数末尾临时打印 `mask.to(torch.int)`，确认是 0/1 图，再删除调试打印。

### 提示 3：可用 API

查 `torch.ones`、`torch.tril`，或 `torch.arange` 与广播比较。创建张量时就放到
传入的 `device`，不要先在 CPU 创建再忘记搬运。

### 常见错误

- 上三角和下三角反了；
- 对角线被屏蔽，导致位置 `t` 连当前 token 都不能看；
- 返回 float mask，但调用处按 bool mask 使用；
- mask 在 CPU、attention scores 在 MPS/CUDA；
- 把 `True` 理解成“屏蔽”，与调用处 `masked_fill(~mask, ...)` 的语义相反。

## EX01_TOKEN_LOSS

### 函数契约

- `logits` 是 `[B,T,V]`；
- `targets` 是 `[B,T]`；
- 返回一个可反传的标量；
- 每个 `(batch, position)` 都是一次 V 类分类，见指南 §6.1、§6.3。

### 先手工对齐

原序列为：

```text
a  b  c  d  e
```

模型输入和目标应是：

```text
input : a  b  c  d
target: b  c  d  e
```

填 shape：

```text
logits [B,T,V] -> [____, V]
targets[B,T]   -> [____]
```

两个空必须表示同样多的分类样本。

### 提示 1：不变量

不能只取最后一个位置，也不能先把 T 个 loss 求和后忘记按 token 平均。
公开检查会验证 4 个位置是否都得到梯度。

### 提示 2：张量结构

词表维 `V` 保持为分类维；把 `B` 和 `T` 合成一个“样本维”。同样方式展平
targets，然后交给普通多分类交叉熵。

### 提示 3：可用 API

使用 `torch.nn.functional.cross_entropy`。先确认它期望哪一维是类别，
再决定 `reshape` 后的形状。不要自己先做 softmax；该函数接收原始 logits。

### 常见错误

- 把 logits 变成 `[B,V,T]` 后又按错误维度计算；
- 只用 `logits[:, -1]`；
- 对 logits 先 softmax，造成重复归一化和数值问题；
- `targets` 没有展平到与 `B*T` 对齐；
- 返回 Python float，反向传播链断掉。

## EX01_AUTOREGRESSIVE_GENERATION

### 函数契约

- 返回序列必须保留原 prompt；
- 每轮只追加一个新 token，共追加 `max_new_tokens` 次；
- 模型输入上下文最多为 `block_size`，但最终返回值不能被裁掉；
- 下一个 token 来自最后一个时间位置的词表分布；
- temperature 必须真实改变分布，见指南 §6.3。

### 先追踪两轮

| 轮次 | 完整输出 | 喂给模型的上下文 | 从哪个位置取 logits |
|---:|---|---|---|
| 0 | prompt | prompt 的末尾至多 `block_size` 个 token | 最后位置 |
| 1 | prompt + 新 token 1 | 新完整输出的末尾窗口 | 最后位置 |
| 2 | prompt + 新 token 1 + 新 token 2 | ____ | ____ |

区分两个变量：一个保存“完整输出”，另一个只是“当前模型上下文窗口”。

### 提示 1：控制流

生成循环的次数由 `max_new_tokens` 决定，不由 prompt 长度决定。每轮调用模型
一次，并把刚得到的 token 接回完整序列。

### 提示 2：一轮生成的六步

1. 从完整序列末尾裁出上下文；
2. forward 得到 `[B,T,V]`；
3. 只取最后一个位置；
4. 用 temperature 调整 logits；
5. 转为概率并采样一个 token；
6. 沿 token 维追加到完整序列。

### 提示 3：可用 API

查张量切片、`torch.softmax`、`torch.multinomial` 和 `torch.cat`。使用
`torch.no_grad()`；temperature 必须为正。不要用固定 `argmax` 冒充采样，
公开检查会比较高低 temperature 的分布行为。

### 常见错误

- 每轮把完整输出裁短，最终丢失 prompt；
- 从所有 T 个位置一起采样；
- 沿 batch 维拼接；
- 每轮仍把原 prompt 喂给模型，没有接回新 token；
- `max_new_tokens=0` 时仍生成一个 token；
- 超过 `block_size` 后位置 embedding 越界。

## 如何根据 check 定位

- mask `FAIL`：先打印 4×4 的 0/1 图，只看对角线和上三角；
- loss `FAIL`：打印展平前后 shape，并检查所有 T 个位置的梯度；
- generation `FAIL`：打印每轮“完整长度 / 上下文长度 / 新 token shape”；
- 三项都 `PASS` 后再训练，训练 loss 不降时才检查数据、学习率或模型接线。
