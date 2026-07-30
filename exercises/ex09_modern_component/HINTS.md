# Ex9 分层提示：RoPE / SwiGLU / GQA

先读指南 §3.3。本题的实验原则是“只替换一个组件”。先完成两个复用项，再从
三个组件中只选一个：

- SwiGLU：控制流最短，适合先练组件替换；
- RoPE：重点是偶/奇维配对与广播 shape；
- GQA：重点是 query heads 与 KV heads 的分组关系。

未选的两项保留 `PENDING` 是正常状态。

## EX09_REUSE_CAUSAL_ATTENTION

### 函数契约

```text
query/key/value: [B,H,T,D]
返回:            [B,H,T,D]
```

必须保持 Ex1 的因果语义。未来位置的 hidden 改变时，更早位置的输出不能改变。

### 提示 1：shape 路线

```text
Q @ K^T → scores [B,H,T,T]
scores × scale
屏蔽未来列
softmax（沿 key 位置）
dropout
weights @ V → [B,H,T,D]
```

先逐行写出每次矩阵乘法最后两维如何变化。

### 提示 2

mask 只需要 `[T,T]`，广播到 B/H。scale 使用 head dimension D，而不是
embedding 总维度。调用处已经提供 `training`，dropout 行为要尊重它。

### 提示 3

可以复用 Ex1 的显式 scores/mask 路径；不要在“复用题”里同时改成另一套 mask
约定。完成后先只跑 causal-attention check，再碰现代组件。

## EX09_REUSE_TOKEN_LOSS

直接迁移 Ex1 已通过检查的 `[B,T,V] + [B,T] → scalar` 实现。baseline 与
component 必须调用同一个函数，避免 loss 差异来自评估口径。

## EX09_ROPE

### 先理解一对二维分量

RoPE 把 head dimension 的相邻两维看成二维向量。对角度 `θ` 的旋转满足：

```text
x_even' = x_even × cos(θ) - x_odd × sin(θ)
x_odd'  = x_even × sin(θ) + x_odd × cos(θ)
```

先验证两个性质：

- position 0 时 `θ=0`，向量不变；
- 旋转前后这一对分量的平方和不变。

### shape 工作表

```text
Q/K                 [B,H,T,D]
pair index          [D/2]
position            [T]
angles              [T,D/2]
广播 cos/sin 到     [1,1,T,D/2]
```

### 提示 1

先拒绝奇数 `head_dim`。把 even/odd 分量分别切出来，旋转后再按原顺序交错合并，
输出 shape/dtype/device 必须与输入一致。

### 提示 2

不同 pair 使用按 `base` 几何变化的频率，不同 position 再乘该频率形成 angle。
先单独打印 `T=3,D=4` 的 angle shape，不要直接在四维张量里猜广播。

### 提示 3：可用 API

`torch.arange`、幂运算、`torch.cos/sin`、切片 `0::2/1::2`、`torch.stack`
或交错赋值都可。创建频率时要继承 Q/K 的 device，并注意计算 dtype。

### 常见错误

- 把相邻 pair 写成前半维/后半维配对；
- Q 旋转了、K 没旋转或角度不同；
- position 维与 head 维广播反了；
- 旋转后把 even 全放前半、odd 全放后半；
- 在 RoPE variant 中仍叠加绝对 position embedding（外层已替你处理）。

## EX09_SWIGLU

### 先画两支

```text
hidden ─→ gate projection ─→ SiLU ─┐
                                   × ─→ down projection ─→ dropout
hidden ─→ up projection ───────────┘
```

### 提示 1

两个上投影输出 shape 完全相同，逐元素相乘后才进入 down projection。不是拼接，
也不是两个分支相加。

### 提示 2

`__init__` 已创建三组线性层并做近似参数量匹配。forward 只负责按图连接，不要
重新创建层，否则参数不会被 optimizer 持久跟踪。

### 提示 3

使用 `torch.nn.functional.silu` 或等价 module。公开 check 会 backward 并确认
三组权重都有梯度，所以不能绕过其中任一分支。

### 常见错误

- gate 忘记 SiLU；
- gate 与 up 相加；
- down projection 输入用了原 hidden；
- forward 内临时创建 Linear；
- 输出忘记 dropout 或 shape 没回到 embedding size。

## EX09_GQA

### 先算头数

若 `Hq=8, Hkv=2`：

```text
每组 query heads = Hq / Hkv = ____
每个 KV head 被 ____ 个 query heads 共享
```

### shape 工作表

```text
Q projection → [B,T,Hq,D]  → [B,Hq,T,D]
K projection → [B,T,Hkv,D] → [B,Hkv,T,D]
V projection → [B,T,Hkv,D] → [B,Hkv,T,D]
```

attention 前要让每个 query head 能配到所属的 K/V head；可以显式扩展头，也可
保留 group 维计算。先在纸上标出 query head `0..Hq-1` 分别使用哪个 KV head。

### 提示 1

group size 是 `Hq // Hkv`。同组 query 共享 K/V，但每个 query projection 和
输出仍保留自己的 head。

### 提示 2

完成 head 对齐后，attention 流程仍与 causal MHA 相同：scaled scores、future
mask、softmax、dropout、乘 V。最后合并 Hq 与 D 回 `[B,T,C]`。

### 提示 3

先用 `Hkv=Hq` 验证它退化为普通 MHA 的 shape，再用 `Hq=4,Hkv=2`。无论使用
repeat、expand 还是 group reshape，都检查 K/V 的复制只发生在 head 语义上，
没有动 T 维。

### 常见错误

- `embedding_size` 用 Hkv 分 head_dim，而不是 Hq；
- K/V reshape 成 Hq 但元素对应关系错位；
- 合并输出时用了 Hkv×D，丢掉 query heads；
- repeat 时重复了 sequence 维；
- 只通过 shape，未来 token 仍泄漏。

## 公平对照

所选组件的 check 通过后才运行 `compare`。固定 seed、数据、steps、optimizer、
batch 和模型宽度；同时看最终 loss、末 20 步平均和参数量。一次差异不等于普遍
结论，诚实写入 `notes.md`。
