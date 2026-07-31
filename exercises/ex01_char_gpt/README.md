# Ex1：最小 char-level GPT

对应指南 §3.1、§6、§7 与 §12 阶段 A。

一次前向会并行产生 `[B,T,V]` 的预测，但因果掩码保证位置 `t` 看不到未来；监督信号来自每个位置的下一个字符。生成阶段没有训练时的目标序列，所以只能把刚采样出的 token 接回输入，逐步循环。

卡住时按 TODO ID 查 [HINTS.md](HINTS.md)：先做纸上小样例，再逐级看
不变量、shape/控制流和 API 提示。

## 你要完成的三个核心

1. `EX01_CAUSAL_MASK`：因果可见矩阵；
2. `EX01_TOKEN_LOSS`：逐 token 交叉熵；
3. `EX01_AUTOREGRESSIVE_GENERATION`：自回归生成循环。

每个 TODO 都在 [train.py](train.py) 中带有指南章节、渐进提示与完成标准。其余模型、数据、设备、训练循环和采样入口已经搭好。

## 命令

```bash
uv run python exercises/ex01_char_gpt/train.py check
uv run python exercises/ex01_char_gpt/train.py prepare
uv run python exercises/ex01_char_gpt/train.py train
```

`check` 是公开判题器，可以在每完成一个 TODO 后重复运行：

- `PASS`：该项的 shape、数值语义和关键边界检查通过；
- `PENDING`：仍抛出原始 `NotImplementedError`；
- `FAIL`：实现已执行，但违反某条验收性质，后面会给具体原因。

因此不需要先猜“正确输出长什么样”，也不需要每写一项就询问。`check` 不会给出核心实现，只验证外部行为。
mask 项还会用 T=7 与完整模型的“改变未来 token 不得影响此前 logits”不变量
确认 mask 确实接入 attention。生成项会验证滚动上下文、`block_size` 裁剪、
调用前后的 train/eval 模式，以及高/低 temperature 下采样分布确实发生变化，
避免只凭输出长度误把 greedy argmax 当成完整实现。
脚手架还会检查随机初始化 loss 是否接近 `ln(V)`，防止 tied embedding/head
因初始化尺度过大而在训练前就让 logits 饱和。

本机先使用 M1 MPS + fp32。只有实测不能在十分钟内达到目标，才保持同一配置迁移到云 CUDA；迁移时不要同时改 batch、模型大小和学习率，否则无法归因。

## 验收标准

- [ ] `train loss < 1.5`，记录达到该值的 step 与耗时；
- [ ] 生成至少 300 个字符，人工判断是否具有莎翁式姓名、换行或对话结构；
- [ ] causal mask 的 4×4 可视化没有未来泄漏；
- [ ] 生成结果长度和前缀保持断言通过。

运行前自测：为什么训练时能一次计算 T 个位置，而生成时仍必须逐 token 循环？
