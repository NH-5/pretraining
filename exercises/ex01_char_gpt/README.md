# Ex1：最小 char-level GPT

对应指南 §3.1、§6、§7 与 §12 阶段 A。

一次前向会并行产生 `[B,T,V]` 的预测，但因果掩码保证位置 `t` 看不到未来；监督信号来自每个位置的下一个字符。生成阶段没有训练时的目标序列，所以只能把刚采样出的 token 接回输入，逐步循环。

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

本机先使用 M1 MPS + fp32。只有实测不能在十分钟内达到目标，才保持同一配置迁移到云 CUDA；迁移时不要同时改 batch、模型大小和学习率，否则无法归因。

## 验收标准

- [ ] `train loss < 1.5`，记录达到该值的 step 与耗时；
- [ ] 生成至少 300 个字符，人工判断是否具有莎翁式姓名、换行或对话结构；
- [ ] causal mask 的 4×4 可视化没有未来泄漏；
- [ ] 生成结果长度和前缀保持断言通过。

运行前自测：为什么训练时能一次计算 T 个位置，而生成时仍必须逐 token 循环？
