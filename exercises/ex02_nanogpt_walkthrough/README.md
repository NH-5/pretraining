# Ex2：通读并讲解 nanoGPT 内部

对应指南 §3 与 §6。本练习不是再写一遍 Transformer，而是把 Ex1 的代码路径压缩成三条可口述的不变量。

```bash
uv run python exercises/ex02_nanogpt_walkthrough/train.py check
uv run python exercises/ex02_nanogpt_walkthrough/train.py trace
```

在 [answers.py](answers.py) 用自己的话填写三个 TODO，再运行：

```bash
uv run python exercises/ex02_nanogpt_walkthrough/train.py verify
```

## 验收标准

- [ ] 不看代码口述 causal mask；
- [ ] 不看代码口述逐 token loss；
- [ ] 不看代码口述训练并行、推理自回归；
- [ ] 能用 `To be or not to be` 的 trace 指出每个位置的可见前缀和目标。

这里的脚本只能检查答案是否存在；“我能否口述”必须由你亲自判断，不能由自动化测试冒充。
