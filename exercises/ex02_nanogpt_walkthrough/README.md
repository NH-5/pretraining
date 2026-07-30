# Ex2：通读并讲解 nanoGPT 内部

对应指南 §3 与 §6。本练习不是再写一遍 Transformer，而是把 Ex1 的代码路径压缩成三条可口述的不变量。

不知道该怎样组织口述时，按三张口述卡使用 [HINTS.md](HINTS.md)，不要照抄；
最终仍要关掉文件脱稿说明。

```bash
uv run python exercises/ex02_nanogpt_walkthrough/train.py check
uv run python exercises/ex02_nanogpt_walkthrough/train.py trace
```

`check` 会逐项显示三份答案的状态：空白时是 `PENDING`；填写后是
`MANUAL`，提醒你脱稿口述。trace 的前缀/目标对齐由脚本自动判为
`PASS` 或 `FAIL`。

在 [answers.py](answers.py) 用自己的话填写三个 TODO，再运行：

```bash
uv run python exercises/ex02_nanogpt_walkthrough/train.py verify
```

## 验收标准

- [ ] 不看代码口述 causal mask；
- [ ] 不看代码口述逐 token loss；
- [ ] 不看代码口述训练并行、推理自回归；
- [ ] 能用 `To be or not to be` 的 trace 指出每个位置的可见前缀和目标。

脚本不会用关键词冒充语义评分；“我能否口述”必须由你亲自判断。
