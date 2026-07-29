# Ex4：eval、perplexity 与 checkpoint

对应指南 §10.1、§10.3。PPL 是同一分词器下平均交叉熵的指数变换；真正能续训的 checkpoint 必须保存模型、优化器、调度器、step、随机状态，以及能重建模型和 tokenizer 的 metadata，而不只是权重。

```bash
uv run python exercises/ex04_eval_checkpoint/train.py check
uv run python exercises/ex04_eval_checkpoint/train.py demo
```

## 你要完成

- 从 Ex1 复用并再次解释逐 token loss；
- 在 `eval()` 与 `no_grad()` 下平均多个验证 batch；
- 实现 PPL；
- 对称保存/恢复完整训练状态。

## 验收标准

- [ ] 固定验证集重复评估完全一致；
- [ ] `loss=0` 时 `PPL=1`；
- [ ] checkpoint 加载前后验证 loss 完全一致；
- [ ] resume 后的 loss 与参数和“不间断继续训练”一致；
- [ ] resume 后 step 和学习率连续；
- [ ] 能解释 PPL 为什么不能跨分词器比较。

脚本使用无注意力的 tiny model，是为了一次只验证 checkpoint 语义，不把 Ex1 的 mask 问题混进来。
