# Ex9（拓展）：RoPE / SwiGLU / GQA 三选一

对应指南 §3.3。先复用 Ex1 已验证的 causal attention 与逐 token loss，再只替换一个现代组件，其他模型、数据、步数、随机种子和优化器保持不变。

```bash
uv run python exercises/ex09_modern_component/train.py check
uv run python exercises/ex09_modern_component/train.py compare --component rope
```

## 必做与选做

必做：

- `EX09_REUSE_CAUSAL_ATTENTION`
- `EX09_REUSE_TOKEN_LOSS`

三选一：

- `EX09_ROPE`
- `EX09_SWIGLU`
- `EX09_GQA`

另外两项可以保留 TODO。check 模式会展示参数量，避免把“参数更多”误当成组件本身更好。

## 验收标准

- [ ] 组件的 shape、梯度和边界单测通过；
- [ ] baseline 与组件用同样步数和数据；
- [ ] 组件 loss 不退步，或诚实记录退步与可能原因；
- [ ] 能解释所选组件解决的是位置、FFN 表达还是 KV cache 成本。
