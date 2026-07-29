# Ex6：Scaling 算账

对应指南 §8 与 §9.4。这里不是背一个数字，而是把单位从参数/token 推到 FLOPs、秒、GPU·天与 MFU，并检查量纲和单调关系。

```bash
uv run python exercises/ex06_scaling_math/train.py check
uv run python exercises/ex06_scaling_math/train.py self-test
uv run python exercises/ex06_scaling_math/train.py estimate \
  --parameters 8B --tokens 15T \
  --peak-tflops 312 --utilization 0.45 --num-gpus 8
```

## 你要完成

- `C≈6ND`；
- Chinchilla 的约 20 token/参数；
- 由有效 TFLOP/s 换算 GPU·天；
- 由实测 token/s 反推 MFU。

## 验收标准

- [ ] `self-test` 通过；
- [ ] 给定 `(N,D)` 能输出 FLOPs 与 GPU·天；
- [ ] 与一次真实训练的 token/s 对账到同数量级；
- [ ] 能解释训练算力最优和推理成本最优为何不是同一配方。

自测：卡数翻倍后，“墙钟天数”和“GPU·天”分别应该怎样变化？
