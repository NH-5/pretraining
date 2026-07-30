# Ex5：训练稳定性组件

对应指南 §7.2–§7.4 与 §10.2。本练习一次只比较一个变量：学习率曲线、是否 warmup、精度或累积步数，不能同时改。

```bash
uv run python exercises/ex05_training_stability/train.py check
uv run python exercises/ex05_training_stability/train.py run --precision fp32
```

`check` 会把 scheduler 和累积更新分别判定，并把累积结果与等价 global
batch 更新比较。本机 M1 上 bf16 项固定显示 `SKIP`，这是硬件边界，不是
代码失败。

在支持 bf16 的云 CUDA 上，再单独运行：

```bash
uv run python exercises/ex05_training_stability/train.py run \
  --precision bf16 --output out/bf16.json
```

## 你要完成

- `EX05_WARMUP_COSINE`：warmup 后 cosine decay；
- `EX05_GRAD_ACCUMULATION`：多次 micro batch 只做一次 optimizer step，并正确缩放 loss、裁剪梯度。

M1 路径固定 fp32；早期不使用 fp16。内存对比只在 CUDA 上报告 `max_memory_allocated`，避免拿 macOS 统一内存和 CUDA 独立显存做伪对照。

## 验收标准

- [ ] 累积更新与等价 global batch 更新数值接近；
- [ ] lr 在 warmup/cosine 分界连续且非负；
- [ ] 同种子 bf16 与 fp32 loss 同量级；
- [ ] CUDA 上 bf16 峰值显存明显低于 fp32；
- [ ] 关闭 warmup 时只改这一项，并记录真实曲线，不能预写“必然发散”。
