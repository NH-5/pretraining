# Ex0 学习记录

## 2026-07-29：环境基线

执行命令：

```bash
uv add torch
uv run python exercises/ex00_env_check/train.py
```

实测结果：

```text
Python: 3.13.12
PyTorch: 2.13.0
CUDA available: False
MPS built: True
MPS available: True
Selected device: mps
Recommended precision: fp32
Apple GPU model: Apple M1
Apple GPU memory: 16 GB unified memory (shared with CPU; no dedicated CUDA VRAM)
Tensor smoke test: PASS (result sum=54.0)
```

结论：Ex0 的硬件与张量计算验收均通过。本机可以承担 Ex0，并应先实测 Ex1；只有本机无法满足 Ex1 的十分钟验收，或进入 Ex8 多卡练习时，才有明确理由租用云显卡。

非阻塞观察：当前环境尚未加入 NumPy，PyTorch 首次执行张量操作时会提示可选的 NumPy 集成不可用；本练习没有调用 NumPy，因此不影响上述结果。遵守“只加当前练习真实需要的依赖”，本次没有顺手安装它。

附加验证：

- `uv lock --check`：通过，锁文件成功解析 30 个包；
- `uv run python -m compileall -q exercises`：通过；
- MPS 张量测试：退出码 0，结果和为 `54.0`。
