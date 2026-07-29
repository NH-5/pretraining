# 预训练练习导航

练习必须按顺序完成。`check` 只证明脚手架的非 TODO 部分可运行，不能替代每个目录 README 中的训练指标与口述验收。

| 练习 | 主题 | 指南 | 当前核心 TODO |
|---|---|---|---:|
| Ex0 | PyTorch / CUDA / MPS 环境 | §7 | 0 |
| Ex1 | char-level GPT | §3、§6、§7 | 3 |
| Ex2 | nanoGPT 内部口述 | §3、§6 | 3 |
| Ex3 | BPE 与词表权衡 | §5 | 2 |
| Ex4 | eval / PPL / checkpoint | §10 | 5 |
| Ex5 | scheduler / accumulation / bf16 | §7.2–7.4 | 2 |
| Ex6 | Scaling 算账 | §8、§9.4 | 4 |
| Ex7 | 数据过滤与去重 | §4 | 3 |
| Ex8 | manual DP / DDP / FSDP | §9 | 4 |
| Ex9 | RoPE / SwiGLU / GQA 三选一 | §3.3 | 2 个复用 + 三选一 |
| Ex10 | 小型 SFT | §6.3、§11 | 5 |

## 总体结构检查

```bash
uv run python exercises/check_scaffolds.py
```

当前交付状态还可以额外验证所有预期 TODO 都保留着：

```bash
uv run python exercises/check_scaffolds.py --expect-unfinished
```

## 依赖何时加入

- Ex0–Ex2、Ex4–Ex6、Ex8–Ex10：当前 `torch` 与标准库足够搭脚手架；
- Ex3：到达后，经确认 `uv add tokenizers`；
- Ex7：到达后，经确认 `uv add datasets tokenizers`。

不要提前一次性安装后续框架；每次只为当前练习引入真实需要的依赖。

## 完成纪律

1. 先读对应指南章节并回答 README 自测；
2. 一次只填一个 TODO 或改一个实验变量；
3. 先运行目录内的 `check`，再运行训练/实验命令；
4. 把真实数字写入 `notes.md`；
5. 达到 README 验收标准后，才把 [PROGRESS.md](PROGRESS.md) 的 `⏳` 改成 `✅`。
