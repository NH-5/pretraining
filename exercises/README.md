# 预训练练习导航

练习必须按顺序完成。每个 `train.py check` 都是可重复运行的公开行为判题器；
它能检查 shape、数值和关键不变量，但不能替代每个目录 README 中的训练指标与口述验收。

## 卡住时怎样使用提示

Ex1–Ex10 每个目录现在都有一份 `HINTS.md`。它沿用数据结构练习的写法，
不是只说一句“想想某概念”，而是把每个核心 TODO 拆成：

1. **函数契约**：输入、输出和不能破坏的不变量；
2. **纸上小样例**：先手算或画 shape，不急着写代码；
3. **提示 1**：只指出判断方向；
4. **提示 2**：给控制流或张量变形骨架；
5. **提示 3**：列出可能用到的 PyTorch/Python API，但仍不放完整实现；
6. **常见错误与 check 对照**：根据 `FAIL` 信息定位哪条不变量坏了。

推荐顺序是：先直接做 → 跑一次 `check` → 只看当前 TODO 的下一档提示 →
只改这一个 TODO → 再跑 `check`。不要一口气读完所有提示，也不要同时改两项。
当 `check` 出现 `PENDING`、`FAIL` 或 `MANUAL` 时，末尾会打印当前练习的
`HINTS.md` 路径。

| 练习 | 主题 | 指南 | 初始核心 TODO |
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

总体检查只负责目录/入口是否完好；学习过程中应运行当前练习自己的
`train.py check`，由该练习的公开判题器输出以下五种状态。你不需要事先猜
完整标准输出；只需处理自己当前目标对应的状态和说明。

统一状态含义：

- `PASS`：自动行为测试通过；
- `PENDING`：核心 TODO 尚未实现；
- `FAIL`：代码已运行，但 shape、数值或不变量错误；
- `SKIP`：当前缺少可选依赖、数据或云硬件；
- `MANUAL`：自动结构检查已过，仍需口述或人工抽查。

只有出现 `FAIL` 时命令才以非零状态退出；`PENDING / SKIP / MANUAL`
都是学习过程中的正常状态。

当前交付状态还可以额外验证所有预期 TODO 都保留着：

```bash
uv run python exercises/check_scaffolds.py --expect-unfinished
```

`--expect-unfinished` 只用于验收最初交付的空白脚手架。一旦开始填写 TODO，
不要再把它当作学习进度判题器。

## 依赖何时加入

- Ex0–Ex2、Ex4–Ex6、Ex8–Ex10：当前 `torch` 与标准库足够搭脚手架；
- Ex3：到达后，经确认 `uv add tokenizers`；
- Ex7：到达后，经确认 `uv add datasets tokenizers`。

不要提前一次性安装后续框架；每次只为当前练习引入真实需要的依赖。

## 完成纪律

1. 先读对应指南章节并回答 README 自测；
2. 在 `HINTS.md` 先手算小样例，卡住时每次只多看一档；
3. 一次只填一个 TODO 或改一个实验变量；
4. 先运行目录内的 `check`，再运行训练/实验命令；
5. 把真实数字写入 `notes.md`；
6. 达到 README 验收标准后，才把 [PROGRESS.md](PROGRESS.md) 的 `⏳` 改成 `✅`。
