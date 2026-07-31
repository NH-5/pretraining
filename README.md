# pretraining：从零学大模型预训练

这是一个**学习练习仓库**：我在从零学大模型预训练，仓库里放我的代码练习。它不是要交付的产品项目——每个练习都是「脚手架 + TODO 注释 + 验收清单」，核心部分（causal mask、逐 token 交叉熵、自回归生成、warmup、梯度累积、分布式切轴……）留给我亲手填。

配套学习材料是[《大模型预训练入门指南》](./大模型预训练入门指南.md)（下称「指南」），所有概念、章节号、术语、自测题以指南为准。动手前先读对应章节，练习的 TODO 注释里也会标出指南章节号。

## 仓库结构

```
├── 大模型预训练入门指南.md   # 学习材料：预训练五大环节，第 0–14 章
├── AGENTS.md                # 协作规则：教练角色的行事准则
├── 代码审查报告.md            # 判题基础设施的代码审查与修复记录
├── pyproject.toml / uv.lock  # uv 管理依赖（本仓库不作为包发布）
└── exercises/
    ├── checking.py           # 所有练习共享的小型公开判题器
    ├── check_scaffolds.py    # 总体结构检查（目录/入口是否完好）
    ├── README.md             # 练习导航、状态含义、完成纪律
    ├── PROGRESS.md           # 练习进度（由验证结果说了算）
    ├── TODO_INDEX.md         # 学习目标 TODO 索引
    ├── HINTS.md（各练习内）    # 渐进式提示：契约 → 手算样例 → 三档提示
    └── ex00_env_check … ex10_sft   # 十个练习，按顺序完成
```

## 练习路线

| 练习 | 主题 | 指南 |
|---|---|---:|
| Ex0 | PyTorch / CUDA / MPS 环境 | §7 |
| Ex1 | char-level GPT | §3、§6、§7 |
| Ex2 | nanoGPT 内部口述 | §3、§6 |
| Ex3 | BPE 与词表权衡 | §5 |
| Ex4 | eval / PPL / checkpoint | §10 |
| Ex5 | scheduler / accumulation / bf16 | §7.2–7.4 |
| Ex6 | Scaling 算账 | §8、§9.4 |
| Ex7 | 数据过滤与去重 | §4 |
| Ex8 | manual DP / DDP / FSDP | §9 |
| Ex9 | RoPE / SwiGLU / GQA 三选一 | §3.3 |
| Ex10 | 小型 SFT | §6.3、§11 |

练习必须按顺序完成，每个练习都要能在单卡（或 CPU）几分钟到几十分钟内跑出有意义的结果。

## 环境与安装

- Python ≥ 3.13，依赖由 [uv](https://docs.astral.sh/uv/) 管理
- 依赖：`numpy`、`torch`、`torchvision`（见 `pyproject.toml`）
- Ex3 到达后再装 `tokenizers`，Ex7 到达后再装 `datasets tokenizers`——不要提前一次性安装后续框架

```bash
uv sync            # 安装依赖
uv run python exercises/check_scaffolds.py   # 总体结构检查
```

## 怎么用

1. 先读对应指南章节，回答练习 README 的自测题；
2. 在练习目录的 `HINTS.md` 先手算小样例，卡住时每次只多看一档提示；
3. 一次只填一个 TODO，跑一次公开判题器：

```bash
uv run python exercises/ex01_char_gpt/train.py check   # 以当前练习为准
```

`check` 是可重复运行的公开判题器，检查 shape、数值和关键不变量，输出五种状态：

- `PASS`：自动行为测试通过；
- `PENDING`：核心 TODO 尚未实现；
- `FAIL`：代码已运行，但 shape、数值或不变量错误（唯一以非零状态退出的状态）；
- `SKIP`：当前缺少可选依赖、数据或云硬件；
- `MANUAL`：自动结构检查已过，仍需口述或人工抽查。

判题器不能替代训练指标与口述验收——每个练习目录的 README 都列有训练数字和验收项。

## 当前进度

见 [exercises/PROGRESS.md](exercises/PROGRESS.md)：Ex0 ✅ 已完成；Ex1 ⏳ 核心检查 PASS，等待完整训练达到验收指标；Ex2–Ex10 脚手架已备好，等待逐个完成。

## 协作方式

本仓库遵循 [AGENTS.md](AGENTS.md) 中约定的协作规则：教练只搭脚手架、给渐进提示，不替我写答案；任何「完成」都基于实际跑出的数字，不写「应该能跑」。首次提交由 `git init` 起步，每个练习里程碑提交一次。
