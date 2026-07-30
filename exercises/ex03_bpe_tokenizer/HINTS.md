# Ex3 分层提示：byte-level BPE

先读指南 §5.2、§5.3。开始前回答：

> 词表变大时，为什么序列通常变短，但 embedding/unembedding 参数会变多？

只有到达本练习并确认后才执行 `uv add tokenizers`。未安装时 `check` 的
`SKIP` 是正常状态。

## 推荐顺序

1. 用很小的双语文本训练一个 300 词表 tokenizer；
2. 通过 UTF-8 往返检查；
3. 再训练两档更有意义的词表；
4. 最后写 chars/token 结论。

每次只做一步：

```bash
uv run python exercises/ex03_bpe_tokenizer/train.py check
```

## EX03_BUILD_BPE

### 函数契约

- 输入：一个或多个语料文件、目标 `vocab_size`、保存路径；
- 输出：实际保存的 `Path`；
- 至少注册 `<bos>`、`<eos>`、`<pad>`；
- byte-level 路径必须能表示任意 UTF-8 字节；
- encode→decode 必须保留空格、换行、中文和中英混排。

### 先画组件图

```text
文本文件
  ↓
ByteLevel pre-tokenizer
  ↓
BPE model + BpeTrainer 学习词表/merge
  ↓
ByteLevel decoder
  ↓
tokenizer.json
```

在代码旁先列出四个对象分别负责什么，不要把 trainer 和 tokenizer 混为一谈。

### 提示 1：组装顺序

先创建“装 BPE 模型的 tokenizer”，再设置预分词器和解码器，之后创建带
`vocab_size` 与 special tokens 的 trainer，最后用语料文件训练并保存。

### 提示 2：对象归属

需要的概念对象分别位于 `tokenizers` 的 model、pre-tokenizer、decoder、
trainer 模块。`require_tokenizers()` 已返回顶层模块，脚手架也已处理依赖缺失。

### 提示 3：可用 API 名

沿着这些名字查当前已安装版本的对象即可：

- `Tokenizer`
- `models.BPE`
- `pre_tokenizers.ByteLevel`
- `decoders.ByteLevel`
- `trainers.BpeTrainer`
- tokenizer 的文件训练与保存方法

先让输出目录存在，再保存。函数返回值应是调用者传入的输出路径，而不是 tokenizer
对象。

### 最小调试语料

先用十几行重复的小语料，不要一上来下载大数据：

```text
A model predicts the next token.
A tokenizer changes sequence length.
语言模型预测下一个词元。
分词器会改变序列长度。
```

手动往返至少测：

```text
Hello, BPE!
中文与 English 混排。
spaces  and
newlines	must survive
```

### 常见错误

- 只设 ByteLevel pre-tokenizer，忘了匹配 decoder；
- special tokens 只是写进文本，没有注册进 trainer；
- 硬编码词表大小，没有使用函数参数；
- `vocab_size < 256 + special-token 数`；
- 保存前未创建父目录；
- decode 时默认跳过了本应保留的普通文本；
- 只测英文，中文或换行往返才暴露问题。

## EX03_INTERPRET_RATIO

### 先读表，不急着写结论

每种语言分别填：

| 语言 | 小词表 tokens | 大词表 tokens | chars/token 变化 | 序列变长/短 |
|---|---:|---:|---|---|
| English | ____ | ____ | ____ | ____ |
| 中文 | ____ | ____ | ____ | ____ |

注意项目报告的是 `characters / tokens`，不是它的倒数。该值越大，通常表示同一
文本用了更少 token。

### 提示 1

先只陈述观察到的数值，不急着解释。中英文分开写，不能假定两者改善比例相同。

### 提示 2

解释必须同时连接两条成本链，见指南 §5.3：

```text
词表 V 改变 → token 数/序列长度改变 → attention/训练计算改变
词表 V 改变 → V × d 改变 → embedding/unembedding 参数改变
```

### 提示 3：结论骨架

可以按下面四句组织，但数值和方向由你的表决定：

1. 从词表 A 到 B，英文 chars/token 从 ____ 到 ____；
2. 中文从 ____ 到 ____，与英文的差异是 ____；
3. 序列成本因此 ____；
4. 代价是 `V×d` 参数从 ____ 变为 ____。

`MANUAL` 表示脚本确认你写了非空结论，但这四点仍要自己口述检查。
