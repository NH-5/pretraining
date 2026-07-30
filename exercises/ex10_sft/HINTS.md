# Ex10 分层提示：小型 SFT

先读指南 §6.3、§11。建议分成两个阶段：

1. 不传 checkpoint，先完成 template、response label mask、response-only loss；
2. Ex4 checkpoint 真实可恢复后，再完成 base adapter 和 SFT checkpoint。

```bash
uv run python exercises/ex10_sft/train.py check
```

没有 checkpoint 时后两项 `SKIP` 是正常状态。

## EX10_FORMAT_INSTRUCTION

### 函数契约

- 返回 `(prompt_text, response_text)` 两个非空字符串；
- prompt 必须包含原 instruction；
- response 必须包含原参考回答；
- prompt 末尾应明确模型接下来进入 assistant response；
- 训练与推理必须调用同一函数。

### 先在纸上定模板

任选一种简单且固定的格式，例如用稳定的 section label。先填：

```text
prompt:
  [固定前缀] __________________
  [instruction] _______________
  [回答开始标记] ______________

response:
  [reference answer] ___________
```

不要让参考回答泄漏进 prompt；也不要在推理时手写另一套模板。

### 提示 1

这个函数只负责文本边界，不负责 tokenize、pad 或 loss mask。保持它纯粹，便于
训练和推理共用。

### 提示 2

公开 check 只验证结构并给 `MANUAL`；你还要手工打印一个训练 prompt 和一个推理
prompt，逐字符确认回答开始前缀完全一致。

## EX10_RESPONSE_LOSS_MASK

这是最容易 off-by-one 的部分。见指南 §6.3：labels 已经相对 inputs 右移。

### 先画完整对齐

假设：

```text
prompt ids  : p0 p1 p2 p3
response ids: r0 r1 eos
all_ids     : p0 p1 p2 p3 r0 r1 eos
```

脚手架已经生成：

| position | input_ids | next_tokens | 这个 target 属于 prompt 还是 response？ | label |
|---:|---|---|---|---|
| 0 | p0 | p1 | ____ | ____ |
| 1 | p1 | p2 | ____ | ____ |
| 2 | p2 | p3 | ____ | ____ |
| 3 | p3 | r0 | ____ | ____ |
| 4 | r0 | r1 | ____ | ____ |
| 5 | r1 | eos | ____ | ____ |

只对 next_tokens 中仍属于 prompt 的目标写 `IGNORE_INDEX`；response 与 eos 保留
原 token id。

### 提示 1

不要直接屏蔽 `prompt_token_count` 个 label。先数一数：右移后
`next_tokens` 里还剩几个 prompt token？

### 提示 2

返回一个新 list，长度必须与 `next_tokens` 一致。可以先复制 token，再对前面
一段做替换；不要修改调用者的输入 list。

### 提示 3：边界样例

分别手算：

- prompt 只有 1 token；
- prompt 4 token、response 1 token + eos；
- prompt 较长但 next_tokens 总长不变。

公开 check 的 `prompt_token_count=1` 样例专门抓“无条件至少 mask 一个”的错误。

## EX10_RESPONSE_ONLY_LOSS

### 函数契约

- logits `[B,T,V]`、labels `[B,T]`；
- 返回可反传标量；
- `IGNORE_INDEX` 位置既不进入平均，也不产生 logits 梯度；
- 其他每个 response/eos 位置都产生梯度。

### 提示 1

复用 Ex1 展平 `B×T` 的结构；区别只是交叉熵需要知道哪个 label 值代表“忽略”。

### 提示 2

使用交叉熵自身的 `ignore_index` 参数，不要先把被忽略位置的 logits 置零。置零
仍可能让这些位置进入 loss 平均。

### 提示 3

先用全零 logits 手算：若 V=4，所有受监督位置的平均 loss 是 `ln(4)`，与被
mask 的 prompt 有几个位置无关。

### 常见错误

- labels 没展平；
- mask logits 而不是 labels；
- 把 pad_id 当 ignore index，但 collate 实际用 `-100`；
- 所有 labels 都被 mask，产生 NaN；
- 对受监督 token 求和而不平均。

## EX10_LOAD_BASE_MODEL

这是一道集成题，不要在 Ex1/Ex4 未通过时硬接。

### 返回契约

`BaseArtifacts` 必须同时提供：

1. 已加载到目标 device 的 `nn.Module`；
2. 有 `pad_id/eos_id/encode/decode` 的 tokenizer adapter；
3. 复用 Ex1 自回归逻辑的 `generate_text(prompt, max_new_tokens)`；
4. 足以追溯原 checkpoint/config/vocab 的 metadata（供保存使用）。

### 提示 1：先检查 checkpoint 行李

打开 Ex4 checkpoint，只打印顶层 keys、config keys 和 vocab 长度，不打印大
tensor。若 metadata 不能重建 Ex1 config 与字符表，应先修 Ex4 保存格式。

### 提示 2：special token 扩词表

原 char vocab 没有 pad/eos。需要规划两个新 ID，并让以下三处一致：

```text
tokenizer vocab size
model token embedding 行数
output head 行数
```

复制旧行，初始化新行，再恢复 tied embedding/head 的共享关系。旧 token ID
不能重排，否则 checkpoint 权重语义全变。

### 提示 3：adapter 分层

先单独实现并测 tokenizer adapter：

```text
decode(encode(old-vocab text)) == text
pad_id/eos_id 在扩展词表范围内且不同
```

再构造 model，最后把已通过 Ex1 check 的 generation 包成 closure。每层单测通过
后再组装 `BaseArtifacts`。

### 常见错误

- 只扩 embedding，输出 head 仍是旧 V；
- 扩展后忘了重新 weight tying；
- 新 special ID 与旧字符 ID 冲突；
- generate closure 捕获了 CPU model，但 artifacts.model 在 CUDA；
- 重新初始化整个 model 后只加载部分层却忽略 missing keys；
- 训练/推理模板没有共同经过 `format_instruction`。

## EX10_SAVE_SFT

复用 Ex4 的 checkpoint 原则，但 metadata 还应包含：

| 内容 | 用途 |
|---|---|
| SFT model state | 恢复微调后参数 |
| optimizer state | 若要继续 SFT |
| epoch | 训练进度 |
| 扩展后的 tokenizer/vocab | 保持 token ID |
| pad/eos ID | 重建 batching/generation |
| base checkpoint 来源 | 追溯底模 |
| model config/template version | 新进程重建 |

### 提示 1

先把“新进程加载时需要什么”倒推成 payload，不要以“文件能创建”作为完成标准。

### 提示 2

保存后启动一个全新 Python 进程，加载同一条 eval instruction，用固定 seed 比较
保存前后的生成。只有一致才证明 model/tokenizer/template 都恢复正确。

## 最终人工对照

对同一条 eval instruction 保存：

```text
base before SFT:
after SFT before save:
after reload:
```

第二、三项应一致；与第一项的差异只能说明 tiny 数据让行为发生了变化，不能声称
模型获得了通用指令遵循能力。
