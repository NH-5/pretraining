# Ex3：换成真实 BPE 分词

对应指南 §5。小词表会拉长序列，大词表会扩大 `V × d` 的 embedding/unembedding；多语言覆盖不好时，中文还可能被拆成多个 UTF-8 字节 token。

## 依赖边界

当前没有提前安装可选依赖。到达本练习后，先说明用途并确认，再执行：

```bash
uv add tokenizers
```

脚手架采用 Hugging Face 当前的 `Tokenizer + BPE + ByteLevel + BpeTrainer` 组合，并保存单个 `tokenizer.json`。

## 命令

```bash
uv run python exercises/ex03_bpe_tokenizer/train.py check

uv run python exercises/ex03_bpe_tokenizer/train.py train \
  --corpus path/to/corpus.txt \
  --vocab-size 8000 \
  --output out/bpe-8k.json

uv run python exercises/ex03_bpe_tokenizer/train.py compare \
  --tokenizer 8k=out/bpe-8k.json \
  --tokenizer 16k=out/bpe-16k.json
```

## 你要完成

- `EX03_BUILD_BPE`：组装、训练和保存 byte-level BPE；
- `EX03_INTERPRET_RATIO`：解释不同词表下中英文的“字/token”差异。

## 验收标准

- [ ] `vocab_size` 至少两档且确实生效；
- [ ] UTF-8 中英文 encode→decode 往返；
- [ ] 输出中英文 chars/token 对照表；
- [ ] 结论同时讨论序列长度和 embedding 参数量。
