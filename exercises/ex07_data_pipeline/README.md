# Ex7：数据 pipeline（过滤与去重）

对应指南 §4。脚本严格按“规范化 → 质量过滤 → exact dedup → 可选 fuzzy dedup → token 计数 → 人工抽查”记录每一步，不能只看最终文件大小。

## 依赖边界

到达本练习后，经确认再安装：

```bash
uv add datasets tokenizers
```

`prepare_data.py` 使用 Hugging Face 当前的 `load_dataset(..., streaming=True)`，数据集、config、split 都从命令行传入，不把可能变化的 FineWeb config 名写死。

## 命令

```bash
uv run python exercises/ex07_data_pipeline/train.py check

uv run python exercises/ex07_data_pipeline/prepare_data.py \
  --dataset HuggingFaceFW/fineweb --config <实际选择的配置> \
  --limit 1000 --output data/fineweb_slice.jsonl

uv run python exercises/ex07_data_pipeline/train.py run \
  --input data/fineweb_slice.jsonl \
  --output out/fineweb_clean.jsonl \
  --audit out/audit_5.jsonl \
  --tokenizer out/bpe.json

# 启用 fuzzy dedup 时必须同时保留被删除文本对
uv run python exercises/ex07_data_pipeline/train.py run \
  --input data/fineweb_slice.jsonl \
  --output out/fineweb_clean_fuzzy.jsonl \
  --audit out/audit_5_fuzzy.jsonl \
  --tokenizer out/bpe.json \
  --fuzzy --fuzzy-audit out/fuzzy_removed_pairs.jsonl
```

`check` 会自动验收规范化、保序 exact dedup；质量规则运行后仍标
`MANUAL`，要求你抽查误杀。未做可选 fuzzy dedup 时显示 `SKIP`，不会阻塞
必做部分。

## 你要完成

- 一条可解释、可单独归因的质量过滤规则；
- 保序 exact dedup；
- fuzzy dedup 是拓展 TODO，启用前必须先定义阈值和误杀审计。

## 验收标准

- [ ] 报处理前后 document 与 token 数；
- [ ] 报 exact/fuzzy 各自移除数；
- [ ] 人工抽查输出 5 条；
- [ ] 对被 fuzzy dedup 的至少 3 对文本做人工核验；
- [ ] 不下载 1GB 以上数据来做第一次调试，先用小切片跑通。
