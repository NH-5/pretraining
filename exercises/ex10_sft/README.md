# Ex10（拓展）：小 SFT 把底模变助手

对应指南 §11。预训练决定底模能力，SFT 用固定的“指令 → 回答”格式把能力导向可用行为；它不是重新注入基础知识。

## 前置条件

先完成 Ex1 的模型/生成与 Ex4 的可恢复 checkpoint。`model_adapter.py` 故意不替你猜 checkpoint 结构，而要求复用自己已经验证的版本。

样例先使用 tiny Shakespeare 字符表能够覆盖的英文字符，避免把“未知中文字符”误诊成 SFT 训练问题。若后续已有 Ex3 的多语言 tokenizer 和与之匹配的预训练模型，再替换成中文指令。

Ex1 的原始字符词表没有 `<pad>/<eos>`。接线时必须显式增加 special-token ID 并扩展 tied embedding/head；checkpoint metadata 需要记录扩展后的词表。

```bash
uv run python exercises/ex10_sft/train.py check
uv run python exercises/ex10_sft/train.py run \
  --checkpoint path/to/base.pt \
  --output out/sft.pt
```

## 你要完成

- 固定训练/推理共用的 instruction template；
- 正确推导右移后哪些 label 属于 prompt，并设为 `IGNORE_INDEX`；
- 接入 Ex1/Ex4 的底模、tokenizer 和 generate；
- 只对 response token 计算 loss；
- 保存可复现的 SFT checkpoint。

## 验收标准

- [ ] prompt/pad token 不贡献 loss；
- [ ] 同一模板用于训练与推理；
- [ ] 给出至少一条相同指令的 SFT 前/后生成；
- [ ] 后模型能遵循一条简单指令；
- [ ] 诚实说明 tiny 数据只证明行为可塑，不证明通用对齐能力。

自测：如果 prompt token 也计入 loss，模型会额外被训练成什么行为？
