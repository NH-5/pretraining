"""The learner's three spoken/written explanations for Ex2."""

# TODO(你)[EX02_EXPLAIN_CAUSAL_MASK]: 见指南 §3.1。
#   用自己的话说明位置 t 能看谁、不能看谁，以及未来泄漏会怎样。
#   完成标准:不看代码，能结合 4-token 例子口述。
CAUSAL_MASK_EXPLANATION = ""

# TODO(你)[EX02_EXPLAIN_TOKEN_LOSS]: 见指南 §6.1、§6.3。
#   说明一次 forward 为什么提供 T 个分类监督，而不是一个句子标签。
#   完成标准:能准确说出 logits/targets 的概念形状。
TOKEN_LOSS_EXPLANATION = ""

# TODO(你)[EX02_EXPLAIN_TRAIN_VS_INFERENCE]: 见指南 §6.3。
#   说明训练为何位置并行、生成为何时间步串行。
#   完成标准:能指出推理时下一个输入来自模型刚才的输出。
TRAIN_VS_INFERENCE_EXPLANATION = ""
