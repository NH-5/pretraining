# Ex1 学习记录

状态：⏳ 三个核心行为检查全部通过；20-step MPS 冒烟训练通过，
尚未达到 `train loss < 1.5` 的完整验收标准。

## 实验记录

| 日期 | 设备 | 唯一改动 | step | train loss | 用时 | 生成观察 |
|---|---|---|---:|---:|---:|---|
| 2026-07-30 | Apple M1 MPS | 修复非核心脚手架的 GPT 小尺度初始化 | 1→20 | 4.2070→3.3297 | 9.4 s | 不再坍缩为单字符；20 步样本仍是乱码 |

## 我的解释

- causal mask 为什么必要：待填写
- 为什么是逐 token loss：待填写
- 训练并行、推理自回归的差异：待填写

## 踩坑

- 2026-07-30：生成循环把不断增长的完整 prompt 送入模型；测试模型在长度超过
  `block_size=4` 时真实报错：
  `generate must crop its context to model.block_size before forward`。
- 2026-07-30：默认 `nn.Embedding` 初始化尺度约为 1，又与输出 head 做了
  weight tying，造成初始 logits 饱和，step 1 loss=`81.0888` 且生成重复 `e`。
  改为标准差 `0.02` 的小尺度初始化后，step 1 loss 恢复为 `4.2070`，
  接近 `ln(65)=4.1744`。
