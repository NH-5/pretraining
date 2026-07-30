# 练习进度

Ex0 ✅ — 2026-07-29 — PyTorch 2.13.0；CUDA=False；MPS=True；Apple M1 / 16 GB 统一内存；张量测试 PASS — M1 使用统一内存，不能误报成 CUDA 独立显存。

Ex1 ⏳ — 2026-07-30 — 三个核心检查 PASS；M1 20-step loss `4.2070→3.3297`（9.4 s），样本不再单字符坍缩 — 等待完整训练达到 `loss < 1.5` 并抽查莎翁式文本。

Ex2 ⏳ — 脚手架已备好 — 等待三项脱离代码的口述验收。

Ex3 ⏳ — 脚手架已备好 — 到达后再确认安装 `tokenizers`，等待两档词表对照。

Ex4 ⏳ — 脚手架已备好 — 等待 eval/PPL 与完整 checkpoint resume 验证。

Ex5 ⏳ — 脚手架已备好 — 等待 warmup+cosine、梯度累积与 fp32/bf16 对照。

Ex6 ⏳ — 脚手架已备好 — 等待 Scaling 公式与实测吞吐对账。

Ex7 ⏳ — 脚手架已备好 — 到达后再确认安装数据依赖，等待 FineWeb 小切片实测。

Ex8 ⏳ — 脚手架已备好 — 等待 2+ CUDA 卡的 DDP/FSDP 等价性、显存、吞吐与 MFU。

Ex9 ⏳ — 脚手架已备好 — 等待 RoPE/SwiGLU/GQA 三选一的同条件对照。

Ex10 ⏳ — 脚手架已备好 — 等待底模 checkpoint、response-only loss 与 SFT 前后对比。
