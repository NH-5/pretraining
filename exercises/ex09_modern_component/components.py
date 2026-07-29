"""Modern decoder components for Ex9; each core formula is deliberately TODO."""

from __future__ import annotations

import math

import torch
from torch import nn


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position encoding to [B,H,T,D] query/key tensors."""
    # TODO(你)[EX09_ROPE]: 见指南 §3.3。
    #   方向:head_dim 按偶/奇维成对旋转；角度由位置和频率共同决定。
    #   边界:head_dim 必须为偶数，输出 shape/dtype/device 与输入一致。
    #   完成标准:position=0 不改变向量；每对二维分量的 L2 范数保持。
    raise NotImplementedError("TODO[EX09_ROPE]")


class SwiGLU(nn.Module):
    """Parameter-matched SwiGLU feed-forward block."""

    def __init__(self, embedding_size: int, dropout: float) -> None:
        super().__init__()
        # 8d/3 roughly matches the parameters of a 4d two-matrix GELU FFN.
        hidden_size = math.ceil((8 * embedding_size / 3) / 8) * 8
        self.gate_projection = nn.Linear(embedding_size, hidden_size, bias=False)
        self.up_projection = nn.Linear(embedding_size, hidden_size, bias=False)
        self.down_projection = nn.Linear(hidden_size, embedding_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # TODO(你)[EX09_SWIGLU]: 见指南 §3.3。
        #   方向:一支经过 SiLU 形成 gate，逐元素乘另一支，再 down projection。
        #   完成标准:输入输出 shape 相同，梯度能到达三组权重。
        raise NotImplementedError("TODO[EX09_SWIGLU]")


class GroupedQueryAttention(nn.Module):
    """GQA with fewer key/value heads than query heads."""

    def __init__(
        self,
        *,
        embedding_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_size % num_query_heads != 0:
            raise ValueError("embedding_size must divide num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must divide evenly by num_kv_heads.")
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embedding_size // num_query_heads
        self.query_projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key_projection = nn.Linear(
            embedding_size,
            num_kv_heads * self.head_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            embedding_size,
            num_kv_heads * self.head_dim,
            bias=False,
        )
        self.output_projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.dropout = dropout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # TODO(你)[EX09_GQA]: 见指南 §3.3。
        #   方向:Q 有 Hq 个头，K/V 只有 Hkv 个头；按 group 共享 K/V。
        #   要求:仍是 causal attention，不能因 repeat/expand 引入未来信息。
        #   完成标准:输出 [B,T,C]；Hkv=Hq 时退化为普通 MHA。
        raise NotImplementedError("TODO[EX09_GQA]")
