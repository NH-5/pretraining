"""A tiny comparison model that swaps exactly one Ex9 component."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from components import GroupedQueryAttention, SwiGLU, apply_rope


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 64
    block_size: int = 32
    embedding_size: int = 96
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    dropout: float = 0.0


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Reuse the causal attention semantics already verified in Ex1."""
    # TODO(你)[EX09_REUSE_CAUSAL_ATTENTION]: 复制你在 Ex1 验证过的 mask 逻辑。
    #   完成标准:同一个未来泄漏测试继续通过；不要在 Ex9 顺手换 mask 语义。
    raise NotImplementedError("TODO[EX09_REUSE_CAUSAL_ATTENTION]")


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reuse the next-token loss already verified in Ex1."""
    # TODO(你)[EX09_REUSE_TOKEN_LOSS]: 复制你在 Ex1 验证过的逐 token loss。
    #   完成标准:baseline/component 两边调用完全同一个函数。
    raise NotImplementedError("TODO[EX09_REUSE_TOKEN_LOSS]")


class BaselineAttention(nn.Module):
    def __init__(self, config: ModelConfig, *, use_rope: bool) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_size // config.num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(config.embedding_size, 3 * config.embedding_size)
        self.output = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = config.dropout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = map(split, (query, key, value))
        if self.use_rope:
            query, key = apply_rope(query, key)
        attended = causal_attention(
            query,
            key,
            value,
            dropout=self.dropout,
            training=self.training,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_size,
        )
        return self.output(attended)


class BaselineFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            nn.GELU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, component: str) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_size)
        if component == "gqa":
            self.attention: nn.Module = GroupedQueryAttention(
                embedding_size=config.embedding_size,
                num_query_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                dropout=config.dropout,
            )
        else:
            self.attention = BaselineAttention(
                config,
                use_rope=component == "rope",
            )
        self.ffn_norm = nn.LayerNorm(config.embedding_size)
        self.feed_forward: nn.Module = (
            SwiGLU(config.embedding_size, config.dropout)
            if component == "swiglu"
            else BaselineFeedForward(config)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig, component: str) -> None:
        super().__init__()
        if component not in {"baseline", "rope", "swiglu", "gqa"}:
            raise ValueError(component)
        self.component = component
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = (
            None
            if component == "rope"
            else nn.Embedding(config.block_size, config.embedding_size)
        )
        self.blocks = nn.ModuleList(
            Block(config, component) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_size)
        self.head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            hidden = hidden + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.final_norm(hidden))
