"""Learning-target utilities for Ex4 evaluation and resumable checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn


BatchFactory = Callable[[int], tuple[torch.Tensor, torch.Tensor]]


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reuse the Ex1 loss only after the learner has completed it there."""
    # TODO(你)[EX04_REUSE_TOKEN_LOSS]: 复制并解释你在 Ex1 验证过的实现。
    #   完成标准:logits=[B,T,V]、targets=[B,T] 时返回可反传标量。
    raise NotImplementedError("TODO[EX04_REUSE_TOKEN_LOSS]")


@torch.no_grad()
def evaluate_average_loss(
    model: nn.Module,
    batch_factory: BatchFactory,
    *,
    num_batches: int,
) -> float:
    """Evaluate without changing parameters or training-mode behavior."""
    # TODO(你)[EX04_EVAL_AVERAGE]: 见指南 §10.1。
    #   方向:临时切 eval，累计多个 batch 的 token loss，再恢复原模式。
    #   完成标准:两次固定种子评估一致，且评估前后参数逐项不变。
    raise NotImplementedError("TODO[EX04_EVAL_AVERAGE]")


def perplexity_from_loss(average_loss: float) -> float:
    """Convert average cross-entropy to perplexity."""
    # TODO(你)[EX04_PERPLEXITY]: 见指南 §10.1。
    #   完成标准:loss=0 时 PPL=1；并能解释为何不可跨分词器比较。
    raise NotImplementedError("TODO[EX04_PERPLEXITY]")


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    validation_loss: float,
    metadata: dict[str, Any],
) -> None:
    """Persist every state required for a genuine training resume."""
    # TODO(你)[EX04_SAVE_CHECKPOINT]: 见指南 §10.3。
    #   必须包含:model、optimizer、scheduler、step、validation_loss、CPU RNG、metadata。
    #   metadata 至少足以重建模型 config 与 tokenizer 词表，而不只是辨认文件名。
    #   完成标准:新进程可从同一步继续；不能只保存 model.state_dict()。
    raise NotImplementedError("TODO[EX04_SAVE_CHECKPOINT]")


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float, dict[str, Any]]:
    """Restore state and return step, validation loss, and reconstruction metadata."""
    # TODO(你)[EX04_LOAD_CHECKPOINT]: 见指南 §10.3。
    #   方向:加载与 save 对称的所有 state，并恢复 RNG。
    #   完成标准:加载前后的固定验证 loss 完全一致，scheduler 的 lr 连续。
    raise NotImplementedError("TODO[EX04_LOAD_CHECKPOINT]")
