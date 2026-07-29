"""Adapter boundary between the completed base-model exercises and Ex10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from utils import TokenizerAdapter


@dataclass(frozen=True)
class BaseArtifacts:
    model: nn.Module
    tokenizer: TokenizerAdapter
    generate_text: Callable[[str, int], str]


def load_base_artifacts(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> BaseArtifacts:
    """Load the learner's Ex1/Ex4 model, tokenizer metadata, and generator."""
    # TODO(你)[EX10_LOAD_BASE_MODEL]: 复用 Ex1 结构、Ex4 checkpoint、Ex1 generate。
    #   完成标准:不训练时，同一 prompt 的加载前后生成结果一致。
    #   提示:checkpoint 必须携带足以重建 config 与字符表的 metadata。
    #   Ex1 字符表没有 pad/eos；需显式扩展词表和 tied embedding/head，不能越界。
    raise NotImplementedError("TODO[EX10_LOAD_BASE_MODEL]")
