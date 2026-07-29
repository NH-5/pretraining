"""Dataset and batching contracts for the Ex10 SFT exercise."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import torch


IGNORE_INDEX = -100


@dataclass(frozen=True)
class InstructionExample:
    split: str
    instruction: str
    response: str


class TokenizerAdapter(Protocol):
    pad_id: int
    eos_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor


def load_examples(path: Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            try:
                example = InstructionExample(
                    split=str(payload["split"]),
                    instruction=str(payload["instruction"]),
                    response=str(payload["response"]),
                )
            except KeyError as error:
                raise ValueError(f"{path}:{line_number} missing {error.args[0]}") from error
            if not example.instruction or not example.response:
                raise ValueError(f"{path}:{line_number} has an empty field.")
            examples.append(example)
    return examples


def format_instruction(example: InstructionExample) -> tuple[str, str]:
    """Return the prompt text and response text as separate strings."""
    # TODO(你)[EX10_FORMAT_INSTRUCTION]: 见指南 §11。
    #   方向:定义固定模板，明确哪里结束 prompt、哪里开始 assistant response。
    #   完成标准:同一模板用于训练和推理，不能训练一种格式、评估另一种。
    raise NotImplementedError("TODO[EX10_FORMAT_INSTRUCTION]")


def response_only_labels(
    next_tokens: list[int],
    *,
    prompt_token_count: int,
) -> list[int]:
    """Mask target positions that teach the model to reproduce the prompt."""
    # TODO(你)[EX10_RESPONSE_LOSS_MASK]: 见指南 §6.3、§11。
    #   注意:next_tokens 已相对 input_ids 右移一位，推导 prompt 对应几个 label。
    #   使用 IGNORE_INDEX 屏蔽 prompt，只监督 assistant response 与 eos。
    #   完成标准:人工画一个 4-token prompt + 2-token response 的对齐表。
    raise NotImplementedError("TODO[EX10_RESPONSE_LOSS_MASK]")


def encode_example(
    example: InstructionExample,
    tokenizer: TokenizerAdapter,
) -> EncodedExample:
    prompt, response = format_instruction(example)
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [tokenizer.eos_id]
    all_ids = prompt_ids + response_ids
    if len(all_ids) < 2:
        raise ValueError("Encoded example is too short.")
    input_ids = all_ids[:-1]
    next_tokens = all_ids[1:]
    labels = response_only_labels(
        next_tokens,
        prompt_token_count=len(prompt_ids),
    )
    if len(input_ids) != len(labels):
        raise RuntimeError("input_ids and labels must have the same length.")
    return EncodedExample(input_ids=input_ids, labels=labels)


def collate(
    examples: list[EncodedExample],
    *,
    pad_id: int,
    device: torch.device,
) -> Batch:
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    max_length = max(len(example.input_ids) for example in examples)
    inputs = torch.full(
        (len(examples), max_length),
        fill_value=pad_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (len(examples), max_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        inputs[row, :length] = torch.tensor(example.input_ids)
        labels[row, :length] = torch.tensor(example.labels)
    return Batch(input_ids=inputs.to(device), labels=labels.to(device))
