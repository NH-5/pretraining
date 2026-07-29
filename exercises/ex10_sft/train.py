"""Ex10: turn a completed base LM into a tiny instruction follower."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from model_adapter import BaseArtifacts, load_base_artifacts
from utils import (
    Batch,
    InstructionExample,
    collate,
    encode_example,
    format_instruction,
    load_examples,
)


TODO_IDS = (
    "EX10_FORMAT_INSTRUCTION",
    "EX10_RESPONSE_LOSS_MASK",
    "EX10_LOAD_BASE_MODEL",
    "EX10_RESPONSE_ONLY_LOSS",
    "EX10_SAVE_SFT",
)
DEFAULT_DATA = Path(__file__).with_name("sample_instructions.jsonl")


@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    max_new_tokens: int = 32
    seed: int = 41


def response_only_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute token loss while respecting IGNORE_INDEX prompt/pad labels."""
    # TODO(你)[EX10_RESPONSE_ONLY_LOSS]: 见指南 §6.3、§11。
    #   复用 Ex1 的逐 token loss，但让 IGNORE_INDEX 不贡献 loss。
    #   完成标准:改变被 mask 的 prompt token 不改变标量 loss。
    raise NotImplementedError("TODO[EX10_RESPONSE_ONLY_LOSS]")


def save_sft_checkpoint(
    output_path: Path,
    *,
    artifacts: BaseArtifacts,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Save enough state to reproduce the SFT model and tokenizer."""
    # TODO(你)[EX10_SAVE_SFT]: 复用 Ex4 的 checkpoint 原则。
    #   完成标准:新进程加载后生成与保存前一致，并保留 base checkpoint 来源。
    raise NotImplementedError("TODO[EX10_SAVE_SFT]")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(
    examples: list[InstructionExample],
    artifacts: BaseArtifacts,
    *,
    batch_size: int,
    device: torch.device,
) -> list[Batch]:
    encoded = [encode_example(example, artifacts.tokenizer) for example in examples]
    return [
        collate(
            encoded[start : start + batch_size],
            pad_id=artifacts.tokenizer.pad_id,
            device=device,
        )
        for start in range(0, len(encoded), batch_size)
    ]


def generate_eval_examples(
    examples: list[InstructionExample],
    artifacts: BaseArtifacts,
    *,
    max_new_tokens: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for example in examples:
        prompt, _ = format_instruction(example)
        outputs[example.instruction] = artifacts.generate_text(prompt, max_new_tokens)
    return outputs


def run_sft(
    *,
    checkpoint: Path,
    data_path: Path,
    output_path: Path,
    config: SFTConfig,
) -> None:
    torch.manual_seed(config.seed)
    device = choose_device()
    artifacts = load_base_artifacts(checkpoint, device=device)
    examples = load_examples(data_path)
    train_examples = [example for example in examples if example.split == "train"]
    eval_examples = [example for example in examples if example.split == "eval"]
    if not train_examples or not eval_examples:
        raise ValueError("Need both train and eval examples.")

    before = generate_eval_examples(
        eval_examples,
        artifacts,
        max_new_tokens=config.max_new_tokens,
    )
    optimizer = torch.optim.AdamW(
        artifacts.model.parameters(),
        lr=config.learning_rate,
    )
    train_batches = batches(
        train_examples,
        artifacts,
        batch_size=config.batch_size,
        device=device,
    )
    artifacts.model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_losses: list[float] = []
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            logits = artifacts.model(batch.input_ids)
            loss = response_only_loss(logits, batch.labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch == 1 or epoch % 5 == 0:
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"epoch={epoch:03d} sft_loss={mean_loss:.5f}")

    artifacts.model.eval()
    after = generate_eval_examples(
        eval_examples,
        artifacts,
        max_new_tokens=config.max_new_tokens,
    )
    print("--- before / after ---")
    for instruction in before:
        print(f"instruction: {instruction}")
        print(f"before:      {before[instruction]}")
        print(f"after:       {after[instruction]}")
    save_sft_checkpoint(
        output_path,
        artifacts=artifacts,
        optimizer=optimizer,
        epoch=config.epochs,
    )


def check_scaffold(data_path: Path) -> None:
    examples = load_examples(data_path)
    splits = {example.split for example in examples}
    if not {"train", "eval"}.issubset(splits):
        raise RuntimeError("Sample data must contain train and eval splits.")
    print("Ex10 scaffold: PASS")
    print(f"sample examples={len(examples)}")
    print("The sample is intentionally tiny; it validates wiring, not general ability.")
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--epochs", type=int, default=SFTConfig.epochs)
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold(args.data)
    else:
        run_sft(
            checkpoint=args.checkpoint,
            data_path=args.data,
            output_path=args.output,
            config=SFTConfig(epochs=args.epochs),
        )


if __name__ == "__main__":
    main()
