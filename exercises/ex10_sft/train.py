"""Ex10: turn a completed base LM into a tiny instruction follower."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.checking import CheckCase, ManualCheck, SkipCheck, run_checks
from model_adapter import BaseArtifacts, load_base_artifacts
from utils import (
    Batch,
    IGNORE_INDEX,
    InstructionExample,
    collate,
    encode_example,
    format_instruction,
    load_examples,
    response_only_labels,
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


def _check_sample_data(data_path: Path) -> str:
    examples = load_examples(data_path)
    splits = {example.split for example in examples}
    if not {"train", "eval"}.issubset(splits):
        raise RuntimeError("Sample data must contain train and eval splits.")
    return (
        f"sample examples={len(examples)}; train/eval splits present; "
        "this validates wiring, not general instruction-following ability"
    )


def _check_instruction_format() -> None:
    example = InstructionExample(
        split="train",
        instruction="把下面一句话改成过去时。",
        response="模型完成了改写。",
    )
    prompt, response = format_instruction(example)
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("format_instruction must return a non-empty prompt string.")
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("format_instruction must return a non-empty response string.")
    if example.instruction not in prompt:
        raise RuntimeError("The formatted prompt lost the instruction text.")
    if example.response not in response:
        raise RuntimeError("The formatted response lost the reference answer.")
    raise ManualCheck(
        "Prompt/response boundary is structurally valid. Confirm the identical "
        "prompt template is used for both SFT training and inference."
    )


def _check_response_mask() -> str:
    next_tokens = [11, 12, 13, 14, 15]
    labels = response_only_labels(next_tokens, prompt_token_count=4)
    expected = [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 14, 15]
    if labels != expected:
        raise RuntimeError(
            "4-token prompt alignment is wrong: "
            f"expected {expected}, got {labels}."
        )
    if response_only_labels([21, 22], prompt_token_count=1) != [21, 22]:
        raise RuntimeError("A 1-token prompt should leave both shifted targets visible.")
    return f"next_tokens={next_tokens} -> labels={labels}"


def _check_response_only_loss() -> str:
    labels = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX, 2, 1, 3]],
        dtype=torch.long,
    )
    logits = torch.zeros(1, 5, 4, requires_grad=True)
    loss = response_only_loss(logits, labels)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Response-only loss must be one finite scalar.")
    if not math.isclose(loss.item(), math.log(4.0), rel_tol=1e-6):
        raise RuntimeError(
            "Zero logits over 4 classes should have loss ln(4) when averaging "
            "only the 3 supervised response positions."
        )
    loss.backward()
    if logits.grad is None:
        raise RuntimeError("Response-only loss did not create logits gradients.")
    if torch.any(logits.grad[:, :2, :] != 0):
        raise RuntimeError("Ignored prompt positions contributed gradients.")
    if not torch.all(logits.grad[:, 2:, :].abs().sum(dim=-1) > 0):
        raise RuntimeError("Every supervised response position needs a gradient.")

    good_logits = torch.full((1, 5, 4), -4.0)
    for position in range(2, 5):
        good_logits[0, position, labels[0, position]] = 4.0
    changed_prompt_logits = good_logits.clone()
    changed_prompt_logits[:, :2, :] = torch.tensor(
        [[[100.0, -100.0, 50.0, -50.0], [-80.0, 90.0, -70.0, 60.0]]]
    )
    good_loss = response_only_loss(good_logits, labels)
    changed_prompt_loss = response_only_loss(changed_prompt_logits, labels)
    if not torch.allclose(good_loss, changed_prompt_loss, rtol=0.0, atol=0.0):
        raise RuntimeError("Changing masked prompt logits changed the scalar loss.")
    wrong_labels = labels.clone()
    wrong_labels[0, 2] = 0
    wrong_loss = response_only_loss(good_logits, wrong_labels)
    if not wrong_loss > good_loss:
        raise RuntimeError("A wrong supervised response target should increase loss.")
    return "prompt gradients=0; response gradients>0; masked logits do not change loss"


def _load_artifacts_for_check(checkpoint: Path | None) -> BaseArtifacts:
    if checkpoint is None:
        raise SkipCheck(
            "Pass --checkpoint /path/to/completed-ex4.pt to validate the base adapter."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return load_base_artifacts(checkpoint, device=torch.device("cpu"))


def _check_base_adapter(checkpoint: Path | None) -> None:
    artifacts = _load_artifacts_for_check(checkpoint)
    if not isinstance(artifacts.model, nn.Module):
        raise RuntimeError("BaseArtifacts.model must be a torch.nn.Module.")
    for attribute in ("pad_id", "eos_id", "encode", "decode"):
        if not hasattr(artifacts.tokenizer, attribute):
            raise RuntimeError(f"Tokenizer adapter lacks {attribute}.")
    if not callable(artifacts.generate_text):
        raise RuntimeError("BaseArtifacts.generate_text must be callable.")
    raise ManualCheck(
        "Base model/tokenizer/generator loaded. Compare generation before and "
        "after loading with the same seed to finish the resume check."
    )


def _check_sft_checkpoint(checkpoint: Path | None) -> None:
    artifacts = _load_artifacts_for_check(checkpoint)
    optimizer = torch.optim.AdamW(artifacts.model.parameters(), lr=1e-4)
    with TemporaryDirectory(prefix="ex10-check-") as temporary_directory:
        output_path = Path(temporary_directory) / "sft.pt"
        save_sft_checkpoint(
            output_path,
            artifacts=artifacts,
            optimizer=optimizer,
            epoch=3,
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("save_sft_checkpoint did not create a non-empty file.")
    raise ManualCheck(
        "SFT checkpoint file was created. Load it in a new process and compare "
        "generation to verify model/tokenizer/base-source metadata."
    )


def check_scaffold(data_path: Path, checkpoint: Path | None) -> None:
    run_checks(
        "Ex10 learning-target checks:",
        [
            CheckCase("sample data wiring", lambda: _check_sample_data(data_path)),
            CheckCase("EX10_FORMAT_INSTRUCTION", _check_instruction_format),
            CheckCase("EX10_RESPONSE_LOSS_MASK", _check_response_mask),
            CheckCase("EX10_RESPONSE_ONLY_LOSS", _check_response_only_loss),
            CheckCase(
                "EX10_LOAD_BASE_MODEL",
                lambda: _check_base_adapter(checkpoint),
            ),
            CheckCase(
                "EX10_SAVE_SFT",
                lambda: _check_sft_checkpoint(checkpoint),
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    check_parser.add_argument("--checkpoint", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--epochs", type=int, default=SFTConfig.epochs)
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold(args.data, args.checkpoint)
    else:
        run_sft(
            checkpoint=args.checkpoint,
            data_path=args.data,
            output_path=args.output,
            config=SFTConfig(epochs=args.epochs),
        )


if __name__ == "__main__":
    main()
