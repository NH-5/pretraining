"""Ex3: train configurable byte-level BPE tokenizers and compare languages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.checking import CheckCase, ManualCheck, SkipCheck, run_checks
from utils import RatioRow, format_table, load_tokenizer, measure, require_tokenizers


SAMPLES = {
    "English": (
        "A language model predicts the next token from the tokens that came before it."
    ),
    "中文": "语言模型根据前面的词元预测下一个词元，并从大量文本中学习规律。",
}


class _CharacterTokenizerForCheck:
    """A dependency-free fake used only to verify reporting code."""

    class Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    def encode(self, text: str) -> "_CharacterTokenizerForCheck.Encoding":
        return self.Encoding(list(range(len(text))))


def train_byte_level_bpe(
    corpus_files: list[Path],
    *,
    vocab_size: int,
    output_path: Path,
) -> Path:
    """Train and save one Hugging Face byte-level BPE tokenizer."""
    require_tokenizers()
    if vocab_size < 259:
        raise ValueError(
            "Byte-level BPE vocab_size must fit 256 bytes plus bos/eos/pad."
        )
    missing = [path for path in corpus_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing corpus files: {missing}")

    # TODO(你)[EX03_BUILD_BPE]: 见指南 §5.2、§5.3。
    #   方向:组合 Tokenizer(BPE)、ByteLevel pre-tokenizer/decoder 和 BpeTrainer。
    #   结构:特殊 token 至少包含 <bos>/<eos>/<pad>；vocab_size 来自参数。
    #   完成标准:保存 tokenizer.json，任意 UTF-8 文本 encode→decode 可往返。
    raise NotImplementedError("TODO[EX03_BUILD_BPE]")


def compare_tokenizers(specs: list[tuple[str, Path]]) -> list[RatioRow]:
    rows: list[RatioRow] = []
    for name, path in specs:
        tokenizer = load_tokenizer(path)
        for language, text in SAMPLES.items():
            rows.append(
                measure(
                    tokenizer,
                    tokenizer_name=name,
                    language=language,
                    text=text,
                )
            )
    return rows


def interpret_ratio(rows: list[RatioRow]) -> str:
    """Summarize the sequence-length/embedding trade-off seen in the table."""
    # TODO(你)[EX03_INTERPRET_RATIO]: 见指南 §5.3。
    #   比较至少两个 vocab_size，并分别解释中英文 chars/token 的变化。
    #   完成标准:结论同时提到序列长度成本和 V*d embedding 参数成本。
    raise NotImplementedError("TODO[EX03_INTERPRET_RATIO]")


def parse_spec(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/tokenizer.json")
    return name, Path(raw_path)


def _check_report_table() -> str:
    fake = _CharacterTokenizerForCheck()
    rows = [
        measure(fake, tokenizer_name="character", language=language, text=text)
        for language, text in SAMPLES.items()
    ]
    table = format_table(rows)
    if "chars/token" not in table:
        raise RuntimeError("Comparison table wiring failed.")
    return table


def _check_bpe_round_trip() -> str:
    try:
        tokenizers = require_tokenizers()
    except RuntimeError as error:
        raise SkipCheck(str(error)) from error

    with TemporaryDirectory(prefix="ex03-check-") as temporary_directory:
        directory = Path(temporary_directory)
        corpus_path = directory / "corpus.txt"
        output_path = directory / "tokenizer.json"
        corpus_path.write_text(
            "\n".join(
                [
                    "A language model predicts the next token.",
                    "Byte pair encoding learns reusable pieces.",
                    "语言模型根据前面的词元预测下一个词元。",
                    "分词方式会改变序列长度和词表大小。",
                ]
                * 8
            ),
            encoding="utf-8",
        )
        saved_path = Path(
            train_byte_level_bpe(
                [corpus_path],
                vocab_size=300,
                output_path=output_path,
            )
        )
        if not saved_path.is_file() or saved_path.stat().st_size == 0:
            raise RuntimeError("train_byte_level_bpe did not save a tokenizer file.")

        try:
            serialized = json.loads(saved_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Saved tokenizer is not valid UTF-8 JSON.") from error
        model_payload = serialized.get("model")
        if not isinstance(model_payload, dict) or model_payload.get("type") != "BPE":
            raise RuntimeError("Saved tokenizer must contain a BPE model.")
        merges = model_payload.get("merges")
        if not isinstance(merges, list) or not merges:
            raise RuntimeError(
                "The BPE model learned no merge rules; a reversible byte codec "
                "alone does not satisfy this exercise."
            )

        tokenizer = tokenizers.Tokenizer.from_file(str(saved_path))
        missing_specials = [
            token
            for token in ("<bos>", "<eos>", "<pad>")
            if tokenizer.token_to_id(token) is None
        ]
        if missing_specials:
            raise RuntimeError(f"Missing special tokens: {missing_specials}")
        if tokenizer.get_vocab_size() < 259:
            raise RuntimeError(
                "Byte-level vocabulary must cover all 256 byte symbols plus "
                "the 3 required special tokens."
            )
        if tokenizer.get_vocab_size() > 300:
            raise RuntimeError("Saved vocabulary exceeds the requested vocab_size=300.")

        samples = (
            "Hello, byte-level BPE!",
            "中文与 English 可以一起往返。",
            "spaces  and\nnewlines\tmust survive",
        )
        for text in samples:
            restored = tokenizer.decode(tokenizer.encode(text).ids)
            if restored != text:
                raise RuntimeError(
                    f"UTF-8 round trip changed text: {text!r} -> {restored!r}"
                )

        compression_probe = (
            "A language model predicts the next token. "
            "Byte pair encoding learns reusable pieces."
        )
        encoded_length = len(tokenizer.encode(compression_probe).ids)
        byte_length = len(compression_probe.encode("utf-8"))
        if encoded_length >= byte_length:
            raise RuntimeError(
                "Learned tokenizer did not compress a corpus-like probe below "
                f"its {byte_length}-byte representation (got {encoded_length} tokens)."
            )
    return (
        f"saved BPE with {len(merges)} merges and full byte coverage; "
        f"probe {byte_length} bytes -> {encoded_length} tokens; "
        "3 UTF-8 round trips passed"
    )


def _check_interpretation() -> None:
    rows = [
        RatioRow("small", "English", characters=80, tokens=40),
        RatioRow("small", "中文", characters=30, tokens=60),
        RatioRow("large", "English", characters=80, tokens=24),
        RatioRow("large", "中文", characters=30, tokens=26),
    ]
    explanation = interpret_ratio(rows)
    if not isinstance(explanation, str) or not explanation.strip():
        raise RuntimeError("interpret_ratio must return a non-empty explanation.")
    raise ManualCheck(
        f"Explanation found ({len(explanation.strip())} characters). "
        "Check aloud that it covers sequence length and V*d embedding cost."
    )


def check_scaffold() -> None:
    run_checks(
        "Ex3 learning-target checks:",
        [
            CheckCase("comparison report wiring", _check_report_table),
            CheckCase("EX03_BUILD_BPE", _check_bpe_round_trip),
            CheckCase("EX03_INTERPRET_RATIO", _check_interpretation),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    train_parser.add_argument("--vocab-size", type=int, required=True)
    train_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--tokenizer",
        type=parse_spec,
        action="append",
        required=True,
        help="Repeat NAME=/path/tokenizer.json for each vocabulary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        check_scaffold()
    elif args.command == "train":
        saved = train_byte_level_bpe(
            args.corpus,
            vocab_size=args.vocab_size,
            output_path=args.output,
        )
        print(f"Saved tokenizer to {saved}")
    else:
        rows = compare_tokenizers(args.tokenizer)
        print(format_table(rows))
        print("\nInterpretation:")
        print(interpret_ratio(rows))


if __name__ == "__main__":
    main()
