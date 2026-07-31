"""Ex9: compare one modern decoder component against a fixed baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components import GroupedQueryAttention, SwiGLU, apply_rope
from exercises.checking import CheckCase, run_checks
from model import ModelConfig, TinyGPT, causal_attention, next_token_loss


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-4
    seed: int = 29


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(train_config.seed + step)
    token_stream = torch.empty(
        train_config.batch_size,
        model_config.block_size + 1,
        dtype=torch.long,
    )
    token_stream[:, :2] = torch.randint(
        0,
        model_config.vocab_size,
        (train_config.batch_size, 2),
        generator=generator,
    )
    # The next token depends on two preceding tokens, so attention is genuinely useful.
    for position in range(2, model_config.block_size + 1):
        token_stream[:, position] = (
            token_stream[:, position - 1] + token_stream[:, position - 2]
        ) % model_config.vocab_size
    inputs = token_stream[:, :-1]
    targets = token_stream[:, 1:]
    return inputs.to(device), targets.to(device)


def train_variant(
    component: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> list[float]:
    torch.manual_seed(train_config.seed)
    device = choose_device()
    model = TinyGPT(model_config, component)
    if initial_state is not None:
        _load_compatible_initial_state(model, initial_state)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
    )
    losses: list[float] = []
    for step in range(train_config.steps):
        inputs, targets = make_batch(
            model_config,
            train_config,
            step=step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _load_compatible_initial_state(
    model: TinyGPT,
    initial_state: dict[str, torch.Tensor],
) -> int:
    """Copy every same-name/same-shape baseline tensor into one variant."""
    target_state = model.state_dict()
    copied = 0
    for name, target in target_state.items():
        source = initial_state.get(name)
        if (
            source is None
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            continue
        target_state[name] = source.detach().clone()
        copied += 1
    model.load_state_dict(target_state)
    return copied


def _check_model_scaffold() -> str:
    config = ModelConfig()
    counts: dict[str, int] = {}
    torch.manual_seed(900)
    baseline = TinyGPT(config, "baseline")
    baseline_state = {
        name: tensor.detach().clone()
        for name, tensor in baseline.state_dict().items()
    }
    shared_counts: dict[str, int] = {}
    for component in ("baseline", "rope", "swiglu", "gqa"):
        torch.manual_seed(900 + len(counts) + 1)
        model = TinyGPT(config, component)
        shared_counts[component] = _load_compatible_initial_state(
            model,
            baseline_state,
        )
        counts[component] = sum(parameter.numel() for parameter in model.parameters())
        for name, tensor in model.state_dict().items():
            source = baseline_state.get(name)
            if source is not None and source.shape == tensor.shape:
                if not torch.equal(tensor, source):
                    raise RuntimeError(
                        f"Compatible initialization tensor {name!r} was not shared."
                    )
    if any(count <= 0 for count in counts.values()):
        raise RuntimeError("A component model has no parameters.")
    return "; ".join(
        f"{name}={counts[name]:,} params/{shared_counts[name]} shared tensors"
        for name in counts
    )


def _check_causal_attention() -> str:
    torch.manual_seed(901)
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 3)
    original = causal_attention(
        query,
        key,
        value,
        dropout=0.0,
        training=False,
    )
    if original.shape != query.shape:
        raise RuntimeError(
            f"Causal attention output shape {original.shape} != {query.shape}."
        )
    changed_value = value.clone()
    changed_value[:, :, 2:, :] += 100.0
    changed = causal_attention(
        query,
        key,
        changed_value,
        dropout=0.0,
        training=False,
    )
    if not torch.allclose(
        original[:, :, :2, :],
        changed[:, :, :2, :],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError("Changing future values changed an earlier attention output.")
    return "output shape preserved; positions 0-1 cannot see changed positions 2-3"


def _check_next_token_loss() -> str:
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 3]])
    loss = next_token_loss(logits, targets)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Loss must be one finite scalar.")
    expected = torch.tensor(5.0).log().item()
    if not torch.isclose(
        loss.detach(),
        torch.tensor(expected, dtype=loss.dtype, device=loss.device),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Loss must average over all B*T token positions: "
            f"uniform V=5 expected ln(5)={expected:.6f}, got {loss.item():.6f}."
        )
    loss.backward()
    if logits.grad is None or not torch.all(logits.grad.abs().sum(dim=-1) > 0):
        raise RuntimeError("Every token position must contribute a gradient.")

    good_logits = torch.full((1, 4, 5), -4.0)
    good_logits.scatter_(-1, targets.unsqueeze(-1), 4.0)
    wrong_targets = targets.clone()
    wrong_targets[0, 2] = 4
    if not next_token_loss(good_logits, wrong_targets) > next_token_loss(
        good_logits,
        targets,
    ):
        raise RuntimeError("One wrong token target should increase loss.")
    return "mean loss=ln(5); all 4 positions have gradients; wrong target raises loss"


def _check_rope() -> str:
    torch.manual_seed(902)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 5, 8)
    rotated_query, rotated_key = apply_rope(query, key)
    for name, original, rotated in (
        ("query", query, rotated_query),
        ("key", key, rotated_key),
    ):
        if (
            rotated.shape != original.shape
            or rotated.dtype != original.dtype
            or rotated.device != original.device
        ):
            raise RuntimeError(f"RoPE changed {name} shape/dtype/device.")
        if not torch.allclose(
            rotated[:, :, 0, :],
            original[:, :, 0, :],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(f"RoPE position 0 changed the {name}.")
        original_pair_norms = original.reshape(2, 3, 5, 4, 2).norm(dim=-1)
        rotated_pair_norms = rotated.reshape(2, 3, 5, 4, 2).norm(dim=-1)
        if not torch.allclose(
            rotated_pair_norms,
            original_pair_norms,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"RoPE did not preserve paired {name} L2 norms.")
        if torch.allclose(rotated[:, :, 1:, :], original[:, :, 1:, :]):
            raise RuntimeError(f"RoPE did not rotate nonzero-position {name} values.")

    shared = torch.randn(1, 2, 5, 8)
    shared_query, shared_key = apply_rope(shared, shared)
    if not torch.allclose(shared_query, shared_key, rtol=0.0, atol=0.0):
        raise RuntimeError("RoPE must apply the same rotation rule to query and key.")

    base_100_query, base_100_key = apply_rope(query, key, base=100.0)
    if torch.allclose(base_100_query[:, :, 1:, :], rotated_query[:, :, 1:, :]):
        raise RuntimeError("RoPE ignored its base argument for query.")
    if torch.allclose(base_100_key[:, :, 1:, :], rotated_key[:, :, 1:, :]):
        raise RuntimeError("RoPE ignored its base argument for key.")

    try:
        apply_rope(query[..., :7], key[..., :7])
    except ValueError:
        pass
    else:
        raise RuntimeError("RoPE must reject an odd head dimension with ValueError.")
    return (
        "Q/K rotation, shape/dtype/device, position-0 identity, pair norms, "
        "base sensitivity, and odd-dimension rejection passed"
    )


def _check_swiglu() -> str:
    torch.manual_seed(903)
    module = SwiGLU(embedding_size=8, dropout=0.0)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = module(hidden)
    if output.shape != hidden.shape:
        raise RuntimeError(f"SwiGLU output shape {output.shape} != {hidden.shape}.")
    output.square().mean().backward()
    for name in ("gate_projection", "up_projection", "down_projection"):
        gradient = getattr(module, name).weight.grad
        if gradient is None or not torch.any(gradient != 0):
            raise RuntimeError(f"SwiGLU gradient did not reach {name}.weight.")

    zero_gate = SwiGLU(embedding_size=8, dropout=0.0)
    with torch.no_grad():
        zero_gate.gate_projection.weight.zero_()
        zero_gate.up_projection.weight.fill_(0.25)
        zero_gate.down_projection.weight.fill_(0.25)
        zero_gate_output = zero_gate(torch.ones(1, 1, 8))
    if not torch.allclose(
        zero_gate_output,
        torch.zeros_like(zero_gate_output),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "A zero gate must suppress the up branch; the two branches may "
            "have been added instead of multiplied."
        )

    controlled = SwiGLU(embedding_size=8, dropout=0.0)
    with torch.no_grad():
        for parameter in controlled.parameters():
            parameter.zero_()
        controlled.gate_projection.weight[0, 0] = 1.25
        controlled.up_projection.weight[0, 0] = -0.75
        controlled.down_projection.weight[0, 0] = 1.0
        probe = torch.zeros(1, 1, 8)
        probe[..., 0] = 1.0
        controlled_output = controlled(probe)
    expected_first = F.silu(torch.tensor(1.25)) * -0.75
    if not torch.allclose(
        controlled_output[..., 0],
        expected_first.expand_as(controlled_output[..., 0]),
        rtol=1e-6,
        atol=1e-7,
    ) or not torch.allclose(
        controlled_output[..., 1:],
        torch.zeros_like(controlled_output[..., 1:]),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Controlled gate/up/down projections do not match SiLU-gated "
            "elementwise multiplication."
        )
    return (
        "shape/backward passed; zero-gate and controlled SiLU×up probes passed"
    )


def _check_gqa() -> str:
    torch.manual_seed(904)
    module = GroupedQueryAttention(
        embedding_size=8,
        num_query_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    )
    module.eval()
    expected_kv_width = module.num_kv_heads * module.head_dim
    if module.num_query_heads != 4 or module.num_kv_heads != 2:
        raise RuntimeError("GQA did not retain the configured query/KV head counts.")
    if expected_kv_width >= 8:
        raise RuntimeError("The checker configuration must use fewer KV than Q heads.")
    if (
        module.query_projection.out_features != 8
        or module.key_projection.out_features != expected_kv_width
        or module.value_projection.out_features != expected_kv_width
    ):
        raise RuntimeError(
            "K/V projections must be narrower than Q when num_kv_heads "
            "< num_query_heads."
        )
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = module(hidden)
    if output.shape != hidden.shape:
        raise RuntimeError(f"GQA output shape {output.shape} != {hidden.shape}.")
    output.square().mean().backward()
    for name in (
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
    ):
        gradient = getattr(module, name).weight.grad
        if gradient is None or not torch.any(gradient != 0):
            raise RuntimeError(f"GQA gradient did not reach {name}.weight.")

    with torch.no_grad():
        clean_hidden = hidden.detach()
        original = module(clean_hidden)
        changed_hidden = clean_hidden.clone()
        changed_hidden[:, 2:, :] += 100.0
        changed = module(changed_hidden)
    if not torch.allclose(
        original[:, :2, :],
        changed[:, :2, :],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError("GQA leaked changed future tokens into earlier outputs.")
    return (
        f"Q heads=4, KV heads=2, KV width={expected_kv_width}; "
        "shape/backward and causal-boundary checks passed"
    )


def check_scaffold() -> None:
    run_checks(
        "Ex9 learning-target checks (RoPE/SwiGLU/GQA choose one):",
        [
            CheckCase("model construction", _check_model_scaffold),
            CheckCase("EX09_REUSE_CAUSAL_ATTENTION", _check_causal_attention),
            CheckCase("EX09_REUSE_TOKEN_LOSS", _check_next_token_loss),
            CheckCase("EX09_ROPE", _check_rope),
            CheckCase("EX09_SWIGLU", _check_swiglu),
            CheckCase("EX09_GQA", _check_gqa),
        ],
    )


def compare(component: str) -> None:
    model_config = ModelConfig()
    train_config = TrainConfig()
    torch.manual_seed(train_config.seed)
    baseline_template = TinyGPT(model_config, "baseline")
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in baseline_template.state_dict().items()
    }
    baseline_losses = train_variant(
        "baseline",
        model_config,
        train_config,
        initial_state=initial_state,
    )
    component_losses = train_variant(
        component,
        model_config,
        train_config,
        initial_state=initial_state,
    )
    tail = min(20, train_config.steps)
    baseline_tail_mean = sum(baseline_losses[-tail:]) / tail
    component_tail_mean = sum(component_losses[-tail:]) / tail
    print(f"steps={train_config.steps}")
    print(f"baseline final loss={baseline_losses[-1]:.6f}")
    print(f"{component} final loss={component_losses[-1]:.6f}")
    print(f"delta={component_losses[-1] - baseline_losses[-1]:+.6f}")
    print(f"baseline last-{tail} mean={baseline_tail_mean:.6f}")
    print(f"{component} last-{tail} mean={component_tail_mean:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--component",
        choices=("rope", "swiglu", "gqa"),
        required=True,
    )
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold()
    else:
        compare(args.component)


if __name__ == "__main__":
    main()
