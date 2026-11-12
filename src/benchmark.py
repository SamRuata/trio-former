"""Benchmark the hand-written block against torch.nn.TransformerEncoderLayer.

    python -m src.benchmark

Three things are produced:

  1. A parity check. Both blocks are given byte-identical weights and the same
     input; their outputs must agree to floating-point tolerance. This is the
     correctness proof -- a loss curve that merely *looks* similar proves very
     little, but two implementations agreeing to 1e-6 on every element proves
     the arithmetic is the same.
  2. Training-loss curves on a small synthetic task, from the same
     initialization and the same batch order.
  3. Single-step timing: forward, and forward plus backward.

Matched initialization is the whole point. torch's layer packs Q, K and V into
one `in_proj_weight` of shape [3D, D]; this module keeps them as three Linear
layers so the copy is a slice rather than a reinterpretation. The remaining
differences are configuration, not architecture: torch defaults to ReLU and
post-norm, so it is constructed with gelu and norm_first=True to match.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from src.transformer import TransformerEncoderBlock

OUT_DIR = Path("docs")

D_MODEL, N_HEADS, D_FF = 64, 4, 256
SEQ_LEN, BATCH, VOCAB = 16, 64, 2
STEPS, LR, SEED = 120, 1e-3, 42
TIMING_REPS, TIMING_WARMUP = 50, 10


# ---------------------------------------------------------------------------
# Matched initialization
# ---------------------------------------------------------------------------
@torch.no_grad()
def copy_weights(torch_layer: nn.TransformerEncoderLayer,
                 ours: TransformerEncoderBlock) -> None:
    """Copy every parameter from torch's layer into ours.

    torch stores the three input projections fused:

        in_proj_weight   [3D, D]   rows 0:D are Q, D:2D are K, 2D:3D are V
        in_proj_bias     [3D]

    so each of our three Linear layers takes one slice. Everything else maps
    one-to-one.
    """
    d = ours.attn.d_model
    w, b = torch_layer.self_attn.in_proj_weight, torch_layer.self_attn.in_proj_bias

    ours.attn.q_proj.weight.copy_(w[:d])
    ours.attn.k_proj.weight.copy_(w[d:2 * d])
    ours.attn.v_proj.weight.copy_(w[2 * d:])
    ours.attn.q_proj.bias.copy_(b[:d])
    ours.attn.k_proj.bias.copy_(b[d:2 * d])
    ours.attn.v_proj.bias.copy_(b[2 * d:])

    ours.attn.out_proj.weight.copy_(torch_layer.self_attn.out_proj.weight)
    ours.attn.out_proj.bias.copy_(torch_layer.self_attn.out_proj.bias)

    ours.ffn.net[0].weight.copy_(torch_layer.linear1.weight)
    ours.ffn.net[0].bias.copy_(torch_layer.linear1.bias)
    ours.ffn.net[3].weight.copy_(torch_layer.linear2.weight)
    ours.ffn.net[3].bias.copy_(torch_layer.linear2.bias)

    for mine, theirs in ((ours.norm1, torch_layer.norm1),
                         (ours.norm2, torch_layer.norm2)):
        mine.weight.copy_(theirs.weight)
        mine.bias.copy_(theirs.bias)


def build_pair() -> Tuple[nn.TransformerEncoderLayer, TransformerEncoderBlock]:
    torch.manual_seed(SEED)
    reference = nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    mine = TransformerEncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    copy_weights(reference, mine)
    return reference.eval(), mine.eval()


# ---------------------------------------------------------------------------
# 1. Parity
# ---------------------------------------------------------------------------
def parity(reference, mine) -> Dict[str, float]:
    torch.manual_seed(7)
    x = torch.randn(8, SEQ_LEN, D_MODEL)
    with torch.no_grad():
        a = reference(x)
        b, _ = mine(x)
    diff = (a - b).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_1e5": bool(torch.allclose(a, b, atol=1e-5)),
        "our_params": sum(p.numel() for p in mine.parameters()),
        "torch_params": sum(p.numel() for p in reference.parameters()),
    }


# ---------------------------------------------------------------------------
# 2. Training curves
# ---------------------------------------------------------------------------
class Classifier(nn.Module):
    """Embedding -> one encoder block -> mean pool -> linear.

    The task is majority vote: given a sequence of two symbols, is symbol 1 in
    the majority? It needs information combined across every position, which
    is what an attention block is for, and it is small enough to converge in
    about a hundred steps on a laptop.
    """

    def __init__(self, block: nn.Module, is_torch_layer: bool):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.block = block
        self.head = nn.Linear(D_MODEL, 2)
        self.is_torch_layer = is_torch_layer

    def forward(self, ids: Tensor) -> Tensor:
        h = self.embed(ids)                               # [B,T,D]
        h = self.block(h) if self.is_torch_layer else self.block(h)[0]
        return self.head(h.mean(dim=1))                   # [B,2]


def make_batch(generator: torch.Generator) -> Tuple[Tensor, Tensor]:
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), generator=generator)
    labels = (ids.sum(dim=1) * 2 > SEQ_LEN).long()
    return ids, labels


def train(model: nn.Module, tag: str) -> List[float]:
    """Train from a fixed seed on a fixed batch sequence. Returns the losses."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)

    losses = []
    for step in range(STEPS):
        ids, labels = make_batch(generator)
        optimizer.zero_grad()
        loss = loss_fn(model(ids), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if (step + 1) % 40 == 0:
            print(f"  {tag:<8} step {step + 1:>3}  loss {float(loss):.4f}")
    return losses


def matched_classifiers() -> Tuple[nn.Module, nn.Module]:
    """Two classifiers whose every parameter starts identical."""
    reference, mine = build_pair()
    torch.manual_seed(SEED + 1)
    a = Classifier(reference, is_torch_layer=True)
    torch.manual_seed(SEED + 1)          # same embedding and head
    b = Classifier(mine, is_torch_layer=False)
    with torch.no_grad():
        b.embed.weight.copy_(a.embed.weight)
        b.head.weight.copy_(a.head.weight)
        b.head.bias.copy_(a.head.bias)
    return a, b


# ---------------------------------------------------------------------------
# 3. Timing
# ---------------------------------------------------------------------------
def time_block(block: nn.Module, is_torch_layer: bool,
               backward: bool) -> Dict[str, float]:
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=backward)

    def once():
        out = block(x) if is_torch_layer else block(x)[0]
        if backward:
            out.sum().backward()

    for _ in range(TIMING_WARMUP):
        once()

    samples = []
    for _ in range(TIMING_REPS):
        start = time.perf_counter()
        once()
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "p95_ms": samples[int(0.95 * len(samples))],
    }


# ---------------------------------------------------------------------------
def plot(ours: List[float], theirs: List[float], timing: Dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_time) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    ax_loss.plot(theirs, color="#6b7280", linewidth=2.5, alpha=0.7,
                 label="torch.nn.TransformerEncoderLayer")
    ax_loss.plot(ours, color="#2563eb", linewidth=1.3,
                 label="hand-written block")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Training loss, matched initialization")
    ax_loss.legend(frameon=False, fontsize=8)
    ax_loss.grid(alpha=0.25, linewidth=0.5)
    ax_loss.spines[["top", "right"]].set_visible(False)

    labels = ["forward", "forward+backward"]
    theirs_ms = [timing["torch_forward"]["median_ms"],
                 timing["torch_fwd_bwd"]["median_ms"]]
    ours_ms = [timing["ours_forward"]["median_ms"],
               timing["ours_fwd_bwd"]["median_ms"]]
    pos = range(len(labels))
    ax_time.bar([p - 0.2 for p in pos], theirs_ms, width=0.38,
                color="#6b7280", label="torch.nn")
    ax_time.bar([p + 0.2 for p in pos], ours_ms, width=0.38,
                color="#2563eb", label="ours")
    ax_time.set_xticks(list(pos), labels)
    ax_time.set_ylabel("median ms per step")
    ax_time.set_title(f"Single-step timing (batch {BATCH}, T={SEQ_LEN})")
    ax_time.legend(frameon=False, fontsize=8)
    ax_time.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "benchmark.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def main() -> None:
    print("parity check")
    reference, mine = build_pair()
    p = parity(reference, mine)
    print(f"  parameters      ours {p['our_params']:,}  "
          f"torch {p['torch_params']:,}")
    print(f"  max |diff|      {p['max_abs_diff']:.3e}")
    print(f"  allclose(1e-5)  {p['allclose_1e5']}")

    print("\ntraining, matched initialization and batch order")
    torch_model, our_model = matched_classifiers()
    theirs = train(torch_model, "torch")
    ours = train(our_model, "ours")

    tail = lambda xs: sum(xs[-10:]) / 10
    print(f"\n  final loss (mean of last 10)   torch {tail(theirs):.4f}   "
          f"ours {tail(ours):.4f}   delta {abs(tail(ours) - tail(theirs)):.4f}")

    print("\ntiming")
    reference, mine = build_pair()
    timing = {
        "torch_forward": time_block(reference, True, backward=False),
        "ours_forward": time_block(mine, False, backward=False),
        "torch_fwd_bwd": time_block(reference, True, backward=True),
        "ours_fwd_bwd": time_block(mine, False, backward=True),
    }
    for name, t in timing.items():
        print(f"  {name:<16} median {t['median_ms']:.2f} ms   "
              f"min {t['min_ms']:.2f}   p95 {t['p95_ms']:.2f}")

    overhead = (timing["ours_forward"]["median_ms"]
                / timing["torch_forward"]["median_ms"])
    print(f"\n  forward overhead vs torch.nn: {overhead:.2f}x")

    path = plot(ours, theirs, timing)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps({
        "parity": p,
        "final_loss_torch": tail(theirs),
        "final_loss_ours": tail(ours),
        "timing": timing,
        "forward_overhead": overhead,
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                   "seq_len": SEQ_LEN, "batch": BATCH, "steps": STEPS,
                   "lr": LR, "seed": SEED},
    }, indent=1), encoding="utf-8")
    print(f"  wrote {path} and {OUT_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()