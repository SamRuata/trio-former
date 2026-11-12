"""Unit tests for the hand-written transformer block.

The four the brief names: output shape, attention rows summing to 1, gradient
flow end to end, and permutation equivariance only without positional
encoding. The rest check the properties the module's docstrings claim, so the
documentation cannot drift away from the code.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.transformer import (LearnedPositionalEncoding, MultiHeadAttention,
                             TransformerEncoderBlock, apply_rotary,
                             positional_encoding_study, rotary_cache,
                             scaled_dot_product_attention,
                             sinusoidal_positional_encoding)

B, T, D, H = 2, 12, 64, 4
DH = D // H


@pytest.fixture
def block():
    torch.manual_seed(0)
    return TransformerEncoderBlock(D, H)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, D)


# ---------------------------------------------------------------------------
# (a) output shape
# ---------------------------------------------------------------------------
def test_output_shape_matches_input(block, x):
    """Shape preservation is what lets blocks stack."""
    y, attn = block(x)
    assert y.shape == x.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)


@pytest.mark.parametrize("seq_len", [1, 3, 37, 128])
def test_shape_holds_at_any_sequence_length(block, seq_len):
    y, attn = block(torch.randn(B, seq_len, D))
    assert y.shape == (B, seq_len, D)
    assert attn.shape == (B, H, seq_len, seq_len)


# ---------------------------------------------------------------------------
# (b) attention weights sum to 1
# ---------------------------------------------------------------------------
def test_attention_rows_sum_to_one(block, x):
    """Along the KEY axis: each query distributes exactly one unit of weight."""
    _, attn = block(x)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)
    assert (attn >= 0).all()


def test_attention_does_not_sum_to_one_along_the_query_axis(block, x):
    """Guards against a transposed softmax, which still looks plausible."""
    _, attn = block(x)
    column_sums = attn.sum(dim=-2)
    assert not torch.allclose(column_sums, torch.ones(B, H, T), atol=1e-3)


def test_masked_keys_receive_no_weight():
    torch.manual_seed(0)
    attn_layer = MultiHeadAttention(D, H)
    x = torch.randn(1, 6, D)
    mask = torch.tensor([[True, True, True, False, False, False]])
    _, weights = attn_layer(x, mask)

    assert float(weights[..., 3:].sum()) < 1e-6
    assert torch.allclose(weights[..., :3].sum(-1), torch.ones(1, H, 6), atol=1e-6)


def test_matches_torch_reference_attention():
    """Our scaled_dot_product_attention against PyTorch's own."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, H, T, DH) for _ in range(3))

    ours, _ = scaled_dot_product_attention(q, k, v)
    theirs = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, theirs, atol=1e-5)


# ---------------------------------------------------------------------------
# (c) gradient flow
# ---------------------------------------------------------------------------
def test_gradient_reaches_every_parameter(block, x):
    """A parameter with no gradient is a parameter that is not being trained."""
    y, _ = block(x)
    y.sum().backward()

    missing = [name for name, p in block.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"

    dead = [name for name, p in block.named_parameters()
            if float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"gradient reached but was exactly zero: {dead}"


def test_gradient_reaches_the_input(block, x):
    """End to end: the block is differentiable with respect to what it was given."""
    x = x.clone().requires_grad_(True)
    y, _ = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0


def test_gradients_are_finite(block, x):
    y, _ = block(x)
    y.pow(2).mean().backward()
    for name, p in block.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


# ---------------------------------------------------------------------------
# (d) permutation equivariance, only without positional encoding
# ---------------------------------------------------------------------------
def test_permutation_equivariant_without_positional_encoding(block, x):
    """Attention averages, and an average does not care about order.

    Shuffle the tokens and the outputs shuffle with them, unchanged. This is
    not a quirk -- it is why positional information has to be injected, and
    the reason the next two tests expect the opposite.
    """
    perm = torch.randperm(T)
    y_then_shuffle, _ = block(x)
    y_shuffle_then, _ = block(x[:, perm])
    assert torch.allclose(y_then_shuffle[:, perm], y_shuffle_then, atol=1e-5)


def test_not_equivariant_once_sinusoidal_is_added(block, x):
    """Adding absolute position breaks it, which is the point of adding it."""
    pe = sinusoidal_positional_encoding(T, D).unsqueeze(0)
    perm = torch.randperm(T)

    y_then_shuffle, _ = block(x + pe)
    y_shuffle_then, _ = block(x[:, perm] + pe)
    assert not torch.allclose(y_then_shuffle[:, perm], y_shuffle_then, atol=1e-4)


def test_not_equivariant_with_rotary(x):
    """Rotary breaks it from inside attention rather than at the input."""
    torch.manual_seed(0)
    rotary_block = TransformerEncoderBlock(D, H, position_mode="rotary")
    perm = torch.randperm(T)

    y_then_shuffle, _ = rotary_block(x)
    y_shuffle_then, _ = rotary_block(x[:, perm])
    assert not torch.allclose(y_then_shuffle[:, perm], y_shuffle_then, atol=1e-4)


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------
def test_sinusoidal_shape_and_range():
    pe = sinusoidal_positional_encoding(50, 32)
    assert pe.shape == (50, 32)
    assert float(pe.abs().max()) <= 1.0
    assert not torch.allclose(pe[0], pe[1])


def test_sinusoidal_is_deterministic_and_extrapolates():
    """No parameters, so position 10_000 is as available as position 3."""
    a = sinusoidal_positional_encoding(16, 32)
    b = sinusoidal_positional_encoding(10_016, 32)[:16]
    assert torch.allclose(a, b, atol=1e-6)


def test_sinusoidal_rejects_odd_width():
    with pytest.raises(ValueError):
        sinusoidal_positional_encoding(8, 33)


def test_learned_encoding_cannot_extrapolate():
    """The trade-off against sinusoidal, asserted rather than described."""
    enc = LearnedPositionalEncoding(max_len=16, d_model=32)
    assert enc(16).shape == (16, 32)
    with pytest.raises(ValueError):
        enc(17)


def test_rotary_score_depends_only_on_relative_position():
    r"""The identity (R_m q).(R_n k) = q^T R_(n-m) k, checked numerically.

    The same query and key, the same distance apart, scored at five different
    absolute positions. Under rotary the five scores are identical; under an
    additive encoding they are not. This is the measurement the README's
    positional-encoding table cites.
    """
    study = positional_encoding_study()
    assert study["rotary_spread"] < 1e-5
    assert study["sinusoidal_spread"] > 1e-2


def test_rotary_preserves_norm():
    """A rotation changes direction, not length."""
    torch.manual_seed(0)
    x = torch.randn(1, H, T, DH)
    cos, sin = rotary_cache(T, DH)
    assert torch.allclose(apply_rotary(x, cos, sin).norm(dim=-1),
                          x.norm(dim=-1), atol=1e-5)


def test_rotary_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        rotary_cache(8, 15)


# ---------------------------------------------------------------------------
# Multi-head structure
# ---------------------------------------------------------------------------
def test_heads_must_divide_the_model_width():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=64, n_heads=5)


def test_split_and_merge_are_inverse():
    """The 'concatenation' in the formula is this reshape, and it loses nothing."""
    attn = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    assert torch.allclose(attn._merge(attn._split(x)), x)


def test_head_count_does_not_change_parameter_count():
    """Splitting into heads buys parallel attention patterns for free."""
    counts = {h: sum(p.numel() for p in MultiHeadAttention(D, h).parameters())
              for h in (1, 2, 4, 8)}
    assert len(set(counts.values())) == 1, counts


def test_heads_learn_different_patterns():
    """With random weights the heads should not already agree."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(D, H)
    _, weights = attn(torch.randn(1, T, D))
    head0, head1 = weights[0, 0], weights[0, 1]
    assert not torch.allclose(head0, head1, atol=1e-3)


# ---------------------------------------------------------------------------
# Block wiring
# ---------------------------------------------------------------------------
def test_block_is_pre_norm():
    r"""The residual path must carry no LayerNorm.

    Pre-norm computes ``x + Attn(LN(x))``, so with the sublayers zeroed the
    block is exactly the identity. Post-norm computes ``LN(x + Attn(x))``,
    which normalizes even when the sublayers contribute nothing -- so this
    test fails if the block is ever rewired.
    """
    torch.manual_seed(0)
    b = TransformerEncoderBlock(D, H)
    with torch.no_grad():
        b.attn.out_proj.weight.zero_()
        b.attn.out_proj.bias.zero_()
        b.ffn.net[-1].weight.zero_()
        b.ffn.net[-1].bias.zero_()

    x = torch.randn(B, T, D)
    y, _ = b(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_dropout_is_inactive_in_eval():
    torch.manual_seed(0)
    b = TransformerEncoderBlock(D, H, dropout=0.5).eval()
    x = torch.randn(B, T, D)
    a, _ = b(x)
    c, _ = b(x)
    assert torch.allclose(a, c)