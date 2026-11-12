"""A transformer encoder block written from scratch, with shapes spelled out.

Every tensor is annotated with its shape in the comments, using:

    B   batch
    T   sequence length (tokens)
    D   model width (d_model)
    H   number of heads
    Dh  per-head width, D // H
    F   feed-forward width

Three positional schemes are implemented rather than described, because the
comparison between them is the point: sinusoidal and learned encodings are
ADDED to the input and therefore carry absolute position, while rotary rotates
the query and key inside attention and therefore makes a score depend only on
the distance between two tokens. `positional_encoding_study` measures that
difference instead of asserting it.

The block is pre-norm (`norm -> sublayer -> add`), which is what makes deep
transformers trainable without a learning-rate warmup: the residual path from
input to output carries no normalization, so gradients reach the early layers
undiminished. Post-norm, the original 2017 arrangement, needs warmup to avoid
diverging.
"""
from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

PositionMode = Literal["none", "rotary"]


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
def sinusoidal_positional_encoding(seq_len: int, d_model: int,
                                   base: float = 10_000.0,
                                   device: Optional[torch.device] = None) -> Tensor:
    r"""Fixed sinusoidal position signal. Returns ``[T, D]``.

        PE(pos, 2i)   = sin( pos / base^(2i/D) )
        PE(pos, 2i+1) = cos( pos / base^(2i/D) )

    Each pair of channels is a rotating vector whose angular frequency falls
    geometrically with depth, so the low channels distinguish nearby positions
    and the high channels distinguish distant ones. Nothing is learned, so it
    extrapolates to sequences longer than anything seen in training -- which a
    learned table cannot do.

    It is still ABSOLUTE: the encoding for position 7 is a fixed vector, and
    the model has to learn that 7 relates to 9 the way 107 relates to 109.
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)           # [D/2]
    inv_freq = torch.exp(-math.log(base) * i / d_model)                           # [D/2]

    pe = torch.zeros(seq_len, d_model, device=device)                             # [T,D]
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)
    return pe


class LearnedPositionalEncoding(nn.Module):
    """A lookup table of position vectors, trained with everything else.

    Strictly more expressive than the sinusoid at any position it has seen,
    and useless past ``max_len`` -- there is no row to look up. This is the
    trade the comparison table in the README turns on.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.table = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, seq_len: int) -> Tensor:                    # -> [T,D]
        if seq_len > self.max_len:
            raise ValueError(
                f"learned encoding has {self.max_len} positions, asked for "
                f"{seq_len}; this is the extrapolation failure sinusoidal and "
                f"rotary encodings do not have")
        idx = torch.arange(seq_len, device=self.table.weight.device)
        return self.table(idx)


def rotary_cache(seq_len: int, head_dim: int, base: float = 10_000.0,
                 device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """Precompute the rotation angles for RoPE. Returns ``(cos, sin)``, each ``[T, Dh/2]``."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for rotary, got {head_dim}")

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=torch.float32) / head_dim))   # [Dh/2]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)               # [T]
    freqs = torch.outer(t, inv_freq)                                            # [T,Dh/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each adjacent channel pair of ``x`` by its position's angle.

    ``x`` is ``[B, H, T, Dh]``; ``cos`` and ``sin`` are ``[T, Dh/2]``.
    Returns ``[B, H, T, Dh]``.

    Treating channels (2i, 2i+1) as a 2-D vector and rotating it by angle
    ``pos * theta_i`` gives the identity that makes rotary different in kind
    from an additive encoding:

        (R_m q) . (R_n k) = q^T R_(n-m) k

    The score between two tokens depends on ``n - m`` alone. Absolute position
    cancels, so a pair ten tokens apart looks the same at the start of a
    document as at the end -- which is why rotary extrapolates past the
    training length where a learned table cannot, and holds relative structure
    the sinusoid only approximates.
    """
    x1 = x[..., 0::2]                                  # [B,H,T,Dh/2]
    x2 = x[..., 1::2]                                  # [B,H,T,Dh/2]
    cos = cos.unsqueeze(0).unsqueeze(0)                # [1,1,T,Dh/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.stack((x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos), dim=-1)   # [B,H,T,Dh/2,2]
    return rotated.flatten(-2)                             # [B,H,T,Dh]


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""One attention operation. Returns ``(context, weights)``.

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Shapes, with any number of leading batch-like dimensions:

        q        [..., T_q, Dh]
        k, v     [..., T_k, Dh]
        mask     [..., T_q, T_k] or broadcastable; True means "may attend"
        context  [..., T_q, Dh]
        weights  [..., T_q, T_k]   each row sums to 1

    **The divisor.** With unit-variance components, ``q . k`` is a sum of
    ``d_k`` products, so its standard deviation is ``sqrt(d_k)``. At d_k = 64
    the raw scores spread over roughly +/- 8; a softmax over scores that wide
    saturates to a one-hot and the gradient through every losing position
    vanishes. Dividing restores unit variance.

    Masked positions are filled with the dtype minimum BEFORE the softmax, not
    zeroed after it. Zeroing afterwards leaves the surviving weights summing to
    less than one, silently rescaling the context vector.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # [...,T_q,T_k]

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)                     # [...,T_q,T_k]
    context = weights @ v                                       # [...,T_q,Dh]
    return context, weights


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    r"""H scaled dot-product attentions in parallel, concatenated.

        head_i      = Attention(Q W_i^Q, K W_i^K, V W_i^V)
        MultiHead   = Concat(head_1, ..., head_H) W^O

    **Shape walk-through**, the part worth being explicit about. The heads are
    not H separate projections in a Python loop; they are one projection of
    width D, reshaped. For input ``x`` of ``[B, T, D]`` with H heads and
    ``Dh = D / H``:

        x                      [B, T, D]
        q = x W_Q              [B, T, D]
        reshape                [B, T, H, Dh]
        transpose              [B, H, T, Dh]     <- H independent attentions
        scores = q k^T / sqrt  [B, H, T, T]
        context = weights @ v  [B, H, T, Dh]
        transpose              [B, T, H, Dh]
        reshape  (concat)      [B, T, D]         <- the concatenation IS the reshape
        out = . W_O            [B, T, D]

    The concatenation in the formula and the `reshape` in the code are the
    same operation: laying H vectors of width Dh end to end gives one of width
    D. Nothing is copied.

    Each head sees the full width D on the way in and gets Dh of it on the way
    out, so total compute is independent of H -- splitting into heads buys
    several attention patterns over the same sequence for free, which is why
    one head can track syntax while another tracks coreference.

    Projections are kept as three separate ``Linear`` layers rather than one
    fused ``in_proj`` so that weights can be copied slice-for-slice from
    ``torch.nn.MultiheadAttention`` for the matched-initialization benchmark.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 position_mode: PositionMode = "none", bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} must divide by n_heads {n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.position_mode = position_mode

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: Tensor) -> Tensor:
        """[B,T,D] -> [B,H,T,Dh]"""
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        """[B,H,T,Dh] -> [B,T,D]. This reshape is the concatenation."""
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """``x`` ``[B,T,D]`` -> ``(output [B,T,D], weights [B,H,T,T])``."""
        q = self._split(self.q_proj(x))                        # [B,H,T,Dh]
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        if self.position_mode == "rotary":
            cos, sin = rotary_cache(x.size(1), self.head_dim, device=x.device)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        attn_mask = None
        if mask is not None:
            # [B,T] key mask -> [B,1,1,T], broadcast across heads and queries.
            attn_mask = mask[:, None, None, :]

        context, weights = scaled_dot_product_attention(q, k, v, attn_mask)
        out = self.out_proj(self._merge(self.dropout(context)))  # [B,T,D]
        return out, weights


# ---------------------------------------------------------------------------
# Feed-forward and the block
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise MLP: ``D -> F -> D``, applied identically at every position.

    Attention moves information between positions; this is the only place a
    position's own representation is transformed non-linearly. F is
    conventionally 4D, which is where most of a transformer's parameters live.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),       # [B,T,D] -> [B,T,F]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),       # [B,T,F] -> [B,T,D]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    r"""One pre-norm encoder block.

        h = x + Attn(LN(x))
        y = h + FFN(LN(h))

    Pre-norm, not the 2017 post-norm (``LN(x + Attn(x))``). The difference is
    the residual path: here it runs from input to output with no normalization
    on it, so a gradient reaches the first layer at full size. Post-norm puts a
    LayerNorm on that path at every block, which is why the original
    transformers needed a learning-rate warmup and why essentially every model
    since 2020 is pre-norm.

    Input and output are both ``[B, T, D]`` -- the block is shape-preserving,
    which is what lets blocks stack.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None,
                 dropout: float = 0.0, position_mode: PositionMode = "none",
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, position_mode)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """``x`` ``[B,T,D]`` -> ``(output [B,T,D], attention [B,H,T,T])``."""
        attended, weights = self.attn(self.norm1(x), mask)
        x = x + self.drop1(attended)                 # residual 1
        x = x + self.drop2(self.ffn(self.norm2(x)))  # residual 2
        return x, weights


# ---------------------------------------------------------------------------
# Measurement, for the README's comparison table
# ---------------------------------------------------------------------------
def positional_encoding_study(d_model: int = 64, n_heads: int = 4,
                              seq_len: int = 64,
                              seed: int = 0) -> Dict[str, object]:
    """Measure the property that separates the three schemes.

    Takes one fixed query/key pair and scores it at several absolute positions
    while holding the DISTANCE between them constant. Under rotary the score is
    the same every time; under an additive encoding it is not. Returns the
    spread of those scores, which is the number the README cites.
    """
    torch.manual_seed(seed)
    head_dim = d_model // n_heads
    q0 = torch.randn(1, 1, 1, head_dim)
    k0 = torch.randn(1, 1, 1, head_dim)
    distance = 8
    starts = [0, 8, 16, 32, 48]

    # --- rotary: rotate each by its absolute position, then score ----------
    cos, sin = rotary_cache(seq_len, head_dim)
    rotary_scores = []
    for s in starts:
        qr = apply_rotary(q0, cos[s:s + 1], sin[s:s + 1])
        kr = apply_rotary(k0, cos[s + distance:s + distance + 1],
                          sin[s + distance:s + distance + 1])
        rotary_scores.append(float((qr * kr).sum()) / math.sqrt(head_dim))

    # --- sinusoidal: add the encoding to each, then score ------------------
    pe = sinusoidal_positional_encoding(seq_len, head_dim)
    sin_scores = []
    for s in starts:
        qs = q0.squeeze() + pe[s]
        ks = k0.squeeze() + pe[s + distance]
        sin_scores.append(float((qs * ks).sum()) / math.sqrt(head_dim))

    def spread(xs):
        return max(xs) - min(xs)

    return {
        "distance": distance,
        "start_positions": starts,
        "rotary_scores": [round(s, 6) for s in rotary_scores],
        "rotary_spread": round(spread(rotary_scores), 6),
        "sinusoidal_scores": [round(s, 6) for s in sin_scores],
        "sinusoidal_spread": round(spread(sin_scores), 6),
        "note": ("Same distance, different absolute positions. Rotary scores "
                 "are identical; additive-sinusoidal scores are not."),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H = 2, 10, 64, 4

    block = TransformerEncoderBlock(D, H)
    x = torch.randn(B, T, D)
    y, attn = block(x)

    print(f"input   {tuple(x.shape)}")
    print(f"output  {tuple(y.shape)}   shape preserved: {y.shape == x.shape}")
    print(f"attn    {tuple(attn.shape)}   [B, H, T, T]")
    print(f"rows sum to 1: {torch.allclose(attn.sum(-1), torch.ones(B, H, T), atol=1e-6)}")
    print(f"parameters: {sum(p.numel() for p in block.parameters()):,}")

    study = positional_encoding_study()
    print(f"\nscoring one q/k pair {study['distance']} apart, at starts "
          f"{study['start_positions']}:")
    print(f"  rotary       spread {study['rotary_spread']:.2e}  "
          f"{study['rotary_scores'][:3]}")
    print(f"  sinusoidal   spread {study['sinusoidal_spread']:.2e}  "
          f"{[round(s, 3) for s in study['sinusoidal_scores'][:3]]}")