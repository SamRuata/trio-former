# Model Card — TrioFormer

**Service:** https://trio-former-api.onrender.com
**Owner:** Lal Ruata, CST-435
**Last updated:** November 2026

This service serves four models: three pretrained archetypes it did not train, and one encoder block written from scratch in this repository. They are documented together because the comparison between them is the product.

---

## 1. The hand-written block — `src/transformer.py`

The only model here whose weights and arithmetic are fully accounted for.

| | |
|---|---|
| Architecture | pre-norm encoder block: `x + Attn(LN(x))`, then `x + FFN(LN(x))` |
| Attention | single-head-capable multi-head scaled dot-product, written out, not `nn.MultiheadAttention` |
| Positional schemes | sinusoidal, learned, and rotary — all three implemented |
| Parameters | 49,984 at `d_model=64, n_heads=4, d_ff=256` |
| Training | **none.** Served with freshly initialised weights from a fixed seed |
| Endpoint | `POST /encoder_attention` |

**Verified against `torch.nn.TransformerEncoderLayer`** under matched initialization — weights copied slice-for-slice, then both trained on the same batch order:

| | hand-written | `torch.nn` |
|---|---|---|
| Parameters | 49,984 | 49,984 |
| Max absolute output difference | **4.77e-07** | — |
| Loss @ step 40 / 80 / 120 | 0.0343 / 0.0021 / 0.0006 | 0.0343 / 0.0021 / 0.0006 |
| Forward, median | 2.68 ms | 2.42 ms |
| Forward + backward, median | 7.85 ms | 7.99 ms |

Identical to floating-point noise, not merely comparable. CI re-runs this check on every push and fails the build on divergence.

**What `/encoder_attention` is and is not.** The block has never been trained. Token embeddings are drawn deterministically from each token's SHA-256, so the same sentence always produces the same heatmap, but those vectors carry no meaning. The endpoint shows the *structure* attention computes — that every position attends to every other, that each row sums to one, that rotary changes the pattern and sinusoidal does not change it the same way. It shows nothing about language. Any interface built on it should say so.

**Measured properties**, asserted in `tests/test_transformer.py`:

- Output shape equals input shape at every sequence length tested.
- Attention rows sum to 1 along the key axis; column sums do not, which catches a transposed softmax.
- Gradients reach every parameter and the input, and are finite.
- The block is permutation-equivariant **without** positional encoding, and is not once sinusoidal or rotary is applied.
- Zeroing both sublayers reduces the block to exactly the identity — true of pre-norm, false of post-norm.
- Head count does not change parameter count.
- Rotary preserves vector norm and encodes only relative position: one query/key pair held 8 apart and scored at five absolute positions gives a score spread of **0.00e+00**, against **1.38** for additive sinusoidal.

---

## 2. The three pretrained archetypes

**The deployed checkpoints are not the ones this project was developed against.** Both sets are documented because the substitution is a finding, not an embarrassment.

| archetype | endpoint | developed locally with | **deployed on Render with** |
|---|---|---|---|
| encoder | `/bert_fillmask` | `distilbert-base-uncased` (66M) | `prajjwal1/bert-mini` (11M) |
| decoder | `/gpt2_continue` | `distilgpt2` (82M) | `sshleifer/tiny-gpt2` (~2M) |
| encoder–decoder | `/t5_summarize` | `t5-small` (60M) | `t5-small` (60M) |

### Why the substitution

Render's free tier is **512 MB of RAM and 0.1 CPU**. PyTorch alone is about 250 MB resident. The service keeps one model loaded at a time and evicts it when another archetype is requested, and even so:

- `distilbert-base-uncased` loaded and served exactly one request, then the container was killed on the next.
- `torch_dtype=bfloat16` made it worse, not better — transformers materialises the float32 weights *then* casts, so the load spike is fp32 + bf16 + torch at once. `low_cpu_mem_usage=True` fixed the spike and the steady state still did not hold.
- `google/t5-efficient-tiny` fit easily and returned **empty strings**. It is a pretrained-only checkpoint that was never fine-tuned for summarization, so it emits end-of-sequence immediately. The architecture does not give you the task; the supervised training does.

Model ids are environment variables precisely so this swap is a dashboard edit rather than a commit.

### Performance

Measured server-side around the whole request path, logged to Supabase, aggregated by the `benchmarks` materialized view, and separated by environment because the same checkpoint on a laptop and on 0.1 CPU are not the same measurement.

**Local** — real checkpoints, developer laptop:

| archetype | model | warm mean | cold mean |
|---|---|---|---|
| encoder | `distilbert-base-uncased` | **32 ms** | ~900 ms |
| decoder | `distilgpt2` (40 tokens) | **897 ms** | ~1.8 s |
| encoder–decoder | `t5-small` (60 tokens) | **1374 ms** | ~1.8 s |

**Render** — reduced checkpoints, free tier:

| archetype | model | warm mean | cold mean | of which load |
|---|---|---|---|---|
| encoder | `prajjwal1/bert-mini` | **18 ms** | 35.8 s | 29.6 s |
| decoder | `sshleifer/tiny-gpt2` (20 tokens) | **384 ms** | 6.6 s | 6.1 s |
| encoder–decoder | `t5-small` (16 tokens) | **51,482 ms** | 68.6 s | 14.9 s |

Two things read straight off those numbers. The encoder's cost is one forward pass and survives anything — 18 ms on a tenth of a CPU. The generative archetypes cost one forward pass **per token**, sequentially, and that is what the free tier cannot absorb: cutting T5's output from 40 tokens to 16 took it from 137 s to 51 s, almost exactly linearly.

### Known failure modes

- **`sshleifer/tiny-gpt2` is a test fixture, not a language model.** It is a randomly-initialised two-million-parameter GPT-2 published for unit-testing the library. Its output is noise. The *decoder archetype* is being demonstrated honestly — causal masking, per-token cost — but nothing it writes means anything.
- **`distilbert-base-uncased` says the capital of France is `marseille`** at 0.143, with `paris` fourth at 0.086. A factual error on something a child knows, delivered with a confidence number. `prajjwal1/bert-mini`, being eleven times smaller, is worse.
- **T5 at 16 output tokens truncates mid-thought.** The cap is a hosting constraint, not an editorial choice.
- **No abstain path.** Every endpoint returns an answer. A paragraph with no masked token is rejected at validation; a paragraph the model has no business judging is not.
- **Confidence figures are softmax outputs, not calibrated frequencies.** 0.143 does not mean 14.3% of such answers are right.
- **The first request after idle costs 50 seconds or more**, before any model load, because free instances spin down.

---

## Intended use

A teaching demo for a learning-and-development briefing: show that an encoder, a decoder and an encoder–decoder answer different questions at very different costs, with the latencies recorded rather than asserted.

**Out of scope.** Anything consequential, and anything that depends on the *content* of what these models produce. The deployed checkpoints are reduced to the point where only the architectural behaviour is meaningful. Use the latency numbers; do not use the summaries.

---

## Privacy: what is and is not stored

**Never written to the database:** the text of any paragraph submitted to the service. Only its SHA-256, character count and word count.

**Written to the database:** which checkpoint served the call, its archetype and task, the model's output, `latency_ms`, `load_ms`, `cold_start`, the environment, and a timestamp.

Nothing identifies who submitted anything — there are no user ids, sessions or addresses in the schema.

**The caveat that matters.** Outputs are stored, and a T5 summary is a lossy copy of the paragraph it summarised. Hashing the input does **not** make that row non-identifying. BERT's output is reduced to candidate tokens and scores before storage, and GPT-2's prompt is stripped so only generated text is kept, but a summarizer cannot be sanitised that way without destroying the thing being logged. A deployment that told users "we only keep a hash" would be saying something false.

This is enforced where it can be: a test fails if any string in `inputs` exceeds 64 characters or contains a space, on the grounds that a hash does neither.

---

## Ethical considerations

Three industrial pretrained models are four lines of code here. I did not train them, cannot inspect them, and could not reproduce them. Everything this service knows about language came from somewhere I cannot audit, and the one model I *can* account for completely is the one that knows nothing.

That asymmetry is the point of the product and the subject of the report. The free tier made it sharper than intended: a service can be stood up in an afternoon, and the thing that stops it is not understanding, or permission, or review — it is 512 megabytes.

The fuller argument, including what the lowered bar demands in return, is in the Engineering Report section of [`README.md`](README.md).