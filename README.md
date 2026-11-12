# TrioFormer — A Side-by-Side Transformer Inference Service

**Name:** Lal Ruata
**Course / Section:** CST-435 
**GitHub:** https://github.com/SamRuata/trio-former
**Render.com (API):** https://trio-former-api.onrender.com — docs at [`/docs`](https://trio-former-api.onrender.com/docs), health at [`/healthz`](https://trio-former-api.onrender.com/healthz)
**Supabase project ref:** `nkopsjdeqfgqfwozqgsk`
**Product presentation (video):** [FILL AFTER RECORDING]
**Engineering report:** [below, in this README](#engineering-report)
**Model card:** [`MODEL_CARD.md`](MODEL_CARD.md)

> The customer-facing interface is **not** part of this product. It is built separately as Product 5B ("TrioDeck") against this service, so this API is the deliverable: reachable, documented at `/docs`, and stable enough for a classmate's interface to call.

> The Render API runs on the free tier and spins down when idle. The first request after a quiet period can take 50 seconds or more.

---

## Product pitch

A learning-and-development team is briefing engineering leadership on *what kind of transformer fits which job*. Slide decks about "encoders versus decoders" do not settle that argument; numbers do.

TrioFormer takes one paragraph and runs it through all three archetypes at once — BERT fills a masked token, GPT-2 continues the text, T5 summarizes it — and records what each one cost. Every call writes its server-side latency to Postgres, and a materialized view aggregates by model, so the recommendation the team eventually makes is built from measurements of their own traffic rather than from vendor claims.

A fourth endpoint, `/encoder_attention`, runs a transformer encoder block written from scratch in this repository and returns its attention matrices. It exists so the briefing can show what attention actually computes, in code the team can read, beside what three industrial models do with the same words.

**Customer story.** *As a learning-and-development lead, I want one paragraph scored by an encoder, a decoder and an encoder-decoder with their latencies recorded, so that I can tell engineering leadership which architecture fits which job and what each one costs us.*

No paragraph submitted to the service is stored. Only its SHA-256.

---

## Architecture

```
┌────────────────────────────┐       ┌──────────────────────────────┐
│  Product 5B — TrioDeck     │       │  FastAPI on Render.com       │
│  (built separately, by     │ ────▶ │  api/main.py                 │
│   each student, against    │ HTTPS │                              │
│   this service)            │ ◀──── │  ┌────────────────────────┐  │
└────────────────────────────┘ JSON  │  │ one-slot registry      │  │
                                     │  │ 512 MB: only ONE model │  │
                                     │  │ resident at a time     │  │
                                     │  └────────────────────────┘  │
                                     │   BERT    encoder            │
                                     │   GPT-2   decoder            │
                                     │   T5      encoder-decoder    │
                                     │   src/transformer.py  (ours) │
                                     └──────────────┬───────────────┘
                                                    │ service-role
                                                    ▼
                           ┌────────────────────────────────────────┐
                           │  Supabase Postgres                     │
                           │                                        │
                           │  inputs      sha256, n_chars, n_words  │
                           │              (never the text)          │
                           │  inferences  model, output, latency_ms,│
                           │              load_ms, cold_start       │
                           │       │                                │
                           │       ▼  aggregated by model           │
                           │  benchmarks  (materialized view)       │
                           │       └─▶ GET /benchmarks              │
                           └────────────────────────────────────────┘
```

The data layer is where the cost story lives. A latency printed in a response is gone the moment the tab closes; a latency in `inferences` accumulates, and the `benchmarks` view turns a week of real traffic into the table a recommendation can rest on.

---

## The hand-written block

`src/transformer.py` implements scaled dot-product attention, multi-head attention, three positional schemes, and a pre-norm encoder block. Every tensor is annotated with its shape.

### Multi-head attention as parallel scaled dot-products

A single attention is

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

and multi-head attention is **H of those in parallel, concatenated**:

```
head_i    = Attention(Q W_i^Q, K W_i^K, V W_i^V)
MultiHead = Concat(head_1, …, head_H) W^O
```

The shape walk, for `[B, T, D]` with H heads and `Dh = D / H`:

| step | shape | note |
|---|---|---|
| `x` | `[B, T, D]` | |
| `q = x W_Q` | `[B, T, D]` | one projection, not H |
| reshape | `[B, T, H, Dh]` | |
| transpose | `[B, H, T, Dh]` | H independent attentions |
| `q kᵀ / √d_k` | `[B, H, T, T]` | the heatmap |
| `weights @ v` | `[B, H, T, Dh]` | |
| transpose, reshape | `[B, T, D]` | **this reshape *is* the concatenation** |
| `· W_O` | `[B, T, D]` | shape preserved, so blocks stack |

The concatenation in the formula and the reshape in the code are the same operation — laying H vectors of width `Dh` end to end gives one of width `D`. Nothing is copied, and total compute is independent of H, so splitting into heads buys several attention patterns over the same sequence for free.

**Why √d_k.** With zero-mean, unit-variance components, `q·k` is a sum of `d_k` products, so `σ(q·k) = √d_k`. At `d_k = 128` the raw scores spread over roughly ±11; a softmax over scores that wide collapses to a one-hot and the gradient through every losing position vanishes. Measured on this implementation: dot-product σ falls from **11.39 to 1.01**, the largest softmax weight from **0.611 to 0.108**, and entropy rises from **0.67 to 3.69 nats** against a uniform ceiling of 4.16. `GET /scaling_demo` recomputes these live.

### Positional encodings: sinusoidal, learned, rotary

Self-attention is permutation-equivariant — a unit test in `tests/test_transformer.py` shuffles the input and shows the output shuffles with it, unchanged. Position has to be injected, and *how* is the live question.

| | Sinusoidal | Learned | Rotary (RoPE) |
|---|---|---|---|
| Where it acts | added to the input | added to the input | rotates Q and K inside attention |
| Parameters | none | `max_len × d_model` | none |
| Position encoded | absolute | absolute | **relative** |
| Past training length | extrapolates | **impossible** — no row to look up | extrapolates |
| Score for a fixed pair, same distance, different absolute positions | varies | varies | **identical** |
| Implemented here | yes | yes | yes |

The last row is the argument, and it is measured rather than asserted. Take one query/key pair, hold them eight positions apart, and score them starting at positions 0, 8, 16, 32 and 48. Under rotary the five scores have a spread of **0.00e+00**; under additive sinusoidal the spread is **1.38**. Rotary satisfies

```
(R_m q) · (R_n k) = qᵀ R_(n−m) k
```

so absolute position cancels and only the distance survives. A pair ten tokens apart looks the same at the start of a document as at the end. That is why rotary dominates modern LLMs: it gives relative structure for free *and* extrapolates past the training length, which learned tables cannot do at all and sinusoids only approximate.

### Verified against `torch.nn.TransformerEncoderLayer`

Weights are copied slice-for-slice from torch's layer (it packs Q, K and V into one `in_proj_weight` of `[3D, D]`; this implementation keeps three `Linear` layers so the copy is a slice), then both are trained from the same initialization on the same batch order.

| | hand-written | `torch.nn` |
|---|---|---|
| Parameters | 49,984 | 49,984 |
| Max abs. output difference | **4.77e-07** | — |
| Loss @ step 40 / 80 / 120 | 0.0343 / 0.0021 / 0.0006 | 0.0343 / 0.0021 / 0.0006 |
| Forward, median | 2.68 ms | 2.42 ms |
| Forward+backward, median | 7.85 ms | 7.99 ms |

Not merely comparable — the same arithmetic to floating-point noise, and the loss curves are identical at every checkpoint. Forward is 1.11× torch's; forward+backward came out marginally faster, which is within run-to-run noise and is not claimed as a win. `docs/benchmark.png` plots both. CI re-runs the parity check on every push and fails the build if the block ever diverges.

---

## Archetype recommendation matrix

| | **Encoder** (BERT) | **Decoder** (GPT-2) | **Encoder–decoder** (T5) |
|---|---|---|---|
| Attention | bidirectional | causal, `t` sees only `≤ t` | bidirectional encode, causal decode + cross-attention |
| Native task | understand a fixed input | continue a sequence | transduce input → different output |
| Output shape | a label or a token | open-ended text | open-ended text, grounded in the source |
| Cost driver | one forward pass | one pass **per generated token** | encoder pass **+** one pass per output token |
| Warm latency here | **32 ms** | **897 ms** | **1374 ms** |
| Choose it for | classification, routing, retrieval, extraction at volume | drafting, completion, chat | summarization, translation, reformatting |
| Avoid it for | anything that must generate | anything needing the right-hand context | anything a single pass could answer |

---

## Data persisted / not persisted

**Persisted**

- `inputs` — one row per distinct paragraph: its SHA-256, character count, word count, first-seen timestamp. The hash lets the same paragraph be recognised across models, which is what makes a fair latency comparison possible.
- `inferences` — one row per call: model id, archetype, task, the model's output, `latency_ms`, `load_ms`, `cold_start`, timestamp.
- `benchmarks` — a materialized view aggregating by model, refreshed by `GET /benchmarks` so a caller sees the call they just made.

**Deliberately not persisted**

- **The paragraph.** Only its hash.
- Who submitted it. There are no user ids, sessions or IP addresses anywhere in the schema.

**The honest caveat.** `inferences` stores outputs, and a T5 summary is a lossy copy of the paragraph it summarised. Hashing the input does **not** make that row non-identifying. BERT's output is reduced to candidate tokens and scores before storage, and GPT-2's prompt is stripped so only generated text is kept — but the summarizer cannot be sanitised that way without destroying the thing being logged. This is stated rather than papered over.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/bert_fillmask` | encoder — fill one `[MASK]`, top-k candidates |
| `POST` | `/gpt2_continue` | decoder — continue the text |
| `POST` | `/t5_summarize` | encoder–decoder — summarize |
| `POST` | `/encoder_attention` | the hand-written block; returns `[H][T][T]` attention |
| `GET` | `/benchmarks` | the materialized view, refreshed first |
| `GET` | `/scaling_demo` | measured effect of the √d_k divisor |
| `GET` | `/healthz` | model loader **and** data client, reported separately |
| `GET` | `/version` | git sha, torch, transformers, dtype |

Every model endpoint logs the input hash, the output and the server-side latency **before** responding. Validation rejects empty input, missing or duplicated `[MASK]`, oversized paragraphs and out-of-range generation parameters with a 422 rather than a 500 from inside the model.

## Tests

`pytest -q` — 28 tests on the block alone. The four the brief names: output shape preserved, attention rows summing to 1 along the key axis, gradient reaching every parameter and the input, and permutation equivariance **only** without positional encoding (with matching tests that sinusoidal and rotary each break it). Beyond those: parity against `torch.nn.functional.scaled_dot_product_attention`, masked keys receiving zero weight, rotary preserving norm, learned encodings raising past `max_len`, head count not changing parameter count, and a check that the block is genuinely pre-norm — zero the sublayers and it must be exactly the identity, which post-norm would not be.

The service suite adds validation, the `/encoder_attention` contract, an `inputs`→`inferences` round trip, an assertion that no string in `inputs` is longer than 64 characters or contains a space (a hash does not), and a data-layer outage that must not fail an inference.

CI runs on every push, excluding the weight-downloading tests, and re-verifies the `torch.nn` parity.

## Running locally

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                 # SUPABASE_URL + service-role key
# apply db/migrations/*.sql in the Supabase SQL editor
python -m src.transformer            # block self-check
python -m src.benchmark              # parity + curves + timing
uvicorn api.main:app --reload --port 8000
python -m src.warmup --interleave    # populate the benchmarks view
```

---

# Engineering Report

## Engineering analysis

The three archetypes differ in one thing — which positions a token may attend to — and everything else follows from it.

**The encoder** attends in both directions, so every position sees every other. Filling a blank is its training objective, and it can use the words *after* the blank, which nothing else here can. It cannot generate: it has no mechanism for producing a token that was not already a position in its input.

**The decoder** is causally masked, so position *t* sees only positions ≤ *t*. That restriction is what lets it be trained to predict every next token at once and to generate indefinitely at inference — and it is exactly why it cannot fill a blank using the right-hand context. The same mask buys generation and forbids bidirectional reading.

**The encoder–decoder** keeps both. The encoder reads the paragraph bidirectionally, the decoder writes a new and shorter sequence while cross-attending to that reading. Neither half alone does this: an encoder cannot generate, and a decoder has no separate representation of the source to attend back to.

**Compute profile, from the `benchmarks` view.** Warm, on identical input:

| archetype | model | warm mean | vs encoder |
|---|---|---|---|
| encoder | `distilbert-base-uncased` | **32 ms** | 1× |
| decoder | `distilgpt2` | **897 ms** | 28× |
| encoder–decoder | `t5-small` | **1374 ms** | 43× |

The spread is structural, not incidental. The encoder answers in one forward pass. The decoder produced 40 tokens in 897 ms — about **22 ms per token**, each requiring the one before it, which no hardware parallelises away. T5 pays an encoder pass *plus* its own decode loop, so it lands highest. The cost of these models is governed by how many tokens they must emit, and only the encoder emits none.

The hosting tier adds a second cost. At 512 MB only one model stays resident, so switching archetypes evicts and reloads — around 300–750 ms locally. Interleaving the three, which is precisely what this product is *for*, makes every call a cold start. **The product's core use case is the architecture's worst case**, and the `cold_start` column exists so that fact shows up in the numbers instead of being averaged away.

**Where each one belongs.** Route 50,000 support tickets overnight and the encoder finishes in 27 minutes; the decoder would take twelve and a half hours to answer a question that never needed generated text. Draft a reply for a human to edit and only the decoder will do. Compress a document into a fixed shape and the encoder–decoder is the one with both halves.

**Against a frontier API.** At a representative rate of $3 per million input tokens and $15 per million output (rates move; check current pricing), a thousand summarizations of this paragraph cost roughly **$1** and return far better summaries, instantly, with no memory ceiling. Self-hosting costs $0 and 1.4 seconds per call, on models that are visibly worse. The free tier is cheaper only in dollars, and dearer in latency, quality and engineering hours.

## Reflection: the lowered bar

This service is about 250 lines. Each model is one call to `pipeline()`. I did not train any of them, cannot inspect them, and could not reproduce them. The bar to ship a transformer-powered tool is now an afternoon and no money at all.

Three responsibilities arrive with that.

**Provenance.** I can name the checkpoint and nothing more. Asked for the capital of France, DistilBERT answers *marseille* at 0.143, with *paris* fourth at 0.086 — a factual error on something a child knows, delivered with a confidence number, inside a demo built for engineering leadership. I cannot fix it, explain it, or enumerate what else the model believes. Shipping it means shipping that.

**Auditability.** `inferences` records which checkpoint served which call, what it produced, what it cost, and whether it paid a cold start. That is enough to reconstruct a decision six months later. It records nothing about who asked, because nothing in this product needs to know.

**Disclosure.** What `inferences` deliberately does *not* record is the paragraph. But outputs are stored, and a T5 summary is a lossy copy of its input, so the hash does not make that row anonymous. A deployment that told users "we only keep a hash" would be telling them something false. Saying so in the schema comments and here is the minimum; the alternative is not logging summaries at all, which would remove the audit trail the product exists to provide.

## Worldview reflection

"Everyone to whom much was given, of him much will be required" (Luke 12:48, ESV). What was given here is other people's work — years of research, millions of dollars of compute, corpora I have never seen — handed over for free, to anyone, in one line.

**The responsibility I accept** is the record. Every call is logged with the model that served it, its latency and its cost, so any claim this service makes about any archetype can be checked against what actually happened. The failures are in the product rather than hidden behind it: the memory ceiling is documented, the wrong answer about Paris is in this README, and the tests enforce the privacy invariant rather than merely describing it.

**The responsibility I am deferring** is everything about the outputs themselves. There is no abstain path — every call returns an answer, including the wrong ones. The confidence figures are softmax values, not calibrated frequencies. Nothing checks whether a T5 summary is faithful to the paragraph it summarised, and nothing filters what GPT-2 generates. There is no retention policy on `inferences`; it grows forever. Each of those is a real obligation of anyone deploying this for real users, and each is left undone here.

Naming them is not the same as discharging them. But a tool whose limits are written down can be used carefully, and one whose limits are undocumented will be trusted exactly as far as its interface looks polished — which, with three industrial models behind a clean `/docs` page, is much further than it deserves.