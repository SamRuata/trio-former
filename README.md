# TrioFormer — A Side-by-Side Transformer Inference Service

**Name:** Lal Ruata
**Course / Section:** CST-435 
**GitHub:** https://github.com/SamRuata/trio-former
**Render.com (API):** https://trio-former-api.onrender.com — docs at [`/docs`](https://trio-former-api.onrender.com/docs), health at [`/healthz`](https://trio-former-api.onrender.com/healthz)
**Supabase project ref:** `nkopsjdeqfgqfwozqgsk`
**Product presentation (video):** https://youtu.be/WSJpbhy9xbU
**Engineering report:** [below, in this README](#engineering-report)
**Model card:** [`MODEL_CARD.md`](MODEL_CARD.md)
**Individual contribution:** [below, in this README](#individual-contribution)
**Product 5B — TrioDeck (interface tier):** [below, in this README](#product-5b--triodeck-the-interface-tier)

> The customer-facing interface is **not** part of this product. It is built separately as Product 5B ("TrioDeck") against this service, so this API is the deliverable: reachable, documented at `/docs`, and stable enough for a classmate's interface to call.



> The Render API runs on the free tier and spins down when idle. The first request after a quiet period can take 50 seconds or more, before any model is loaded.

---

## Product pitch

A learning-and-development team is briefing engineering leadership on *what kind of transformer fits which job*. Slide decks about "encoders versus decoders" do not settle that argument; numbers do.

TrioFormer takes one paragraph and runs it through all three archetypes — BERT fills a masked token, GPT-2 continues the text, T5 summarizes it — and records what each one cost. Every call writes its server-side latency to Postgres, and a materialized view aggregates by model and environment, so the recommendation the team eventually makes is built from measurements of their own traffic rather than from vendor claims.

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
                           │              load_ms, cold_start, env  │
                           │       │                                │
                           │       ▼  aggregated by env and model   │
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

Self-attention is permutation-equivariant — a unit test shuffles the input and shows the output shuffles with it, unchanged. Position has to be injected, and *how* is the live question.

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
| Warm latency, local | **32 ms** | **897 ms** | **1374 ms** |
| Warm latency, free tier | **18 ms** | **384 ms** | **51,482 ms** |
| Choose it for | classification, routing, retrieval, extraction at volume | drafting, completion, chat | summarization, translation, reformatting |
| Avoid it for | anything that must generate | anything needing the right-hand context | anything a single pass could answer |

---

## Data persisted / not persisted

**Persisted**

- `inputs` — one row per distinct paragraph: its SHA-256, character count, word count, first-seen timestamp. The hash lets the same paragraph be recognised across models, which is what makes a fair latency comparison possible.
- `inferences` — one row per call: model id, archetype, task, the model's output, `latency_ms`, `load_ms`, `cold_start`, `environment`, timestamp.
- `benchmarks` — a materialized view aggregating by environment and model, refreshed by `GET /benchmarks` so a caller sees the call they just made.

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
| `GET` | `/version` | git sha, torch, transformers, dtype, environment, token caps |

Every model endpoint logs the input hash, the output and the server-side latency **before** responding. Validation rejects empty input, missing or duplicated `[MASK]`, oversized paragraphs and out-of-range generation parameters with a 422 rather than a 500 from inside the model.

## Tests

`pytest -q` — 28 tests on the block alone. The four the brief names: output shape preserved, attention rows summing to 1 along the key axis, gradient reaching every parameter and the input, and permutation equivariance **only** without positional encoding (with matching tests that sinusoidal and rotary each break it). Beyond those: parity against `torch.nn.functional.scaled_dot_product_attention`, masked keys receiving zero weight, rotary preserving norm, learned encodings raising past `max_len`, head count not changing parameter count, and a check that the block is genuinely pre-norm — zero the sublayers and it must be exactly the identity, which post-norm would not be.

The service suite adds validation, the `/encoder_attention` contract, an `inputs`→`inferences` round trip, an assertion that no string in `inputs` is longer than 64 characters or contains a space (a hash does neither), and a data-layer outage that must not fail an inference.

CI runs on every push, excluding the weight-downloading tests, and re-verifies the `torch.nn` parity.

## Running locally

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                 # SUPABASE_URL + service-role key
# apply db/migrations/*.sql in the Supabase SQL editor
python -m src.transformer            # block self-check
python -m src.benchmark              # parity + curves + timing
uvicorn api.main:app --port 8000
python -m src.warmup --interleave    # populate the benchmarks view
```

---

# Engineering Report

## Engineering analysis

The three archetypes differ in one thing — which positions a token may attend to — and everything else follows from it.

**The encoder** attends in both directions, so every position sees every other. Filling a blank is its training objective, and it can use the words *after* the blank, which nothing else here can. It cannot generate: it has no mechanism for producing a token that was not already a position in its input.

**The decoder** is causally masked, so position *t* sees only positions ≤ *t*. That restriction is what lets it be trained to predict every next token at once and to generate indefinitely at inference — and it is exactly why it cannot fill a blank using the right-hand context. The same mask buys generation and forbids bidirectional reading.

**The encoder–decoder** keeps both. The encoder reads the paragraph bidirectionally, the decoder writes a new and shorter sequence while cross-attending to that reading. Neither half alone does this: an encoder cannot generate, and a decoder has no separate representation of the source to attend back to.

**Compute profile, from the `benchmarks` view.** The same checkpoint on a laptop and on a 0.1 CPU instance are different measurements, so the view separates them.

*Local, full checkpoints:*

| archetype | model | warm mean | vs encoder |
|---|---|---|---|
| encoder | `distilbert-base-uncased` | **32 ms** | 1× |
| decoder | `distilgpt2`, 40 tokens | **897 ms** | 28× |
| encoder–decoder | `t5-small`, 60 tokens | **1374 ms** | 43× |

*Render free tier, reduced checkpoints:*

| archetype | model | warm mean | cold mean | of which load |
|---|---|---|---|---|
| encoder | `prajjwal1/bert-mini` | **18 ms** | 35.8 s | 29.6 s |
| decoder | `sshleifer/tiny-gpt2`, 20 tokens | **384 ms** | 6.6 s | 6.1 s |
| encoder–decoder | `t5-small`, 16 tokens | **51,482 ms** | 68.6 s | 14.9 s |

The spread is structural. The encoder answers in one forward pass and survives anything — 18 ms on a tenth of a CPU. The decoder produced 40 tokens in 897 ms locally, about **22 ms per token**, each requiring the one before it. T5 pays an encoder pass *plus* its own decode loop. Cutting its output from 40 tokens to 16 took it from 137 s to 51 s on Render — almost exactly linear.

The hosting tier adds a second cost. At 512 MB only one model stays resident, so switching archetypes evicts and reloads — up to 30 seconds on Render. Interleaving the three, which is precisely what this product is *for*, makes every call a cold start. **The product's core use case is the architecture's worst case**, and `cold_start` exists so that shows up in the numbers instead of being averaged away.

**The deployed models are not the ones I developed against.** `distilbert-base-uncased` loaded and served exactly one request before the container was killed; `bfloat16` made the load spike worse, not better, because transformers builds the float32 weights before casting; and the smallest T5 that fit returned empty strings, never having been fine-tuned to summarize. The archetypes are unchanged, but the deployed outputs are far worse.

**Where each belongs.** Route 50,000 tickets overnight and the encoder finishes in 27 minutes; the decoder would take twelve hours to answer a question that never needed generated text. Draft a reply for a human to edit and only the decoder will do. Compress a document into a fixed shape and the encoder–decoder is the one with both halves. **Against a frontier API** at a representative $3 per million input tokens and $15 per million output, a thousand summarizations of this paragraph cost roughly **$1** and return far better summaries, instantly. The free tier is cheaper only in dollars.

## Reflection: the lowered bar

This service is about 250 lines. Each model is one call to `pipeline()`. I did not train any of them, cannot inspect them, and could not reproduce them. The bar to ship a transformer-powered tool is an afternoon and no money.

**Provenance.** I can name the checkpoint and nothing more. Asked for the capital of France, DistilBERT answers *marseille* at 0.143, with *paris* fourth — a factual error on something a child knows, with a confidence number attached, inside a demo for engineering leadership. The deployed checkpoint is eleven times smaller and worse. I cannot fix that, explain it, or enumerate what else these models believe.

**Auditability.** `inferences` records which checkpoint served which call, what it produced, what it cost, and whether it paid a cold start — enough to reconstruct a decision months later. It records nothing about who asked, because nothing here needs to know.

**Disclosure.** What `inferences` deliberately does *not* record is the paragraph. But outputs are stored, and a summary is a lossy copy of its input, so the hash does not make that row anonymous. A deployment claiming "we only keep a hash" would be saying something false. The alternative — not logging summaries — would remove the audit trail the product exists to provide.

## Worldview reflection

"Everyone to whom much was given, of him much will be required" (Luke 12:48, ESV). What was given here is other people's work — years of research, millions of dollars of compute, corpora I have never seen — handed over free, to anyone, in one line.

**The responsibility I accept** is the record. Every call is logged with the model that served it, its latency and its cost, so any claim this service makes can be checked against what happened. The failures are in the product rather than behind it: the memory ceiling, the substituted checkpoints and the wrong answer about Paris are all in this README, and the tests enforce the privacy invariant rather than describing it.

**The responsibility I am deferring** is everything about the outputs. There is no abstain path — every call returns an answer, including the wrong ones. Confidence figures are softmax values, not calibrated frequencies. Nothing checks whether a summary is faithful, and nothing filters what the decoder generates. There is no retention policy on `inferences`; it grows forever.

Naming those is not discharging them. But a tool whose limits are written down can be used carefully, and one whose limits are undocumented will be trusted as far as its interface looks polished — which, with three industrial models behind a clean `/docs` page, is much further than it deserves.

---

## Individual contribution

Built solo: the transformer block, the schema, the service, the tests, the CI, the deployment, and this report.

**Decisions I made, and why**

- **Wrote the attention block to be checkable against PyTorch.** Keeping Q, K and V as three separate `Linear` layers instead of one fused projection is slightly less efficient and makes a slice-for-slice weight copy from `torch.nn.TransformerEncoderLayer` possible. That is what turns "my block looks right" into a 4.77e-07 maximum difference and identical loss curves, and CI now fails if it ever drifts.
- **Implemented all three positional schemes rather than describing two.** The claim that rotary encodes relative position is testable, so I tested it: one query/key pair, eight apart, scored at five absolute positions, gives a spread of 0.00 under rotary and 1.38 under sinusoidal.
- **Separated `load_ms` from `latency_ms`.** My first version timed only the forward pass, so a call that spent four seconds pulling weights into memory reported the 120 ms it spent thinking. On a one-slot registry that load is the dominant cost of switching archetypes.
- **Tagged every inference with its environment.** The benchmarks view originally grouped by model alone, so one row showed a mean of 1861 ms over a p50 of 37 ms and a p95 of 5618 ms — my laptop and a 0.1 CPU instance averaged together, describing neither.
- **Reported that the deployed models are not the ones I developed against**, rather than quietly swapping them and showing laptop numbers.

**Problems I hit and fixed**

The free tier was the whole afternoon. `distilbert-base-uncased` loaded and served one request, then died on the next. Switching to `bfloat16` made it worse — transformers builds the float32 weights before casting, so the load spike is larger, not smaller — and `low_cpu_mem_usage=True` fixed the spike without fixing the steady state. The smallest T5 that fit returned empty strings, because it had never been fine-tuned for summarization. What finally worked was reduced checkpoints plus a per-deployment cap on generation length, since each generated token is a full forward pass and forty of them on a tenth of a CPU is over two minutes. Dropping T5 from 40 output tokens to 16 took it from 137 s to 51 s, almost exactly linearly — which is the per-token cost argument, measured by accident.

**Full task list:** Supabase project and schema · `inputs` / `inferences` / `benchmarks` materialized view with a refresh function · scaled dot-product attention · multi-head attention · sinusoidal, learned and rotary positional encodings · pre-norm encoder block · 28 unit tests including the four required · matched-initialization benchmark against `torch.nn` with loss curves and timing · FastAPI service with seven endpoints and a validation layer · one-slot model registry with eviction · server-side latency logging · GitHub Actions CI · Render deployment and the memory investigation above · model card · engineering report · video.

---

# Product 5B — TrioDeck, the interface tier

*Individual assignment, built against the Product 5 service above.*

**GitHub:** https://github.com/SamRuata/trio-former — the interface is `ui/app.py` in this repository
**Deployed interface (Streamlit Community Cloud):** https://trio-former-mjjvtmkh4rngf5evhcb23s.streamlit.app
**Team Render.com service it targets:** https://trio-former-api.onrender.com
**Supabase project ref:** `nkopsjdeqfgqfwozqgsk`
**Deployment reflection:** [below](#deployment-reflection)

TrioDeck is the browser link the learning-and-development team actually gets. It exercises every endpoint the service exposes, renders the three archetypes side by side with their measured latencies, and charts the benchmarks view over time. It runs on Streamlit Community Cloud against the deployed Render service — not against a local one.

## The seven tabs

| tab | what it does |
|---|---|
| **Concepts** | LaTeX derivation of multi-head attention as parallel scaled dot-products, with the full shape walk, the √d_k justification against live figures from `/scaling_demo`, and the sinusoidal / learned / rotary comparison |
| **My Encoder Block** | Per-head attention heatmaps from `/encoder_attention`, with a toggle that turns rotary positional encoding on and off so the permutation-equivariance claim is visible rather than asserted |
| **BERT — Fill Mask** | `/bert_fillmask`, top-k candidates with scores |
| **GPT-2 — Continue Text** | `/gpt2_continue`, with token count and temperature |
| **T5 — Summarize** | `/t5_summarize`, with the compression ratio |
| **Compare** | One paragraph through all three archetypes, each with both latency figures side by side |
| **Latency History** | Median latency per model per hour, charted from the `benchmarks_timeline` materialized view, plus the current aggregate |

Every tab carries a "When to use this archetype" panel written for a non-technical reader, and the three archetype summaries are permanently in the sidebar.

## The tier boundary

**No model runs in this tier.** `ui/app.py` imports `streamlit`, `requests`, `pandas`, `matplotlib` and `supabase`. There is no `torch`, no `transformers`, and no weights in `ui/`. Every inference is an HTTPS call to the FastAPI service and arrives back as JSON.

**No SQL is written in this tier.** The only database access is `.select()` against two materialized views, `benchmarks` and `benchmarks_timeline`. There is no `insert`, `update`, `upsert` or `delete` anywhere in `ui/`. Every write to Postgres happens server-side, inside the API, before it responds.

**Only the anon key is here.** `API_URL`, `SUPABASE_URL` and `SUPABASE_ANON_KEY` are read from Streamlit's secrets manager. The service-role key exists only in Render's environment and appears nowhere in this tier or in source control; `.streamlit/` is git-ignored and no key has ever been committed.

**Aggregates only.** The interface cannot read `inferences` even with the anon key, because that table holds T5 summaries — which are lossy copies of what users typed. The two views it reads expose counts and percentiles and nothing else, so no individual call is recoverable through the browser.

## Failing honestly

The client classifies what went wrong rather than printing a traceback:

- **A timeout** is reported as a probable cold start, with an explanation that Render's free tier spins down and a suggestion to retry — because retrying usually works.
- **A 502** is reported as a restarting container, which on this tier usually means a model load exhausted 512 MB.
- **A connection failure** is reported as unreachable, naming the URL it tried.
- **A 422** is reported as a rejected input, with the service's own validation message, and states that nothing reached a model.

Nothing is cached across reruns except the static `/scaling_demo` figures, which are labelled with their fetch time. When a call fails, the tab shows the error instead of the previous answer — a stale result presented as live would be worse than no result at all. Once the service responds again, the next click succeeds without a reload.

## Deployment Reflection

**Measurement.** On the deployed stack, a cold start took **42,757 ms** end to end. The service reported **41,800 ms** of that, and said **30,194 ms** was loading the model into memory rather than inference. Warm, the same request reported **95 ms** server-side against **671 ms** in the browser — a gap of **576 ms**, and **549 ms** on a repeat — a property of the path, not one bad sample. That gap is not model time. It is TLS setup, two public-internet hops between Streamlit's host and Render's, Render's proxy, and queueing on a tenth of a CPU. The model's share of the wait was 95 of 671 milliseconds.

**Architecture.** Harder: failure now has shapes that cannot exist in one process. The interface can be perfectly healthy while the service is asleep, so I had to classify a cold start apart from an outage and tell the user which one they hit. Possible: the interface deploys on its own schedule against a stable contract, a classmate can build a different front end against the same API without touching it. And the service-role key never reaches a browser, because the browser was never given a way to write.

**Worldview.** The number I could most easily have hidden is that 576 ms gap. Reporting only the server's 95 ms would have been true, and far more flattering. But the user waited 671 ms, and a dashboard that measures the part I control and labels it the part they experience is a false balance: a weight calibrated to make the merchant look good. *"A false balance is an abomination to the LORD, but a just weight is his delight"* (Proverbs 11:1, ESV). So both numbers are on screen, labelled, with the difference shown as its own figure — including when the difference is embarrassing.