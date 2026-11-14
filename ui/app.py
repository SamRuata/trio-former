"""TrioDeck -- the customer-facing tier for TrioFormer (Product 5B).

THE TIER BOUNDARY, stated here and enforced by what this file does not do:

  * No model code. There is no torch import, no transformers import, and no
    weights anywhere in ui/. Every inference crosses the network to the
    FastAPI service on Render and comes back as JSON.
  * No SQL writes. The only database access is `.select()` against two
    materialized views, using the ANON key. The service-role key does not
    exist in this tier -- it lives in Render's environment, server-side.
  * No fabricated results. Nothing is cached across reruns except the
    static /scaling_demo figures, which are labelled with their fetch time.
    If the service is unreachable the tab says so; it never falls back to a
    stored answer presented as live.

Secrets come from Streamlit's secrets manager (.streamlit/secrets.toml
locally, the Cloud secrets UI when deployed) and never from source control.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="TrioDeck", layout="wide")


def secret(name: str, default=None):
    """Read a secret without exploding when no secrets file exists.

    `st.secrets` raises on access if the file is absent, which would take
    down every tab rather than the one feature that needed the value.
    """
    try:
        return st.secrets[name]
    except Exception:                                          # noqa: BLE001
        return default


API_URL = str(secret("API_URL", "http://127.0.0.1:8000")).rstrip("/")
SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_ANON_KEY = secret("SUPABASE_ANON_KEY")

INK, BLUE, GREEN, AMBER, RED = "#1f2937", "#2563eb", "#15803d", "#b45309", "#b91c1c"

ARCHETYPE_GUIDANCE = {
    "encoder": (
        "**When to use an encoder.** Reach for this when you need to "
        "*understand* something that is already written, and the answer is a "
        "label or a single word rather than new prose — sorting support "
        "tickets, pulling names out of a contract, deciding whether a review "
        "is positive. It reads the whole passage at once, in both directions, "
        "so it can use what comes after a gap as well as before it. It is by "
        "far the cheapest of the three because it answers in one pass, "
        "however long the text. It cannot write anything."),
    "decoder": (
        "**When to use a decoder.** Reach for this when you need new text "
        "written — a draft reply, a completion, a chat turn. It reads strictly "
        "left to right, which is what lets it keep going indefinitely, and is "
        "also why it cannot look ahead to fill a blank. The cost is the catch: "
        "it does a full pass of work for *every word it writes*, one after "
        "another, so a long answer is not slightly more expensive than a short "
        "one, it is proportionally more expensive."),
    "encoder-decoder": (
        "**When to use an encoder–decoder.** Reach for this when the output is "
        "new text that must stay faithful to an input — a summary, a "
        "translation, a reformat. It reads the source the way an encoder does, "
        "then writes the way a decoder does, referring back to its reading as "
        "it goes. It is the most expensive of the three, because it pays for "
        "the reading *and* for every word it produces."),
}


# ---------------------------------------------------------------------------
# Network client -- the only way this tier touches a model
# ---------------------------------------------------------------------------
class ServiceError(Exception):
    """Carries a classification so the UI can say what to do next."""

    def __init__(self, kind: str, message: str, hint: str):
        super().__init__(message)
        self.kind = kind          # cold_start | unreachable | rejected | failed
        self.message = message
        self.hint = hint


def call_api(method: str, path: str, payload: Optional[Dict] = None,
             timeout: int = 180) -> Tuple[Dict, int]:
    """Call the service. Returns ``(json, end_to_end_ms)`` or raises.

    The elapsed time measured here is what the BROWSER waited: network
    transit each way, any queueing at the service, and the server-side work.
    It is deliberately not the same number the service reports, and the two
    are always shown separately.
    """
    url = f"{API_URL}{path}"
    started = time.perf_counter()
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        raise ServiceError(
            "cold_start",
            f"The service did not answer within {timeout} seconds.",
            "Render's free tier spins the container down when idle and the "
            "first request has to wake it, load a model, and only then "
            "answer. Try once more — a second attempt usually succeeds.")
    except requests.exceptions.ConnectionError:
        raise ServiceError(
            "unreachable",
            "Could not reach the service at all.",
            f"Check that {API_URL} is deployed and awake. Nothing is shown "
            f"below because there is no live result to show.")
    except Exception as exc:                                   # noqa: BLE001
        raise ServiceError("failed", f"{type(exc).__name__}: {exc}",
                           "Unexpected client-side failure.")

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if r.status_code == 422:
        detail = r.json().get("detail")
        raise ServiceError("rejected", f"The service rejected the input: {detail}",
                           "Fix the input above and try again. Nothing was sent "
                           "to a model.")
    if r.status_code == 502:
        raise ServiceError(
            "cold_start",
            "The service returned 502 — the container is restarting.",
            "This usually means it was asleep, or a model load exhausted its "
            "512 MB. Wait a few seconds and try again.")
    if r.status_code >= 500:
        raise ServiceError("failed", f"The service returned {r.status_code}.",
                           r.text[:300])
    if r.status_code >= 400:
        raise ServiceError("rejected", f"The service returned {r.status_code}.",
                           r.text[:300])

    return r.json(), elapsed_ms


def show_error(err: ServiceError) -> None:
    """Say what failed and what to do. Never leave a stale result on screen."""
    if err.kind == "cold_start":
        st.warning(f"**{err.message}**\n\n{err.hint}")
    elif err.kind == "rejected":
        st.info(f"**{err.message}**\n\n{err.hint}")
    else:
        st.error(f"**{err.message}**\n\n{err.hint}")


def latency_pair(server_ms: int, browser_ms: int, load_ms: int = 0,
                 cold: bool = False) -> None:
    """The two figures, labelled, never conflated."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Server-side", f"{server_ms:,} ms",
              help="Measured inside the service, around the model call. "
                   "Reported by the API as latency_ms and written to Postgres.")
    c2.metric("Browser end-to-end", f"{browser_ms:,} ms",
              help="Measured in this app, from sending the request to "
                   "receiving the response. Includes network transit both "
                   "ways and any queueing at the service.")
    gap = browser_ms - server_ms
    c3.metric("Gap", f"{gap:,} ms",
              help="Network transit, TLS, and queueing. Not model time.")
    if cold:
        st.caption(f"Cold start: {load_ms:,} ms of the server-side figure was "
                   f"loading the model into memory, not inference. The service "
                   f"keeps one model resident at a time.")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_scaling_demo() -> Tuple[Optional[Dict], str]:
    """The only cached call. Static numbers; the fetch time is displayed."""
    try:
        data, _ = call_api("GET", "/scaling_demo", timeout=90)
        return data, datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    except ServiceError:
        return None, ""


def service_status() -> None:
    """Sidebar health. Distinguishes awake, degraded and unreachable."""
    try:
        health, ms = call_api("GET", "/healthz", timeout=60)
    except ServiceError as err:
        st.sidebar.error(f"Service {err.kind.replace('_', ' ')}")
        st.sidebar.caption(err.hint)
        return

    icon = "ok" if health["status"] == "ok" else "degraded"
    st.sidebar.success(f"Service {icon} · {ms} ms")
    st.sidebar.caption(
        f"model loader {'up' if health['model_loader'] else 'down'} · "
        f"data client {'up' if health['data_client'] else 'down'}\n\n"
        f"resident model: `{health.get('resident_model') or 'none'}`")
    with st.sidebar.expander("Deployed checkpoints"):
        for name, model_id in health["configured"].items():
            st.caption(f"`{name}` → `{model_id}`")


# ---------------------------------------------------------------------------
# Data layer -- read only, anon key, aggregates only
# ---------------------------------------------------------------------------
def read_view(view: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """SELECT from one materialized view with the anon key. No writes exist."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None, ("Supabase credentials are not configured in this "
                      "deployment's secrets.")
    try:
        from supabase import create_client

        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        rows = client.table(view).select("*").execute().data or []
        return pd.DataFrame(rows), None
    except Exception as exc:                                   # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def attention_heatmap(matrix: List[List[float]], tokens: List[str],
                      title: str):
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, aspect="auto")
    ax.set_xticks(range(n), tokens, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n), tokens, fontsize=7)
    ax.set_xlabel("attended to (key)", fontsize=8)
    ax.set_ylabel("attending from (query)", fontsize=8)
    ax.set_title(title, fontsize=9, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
st.title("TrioDeck")
st.caption("Three transformer archetypes, one paragraph, measured costs. "
           "This interface runs no model and writes no data — every answer "
           "below crosses the network to the service.")

with st.sidebar:
    st.subheader("Service")
    service_status()
    st.divider()
    st.subheader("When to use this archetype")
    for key, label in (("encoder", "Encoder — BERT"),
                       ("decoder", "Decoder — GPT-2"),
                       ("encoder-decoder", "Encoder–decoder — T5")):
        with st.expander(label):
            st.markdown(ARCHETYPE_GUIDANCE[key])
    st.divider()
    st.caption(f"API target\n\n`{API_URL}`")

tabs = st.tabs(["Concepts", "My Encoder Block", "BERT — Fill Mask",
                "GPT-2 — Continue Text", "T5 — Summarize", "Compare",
                "Latency History"])

# ---------------------------------------------------------------------------
# 1. Concepts
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Multi-head attention")
    st.markdown("A single attention operation is a weighted average of value "
                "vectors, where the weights come from how well each query "
                "matches each key:")
    st.latex(r"\mathrm{Attention}(Q,K,V) = "
             r"\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")

    st.markdown("Multi-head attention is **H of those in parallel, "
                "concatenated** — not one bigger attention:")
    st.latex(r"\mathrm{head}_i = \mathrm{Attention}"
             r"(QW_i^{Q},\, KW_i^{K},\, VW_i^{V})")
    st.latex(r"\mathrm{MultiHead}(Q,K,V) = "
             r"\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)\,W^{O}")

    st.subheader("Shape analysis")
    st.markdown("With batch $B$, sequence length $T$, model width $D$, "
                "$H$ heads and per-head width $D_h = D/H$:")
    st.dataframe(pd.DataFrame([
        {"step": "x", "shape": "[B, T, D]", "note": "input"},
        {"step": "q = x W_Q", "shape": "[B, T, D]",
         "note": "ONE projection of full width, not H of them"},
        {"step": "reshape", "shape": "[B, T, H, Dh]", "note": "split the width"},
        {"step": "transpose", "shape": "[B, H, T, Dh]",
         "note": "H independent attentions"},
        {"step": "q kᵀ / √d_k", "shape": "[B, H, T, T]",
         "note": "one heatmap per head"},
        {"step": "weights @ v", "shape": "[B, H, T, Dh]", "note": "context"},
        {"step": "transpose, reshape", "shape": "[B, T, D]",
         "note": "this reshape IS the concatenation"},
        {"step": "· W_O", "shape": "[B, T, D]",
         "note": "shape preserved, so blocks stack"},
    ]), hide_index=True, use_container_width=True)
    st.caption("Laying H vectors of width Dh end to end gives one of width D, "
               "so the concatenation in the formula and the reshape in the "
               "code are the same operation. Total compute does not depend on "
               "H — splitting into heads buys several attention patterns over "
               "the same sequence for free.")

    st.subheader("Why divide by √d_k")
    st.markdown("If $q$ and $k$ have independent, zero-mean, unit-variance "
                "components, their dot product is a sum of $d_k$ such "
                "products, so")
    st.latex(r"\mathrm{Var}(q \cdot k) = d_k \quad\Longrightarrow\quad "
             r"\sigma(q \cdot k) = \sqrt{d_k}")
    st.markdown("At $d_k = 128$ that is a spread of about $\\pm 11$. A softmax "
                "over scores that wide is effectively an argmax: one weight "
                "saturates near 1, the rest underflow, and the gradient "
                "through them vanishes. Measured on the live service:")

    demo, fetched_at = fetch_scaling_demo()
    if demo is None:
        st.warning("Could not reach the service for the measured figures. "
                   "The derivation above stands on its own.")
    else:
        st.dataframe(pd.DataFrame([
            {"quantity": "std. dev. of q·k",
             "unscaled": f"{demo['unscaled_std']:.2f}",
             "scaled": f"{demo['scaled_std']:.2f}"},
            {"quantity": "largest softmax weight",
             "unscaled": f"{demo['unscaled_max_softmax']:.3f}",
             "scaled": f"{demo['scaled_max_softmax']:.3f}"},
            {"quantity": "softmax entropy (nats)",
             "unscaled": f"{demo['unscaled_entropy']:.2f}",
             "scaled": f"{demo['scaled_entropy']:.2f}"},
        ]), hide_index=True, use_container_width=True)
        st.caption(f"Uniform attention over the same positions would have "
                   f"entropy {demo['uniform_entropy']:.2f} nats. Fetched at "
                   f"{fetched_at}; cached for ten minutes, since these numbers "
                   f"do not change.")

    st.subheader("Positional encodings")
    st.markdown("Attention is a weighted average, and an average does not "
                "care about order — so position has to be injected. There are "
                "three common ways, and they are not equivalent.")
    st.dataframe(pd.DataFrame([
        {"": "Where it acts", "Sinusoidal": "added to the input",
         "Learned": "added to the input",
         "Rotary (RoPE)": "rotates Q and K inside attention"},
        {"": "Parameters", "Sinusoidal": "none",
         "Learned": "max_len × d_model", "Rotary (RoPE)": "none"},
        {"": "Position encoded", "Sinusoidal": "absolute",
         "Learned": "absolute", "Rotary (RoPE)": "relative"},
        {"": "Past training length", "Sinusoidal": "extrapolates",
         "Learned": "impossible — no row to look up",
         "Rotary (RoPE)": "extrapolates"},
        {"": "Same pair, same distance, different absolute positions",
         "Sinusoidal": "score varies", "Learned": "score varies",
         "Rotary (RoPE)": "score identical"},
    ]), hide_index=True, use_container_width=True)

    st.latex(r"(R_m q) \cdot (R_n k) \;=\; q^{\top} R_{n-m} k")
    st.markdown(
        "That identity is why **rotary dominates modern LLMs**. Rotating the "
        "query and key by an angle proportional to their positions makes "
        "absolute position cancel out of the score — only the distance "
        "survives. A pair ten tokens apart looks the same at the start of a "
        "document as at the end, and because nothing is learned, it keeps "
        "working past the longest sequence ever trained on. A learned table "
        "simply has no row for position 10,000; an additive sinusoid has one, "
        "but the model still has to learn separately that 7→9 is the same "
        "relationship as 107→109.\n\n"
        "Measured on this project's implementation: one query/key pair held "
        "eight positions apart and scored at five different absolute "
        "positions gives a score spread of **0.00e+00** under rotary and "
        "**1.38** under additive sinusoidal.\n\n"
        "*The next tab lets you turn this on and off and watch the attention "
        "pattern change.*")

# ---------------------------------------------------------------------------
# 2. My Encoder Block
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("The hand-written encoder block")
    st.markdown(
        "This is not a pretrained model. It is the transformer encoder block "
        "written from scratch in `src/transformer.py` — scaled dot-product "
        "attention, multi-head projection, pre-norm residuals — with freshly "
        "initialised weights and token embeddings derived deterministically "
        "from each word's SHA-256. The same sentence always gives the same "
        "picture, but those vectors carry no meaning. **What you are seeing "
        "is the structure attention computes, not learned language.**")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        text = st.text_input("Sentence",
                             value="the cat sat on the mat and purred")
    with col_b:
        heads = st.selectbox("Heads", [1, 2, 4, 8], index=2)

    rotary_on = st.toggle(
        "Rotary positional encoding",
        value=False,
        help="Off: the block is permutation-equivariant — shuffle the words "
             "and the output shuffles with them, unchanged. On: position "
             "enters the score and the pattern changes.")

    if st.button("Run the block", type="primary"):
        try:
            body, browser_ms = call_api(
                "POST", "/encoder_attention",
                {"text": text, "n_heads": heads,
                 "position_mode": "rotary" if rotary_on else "none"},
                timeout=120)
        except ServiceError as err:
            show_error(err)
        else:
            st.session_state["block_result"] = (body, browser_ms, rotary_on)

    result = st.session_state.get("block_result")
    if result:
        body, browser_ms, was_rotary = result
        tokens, attn = body["tokens"], body["attention"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens", len(tokens))
        c2.metric("Server-side", f"{body['latency_ms']} ms")
        c3.metric("Browser end-to-end", f"{browser_ms} ms")

        st.caption(f"Positional encoding: "
                   f"**{'rotary' if was_rotary else 'none'}** · "
                   f"d_model {body['d_model']} · {body['n_heads']} heads")

        head_cols = st.columns(min(2, len(attn)))
        for i, head in enumerate(attn[:4]):
            with head_cols[i % len(head_cols)]:
                st.pyplot(attention_heatmap(
                    head, tokens,
                    f"head {i} — {'rotary' if was_rotary else 'no positions'}"))

        st.info(
            "**Toggle rotary on and run it again.** With positions off, every "
            "row of the heatmap is computed from word identity alone, so the "
            "pattern depends only on *which* words are present. With rotary "
            "on, the score between two positions also depends on how far "
            "apart they are, and the diagonal structure that appears is that "
            "distance dependence made visible.")

# ---------------------------------------------------------------------------
# 3. BERT
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("BERT — fill a masked token")
    st.markdown(ARCHETYPE_GUIDANCE["encoder"])
    st.divider()

    text = st.text_area("Text with one `[MASK]`", height=90,
                        value="The capital of France is [MASK].",
                        key="bert_text")
    top_k = st.slider("Candidates", 1, 10, 5, key="bert_k")

    if st.button("Fill the mask", type="primary", key="bert_go"):
        try:
            body, browser_ms = call_api("POST", "/bert_fillmask",
                                        {"text": text, "top_k": top_k})
        except ServiceError as err:
            show_error(err)
        else:
            meta = body["meta"]
            st.success(f"**{body['filled']}**")
            st.dataframe(pd.DataFrame(body["candidates"]),
                         hide_index=True, use_container_width=True)
            latency_pair(meta["latency_ms"], browser_ms, meta["load_ms"],
                         meta["cold_start"])
            st.caption(f"Served by `{meta['model_id']}` · logged: "
                       f"{meta['logged']} · input hash "
                       f"`{meta['input_sha256'][:16]}…` — the text itself is "
                       f"not stored.")

# ---------------------------------------------------------------------------
# 4. GPT-2
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("GPT-2 — continue the text")
    st.markdown(ARCHETYPE_GUIDANCE["decoder"])
    st.divider()

    text = st.text_area("Prompt", height=90,
                        value="The engineering team met on Monday to decide",
                        key="gpt_text")
    c1, c2 = st.columns(2)
    n_tokens = c1.slider("Max new tokens", 1, 60, 20, key="gpt_n",
                         help="The service caps this per deployment. Each "
                              "token is a full forward pass.")
    temp = c2.slider("Temperature", 0.1, 1.5, 0.8, key="gpt_t")

    if st.button("Continue", type="primary", key="gpt_go"):
        try:
            body, browser_ms = call_api(
                "POST", "/gpt2_continue",
                {"text": text, "max_new_tokens": n_tokens,
                 "temperature": temp})
        except ServiceError as err:
            show_error(err)
        else:
            meta = body["meta"]
            st.success(body["continuation"] or "*(empty continuation)*")
            latency_pair(meta["latency_ms"], browser_ms, meta["load_ms"],
                         meta["cold_start"])
            st.caption(f"Served by `{meta['model_id']}`. The deployed "
                       f"checkpoint is a reduced one — see the model card. "
                       f"The archetype's behaviour is real; the words are not "
                       f"worth much.")

# ---------------------------------------------------------------------------
# 5. T5
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("T5 — summarize")
    st.markdown(ARCHETYPE_GUIDANCE["encoder-decoder"])
    st.divider()

    st.warning(
        "On the free tier this takes roughly a minute. That is the point of "
        "the measurement, not a fault: the model does one full forward pass "
        "per word it writes, and the instance has a tenth of a CPU.")

    text = st.text_area(
        "Paragraph", height=160, key="t5_text",
        value="The learning and development team wants a single hosted demo "
              "that takes a paragraph of text and runs it through three "
              "different transformer architectures at once, so that "
              "engineering leadership can see which kind of model fits which "
              "job. They also want a side-by-side latency comparison so the "
              "cost story can be told honestly, and a persistent log of every "
              "comparison so the internal recommendation is built from real "
              "numbers rather than vendor claims.")
    max_len = st.slider("Max summary length", 10, 80, 24, key="t5_len")

    if st.button("Summarize", type="primary", key="t5_go"):
        with st.spinner("Generating — one forward pass per word…"):
            try:
                body, browser_ms = call_api(
                    "POST", "/t5_summarize",
                    {"text": text, "max_length": max_len, "min_length": 8},
                    timeout=300)
            except ServiceError as err:
                show_error(err)
            else:
                meta = body["meta"]
                st.success(body["summary"] or "*(empty summary)*")
                st.caption(f"Compression: {body['compression']:.2f} "
                           f"(summary words ÷ input words)")
                latency_pair(meta["latency_ms"], browser_ms, meta["load_ms"],
                            meta["cold_start"])
                st.caption(f"Served by `{meta['model_id']}`. Note: this "
                           f"output **is** stored, and a summary is a lossy "
                           f"copy of its input — so hashing the paragraph "
                           f"does not make the stored row anonymous.")

# ---------------------------------------------------------------------------
# 6. Compare
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("All three, one input")
    st.markdown(
        "The same paragraph through all three archetypes, with both latency "
        "figures for each. Expect this to take a while: the service holds one "
        "model in memory at a time, so running three in a row means three "
        "model loads — **the thing this product is for is the access pattern "
        "the free tier handles worst.**")

    base = st.text_area(
        "Input", height=120, key="cmp_text",
        value="The engineering team met on Monday to decide how to ship the "
              "demo. They agreed to deploy the service first and the "
              "interface later, so that a classmate could build against it.")
    st.caption("BERT needs a `[MASK]`; one is appended automatically for that "
               "tab only, so all three see the same paragraph otherwise.")

    if st.button("Run all three", type="primary", key="cmp_go"):
        jobs = [
            ("encoder", "BERT", "/bert_fillmask",
             {"text": base + " The decision was [MASK].", "top_k": 3},
             lambda b: ", ".join(c["token"] for c in b["candidates"])),
            ("decoder", "GPT-2", "/gpt2_continue",
             {"text": base, "max_new_tokens": 20},
             lambda b: b["continuation"] or "(empty)"),
            ("encoder-decoder", "T5", "/t5_summarize",
             {"text": base, "max_length": 24, "min_length": 8},
             lambda b: b["summary"] or "(empty)"),
        ]
        rows, slots = [], st.container()
        progress = st.progress(0.0, text="starting…")

        for i, (archetype, label, path, payload, extract) in enumerate(jobs):
            progress.progress(i / len(jobs), text=f"{label} — calling…")
            try:
                body, browser_ms = call_api("POST", path, payload, timeout=300)
            except ServiceError as err:
                with slots:
                    st.subheader(f"{label} — {archetype}")
                    show_error(err)
                continue

            meta = body["meta"]
            rows.append({
                "archetype": archetype,
                "model": meta["model_id"],
                "server latency_ms": meta["latency_ms"],
                "browser end-to-end ms": browser_ms,
                "gap ms": browser_ms - meta["latency_ms"],
                "of which model load ms": meta["load_ms"],
                "cold start": meta["cold_start"],
            })
            with slots:
                st.subheader(f"{label} — {archetype}")
                st.write(extract(body))
                latency_pair(meta["latency_ms"], browser_ms, meta["load_ms"],
                             meta["cold_start"])
                st.divider()

        progress.progress(1.0, text="done")

        if rows:
            st.subheader("Side by side")
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
            st.caption(
                "**Server latency_ms** is measured inside the service around "
                "the model call. **Browser end-to-end** is measured here and "
                "includes TLS, network transit both ways, and any queueing. "
                "The gap between them is not model time. Fetched "
                f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}.")

# ---------------------------------------------------------------------------
# 7. Latency History
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Latency history")
    st.caption("Read directly from Supabase with the **anon** key, from two "
               "materialized views. This tier can only SELECT, and only "
               "aggregates are exposed — no individual call, and no model "
               "output, is readable from here.")

    timeline, err = read_view("benchmarks_timeline")
    if err:
        st.error(f"Could not read the data layer: {err}")
    elif timeline is None or timeline.empty:
        st.info("No calls logged yet. Run something in one of the model tabs.")
    else:
        envs = sorted(timeline["environment"].unique())
        env = st.radio("Environment", envs,
                       index=envs.index("render") if "render" in envs else 0,
                       horizontal=True,
                       help="The same checkpoint on a laptop and on a 0.1 CPU "
                            "instance are different measurements and are not "
                            "averaged together.")
        view = timeline[timeline["environment"] == env].copy()
        view["bucket"] = pd.to_datetime(view["bucket"])

        chart = (view.pivot_table(index="bucket", columns="model_name",
                                  values="median_ms", aggfunc="median")
                 .sort_index())
        st.subheader("Median latency per model, by hour")
        st.line_chart(chart, height=320)
        st.caption("Median rather than mean: one cold start would drag an "
                   "average far enough to hide what a typical call costs.")

        st.subheader("Calls per hour")
        counts = (view.pivot_table(index="bucket", columns="model_name",
                                   values="n_calls", aggfunc="sum")
                  .sort_index())
        st.bar_chart(counts, height=220)

        st.subheader("Current aggregate")
        summary, serr = read_view("benchmarks")
        if serr:
            st.warning(serr)
        elif summary is not None and not summary.empty:
            cols = ["environment", "archetype", "model_id", "n_calls",
                    "n_cold_starts", "mean_warm_ms", "p50_warm_ms",
                    "p95_warm_ms", "mean_cold_ms", "mean_load_ms"]
            st.dataframe(summary[[c for c in cols if c in summary.columns]],
                         hide_index=True, use_container_width=True)

        st.caption(f"Fetched {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}. "
                   f"Nothing on this tab is cached — a failed read shows an "
                   f"error rather than the previous numbers.")