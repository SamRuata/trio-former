"""FastAPI service -- three transformer archetypes behind one contract.

    encoder          BERT        fill a masked token
    decoder          GPT-2       continue the text
    encoder-decoder  T5          summarize the text

plus /encoder_attention, which runs the hand-written block from
src/transformer.py and returns its attention matrices. That endpoint exists so
the interface tier can show what attention looks like in something the author
wrote, next to what three industrial models do with the same paragraph.

**Memory.** Render's free tier is 512 MB and 0.1 CPU. PyTorch alone is roughly
250 MB resident, so the three pipelines cannot all be loaded. The registry
below keeps exactly ONE resident and evicts it when a different archetype is
asked for. Every switch therefore pays a load, recorded separately as
`load_ms` so the cost of the hosting tier never gets mistaken for the cost of
the model.

Model ids come from the environment so the deployment can be moved to smaller
checkpoints without a code change.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import gc
import hashlib
import os
import subprocess
import time
from typing import Dict, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from shared.schemas import (BenchmarkRow, Candidate, ContinueRequest,
                            ContinueResponse, EncoderAttentionRequest,
                            EncoderAttentionResponse, FillMaskRequest,
                            FillMaskResponse, Health, InferenceMeta,
                            SummarizeRequest, SummarizeResponse, Version)

# --- configuration ---------------------------------------------------------
MODELS = {
    "bert_fillmask": {
        "model_id": os.getenv("MODEL_BERT", "distilbert-base-uncased"),
        "task": "fill-mask",
        "archetype": "encoder",
    },
    "gpt2_continue": {
        "model_id": os.getenv("MODEL_GPT2", "distilgpt2"),
        "task": "text-generation",
        "archetype": "decoder",
    },
    "t5_summarize": {
        "model_id": os.getenv("MODEL_T5", "t5-small"),
        "task": "summarization",
        "archetype": "encoder-decoder",
    },
}
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.getenv("MODEL_DTYPE", "float32")]

TAGS = [
    {"name": "models", "description": "The three pretrained archetypes."},
    {"name": "teaching", "description": "The hand-written encoder block."},
    {"name": "benchmarks", "description": "Latency aggregated by model."},
    {"name": "ops", "description": "Liveness and build information."},
]

app = FastAPI(
    title="TrioFormer API",
    version="1.0.0",
    description=(
        "BERT, GPT-2 and T5 behind one contract, with server-side latency "
        "logged for every call. One model is resident at a time; a call that "
        "switches archetype pays a load, reported separately as load_ms."),
    openapi_tags=TAGS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# One-slot model registry
# ---------------------------------------------------------------------------
class Registry:
    """Holds at most one pipeline. Loading a second evicts the first."""

    def __init__(self) -> None:
        self.key: Optional[str] = None
        self.pipe = None
        self.error: Optional[str] = None

    def get(self, name: str) -> Tuple[object, bool]:
        """Return ``(pipeline, cold_start)``, loading and evicting as needed."""
        if self.key == name and self.pipe is not None:
            return self.pipe, False

        if self.pipe is not None:
            self.pipe = None
            self.key = None
            gc.collect()        # release the previous model before allocating

        from transformers import pipeline

        spec = MODELS[name]
        self.pipe = pipeline(spec["task"], model=spec["model_id"],
                             torch_dtype=DTYPE, device=-1)
        self.key = name
        return self.pipe, True


REGISTRY = Registry()


def _supabase():
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _log(name: str, text: str, output: dict, latency_ms: int,
         cold_start: bool, load_ms: int = 0) -> bool:
    """Record one call. Returns whether it was written.

    Never raises, and the client is constructed INSIDE the try: a data-layer
    failure must degrade the log, not the inference. The product is the
    prediction; the log is evidence about it.
    """
    try:
        client = _supabase()
        if client is None:
            return False

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = client.table("inputs").upsert(
            {"input_hash": digest, "n_chars": len(text),
             "n_words": len(text.split())},
            on_conflict="input_hash").execute()
        input_id = row.data[0]["id"]

        spec = MODELS[name]
        client.table("inferences").insert({
            "input_id": input_id,
            "model_name": name,
            "model_id": spec["model_id"],
            "archetype": spec["archetype"],
            "task": spec["task"],
            "output": output,
            "latency_ms": latency_ms,
            "cold_start": cold_start,
            "load_ms": load_ms,
        }).execute()
        return True
    except Exception:                                         # noqa: BLE001
        return False


def _meta(name: str, text: str, latency_ms: int, cold_start: bool,
          load_ms: int, logged: bool) -> InferenceMeta:
    spec = MODELS[name]
    return InferenceMeta(
        model_name=name, model_id=spec["model_id"],
        archetype=spec["archetype"], task=spec["task"],
        latency_ms=latency_ms, load_ms=load_ms, cold_start=cold_start,
        input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        logged=logged)


def _run(name: str, text: str, call):
    """Load if needed, time the whole server-side path, report both parts.

    Returns ``(raw, latency_ms, cold_start, load_ms)``.

    `latency_ms` is total time to serve -- load plus inference -- because that
    is what a caller waits for. `load_ms` is how much of it was getting the
    model into memory, zero on a warm call. Keeping them separate is the
    point: on a one-slot registry the load dominates every archetype switch,
    and folding it into one number would describe the hosting tier rather
    than the model.
    """
    started = time.perf_counter()
    try:
        pipe, cold = REGISTRY.get(name)
    except Exception as exc:                                  # noqa: BLE001
        REGISTRY.error = f"{type(exc).__name__}: {exc}"
        raise HTTPException(
            503, f"could not load {MODELS[name]['model_id']}: {exc}")

    loaded_at = time.perf_counter()
    raw = call(pipe)
    finished = time.perf_counter()

    return (raw,
            int((finished - started) * 1000),
            cold,
            int((loaded_at - started) * 1000))


# ---------------------------------------------------------------------------
# Encoder: BERT
# ---------------------------------------------------------------------------
@app.post("/bert_fillmask", response_model=FillMaskResponse, tags=["models"])
def bert_fillmask(req: FillMaskRequest) -> FillMaskResponse:
    """Fill one masked token. The encoder archetype's native task.

    BERT sees the whole sentence at once and attends in both directions, which
    is why it can use the words *after* the blank. A decoder cannot: at the
    masked position it has not read them yet.
    """
    def call(pipe):
        prompt = req.text.replace("[MASK]", pipe.tokenizer.mask_token)
        return pipe(prompt, top_k=req.top_k)

    raw, latency, cold, load_ms = _run("bert_fillmask", req.text, call)

    candidates = [Candidate(token=r["token_str"].strip(), score=round(r["score"], 6))
                  for r in raw]
    filled = req.text.replace("[MASK]", candidates[0].token)

    # Only the candidate tokens and scores are stored. The filled sentence is
    # the input plus one word, so storing it would undo the hashing.
    logged = _log("bert_fillmask", req.text,
                  {"candidates": [c.model_dump() for c in candidates]},
                  latency, cold, load_ms)

    return FillMaskResponse(candidates=candidates, filled=filled,
                            meta=_meta("bert_fillmask", req.text, latency,
                                       cold, load_ms, logged))


# ---------------------------------------------------------------------------
# Decoder: GPT-2
# ---------------------------------------------------------------------------
@app.post("/gpt2_continue", response_model=ContinueResponse, tags=["models"])
def gpt2_continue(req: ContinueRequest) -> ContinueResponse:
    """Continue the text. The decoder archetype's native task.

    Causal masking means position t may only attend to positions <= t, so the
    model can be trained to predict every next token at once and can generate
    indefinitely at inference. The same mask is why it cannot fill a blank
    using what comes after it.
    """
    def call(pipe):
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        return pipe(req.text, max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature, do_sample=True,
                    truncation=True, return_full_text=False)

    raw, latency, cold, load_ms = _run("gpt2_continue", req.text, call)

    # Small decoders degenerate into runs of newlines; collapse whitespace so
    # the output is readable. The repetition itself is left visible -- it is a
    # real property of the archetype at this size.
    continuation = " ".join(raw[0]["generated_text"].split())

    # The prompt is stripped by return_full_text=False, so what is stored is
    # the model's own output and not a copy of the input.
    logged = _log("gpt2_continue", req.text,
                  {"continuation": continuation,
                   "max_new_tokens": req.max_new_tokens,
                   "temperature": req.temperature},
                  latency, cold, load_ms)

    return ContinueResponse(
        continuation=continuation,
        n_tokens=len(continuation.split()),
        meta=_meta("gpt2_continue", req.text, latency, cold, load_ms, logged))


# ---------------------------------------------------------------------------
# Encoder-decoder: T5
# ---------------------------------------------------------------------------
@app.post("/t5_summarize", response_model=SummarizeResponse, tags=["models"])
def t5_summarize(req: SummarizeRequest) -> SummarizeResponse:
    """Summarize. The encoder-decoder archetype's native task.

    The encoder reads the paragraph bidirectionally; the decoder writes a new,
    shorter sequence while cross-attending to that reading. Neither half alone
    does this: an encoder cannot generate, and a decoder has no separate
    representation of the source to attend back to.
    """
    def call(pipe):
        return pipe(req.text, max_length=req.max_length,
                    min_length=req.min_length, truncation=True,
                    do_sample=False)

    raw, latency, cold, load_ms = _run("t5_summarize", req.text, call)
    summary = raw[0]["summary_text"].strip()

    # NOTE: unlike the other two, this output is a lossy copy of the input.
    # Hashing the input does not make this row non-identifying, and the README
    # says so rather than implying the hash settles it.
    logged = _log("t5_summarize", req.text, {"summary": summary},
                  latency, cold, load_ms)

    words = max(1, len(req.text.split()))
    return SummarizeResponse(
        summary=summary,
        compression=round(len(summary.split()) / words, 3),
        meta=_meta("t5_summarize", req.text, latency, cold, load_ms, logged))


# ---------------------------------------------------------------------------
# The hand-written block
# ---------------------------------------------------------------------------
@app.post("/encoder_attention", response_model=EncoderAttentionResponse,
          tags=["teaching"])
def encoder_attention(req: EncoderAttentionRequest) -> EncoderAttentionResponse:
    """Run the hand-written block on a toy input and return its attention.

    No pretrained weights: the block is freshly initialised from a fixed seed
    and the token embeddings are drawn deterministically from each token's
    SHA-256. The heatmap therefore shows the *structure* attention produces,
    not learned meaning -- which is the honest claim, and the one the
    interface tier should repeat.
    """
    from src.transformer import TransformerEncoderBlock

    tokens = req.text.lower().split()[:24]
    if not tokens:
        raise HTTPException(422, "no tokens in text")

    d_model = 64
    started = time.perf_counter()

    torch.manual_seed(0)
    block = TransformerEncoderBlock(d_model, req.n_heads,
                                    position_mode=req.position_mode).eval()

    rows = []
    for tok in tokens:
        seed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
        g = torch.Generator().manual_seed(seed)
        rows.append(torch.randn(d_model, generator=g))
    x = torch.stack(rows).unsqueeze(0)                    # [1,T,D]

    with torch.no_grad():
        _, weights = block(x)                             # [1,H,T,T]

    latency = int((time.perf_counter() - started) * 1000)
    matrix = [[[round(float(v), 5) for v in row] for row in head]
              for head in weights[0]]

    return EncoderAttentionResponse(
        tokens=tokens, attention=matrix, n_heads=req.n_heads,
        d_model=d_model, position_mode=req.position_mode, latency_ms=latency,
        note=("Untrained weights, deterministic per-token embeddings. Shows "
              "the shape of attention, not learned meaning."))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
@app.get("/benchmarks", response_model=list[BenchmarkRow], tags=["benchmarks"])
def benchmarks() -> list[BenchmarkRow]:
    """Latency aggregated by model, from the Supabase materialized view.

    The view is refreshed first, so a caller sees the call they just made.
    """
    client = _supabase()
    if client is None:
        raise HTTPException(503, "data layer is not configured")
    try:
        client.rpc("refresh_benchmarks").execute()
    except Exception:                                         # noqa: BLE001
        pass          # stale numbers beat a 500
    res = client.table("benchmarks").select("*").execute()
    return [BenchmarkRow(**row) for row in (res.data or [])]


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    """Reports both halves: the model loader and the data client."""
    loader_ok = True
    detail = REGISTRY.error
    try:
        import transformers  # noqa: F401
    except Exception as exc:                                  # noqa: BLE001
        loader_ok = False
        detail = f"{type(exc).__name__}: {exc}"

    data_ok = False
    try:
        client = _supabase()
        if client is not None:
            client.table("inferences").select("id").limit(1).execute()
            data_ok = True
    except Exception:                                         # noqa: BLE001
        data_ok = False

    return Health(
        status="ok" if loader_ok and data_ok else "degraded",
        model_loader=loader_ok,
        resident_model=REGISTRY.key,
        configured={k: v["model_id"] for k, v in MODELS.items()},
        data_client=data_ok,
        detail=detail,
    )


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    import transformers

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                         # noqa: BLE001
        sha = os.getenv("RENDER_GIT_COMMIT", "unknown")[:7] or "unknown"

    url = os.getenv("SUPABASE_URL", "")
    return Version(
        git_sha=sha, torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        supabase_project_ref=url.split("//")[-1].split(".")[0] if url else None,
        dtype=str(DTYPE).replace("torch.", ""))