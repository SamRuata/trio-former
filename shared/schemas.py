"""The API contract. Product 5B is built against this, so it is documentation
as much as validation -- a classmate reads /docs and writes an interface.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_CHARS = 8_000
ALLOW_MODEL_PREFIX = {"protected_namespaces": ()}

Archetype = Literal["encoder", "decoder", "encoder-decoder"]


class TextRequest(BaseModel):
    """Shared body for the three model endpoints."""

    model_config = ALLOW_MODEL_PREFIX

    text: str = Field(..., min_length=1, max_length=MAX_CHARS,
                      description="The paragraph to process. Hashed, never stored.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty or whitespace only")
        return v


class FillMaskRequest(TextRequest):
    """BERT needs a mask token somewhere in the text."""

    top_k: int = Field(5, ge=1, le=20)

    @field_validator("text")
    @classmethod
    def has_mask(cls, v: str) -> str:
        if "[MASK]" not in v:
            raise ValueError("text must contain [MASK] exactly once, e.g. "
                             "'The capital of France is [MASK].'")
        if v.count("[MASK]") > 1:
            raise ValueError("only one [MASK] is supported")
        return v


class ContinueRequest(TextRequest):
    max_new_tokens: int = Field(40, ge=1, le=128)
    temperature: float = Field(0.8, gt=0.0, le=2.0)


class SummarizeRequest(TextRequest):
    max_length: int = Field(60, ge=10, le=200)
    min_length: int = Field(10, ge=1, le=100)


class EncoderAttentionRequest(BaseModel):
    """The teaching endpoint: the hand-written block, not a pretrained model."""

    text: str = Field("the cat sat on the mat and purred",
                      min_length=1, max_length=200)
    n_heads: int = Field(4, ge=1, le=8)
    position_mode: Literal["none", "rotary"] = "none"


class Candidate(BaseModel):
    token: str
    score: float


class InferenceMeta(BaseModel):
    """Attached to every model response: the cost story, per call."""

    model_config = ALLOW_MODEL_PREFIX

    model_name: str
    model_id: str
    archetype: Archetype
    task: str
    latency_ms: int = Field(..., description="Total server-side time to serve "
                                             "this call: load plus inference. "
                                             "Excludes network time.")
    load_ms: int = Field(..., description="Of that latency, how long getting "
                                          "the model into memory took. Zero "
                                          "on a warm call.")
    cold_start: bool = Field(..., description="True when this call paid to "
                                              "load the model.")
    input_sha256: str
    logged: bool = Field(..., description="False when the data layer was "
                                          "unreachable; the inference still "
                                          "succeeded.")


class FillMaskResponse(BaseModel):
    candidates: List[Candidate]
    filled: str = Field(..., description="The sentence with the top candidate "
                                         "substituted. Returned, never stored.")
    meta: InferenceMeta


class ContinueResponse(BaseModel):
    continuation: str = Field(..., description="Generated text only, with the "
                                               "prompt stripped.")
    n_tokens: int
    meta: InferenceMeta


class SummarizeResponse(BaseModel):
    summary: str
    compression: float = Field(..., description="summary words / input words")
    meta: InferenceMeta


class EncoderAttentionResponse(BaseModel):
    tokens: List[str]
    attention: List[List[List[float]]] = Field(
        ..., description="[H][T][T]. Row i of head h is what position i "
                         "attended to. Every row sums to 1.")
    n_heads: int
    d_model: int
    position_mode: str
    latency_ms: int
    note: str


class BenchmarkRow(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    environment: str = Field(..., description="Where the call ran: 'local' or "
                                              "'render'. The same checkpoint "
                                              "on a laptop and on 0.1 CPU are "
                                              "different measurements.")
    model_name: str
    model_id: str
    archetype: str
    task: str
    n_calls: int
    n_cold_starts: int
    mean_warm_ms: Optional[int] = None
    p50_warm_ms: Optional[int] = None
    p95_warm_ms: Optional[int] = None
    min_warm_ms: Optional[int] = None
    max_warm_ms: Optional[int] = None
    mean_cold_ms: Optional[int] = None
    mean_load_ms: Optional[int] = None
    last_served_at: Optional[datetime] = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loader: bool = Field(..., description="Transformers importable and "
                                                "the registry usable.")
    resident_model: Optional[str] = Field(
        None, description="Only one model stays in memory on a 512 MB "
                          "instance; this is the one currently loaded.")
    configured: Dict[str, str]
    data_client: bool
    detail: Optional[str] = None


class Version(BaseModel):
    git_sha: str
    torch_version: str
    transformers_version: str
    supabase_project_ref: Optional[str]
    dtype: str
    environment: str
    max_new_tokens_cap: int = Field(
        ..., description="Generation is clamped to this. Set low on a free "
                         "instance, where each token is a full forward pass.")
    t5_max_length_cap: int