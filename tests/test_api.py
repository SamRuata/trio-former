"""Endpoint contract, validation, and the graceful-degradation path."""
from __future__ import annotations

from tests.conftest import needs_supabase, slow


def test_healthz_reports_both_halves(client):
    """The rubric asks for the model loader AND the data client, separately."""
    body = client.get("/healthz").json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_loader"], bool)
    assert isinstance(body["data_client"], bool)
    assert set(body["configured"]) == {"bert_fillmask", "gpt2_continue",
                                       "t5_summarize"}


def test_version_reports_both_libraries(client):
    body = client.get("/version").json()
    assert body["torch_version"].startswith("2.4.1")
    assert body["transformers_version"].startswith("4.44")


# ---------------------------------------------------------------------------
# The hand-written block, through the API
# ---------------------------------------------------------------------------
def test_encoder_attention_returns_a_square_matrix_per_head(client):
    body = client.post("/encoder_attention",
                       json={"text": "the cat sat on the mat",
                             "n_heads": 4}).json()
    tokens, attn = body["tokens"], body["attention"]

    assert len(attn) == 4
    for head in attn:
        assert len(head) == len(tokens)
        assert all(len(row) == len(tokens) for row in head)
        for row in head:
            assert abs(sum(row) - 1.0) < 1e-3


def test_encoder_attention_is_deterministic(client):
    payload = {"text": "the cat sat on the mat"}
    a = client.post("/encoder_attention", json=payload).json()
    b = client.post("/encoder_attention", json=payload).json()
    assert a["attention"] == b["attention"]


def test_rotary_changes_the_attention_pattern(client):
    base = client.post("/encoder_attention",
                       json={"text": "the cat sat on the mat",
                             "position_mode": "none"}).json()
    rope = client.post("/encoder_attention",
                       json={"text": "the cat sat on the mat",
                             "position_mode": "rotary"}).json()
    assert base["attention"] != rope["attention"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_fillmask_requires_a_mask(client):
    r = client.post("/bert_fillmask", json={"text": "no mask here"})
    assert r.status_code == 422
    assert "MASK" in str(r.json()["detail"])


def test_fillmask_rejects_two_masks(client):
    r = client.post("/bert_fillmask", json={"text": "[MASK] and [MASK]"})
    assert r.status_code == 422


def test_empty_text_is_rejected_everywhere(client):
    for path in ("/gpt2_continue", "/t5_summarize"):
        assert client.post(path, json={"text": "   "}).status_code == 422


def test_overlong_text_is_rejected(client):
    r = client.post("/t5_summarize", json={"text": "word " * 5000})
    assert r.status_code == 422


def test_generation_bounds_are_enforced(client):
    assert client.post("/gpt2_continue",
                       json={"text": "hello", "max_new_tokens": 999}
                       ).status_code == 422
    assert client.post("/gpt2_continue",
                       json={"text": "hello", "temperature": 0}
                       ).status_code == 422


# ---------------------------------------------------------------------------
# Live inference -- opt in with `pytest -m slow`
# ---------------------------------------------------------------------------
@slow
def test_each_archetype_answers_and_reports_its_cost(client):
    cases = [
        ("/bert_fillmask", {"text": "The capital of France is [MASK]."},
         "encoder"),
        ("/gpt2_continue", {"text": "The team met on Monday to decide",
                            "max_new_tokens": 10}, "decoder"),
        ("/t5_summarize", {"text": "The team met on Monday to decide how to "
                                   "ship the demo. They agreed to deploy the "
                                   "service first and the interface later."},
         "encoder-decoder"),
    ]
    for path, payload, archetype in cases:
        body = client.post(path, json=payload).json()
        meta = body["meta"]
        assert meta["archetype"] == archetype
        assert meta["latency_ms"] > 0
        assert len(meta["input_sha256"]) == 64


@slow
def test_the_second_call_to_a_model_is_not_a_cold_start(client):
    payload = {"text": "The capital of France is [MASK]."}
    client.post("/bert_fillmask", json=payload)
    second = client.post("/bert_fillmask", json=payload).json()
    assert second["meta"]["cold_start"] is False