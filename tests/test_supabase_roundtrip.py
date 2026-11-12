"""The data layer: logging, the privacy invariant, and an outage."""
from __future__ import annotations

import hashlib

from tests.conftest import needs_supabase, slow

TEXT = "The capital of France is [MASK]."


@slow
@needs_supabase
def test_an_inference_row_joins_to_its_input(client, supabase):
    client.post("/bert_fillmask", json={"text": TEXT})
    digest = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()

    rows = (supabase.table("inputs").select("*")
            .eq("input_hash", digest).limit(1).execute().data)
    assert rows, "no inputs row was written"

    inferences = (supabase.table("inferences").select("*")
                  .eq("input_id", rows[0]["id"])
                  .order("id", desc=True).limit(1).execute().data)
    assert inferences
    row = inferences[0]
    assert row["archetype"] == "encoder"
    assert row["latency_ms"] > 0


@needs_supabase
def test_the_inputs_table_never_holds_the_prompt(supabase):
    """The invariant, enforced rather than promised.

    `inputs` stores a hash and two integers. Nothing in it should be long
    enough, or word-like enough, to be a paragraph someone typed.
    """
    rows = (supabase.table("inputs").select("*")
            .order("id", desc=True).limit(50).execute().data or [])
    for row in rows:
        for column, value in row.items():
            if isinstance(value, str):
                assert len(value) <= 64, (
                    f"inputs.{column} holds {len(value)} characters -- this "
                    f"table must never contain prompt text")
                assert " " not in value, (
                    f"inputs.{column} contains a space; a hash does not")


@needs_supabase
def test_benchmarks_view_aggregates_by_model(client, supabase):
    rows = client.get("/benchmarks").json()
    for row in rows:
        assert row["n_calls"] >= 1
        assert row["archetype"] in ("encoder", "decoder", "encoder-decoder")
        if row["mean_warm_ms"] is not None:
            assert row["mean_warm_ms"] > 0


def test_inference_survives_a_data_layer_outage(client, monkeypatch):
    """A dead database must not fail an inference.

    The same failure mode that bit Seq-Sight: the client is constructed inside
    the logging try block, so a connection error degrades the log rather than
    the product.
    """
    import api.main as main

    def broken():
        raise RuntimeError("supabase is down")

    monkeypatch.setattr(main, "_supabase", broken)
    body = client.post("/encoder_attention",
                       json={"text": "the cat sat on the mat"})
    assert body.status_code == 200


def test_healthz_reports_degraded_without_a_data_client(client, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_supabase", lambda: None)
    body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert body["data_client"] is False