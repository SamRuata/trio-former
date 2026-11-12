"""Call every model endpoint several times, then print the benchmarks table.

    python -m src.warmup                                   # local
    python -m src.warmup --base https://<app>.onrender.com # deployed

Calls are grouped BY ENDPOINT, not round-robin. The service keeps one model
resident and evicts it when another archetype is asked for, so interleaving
the three would make every call a cold start and leave the warm percentiles
empty. Grouping gives one cold sample and N-1 warm ones per model, which is
what the benchmarks view needs to separate the cost of loading a model from
the cost of running it.

That interleaving penalty is real and is discussed in the README: comparing
all three models on one paragraph -- the product's whole purpose -- is the
access pattern this hosting tier handles worst.
"""
from __future__ import annotations

import argparse
import sys

import requests

CASES = [
    ("/bert_fillmask",
     {"text": "The capital of France is [MASK].", "top_k": 5}),
    ("/gpt2_continue",
     {"text": "The engineering team met on Monday to decide",
      "max_new_tokens": 40}),
    ("/t5_summarize",
     {"text": "The learning and development team wants a single hosted demo "
              "that takes a paragraph of text and runs it through three "
              "different transformer architectures at once, so that "
              "engineering leadership can see which kind of model fits which "
              "job. They also want a side-by-side latency comparison so the "
              "cost story can be told honestly, and a persistent log of every "
              "comparison so the internal recommendation is built from real "
              "numbers rather than vendor claims."}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rounds", type=int, default=4,
                    help="calls per endpoint: 1 cold + (rounds-1) warm")
    ap.add_argument("--interleave", action="store_true",
                    help="also make one round-robin pass, which forces a cold "
                         "start on every call -- the comparison workload")
    ap.add_argument("--timeout", type=int, default=300,
                    help="generous: a cold Render container downloads weights")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    health = requests.get(f"{base}/healthz", timeout=args.timeout).json()
    print(f"health      {health['status']}   loader {health['model_loader']}   "
          f"data {health['data_client']}")
    for name, model_id in health["configured"].items():
        print(f"  {name:<16} {model_id}")

    def fire(path, payload, label):
        r = requests.post(f"{base}{path}", json=payload, timeout=args.timeout)
        if r.status_code != 200:
            print(f"  {label:<10} {path:<20} HTTP {r.status_code}  "
                  f"{r.text[:160]}")
            return
        m = r.json()["meta"]
        print(f"  {label:<10} {path:<20} {m['latency_ms']:>6} ms total  "
              f"{m['load_ms']:>5} ms load  cold={str(m['cold_start']):<5} "
              f"logged={m['logged']}")

    print(f"\ngrouped: {args.rounds} consecutive calls per endpoint")
    for path, payload in CASES:
        for n in range(1, args.rounds + 1):
            fire(path, payload, "cold" if n == 1 else f"warm {n - 1}")

    if args.interleave:
        print("\ninterleaved: one pass across all three, every call cold")
        for path, payload in CASES:
            fire(path, payload, "switch")

    rows = requests.get(f"{base}/benchmarks", timeout=args.timeout).json()
    if not rows:
        print("\nbenchmarks view is empty -- is the data layer reachable?")
        return 1

    order = {"encoder": 0, "decoder": 1, "encoder-decoder": 2}
    rows.sort(key=lambda r: order.get(r["archetype"], 9))

    fmt = lambda v: f"{v} ms" if v is not None else "—"
    print("\n| archetype | model | task | calls | warm mean | warm p50 | "
          "warm p95 | cold mean | load mean |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['archetype']} | `{r['model_id']}` | {r['task']} "
              f"| {r['n_calls']} | {fmt(r['mean_warm_ms'])} "
              f"| {fmt(r['p50_warm_ms'])} | {fmt(r['p95_warm_ms'])} "
              f"| {fmt(r['mean_cold_ms'])} | {fmt(r['mean_load_ms'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())