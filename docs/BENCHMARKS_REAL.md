# BENCHMARKS_REAL.md — the first honest numbers

_2026-07-16. Real models, real public benchmark, third-party relevance
judgments. **These supersede every accuracy number in `BENCHMARKS.md`,
`COMPARISON.md`, `PARITY.md` and `README.md`**, all of which were produced by
the deterministic hash embedder scoring synthetic vectors against synthetic
vectors — a hash function grading itself._

## The harness is calibrated

The single most important number here is not RRO's:

| | nDCG@10 |
|---|---|
| our BM25 on nfcorpus | **0.3115** |
| published BEIR BM25 on nfcorpus | **~0.325** |

Our lexical floor reproduces the literature's lexical floor. That is what makes
everything below arguable rather than self-reported. A harness whose baseline
doesn't match published work is measuring something else, and its headline
number is worthless no matter how good it looks.

**This was wrong at first, in the most flattering possible direction.** The BM25
arm originally faked "lexical only" by handing `hybrid_search` a zero vector —
fusion still ran and blended a degenerate dense ranking into the lexical one,
scoring **0.159**, less than half the real baseline. A baseline broken *low*
makes every arm above it look good. It was fixed to call the estate's real
`lexical_search`, which moved it onto the published number.

## nfcorpus (BEIR) — 3,633 docs, 323 judged queries, graded qrels 0..2

Embedder: Qwen3-Embedding-4B (f16, llama.cpp :8090, 2560-d).
Reranker: llama-nemotron-rerank-1b-v2 (vLLM :8092). `recall_k=100`, `top_k=10`.

| arm | nDCG@10 | Recall@10 | MRR@10 | ms/query | vs prev |
|---|---:|---:|---:|---:|---|
| `bm25` — lexical only | 0.3115 | 0.1519 | 0.5188 | 39.6 | — |
| `dense` — ANN only | **0.4119** | 0.2013 | 0.6237 | 43.5 | +32.3% |
| `hybrid` — dense+BM25, RRF-fused | 0.3902 | 0.1903 | 0.6132 | 43.0 | **−5.3%** |
| `rro` — hybrid + cross-encoder | **0.4288** | 0.2152 | 0.6264 | **1167.5** | +9.9% |

### Finding 1 — hybrid fusion HURTS here (−5.3%)

`hybrid` (0.3902) is **worse than `dense` alone** (0.4119). Reciprocal-rank
fusion blends a 0.31 lexical ranking into a 0.41 dense one and drags it down.

This contradicts the repo's own marketing. `COMPARISON.md` sells "hybrid
dense+BM25 fused **by default**" as a headline advantage over vector-first
stores. On this corpus, with this embedder, the default is a **5.3% nDCG
regression** versus just not fusing. RRF weights the two rankings equally by
construction; when one retriever is materially better, equal weighting is a tax.

Not "hybrid is bad" — it is one corpus. But "fused by default" is now a claim
with a counter-example, and the fusion needs a weight or a gate, not a default.

### Finding 2 — the reranker earns quality and costs 27x latency

`rro` (0.4288) beats `hybrid` by 9.9% and `dense` by 4.1%, and is the best arm.
It also goes from **43 ms → 1168 ms per query**: a 27x cost for +4.1% over
plain dense. That is a real trade, not a free win, and whether it is worth
paying is a product decision that needs the number in front of it.

Note it also *rescues* fusion: the reranker recovers hybrid's regression and
passes dense. The cross-encoder is doing the work the fusion weighting isn't.

### Finding 3 — ingest is ~1000x slower than advertised

**10 docs/sec** (3,633 docs in 349 s), against the README's **10.9k docs/sec**.

Nothing regressed. The old number was measured with a microsecond hash embedder;
this one runs a real 4B model over HTTP. Once a real model is in the path, the
forward pass dominates and every wire/engine choice becomes noise. This is the
first honest ingest measurement RRO has, and the README's figure should be read
as "how fast the estate can index vectors someone else already computed."

## What is NOT yet measured

- **BRIGHT** — the reasoning-intensive benchmark (published SOTA is only ~22.1
  nDCG@10). nfcorpus is a warm-up; BRIGHT is the real target.
- **ANN `ef`/graph re-tuning on real vectors.** The current params were fit to
  synthetic distributions. Untouched here, so `dense`/`hybrid`/`rro` may all
  improve.
- **The 0.6/4/8B tier ladder.** Only 4B (embed) and 1B (rerank) ran.
- **Statistical significance.** 323 queries, single run, no CIs. The −5.3% and
  +9.9% deltas are directional, not established.

## Reproduce

```sh
hf download mteb/nfcorpus --repo-type dataset --local-dir eval-data/nfcorpus

RRO_EMBEDDER=llamacpp RRO_EMBEDDER_ENDPOINT=http://127.0.0.1:8090/v1/embeddings \
RRO_RERANKER=vllm    RRO_RERANKER_ENDPOINT=http://127.0.0.1:8092/rerank \
RRO_EVAL_DATA=eval-data/nfcorpus RRO_EMBED_BATCH=64 \
  cargo run --release --bin rro-eval
```
