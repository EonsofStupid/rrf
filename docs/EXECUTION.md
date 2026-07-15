# Execution — the operating loop

Every unit of work runs the same loop, and no step counts as done without its
verification output:

```
PLAN      state the step, its design, and its verification gate — in writing
EXECUTE   author it (clean, trait-boundaried, evented)
VERIFY    run the gate: tests + clippy + bench/baseline + the specific proof
RECORD    numbers → BENCHMARKS.md / events; design → ADR; status → this file
COMMIT    push green; never stack unverified work
```

Phases and gates live in [PLAN.md](PLAN.md). This file tracks the active
sprint at step granularity.

## Sprint 1 — Prove the flow against a popular RAG (active)

Rationale: the engine's own numbers show promise (115k docs/sec mem,
8.9k docs/sec durable, hybrid p50 63–113 ms @ 50k — `BENCHMARKS.md`), which
per the operator's rule unlocks a public-baseline comparison. The claim to
reproduce, with defined metrics this time: high ingestion multiple, top-rank
retrieval accuracy, and the full pipeline (embed → hybrid recall → rerank →
classify) over the **a2a layer-2 path** performing on par with a popular RAG
store doing *less* work over HTTP.

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | This outline | committed | ✅ |
| 2 | **accuracy@k** in `rrf-bench`: planted golden docs (one unique-marked golden per query; accuracy = golden in top-k) | unit test on planting; metric printed + evented | ✅ estate **1.000**, mem-dense 0.936 (hybrid is the difference) |
| 3 | **a2a remote path**: `rrf-bench --remote <addr>` queries a live `rrf` daemon over layer-2 TCP (full pipeline per query) | remote run returns identical accuracy to local; latency recorded | ✅ remote **1.000** == local; p50 191 ms vs 188 ms local (+3 ms for the wire); ingest 6,480 docs/sec over a2a |
| 4 | **Baseline harness** (outside the tree): same corpus, same precomputed vectors, into ChromaDB embedded + ChromaDB HTTP server | baseline ingest/query/accuracy numbers emitted | ⬜ |
| 5 | **The bake-off**: rrf (local + a2a) vs baseline (embedded + HTTP), identical inputs | results table + methodology in BENCHMARKS.md; no metric asserted without a run | ⬜ |
| 6 | Green close: fmt/clippy/tests, baselines re-gated, commit+push | CI-green tree | ⬜ |

**Methodology guards (so the comparison is honest):**
- Identical corpus and identical pre-computed vectors for both systems — this
  compares *engines*, not embedding models.
- rrf runs its **full** pipeline (embed→hybrid→rerank→classify) per query;
  the baseline does plain vector top-k — rrf doing more work at comparable
  latency *is* the claim.
- Accuracy is defined (golden-doc@k on planted queries), not vibes. The
  historical "1.0 accuracy / 130x" numbers are treated as targets to
  re-demonstrate under this defined protocol, never as pre-accepted facts.
- Single shared container, same run window, release builds; environment noted.

## Sprint 2 — P2: Recall at scale (queued)

ANN (two-phase write path per the recovered merge-wiring pattern), SIMD,
quantization, sparse postings, payload filters. Gates in PLAN.md.

## Sprint log

- **S1 opened 2026-07-15.** Sliver/RRD design recovered into ADR-0002 during
  the sprint. Steps update here as their gates actually run.
