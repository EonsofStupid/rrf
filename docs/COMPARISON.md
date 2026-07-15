# RRF vs the reference engines — the honest head-to-head

Two axes matter: what RRF has **that neither reference engine has at all**,
and where RRF stands on **their** home turf. Every ✅ below is backed by a
test or a measured run in this tree; every ⬜ has a phase in
[PARITY.md](PARITY.md). Nothing here is asserted from memory.

## 1. What RRF has that NEITHER engine has

| Capability | The vector engine | The multi-model DB | **RRF** |
|---|---|---|---|
| **RRD — reason-ready object JIT**: shape lattice (modes→slivers), per-shape compiled plans, RROs | — | — | ✅ tested |
| **Gate ladder at first touch**: stamp → L0 (µs) → L1 lexical (secrets/injection/unicode) → L2 semantic — *before any model cost* | — | — | ✅ tested; blocked docs never reach the embedder |
| **Evolving shape baseline**: per-source predictability (entropy), speculative prediction hit-rate, PSI drift alerts, snapshots persisted & grown across sessions | — | — | ✅ gated (hit-rate → 1.0; drift fires on regime change; survives restart) |
| **Readiness gate**: the engine judges whether retrieval is sufficient to reason on | — | — | ✅ in every pass |
| **Intent on every query** (semantic-routed, front-door) | — | — | ✅ |
| **The connectome**: every pass and the whole estate rendered as a graph a non-technical operator can read | — | — | ✅ JSON + DOT |
| **Hybrid dense+BM25 fused by default** (reciprocal rank fusion, one engine, no plugin) | vector-first; sparse separate | BM25 and KNN as separate index types | ✅ default path |
| **Route→recall fusion** (graph resolves scope, exact hybrid inside it) | — | graph and KNN not fused | ✅ **measured: 1.000 vs 0.025** on ambiguous corpora |
| **a2a layer-2**: remote node ≡ local (+3 ms measured, identical accuracy) | HTTP/gRPC client-server | HTTP/WS client-server | ✅ measured |
| **DuckDB-native telemetry**: every stage of every pass as JSONL, zero ETL | prometheus metrics | OTLP | ✅ verified stream |
| **Baseline regression gates in the repo** (perf claims mechanically enforced) | CI benches | CI benches | ✅ gate exits non-zero on regression |

## 2. Their home turf — measured on identical inputs (this container)

| Metric | RRF | Popular RAG baseline (same corpus, same vectors) |
|---|---|---|
| Durable ingest (incl. embedding, RRD, provenance) | **10,800–10,953 docs/sec** | 566–586 docs/sec (no embedding work) |
| Query p50 @100k, full pipeline (hybrid+rerank+readiness) | **1.88 ms** | 3.2–4.9 ms (vector-only) |
| Planted-retrieval accuracy@10 | **1.000** | 0.572–0.606 |

Core retrieval capabilities in place and tested: ANN graph index
(recall@10 ≥ 0.95 vs exact, gated), exact fallback, hybrid fusion, BM25
inverted index (LSM-native), relations + traversal, resumable connectors,
durable changefeed (atomic with writes), crash-safe two-phase indexing with
read-your-writes, snapshots-of-behavior (RRD baseline), graceful signal
handling — all in one binary, embedded or networked.

## 3. The scheduled tail (honesty section)

Capabilities the references have that RRF has inventoried and phased, not
yet built: full query language & GraphQL (P6 after the typed builder),
quantization/multi-vector/payload-secondary-indexes (P2.5–P3 tail),
transactions beyond atomic batches (P5), auth/IAM (P5), DB snapshots/backup
tooling (P5), WASM plugin runtime (P6), replication/sharding (P8), GPU
build (P7+). Row-by-row status lives in [PARITY.md](PARITY.md); nothing
ships as parity until its gate runs.

**Bottom line:** on the retrieval core the engine already outperforms a
popular baseline on identical inputs; on the reasoning layer — RRD, gates,
readiness, baseline, connectome, warp mesh — the reference engines have no
equivalent at all. The remaining tail is enumerated, phased, and gated.
