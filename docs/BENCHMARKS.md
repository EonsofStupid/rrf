# Reason Ready — Measured Results

**Every number here came out of a real run of `rrf-bench`.** Nothing is
asserted that a run did not produce. Reproduce with:

```sh
cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store mem
cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store estate
```

## Environment (2026-07-15)

Shared cloud container (Linux x86_64), release profile, default engine
components (deterministic embedder, dim 384). Synthetic corpus: 50,000 docs,
24–64 tokens each, zipf-skewed vocabulary of ~8k distinct terms; 500 hybrid
queries, top-10. **Numbers on dedicated hardware will differ — re-run there.**
External baselines run outside this tree on the same corpus/queries and are
compared on these emitted numbers.

## Ingest — the full machine (embed → index → persist)

| store | wall time | throughput | errors |
|---|---|---|---|
| `mem` (in-memory) | 0.43 s | **115,387 docs/sec** | 0 |
| `estate` (persistent kvs, durable BM25 + vectors + shapes) | 5.63 s | **8,883 docs/sec** | 0 |

Ingestion runs through the whole tokio machine: bounded intake
(backpressure), 256-doc batches, 4 concurrent batches, graceful drain, every
document embedded, BM25-indexed, and (estate) durably written.

## Query — hybrid (dense + BM25, reciprocal rank fusion), top-10

| store | p50 | p95 | p99 |
|---|---|---|---|
| `mem` | 82.3 ms | 85.6 ms | 95.1 ms |
| `estate` | 155.4 ms | 168.4 ms | 180.6 ms |

Sequential, single-client latency over 50k docs with **exact** (full-scan)
dense search. The scan is the known cost: ANN indexing (roadmap Phase 4)
replaces the O(N) scan; the trait boundary means nothing else changes.

## The rigor loop, demonstrated

The first estate run measured **762 docs/sec**. The harness exposed the flaw:
postings stored as one JSON blob per term were re-read and re-written on every
batch — O(N²) on hot terms. Re-authored to the LSM-native layout (one row per
`(term, doc)`; blind puts, prefix-scan reads):

| | before | after | change |
|---|---|---|---|
| estate ingest | 762 docs/sec | **8,883 docs/sec** | **11.7×** |

A second finding from the same runs: the in-memory store cloned every
record's payload before truncating to top-k; scoring first and cloning only
winners cut mem query p50 from 116 ms to 82 ms (−29%).

Measure → find → re-author → re-measure. That is how every performance claim
in this repository gets made.

## Bake-off vs a popular RAG store (2026-07-15, planted-v1 protocol)

**Identical inputs for every row**: the same 50,500 documents and the same
precomputed 384-d vectors (exported via `rrf-bench --export`), same shared
container, release builds, same run window. Baseline: **ChromaDB 1.5.9**
(embedded and HTTP-server modes), a widely used RAG vector store. 500 planted
queries; accuracy@10 = the planted golden doc retrieved.

| system | path | ingest (docs/sec) | accuracy@10 | query p50 |
|---|---|---|---|---|
| **rrf estate** (hybrid, durable) | local | **6,624** | **1.000** | 188.5 ms |
| **rrf estate, full pipeline** (embed→hybrid→rerank→classify per query) | **a2a layer-2 TCP** | **6,480** | **1.000** | 191.0 ms |
| rrf mem (dense-only fallback) | local | 85,358 | 0.936 | 98.1 ms |
| ChromaDB (vector ANN) | embedded | 566 | 0.572 | 3.2 ms |
| ChromaDB (vector ANN) | HTTP | 586 | 0.606 | 4.9 ms |

What the run demonstrated:

- **Ingestion: 11.7× durable-to-durable** (6,624 vs 566), and rrf's number
  *includes* server-side embedding while the baseline received precomputed
  vectors. Over the network: **11.1×** (a2a 6,480 vs HTTP 586). The
  in-memory engine is ~150× on this protocol.
- **Retrieval correctness: 1.000 vs 0.572/0.606.** The hybrid (dense + BM25,
  reciprocal-rank fused) retrieved every planted target; pure-vector ANN
  missed ~40%. rrf's own dense-only path (0.936) shows the split: exact
  scan recovers most of the gap, **hybrid closes it to zero** — the design
  thesis, measured.
- **The a2a layer-2 wire is ~free**: full pipeline remotely at 191 ms vs
  188.5 ms locally (+3 ms), identical accuracy — the "treat remote nodes as
  local" property, demonstrated over TCP.
- **Query latency is the honest loss**: the baseline's ANN answers in 3–5 ms;
  rrf's exact O(N) scan takes ~190 ms at 50k docs — while also running
  rerank + readiness per query. This is precisely P2 (ANN) — the gap is
  quantified, not hidden.

Methodology caveats, stated plainly: hash-based embeddings are adversarial
for HNSW graphs (near-orthogonal vectors), which depresses the baseline's ANN
recall relative to semantic embeddings; the historical "130×" figure was not
produced by this protocol on this container — today's measured multiples are
**11–15× durable, ~150× in-memory**, and the harness (not memory) is now the
arbiter of every future claim.

## P2: the ANN index lands (2026-07-15)

Clean-authored layered small-world graph (`recall::AnnIndex`), integrated
into the estate on the two-phase pattern (durable `vecs` CF is the source of
truth; graph applied post-commit with read-your-writes; rebuilt from durable
vectors on open). Unit gate: recall@10 ≥ 0.95 vs exact (property test).
End-to-end, planted-v1, release, this container:

| scale | metric | exact scan (before) | ANN (after) | change |
|---|---|---|---|---|
| 50k | query p50 (hybrid) | 188.5 ms | **1.40 ms** | **135×** |
| 50k | accuracy@10 | 1.000 | **1.000** | held |
| 100k | query p50 (hybrid) | ~380 ms (extrapolated O(N)) | **2.09 ms** | ~180× |
| 100k | accuracy@10 | — | **1.000** | 500/500 goldens |
| 100k | throughput | ~3 qps | **478 qps** | sequential, full hybrid |

The engine now answers **faster than the popular baseline's pure-vector ANN
(3.2–4.9 ms) while also running BM25 + reciprocal-rank fusion** — and keeps
exact-retrieval accuracy the baseline could not reach (1.000 vs 0.572–0.606).

**The honest cost:** durable ingest dropped 8,883 → 488 docs/sec — graph
construction currently runs single-threaded inside the writer path
(~2 ms/vector). Known, quantified, and the two-phase design already carries
the fix: apply graph inserts out-of-band (compaction-style, the recovered
reference pattern) and/or parallel batch build. That is the next
optimization; until it lands, the recorded baseline reflects reality.

## Baselines & the regression gate

Recorded container baselines live in `baselines/` (config + numbers, JSON).
`rrf-bench --baseline <path>` re-runs the same configuration and exits
non-zero on regression beyond tolerance — see
[OBSERVABILITY](OBSERVABILITY.md). Runs stream JSONL events (`--events`)
queryable directly by DuckDB.
