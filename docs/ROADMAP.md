# Reason Ready — Roadmap

Phased, measurable, each phase gated by the one before. Status: 🟢 done ·
🟡 in progress · ⬜ planned.

## Phase 0 — Engine skeleton 🟢
The clean-authored workspace: the four component traits, weightless defaults,
the connectome map, the a2a surface, and the end-to-end flow + `rrf` daemon.
All component tests pass; the demo exercises the pipeline.

## Phase 1 — Rigorous foundation 🟡
Make everything after this measurable and gated.
- ADRs + `ARCHITECTURE` / `TESTING` / `ROADMAP` docs.
- CI: `fmt`, `clippy -D warnings`, `nextest`, coverage, `cargo-deny`.
- Property tests (invariants) + criterion benches on existing crates.
- MSRV pin; supply-chain policy (`deny.toml`).

## Phase 2 — Backend abstraction + bake-off ⬜
- `Generator` trait in `rrf-core`; backend features `candle` / `llamacpp` /
  `vllm` / `candle-vllm`; a provider registry resolving backends from config.
- candle in-process forward passes for the DevPULSE embedder (Qwen) and
  reranker (Nemotron) behind `candle`.
- Bake-off harness: recall@k / nDCG / MRR + p50/p95 latency / throughput / RSS,
  emitting a comparable report across backends.

## Phase 3 — Ingestion at scale + live connectome ⬜
- Tokio, signal-driven ingestion **state machine** with backpressure
  (bounded channels + semaphore) and graceful drain.
- Engine state model: **state · trends · tags · shapes**, observable live.
- Connectome renders memory + state + trends + tags + shapes as one graph.
- Soak / load tests; `tokio-console` wiring; leak checks.

## Phase 4 — Recall depth ⬜
- Chunking + document model; metadata filters end-to-end.
- Pluggable ANN index behind `Recall` (graph/HNSW) for larger working sets.
- Persistence + snapshot/restore.

## Phase 5 — Deploy & operability ⬜
- Static musl binary; distroless image; health/readiness; metrics (Prometheus)
  + optional OpenTelemetry; config via env + file.
- Release automation (semver, `cargo-release`, CHANGELOG); `SECURITY.md`.

## Phase 6 — DevPULSE productionization ⬜
- Wire tuned DevPULSE weights (Qwen embedder, Nemotron reranker, learned
  readiness classifier); quantization; warm-load; A/B behind the trait.
- Python training/tuning pipeline handoff (headless Clyffy).
