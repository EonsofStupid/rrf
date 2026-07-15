# Reason Ready — Architecture

> Recall finds. Reranker orders. Classifier judges readiness. Embedder
> perceives. Connectome shows. One faithful engine, one flow.

Reason Ready (RRF) is an embedded, tokio-native retrieval-and-reasoning engine.
It is clean-authored, single-binary, and stands on its own — no external
database, vector store, or model gateway is required to run it. Model inference
is a *pluggable boundary*, never a hard dependency.

## Design laws

1. **Everything swappable is a trait in `rrf-core`.** The flow depends on
   capabilities, not implementations. Components are chosen at build time
   (cargo features) and/or runtime (config).
2. **Weightless by default.** The whole pipeline compiles and runs with zero
   model weights, so the engine is always testable and always deployable. Model
   backends are additive.
3. **Embedded is not isolated.** The engine runs in-process on tokio, but keeps
   a first-class a2a / node networking surface.
4. **The map is data.** The connectome is a serializable graph; no rendering
   runtime is baked in, so any front end can consume it.
5. **No unsafe.** Every crate is `#![forbid(unsafe_code)]`.

## Crate map

```
rrf-core ──────────────── the contract: domain types + component traits
   ▲  ▲  ▲  ▲
   │  │  │  └── classifier   Reason Ready daemon (readiness judgment + service)
   │  │  └───── reranker     true-relevance ordering (BM25 default; DevPULSE)
   │  └──────── recall       dense vector memory (flat cosine; pluggable ANN)
   └─────────── embedder     perception: text → vectors (deterministic; DevPULSE)

connectome ────────────── the visual/relational map (graph model + render)
rrf-net ───────────────── a2a / node surface (in-proc bus + TCP transport)
rrf-flow ──────────────── orchestrator + `rrf` daemon + demo (depends on all)
```

`rrf-core` is the single source of truth. No crate depends on another
component crate's internals — only on the traits and types in `rrf-core`.
`rrf-flow` is the only crate that composes concrete implementations.

## The flow (one pass)

```
query ─▶ embedder.embed_one ─▶ recall.search(recall_k) ─▶ reranker.rerank(rerank_k)
        ─▶ classifier.classify ─▶ RecallResult ─▶ connectome.map ─▶ ConnectomeGraph
```

Each arrow is a trait call. Replace any stage's implementation and the flow is
unchanged.

## Inference backends (the pluggable boundary)

Model-backed components (`Embedder`, `Reranker`, and the future `Generator`)
resolve to a backend selected by cargo feature + config. See
[ADR-0001](adr/0001-inference-backends.md) for the rationale. The topology:

| Backend | Runs | Best for | Feature |
|---|---|---|---|
| **candle** | in-process (Rust) | small encoders: embed / rerank / classify (DevPULSE) | `candle` |
| **llama.cpp** | local server / FFI | quantized, CPU, edge | `llamacpp` |
| **vLLM** | external GPU server (OpenAI API) | large-LLM generation at scale | `vllm` |
| **candle-vllm** | in-process (Rust, experimental) | Rust-native generation bet | `candle-vllm` |

The engine (memory, retrieval, state, routing, a2a, the deployable binary) is
Rust. Training/tuning of DevPULSE models is Python. We do **not** reimplement
vLLM — we drive it behind the trait. Bake-offs run the same eval set across
backends and compare quality + latency (see [TESTING](TESTING.md)).

## State, trends, tags, shapes → one connectome

The engine tracks its own life as first-class, observable state:

- **State** — the ingestion machine (`Idle → Ingesting → Indexed`) with
  per-batch progress, counts, errors, and timestamps.
- **Trends** — time-series over ingest/query counts, latency, and recall
  quality; the graphs.
- **Tags** — a taxonomy over documents/collections; filterable.
- **Shapes** — the schema/modality fingerprint of ingested content.

The connectome renders memory + state + trends + tags + shapes as one graph, so
a non-technical viewer can *see* what the engine knows and how it is doing.

## Deployment

A single static (musl) binary; distroless/scratch image; config via env + file;
health/readiness endpoints; graceful shutdown on SIGTERM/Ctrl-C. Compile only
the backends you need via features. The artifact drops into a host (Clyffy)
in-process or runs standalone as the `rrf` daemon.
