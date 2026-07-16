//! The `rrf` daemon: the Reason Ready engine as an embedded, signal-driven
//! service with an optional a2a surface.
//!
//! Env:
//! - `RRF_LISTEN` — a2a TCP address (e.g. `127.0.0.1:7878`); unset = disabled.
//! - `RRF_ESTATE` — path to the persistent estate; unset = in-memory.
//! - `RRF_EVENTS` — JSONL event-stream path (DuckDB-ready); unset = disabled.
//! - `RUST_LOG`   — tracing filter (default `info`).
//! - `RRF_EMBEDDER` / `RRF_RERANKER` — model selection; see [`model_registry`].
//!   Unset = the weightless deterministic/lexical defaults.

use std::sync::Arc;

use model_registry::{build_embedder, build_reranker, EmbedderConfig, RerankerConfig};
use rrf_flow::{estate_map, init_tracing, sample_corpus, serve, ReasonReadyFlow, ServeOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    // The event stream: every meaningful transition, consistently emitted,
    // straight into DuckDB via read_json_auto().
    if let Ok(path) = std::env::var("RRF_EVENTS") {
        let sink = rrf_core::events::JsonlSink::open(&path)?;
        rrf_core::events::set_sink(Box::new(sink));
        tracing::info!(path, "event stream enabled (JSONL)");
    }

    // With RRF_ESTATE set, memory is the persistent kvs estate (hybrid
    // dense + lexical recall); otherwise the in-memory default. Swap in
    // DevPULSE components here as they land.
    // RRD is the engine's front door: attached to every flow, baseline
    // restored from the estate so predictions are warm from the first query.
    let rrd = Arc::new(rrd::Rrd::new());

    // Model selection is data, not code (docs/MODELS.md §2): config -> boxed
    // trait. Resolving it HERE means a bad kind or a missing feature fails at
    // startup with an actionable message, instead of the daemon coming up and
    // quietly serving synthetic vectors under a real model's name.
    let embed_cfg = EmbedderConfig::from_env()?;
    let rerank_cfg = RerankerConfig::from_env()?;
    let embedder = build_embedder(&embed_cfg)?;
    let reranker = build_reranker(&rerank_cfg)?;
    tracing::info!(
        embedder = embed_cfg.kind.as_str(),
        model = embedder.model_name(),
        dim = embedder.dim(),
        reranker = rerank_cfg.kind.as_str(),
        device = ?embed_cfg.device,
        batch = embed_cfg.batch,
        "models selected"
    );

    // The estate must outlive the daemon: it owns the out-of-band ANN
    // applier thread (dropping it stops graph maintenance).
    let mut estate_handle: Option<Arc<connxism::Estate>> = None;
    let flow = match std::env::var("RRF_ESTATE").ok() {
        Some(path) => {
            let mut config = connxism::EstateConfig::default();
            if std::env::var("RRF_STRICT").map(|v| v == "1" || v == "true") == Ok(true) {
                config.quotas = connxism::Quotas::strict();
                tracing::info!("strict mode: {:?}", config.quotas);
            }
            let estate = Arc::new(connxism::Estate::open_with(&path, "rrf", config)?);
            if let Some(snap) =
                estate.get_component_json::<rrd::BaselineSnapshot>("rrd:baseline")?
            {
                tracing::info!(
                    version = snap.version,
                    observations = snap.observations,
                    "rrd baseline restored"
                );
                rrd.restore_baseline(snap);
            }
            let map = estate_map(&estate)?;
            tracing::info!(
                estate = %estate.info().name,
                nodes = map.nodes.len(),
                edges = map.edges.len(),
                "opened estate"
            );
            let flow = ReasonReadyFlow::builder()
                .rrd(rrd.clone())
                .recall(Arc::new(estate.recall()))
                .embedder(embedder.clone())
                .reranker(reranker.clone())
                .build();
            estate_handle = Some(estate);
            flow
        }
        None => ReasonReadyFlow::builder()
            .rrd(rrd.clone())
            .embedder(embedder.clone())
            .reranker(reranker.clone())
            .build(),
    };
    let n = flow.index(sample_corpus()).await?;
    tracing::info!(indexed = n, "seeded sample corpus");

    let opts = ServeOptions {
        node_id: std::env::var("RRF_NODE").unwrap_or_else(|_| "rrf".to_string()),
        listen: std::env::var("RRF_LISTEN").ok(),
        estate: estate_handle,
        token: std::env::var("RRF_TOKEN").ok(),
    };

    // Ops HTTP surface (prometheus /metrics + health probes), when asked.
    let mut _ops_task = None;
    if let (Ok(ops_addr), Some(estate)) = (std::env::var("RRF_OPS_ADDR"), opts.estate.clone()) {
        let (bound, task) = rrf_flow::ops::serve_ops(&ops_addr, estate).await?;
        tracing::info!(%bound, "ops surface up: /metrics /healthz /livez /readyz");
        _ops_task = Some(task);
    }

    let estate_for_shutdown = opts.estate.clone();
    serve(Arc::new(flow), opts).await?;

    // Commit the evolved baseline on the way out — the next boot restores it.
    if let Some(estate) = estate_for_shutdown {
        estate.put_component_json("rrd:baseline", &rrd.baseline_snapshot())?;
        tracing::info!("rrd baseline snapshot committed");
    }
    Ok(())
}
