//! The `rrf` daemon: the Reason Ready engine as an embedded, signal-driven
//! service with an optional a2a surface.
//!
//! Env:
//! - `RRF_LISTEN` — a2a TCP address (e.g. `127.0.0.1:7878`); unset = disabled.
//! - `RRF_ESTATE` — path to the persistent estate; unset = in-memory.
//! - `RRF_EVENTS` — JSONL event-stream path (DuckDB-ready); unset = disabled.
//! - `RUST_LOG`   — tracing filter (default `info`).

use std::sync::Arc;

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

    // The estate must outlive the daemon: it owns the out-of-band ANN
    // applier thread (dropping it stops graph maintenance).
    let mut estate_handle: Option<Arc<connxism::Estate>> = None;
    let flow = match std::env::var("RRF_ESTATE").ok() {
        Some(path) => {
            let estate = Arc::new(connxism::Estate::open(&path, "rrf")?);
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
                .build();
            estate_handle = Some(estate);
            flow
        }
        None => ReasonReadyFlow::builder().rrd(rrd.clone()).build(),
    };
    let n = flow.index(sample_corpus()).await?;
    tracing::info!(indexed = n, "seeded sample corpus");

    let opts = ServeOptions {
        node_id: std::env::var("RRF_NODE").unwrap_or_else(|_| "rrf".to_string()),
        listen: std::env::var("RRF_LISTEN").ok(),
        estate: estate_handle,
        token: std::env::var("RRF_TOKEN").ok(),
    };

    let estate_for_shutdown = opts.estate.clone();
    serve(Arc::new(flow), opts).await?;

    // Commit the evolved baseline on the way out — the next boot restores it.
    if let Some(estate) = estate_for_shutdown {
        estate.put_component_json("rrd:baseline", &rrd.baseline_snapshot())?;
        tracing::info!("rrd baseline snapshot committed");
    }
    Ok(())
}
