//! The `rrf` daemon: the Reason Ready engine as an embedded, signal-driven
//! service with an optional a2a surface.
//!
//! Env:
//! - `RRF_LISTEN` — a2a TCP address (e.g. `127.0.0.1:7878`); unset = disabled.
//! - `RUST_LOG`   — tracing filter (default `info`).

use std::sync::Arc;

use rrf_flow::{init_tracing, sample_corpus, serve, ReasonReadyFlow, ServeOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    // Wire the default (weightless) engine and seed the sample corpus so the
    // daemon answers `ask` immediately. Swap in DevPULSE components here.
    let flow = ReasonReadyFlow::default_engine();
    let n = flow.index(sample_corpus()).await?;
    tracing::info!(indexed = n, "seeded sample corpus");

    let opts = ServeOptions {
        node_id: std::env::var("RRF_NODE").unwrap_or_else(|_| "rrf".to_string()),
        listen: std::env::var("RRF_LISTEN").ok(),
    };

    serve(Arc::new(flow), opts).await?;
    Ok(())
}
