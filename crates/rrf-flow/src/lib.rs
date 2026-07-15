//! # rrf-flow
//!
//! The orchestrator. It assembles the components into one Reason Ready pass
//! (embed → recall → rerank → classify → connectome) and runs the embedded,
//! signal-driven daemon with an a2a surface.
//!
//! ```no_run
//! # async fn run() -> rrf_core::Result<()> {
//! use rrf_flow::{ReasonReadyFlow, sample_corpus};
//!
//! let flow = ReasonReadyFlow::default_engine();
//! flow.index(sample_corpus()).await?;
//! let (result, map) = flow.ask_with_map("how do I upgrade postgres safely?").await?;
//! println!("ready = {}", result.readiness.ready);
//! println!("{}", map.to_json()?);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod estate_map;
mod flow;
mod handler;
mod ingest;
mod sample;
mod serve;

pub use estate_map::estate_map;
pub use flow::{FlowBuilder, FlowConfig, ReasonReadyFlow};
pub use handler::FlowNode;
pub use ingest::{
    spawn_ingest, IngestConfig, IngestHandle, IngestPhase, IngestStats, IngestStatus,
};
pub use sample::sample_corpus;
pub use serve::{serve, wait_for_shutdown, ServeOptions};

// Re-export the shared surface so a consumer needs only `rrf-flow`.
pub use rrf_core::{self as core, Candidate, Document, Query, Readiness, RecallResult};

/// Install a default tracing subscriber honouring `RUST_LOG` (default `info`).
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}
