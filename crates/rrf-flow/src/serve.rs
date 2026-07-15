//! The embedded, signal-driven runtime.
//!
//! `serve` runs the engine as a long-lived daemon: it optionally opens the a2a
//! TCP surface and then parks on OS shutdown signals (Ctrl-C / SIGTERM),
//! draining cleanly. This is "embedded" in the sense that matters — it is one
//! tokio process you can drop into a host — without giving up the network.

use std::sync::Arc;

use rrf_core::Result;
use rrf_net::NodeId;

use crate::flow::ReasonReadyFlow;
use crate::handler::FlowNode;

/// How the daemon should run.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// This node's a2a id.
    pub node_id: String,
    /// If set, open the a2a TCP surface on this address (e.g. `127.0.0.1:7878`).
    pub listen: Option<String>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions {
            node_id: "rrf".to_string(),
            listen: None,
        }
    }
}

/// Run the engine until a shutdown signal arrives.
pub async fn serve(flow: Arc<ReasonReadyFlow>, opts: ServeOptions) -> Result<()> {
    for (stage, model) in flow.model_names() {
        tracing::info!(stage, model, "component");
    }

    let node = Arc::new(FlowNode::new(flow.clone(), NodeId::new(&opts.node_id)));

    let _server = match &opts.listen {
        Some(addr) => {
            let (bound, task) = rrf_net::tcp::serve(addr.clone(), node.clone()).await?;
            tracing::info!(%bound, "a2a surface listening");
            Some(task)
        }
        None => {
            tracing::info!("a2a surface disabled (set listen to enable)");
            None
        }
    };

    tracing::info!("reason ready — awaiting shutdown signal (Ctrl-C / SIGTERM)");
    wait_for_shutdown().await;
    tracing::info!("shutdown signal received — stopping");
    Ok(())
}

/// Block until Ctrl-C or (on Unix) SIGTERM.
pub async fn wait_for_shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("cannot install SIGTERM handler: {e}; using Ctrl-C only");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
