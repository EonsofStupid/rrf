//! The DevPULSE (Nemotron) reranker plug-point.
//!
//! Cross-encoder relevance scoring of (query, candidate) pairs. The real model
//! — a Nemotron-family reranker tuned in-house — loads behind the `candle`
//! feature. Until then it reports honestly and callers fall back to
//! [`crate::LexicalReranker`].

use async_trait::async_trait;
use rrf_core::{Candidate, Reranker, Result, RrfError};

/// Spec for a DevPULSE reranker backbone.
#[derive(Debug, Clone)]
pub struct RerankSpec {
    /// Telemetry name, e.g. `devpulse-rerank-nemotron`.
    pub name: String,
    /// Path to the weights.
    pub weights_path: Option<String>,
}

impl RerankSpec {
    /// A Nemotron-family reranker spec (the DevPULSE default lineage).
    pub fn nemotron() -> Self {
        RerankSpec {
            name: "devpulse-rerank-nemotron".to_string(),
            weights_path: None,
        }
    }
}

/// The DevPULSE reranker. Backbone: Nemotron; tuned in-house.
#[derive(Debug, Clone)]
pub struct DevPulseReranker {
    spec: RerankSpec,
    #[allow(dead_code)]
    loaded: bool,
}

impl DevPulseReranker {
    /// Declare a DevPULSE reranker from a spec. Does not load weights.
    pub fn new(spec: RerankSpec) -> Self {
        DevPulseReranker {
            spec,
            loaded: false,
        }
    }

    /// Load the cross-encoder weights (behind the `candle` feature).
    pub fn load(mut self) -> Result<Self> {
        #[cfg(feature = "candle")]
        {
            // TODO(devpulse): load the Nemotron cross-encoder via candle.
            self.loaded = true;
            Ok(self)
        }
        #[cfg(not(feature = "candle"))]
        {
            let _ = &mut self;
            Err(RrfError::Rerank(
                "DevPULSE reranker requires the `candle` feature and tuned weights; \
                 use LexicalReranker until they are wired"
                    .into(),
            ))
        }
    }
}

#[async_trait]
impl Reranker for DevPulseReranker {
    async fn rerank(
        &self,
        _query: &str,
        _candidates: Vec<Candidate>,
        _top_k: usize,
    ) -> Result<Vec<Candidate>> {
        Err(RrfError::Rerank(format!(
            "DevPULSE reranker `{}` has no weights loaded (build with --features candle)",
            self.spec.name
        )))
    }

    fn model_name(&self) -> &str {
        &self.spec.name
    }
}
