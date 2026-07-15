//! The DevPULSE (Qwen) embedder plug-point.
//!
//! This is the seam where the tuned in-house embedding model lands. The real
//! forward pass — loading a Qwen-family embedding backbone and running it —
//! goes behind the `candle` feature so the default workspace build stays
//! weightless. Until weights are wired, the struct carries its spec and
//! reports honestly at call time.

use async_trait::async_trait;
use rrf_core::{Embedder, Embedding, Result, RrfError};

/// Which backbone a DevPULSE embedder wraps.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Human/telemetry name, e.g. `devpulse-embed-qwen3-0.6b`.
    pub name: String,
    /// Output dimensionality of the model.
    pub dim: usize,
    /// Filesystem path to the weights (e.g. a `.safetensors` directory).
    pub weights_path: Option<String>,
}

impl ModelSpec {
    /// A Qwen-family embedding backbone spec (the DevPULSE default lineage).
    pub fn qwen(dim: usize) -> Self {
        ModelSpec {
            name: "devpulse-embed-qwen".to_string(),
            dim,
            weights_path: None,
        }
    }
}

/// The DevPULSE embedder. Backbone: Qwen; tuned in-house.
#[derive(Debug, Clone)]
pub struct DevPulseEmbedder {
    spec: ModelSpec,
    #[allow(dead_code)]
    loaded: bool,
}

impl DevPulseEmbedder {
    /// Declare a DevPULSE embedder from a model spec. Does not load weights.
    pub fn new(spec: ModelSpec) -> Self {
        DevPulseEmbedder {
            spec,
            loaded: false,
        }
    }

    /// Load the weights and prepare for inference.
    ///
    /// Wired behind the `candle` feature; without it, loading is unsupported so
    /// callers fall back to the deterministic embedder.
    pub fn load(mut self) -> Result<Self> {
        #[cfg(feature = "candle")]
        {
            // TODO(devpulse): load the Qwen backbone via candle-transformers from
            // `self.spec.weights_path`, build the tokenizer, warm the graph.
            self.loaded = true;
            Ok(self)
        }
        #[cfg(not(feature = "candle"))]
        {
            let _ = &mut self;
            Err(RrfError::Embed(
                "DevPULSE embedder requires the `candle` feature and tuned weights; \
                 use DeterministicEmbedder until they are wired"
                    .into(),
            ))
        }
    }
}

#[async_trait]
impl Embedder for DevPulseEmbedder {
    fn dim(&self) -> usize {
        self.spec.dim
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Embedding>> {
        #[cfg(feature = "candle")]
        {
            // TODO(devpulse): tokenize + forward + mean/pool + normalize.
            Err(RrfError::Embed(
                "DevPULSE candle forward pass not yet wired".into(),
            ))
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(RrfError::Embed(format!(
                "DevPULSE embedder `{}` has no weights loaded (build with --features candle)",
                self.spec.name
            )))
        }
    }

    fn model_name(&self) -> &str {
        &self.spec.name
    }
}
