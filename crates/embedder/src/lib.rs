//! # embedder
//!
//! Perception for Reason Ready: text → dense vectors, behind the
//! [`rro_core::Embedder`] trait.
//!
//! - [`DeterministicEmbedder`] — weightless feature-hashing default. Runs today.
//! - [`DevPulseEmbedder`] — the tuned in-house model (Qwen backbone), wired
//!   behind the `candle` feature.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "candle")]
mod candle_qwen;
mod deterministic;
mod openai;
mod devpulse;
mod tokenize;

#[cfg(feature = "candle")]
pub use candle_qwen::{CandleQwenEmbedder, QwenEmbedConfig, Qwen3Encoder};
pub use openai::{OpenAiEmbedConfig, OpenAiEmbedder, OpenAiKind};
pub use deterministic::DeterministicEmbedder;
pub use devpulse::{DevPulseEmbedder, ModelSpec};

/// Re-export so downstream crates can name the trait without a second dep.
/// The instruction Qwen3-Embedding prepends to a **query** — never a document.
///
/// Lives at crate level because it is the model's contract, not one backend's
/// detail: candle and the OpenAI-compatible backends must apply the identical
/// prefix or their vectors are not comparable.
pub const DEFAULT_QUERY_TASK: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

pub use rro_core::Embedder;
