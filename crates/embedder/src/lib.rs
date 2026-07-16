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
mod devpulse;
mod tokenize;

#[cfg(feature = "candle")]
pub use candle_qwen::{CandleQwenEmbedder, QwenEmbedConfig, Qwen3Encoder, DEFAULT_QUERY_TASK};
pub use deterministic::DeterministicEmbedder;
pub use devpulse::{DevPulseEmbedder, ModelSpec};

/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Embedder;
