//! # embedder
//!
//! Perception for Reason Ready: text → dense vectors, behind the
//! [`rrf_core::Embedder`] trait.
//!
//! - [`DeterministicEmbedder`] — weightless feature-hashing default. Runs today.
//! - [`DevPulseEmbedder`] — the tuned in-house model (Qwen backbone), wired
//!   behind the `candle` feature.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod deterministic;
mod devpulse;
mod tokenize;

pub use deterministic::DeterministicEmbedder;
pub use devpulse::{DevPulseEmbedder, ModelSpec};

/// Re-export so downstream crates can name the trait without a second dep.
pub use rrf_core::Embedder;
