//! # reranker
//!
//! True-relevance ordering over recall candidates, behind the
//! [`rro_core::Reranker`] trait.
//!
//! - [`LexicalReranker`] — weightless Okapi BM25 default. Runs today.
//! - [`DevPulseReranker`] — the tuned cross-encoder (Nemotron backbone), wired
//!   behind the `candle` feature.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bm25;
#[cfg(feature = "candle")]
mod candle_qwen;
mod devpulse;
mod http;

pub use bm25::LexicalReranker;
#[cfg(feature = "candle")]
pub use candle_qwen::{CandleQwenReranker, CandleRerankConfig, DEFAULT_RERANK_TASK};
pub use devpulse::{DevPulseReranker, RerankSpec};
pub use http::{HttpRerankConfig, HttpRerankKind, HttpReranker};

/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Reranker;
