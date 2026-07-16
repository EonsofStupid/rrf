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
mod http;
mod devpulse;

pub use bm25::LexicalReranker;
pub use http::{HttpRerankConfig, HttpRerankKind, HttpReranker};
pub use devpulse::{DevPulseReranker, RerankSpec};

/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Reranker;
