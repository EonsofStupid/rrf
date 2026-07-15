//! # recall
//!
//! Dense vector memory for Reason Ready — the retrieval core, behind the
//! [`rrf_core::Recall`] trait.
//!
//! [`FlatRecall`] is an exact in-memory store. It is the default engine; larger
//! deployments swap an ANN index in behind the same trait.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod flat;

pub use flat::FlatRecall;

/// Re-export so downstream crates can name the trait without a second dep.
pub use rrf_core::Recall;
