//! The shared vocabulary of the engine: documents, chunks, embeddings,
//! queries, candidates, and the reason-ready verdict.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Arbitrary structured metadata carried alongside content.
pub type Metadata = BTreeMap<String, serde_json::Value>;

/// A stable identifier for any addressable unit — document, chunk, or node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Id(pub String);

impl Id {
    /// Wrap an existing identifier.
    pub fn new(s: impl Into<String>) -> Self {
        Id(s.into())
    }

    /// Mint a fresh random identifier.
    pub fn random() -> Self {
        Id(uuid::Uuid::new_v4().to_string())
    }

    /// Borrow as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Id {
    fn from(s: &str) -> Self {
        Id(s.to_string())
    }
}

impl From<String> for Id {
    fn from(s: String) -> Self {
        Id(s)
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A source document, before chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Stable id.
    pub id: Id,
    /// Raw text.
    pub text: String,
    /// Structured metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl Document {
    /// Convenience constructor with a random id and empty metadata.
    pub fn new(text: impl Into<String>) -> Self {
        Document {
            id: Id::random(),
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// Builder-style id override.
    pub fn with_id(mut self, id: impl Into<Id>) -> Self {
        self.id = id.into();
        self
    }
}

/// A retrievable unit of text (a document, post-chunking).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable id.
    pub id: Id,
    /// The document this chunk came from.
    pub doc_id: Id,
    /// Chunk text.
    pub text: String,
    /// Structured metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

/// A dense vector embedding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Wrap a raw vector.
    pub fn new(v: Vec<f32>) -> Self {
        Embedding(v)
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Borrow as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Dot product (unrolled kernel). Returns 0.0 on dimension mismatch.
    pub fn dot(&self, other: &Embedding) -> f32 {
        if self.0.len() != other.0.len() {
            return 0.0;
        }
        crate::simd::dot(&self.0, &other.0)
    }

    /// L2 norm.
    pub fn norm(&self) -> f32 {
        crate::simd::norm_sq(&self.0).sqrt()
    }

    /// Cosine similarity in `[-1, 1]`; 0.0 if either vector is zero-length.
    pub fn cosine(&self, other: &Embedding) -> f32 {
        let denom = self.norm() * other.norm();
        if denom == 0.0 {
            0.0
        } else {
            self.dot(other) / denom
        }
    }

    /// A unit-length copy (no-op if the vector is zero).
    pub fn normalized(&self) -> Embedding {
        let n = self.norm();
        if n == 0.0 {
            self.clone()
        } else {
            Embedding(self.0.iter().map(|x| x / n).collect())
        }
    }
}

/// A retrieval query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// The natural-language query text.
    pub text: String,
    /// How many candidates to return.
    pub top_k: usize,
    /// Optional metadata equality filter (all keys must match).
    #[serde(default)]
    pub filter: Metadata,
}

impl Query {
    /// A query for the top `k` results, no filter.
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Query {
            text: text.into(),
            top_k,
            filter: Metadata::new(),
        }
    }
}

/// A scored retrieval candidate flowing through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// Id of the underlying record.
    pub id: Id,
    /// The candidate text.
    pub text: String,
    /// Current score. Interpretation depends on the stage that set it
    /// (cosine after recall, relevance after rerank).
    pub score: f32,
    /// Structured metadata carried from the record.
    #[serde(default)]
    pub metadata: Metadata,
}

impl Candidate {
    /// Construct a candidate.
    pub fn new(id: impl Into<Id>, text: impl Into<String>, score: f32) -> Self {
        Candidate {
            id: id.into(),
            text: text.into(),
            score,
            metadata: Metadata::new(),
        }
    }
}

/// The reason-ready verdict produced by the classifier daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Readiness {
    /// Whether the retrieved context is sufficient to reason on.
    pub ready: bool,
    /// Confidence in `[0, 1]`.
    pub confidence: f32,
    /// Short machine label (e.g. `ready`, `insufficient`, `ambiguous`).
    pub label: String,
    /// Human-readable rationale.
    pub rationale: String,
}

impl Readiness {
    /// A ready verdict.
    pub fn ready(confidence: f32, rationale: impl Into<String>) -> Self {
        Readiness {
            ready: true,
            confidence,
            label: "ready".into(),
            rationale: rationale.into(),
        }
    }

    /// A not-ready verdict with an explanatory label.
    pub fn not_ready(
        confidence: f32,
        label: impl Into<String>,
        rationale: impl Into<String>,
    ) -> Self {
        Readiness {
            ready: false,
            confidence,
            label: label.into(),
            rationale: rationale.into(),
        }
    }
}

/// The full result of one flow pass: ranked context plus the readiness gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// The query that produced this result.
    pub query: String,
    /// Ranked candidates after recall + rerank.
    pub candidates: Vec<Candidate>,
    /// The reason-ready verdict.
    pub readiness: Readiness,
}
