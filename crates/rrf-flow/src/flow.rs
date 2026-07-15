//! The Reason Ready flow: the one pass that ties the components together.

use std::sync::Arc;

use classifier::HeuristicClassifier;
use connectome::{Connectome, ConnectomeGraph};
use embedder::DeterministicEmbedder;
use recall::FlatRecall;
use reranker::LexicalReranker;
use rrf_core::{
    Classifier, Document, Embedder, Recall, RecallResult, Reranker, Result, VectorRecord,
};

/// How wide each stage runs.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Candidates pulled from recall (vector) before reranking.
    pub recall_k: usize,
    /// Candidates kept after reranking (and handed to the classifier).
    pub rerank_k: usize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        FlowConfig {
            recall_k: 20,
            rerank_k: 5,
        }
    }
}

/// The assembled engine: embedder → recall → reranker → classifier → connectome.
pub struct ReasonReadyFlow {
    embedder: Arc<dyn Embedder>,
    recall: Arc<dyn Recall>,
    reranker: Arc<dyn Reranker>,
    classifier: Arc<dyn Classifier>,
    connectome: Connectome,
    config: FlowConfig,
}

impl ReasonReadyFlow {
    /// Start building a flow.
    pub fn builder() -> FlowBuilder {
        FlowBuilder::new()
    }

    /// The default, weightless engine: deterministic embedder, flat recall,
    /// BM25 reranker, heuristic reason-ready classifier. Runs today, no weights.
    pub fn default_engine() -> Self {
        FlowBuilder::new().build()
    }

    /// Index documents: embed each, then upsert into recall. Returns the new
    /// total record count.
    pub async fn index(&self, docs: Vec<Document>) -> Result<usize> {
        if docs.is_empty() {
            return self.recall.len().await;
        }
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts).await?;
        let records: Vec<VectorRecord> = docs
            .into_iter()
            .zip(embeddings)
            .map(|(d, e)| {
                let mut r = VectorRecord::new(d.id, e, d.text);
                r.metadata = d.metadata;
                r
            })
            .collect();
        self.recall.upsert(records).await?;
        self.recall.len().await
    }

    /// Run one full pass for `query`: embed → recall → rerank → classify.
    ///
    /// Recall is *hybrid*: stores that maintain a lexical index fuse dense and
    /// lexical rankings (reciprocal rank fusion); pure vector stores fall back
    /// to dense search via the trait's default.
    pub async fn ask(&self, query: &str) -> Result<RecallResult> {
        let q = self.embedder.embed_one(query).await?;
        let recalled = self
            .recall
            .hybrid_search(query, &q, self.config.recall_k)
            .await?;
        let ranked = self
            .reranker
            .rerank(query, recalled, self.config.rerank_k)
            .await?;
        let readiness = self.classifier.classify(query, &ranked).await?;
        Ok(RecallResult {
            query: query.to_string(),
            candidates: ranked,
            readiness,
        })
    }

    /// Build the visual map for a completed pass.
    pub fn connectome(&self, result: &RecallResult) -> ConnectomeGraph {
        self.connectome
            .map(&result.query, &result.candidates, &result.readiness)
    }

    /// Convenience: run a pass and build its map in one call.
    pub async fn ask_with_map(&self, query: &str) -> Result<(RecallResult, ConnectomeGraph)> {
        let result = self.ask(query).await?;
        let map = self.connectome(&result);
        Ok((result, map))
    }

    /// The active configuration.
    pub fn config(&self) -> &FlowConfig {
        &self.config
    }

    /// Names of the active component models, for telemetry / the connectome.
    pub fn model_names(&self) -> [(&'static str, &str); 4] {
        [
            ("embedder", self.embedder.model_name()),
            ("recall", "flat-cosine"),
            ("reranker", self.reranker.model_name()),
            ("classifier", self.classifier.model_name()),
        ]
    }
}

/// Fluent builder for [`ReasonReadyFlow`]. Any component left unset falls back
/// to its weightless default.
pub struct FlowBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    recall: Option<Arc<dyn Recall>>,
    reranker: Option<Arc<dyn Reranker>>,
    classifier: Option<Arc<dyn Classifier>>,
    config: FlowConfig,
}

impl Default for FlowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowBuilder {
    /// A builder with all-default components.
    pub fn new() -> Self {
        FlowBuilder {
            embedder: None,
            recall: None,
            reranker: None,
            classifier: None,
            config: FlowConfig::default(),
        }
    }

    /// Override the embedder (e.g. the DevPULSE / Qwen model).
    pub fn embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(e);
        self
    }

    /// Override the recall store.
    pub fn recall(mut self, r: Arc<dyn Recall>) -> Self {
        self.recall = Some(r);
        self
    }

    /// Override the reranker (e.g. the DevPULSE / Nemotron model).
    pub fn reranker(mut self, r: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(r);
        self
    }

    /// Override the reason-ready classifier.
    pub fn classifier(mut self, c: Arc<dyn Classifier>) -> Self {
        self.classifier = Some(c);
        self
    }

    /// Set the flow widths.
    pub fn config(mut self, config: FlowConfig) -> Self {
        self.config = config;
        self
    }

    /// Assemble the flow, filling any unset component with its default.
    pub fn build(self) -> ReasonReadyFlow {
        ReasonReadyFlow {
            embedder: self
                .embedder
                .unwrap_or_else(|| Arc::new(DeterministicEmbedder::new()) as Arc<dyn Embedder>),
            recall: self
                .recall
                .unwrap_or_else(|| Arc::new(FlatRecall::new()) as Arc<dyn Recall>),
            reranker: self
                .reranker
                .unwrap_or_else(|| Arc::new(LexicalReranker::new()) as Arc<dyn Reranker>),
            classifier: self
                .classifier
                .unwrap_or_else(|| Arc::new(HeuristicClassifier::new()) as Arc<dyn Classifier>),
            connectome: Connectome::new(),
            config: self.config,
        }
    }
}
