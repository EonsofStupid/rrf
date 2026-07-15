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

/// Emit one pipeline-stage event: the flow is one engine, and the event
/// stream shows every stage of every pass (`flow.stage` + `flow.pass`).
fn stage(name: &str, since: std::time::Instant, mut fields: serde_json::Value) {
    if let Some(obj) = fields.as_object_mut() {
        obj.insert(
            "stage".to_string(),
            serde_json::Value::String(name.to_string()),
        );
        obj.insert(
            "ms".to_string(),
            serde_json::json!(since.elapsed().as_micros() as f64 / 1000.0),
        );
    }
    rrf_core::events::emit("flow.stage", fields);
}

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

/// The assembled engine: **RRD first**, then embedder → recall → reranker →
/// classifier → connectome.
pub struct ReasonReadyFlow {
    rrd: Option<Arc<rrd::Rrd>>,
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
        use std::time::Instant;
        let pass = Instant::now();

        // RRD is literally the instant first thing: stamp + gate ladder on
        // the query BEFORE any model cost. A blocked query never reaches the
        // embedder — it returns gated, with the verdict on the record.
        let mut query_rro = None;
        if let Some(rrd) = &self.rrd {
            let t = Instant::now();
            let stamp = rrd::SourceStamp {
                channel: Some("query".to_string()),
                ..rrd::SourceStamp::default()
            };
            let rro = rrd.distill_stamped("query", query, &rrf_core::Metadata::new(), None, stamp);
            stage(
                "rrd",
                t,
                serde_json::json!({ "gate": rro.gate, "mode": rro.mode.name() }),
            );
            if rro.gate == rrd::GateVerdict::Block {
                return Ok(RecallResult {
                    query: query.to_string(),
                    candidates: Vec::new(),
                    readiness: rrf_core::Readiness::not_ready(
                        1.0,
                        "gated",
                        "blocked by the RRD deterministic gate before any model ran",
                    ),
                    intent: Vec::new(),
                });
            }
            query_rro = Some(rro);
        }

        let t = Instant::now();
        let q = self.embedder.embed_one(query).await?;
        stage("embed", t, serde_json::json!({ "dim": q.dim() }));

        // Intent: the L2 half of the query's distillation, on the embedding
        // we just paid for anyway.
        let intent: Vec<String> = match (&self.rrd, &query_rro) {
            (Some(rrd), Some(_)) => rrd.route_tags(&q).into_iter().map(|t| t.tag).collect(),
            _ => Vec::new(),
        };

        let t = Instant::now();
        let recalled = self
            .recall
            .hybrid_search(query, &q, self.config.recall_k)
            .await?;
        stage(
            "recall",
            t,
            serde_json::json!({ "candidates": recalled.len() }),
        );

        let t = Instant::now();
        let ranked = self
            .reranker
            .rerank(query, recalled, self.config.rerank_k)
            .await?;
        stage("rerank", t, serde_json::json!({ "kept": ranked.len() }));

        let t = Instant::now();
        let readiness = self.classifier.classify(query, &ranked).await?;
        stage(
            "classify",
            t,
            serde_json::json!({ "ready": readiness.ready, "confidence": readiness.confidence }),
        );

        rrf_core::events::emit(
            "flow.pass",
            serde_json::json!({
                "total_ms": pass.elapsed().as_millis() as u64,
                "ready": readiness.ready,
                "candidates": ranked.len(),
            }),
        );

        Ok(RecallResult {
            query: query.to_string(),
            candidates: ranked,
            readiness,
            intent,
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
    rrd: Option<Arc<rrd::Rrd>>,
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
            rrd: None,
            embedder: None,
            recall: None,
            reranker: None,
            classifier: None,
            config: FlowConfig::default(),
        }
    }

    /// Attach RRD as the flow's front door (query gating + intent routing).
    pub fn rrd(mut self, rrd: Arc<rrd::Rrd>) -> Self {
        self.rrd = Some(rrd);
        self
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
            rrd: self.rrd,
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
