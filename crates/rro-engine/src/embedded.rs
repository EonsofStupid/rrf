//! Turnkey in-process engine: open an estate, point at HTTP model servers, get a
//! ready [`EmbeddedEngine`]. The assembly the `rro` daemon does, as one call — so
//! a consumer embeds RRO without naming connxism/embedder/reranker/rrd itself.

use std::path::Path;
use std::sync::Arc;

use rro_core::{Document, RecallResult, Result};

use crate::flow::{ObjectBuilder, ReasonReadyObject};

/// An estate plus its assembled flow, held together — the estate stays alive so
/// its out-of-band graph applier keeps running.
pub struct EmbeddedEngine {
    estate: Arc<connxism::Estate>,
    flow: ReasonReadyObject,
}

impl EmbeddedEngine {
    /// Assemble over vLLM model servers + a connxism estate at `path` (named
    /// `name`): RRD front door, HTTP embedder/reranker, the estate's hybrid recall.
    /// Async because the embedder/reranker connect probes the servers (the
    /// embedder reads its dimension) — they must be up.
    pub async fn embed_http(
        path: impl AsRef<Path>,
        name: &str,
        embed_url: &str,
        rerank_url: &str,
    ) -> Result<Self> {
        let estate = Arc::new(connxism::Estate::open(path, name)?);
        let embedder = embedder::OpenAiEmbedder::connect(embedder::OpenAiEmbedConfig::new(
            embed_url,
            embedder::OpenAiKind::Vllm,
        ))
        .await?;
        let reranker = reranker::HttpReranker::connect(reranker::HttpRerankConfig::new(
            rerank_url,
            reranker::HttpRerankKind::Vllm,
        ))
        .await?;
        let flow = ObjectBuilder::new()
            .rrd(Arc::new(rrd::Rrd::new()))
            .recall(Arc::new(estate.recall()))
            .embedder(Arc::new(embedder))
            .reranker(Arc::new(reranker))
            .build();
        Ok(Self { estate, flow })
    }

    /// Ground a query — the full RRO pass (RRD gate → embed → hybrid recall →
    /// rerank → classify).
    pub async fn ask(&self, query: &str) -> Result<RecallResult> {
        self.flow.ask(query).await
    }

    /// Index documents into the estate.
    pub async fn index(&self, docs: Vec<Document>) -> Result<usize> {
        self.flow.index(docs).await
    }

    /// The underlying estate — direct recall + admin (flush/compact/snapshot).
    pub fn estate(&self) -> &connxism::Estate {
        &self.estate
    }
}
