//! Expose the flow to peers over a2a.
//!
//! A [`FlowNode`] wraps a shared flow and answers a2a messages, so a remote (or
//! co-located) agent can `ask` the engine without owning it. Same [`Handler`]
//! contract for local and TCP transports.

use std::sync::Arc;

use async_trait::async_trait;
use rrf_core::{Document, Result};
use rrf_net::{Handler, Message, NodeId};

use crate::flow::ReasonReadyFlow;

/// A network-facing node backed by a [`ReasonReadyFlow`].
pub struct FlowNode {
    flow: Arc<ReasonReadyFlow>,
    me: NodeId,
}

impl FlowNode {
    /// Wrap a flow as an addressable node.
    pub fn new(flow: Arc<ReasonReadyFlow>, me: impl Into<NodeId>) -> Self {
        FlowNode {
            flow,
            me: me.into(),
        }
    }
}

#[async_trait]
impl Handler for FlowNode {
    async fn handle(&self, msg: Message) -> Result<Option<Message>> {
        match msg.verb.as_str() {
            "ping" => Ok(Some(msg.reply(serde_json::json!({
                "pong": true,
                "node": self.me.as_str(),
            })))),

            // `ask` / `recall`: run the flow for `body.query`.
            "ask" | "recall" => {
                let query = msg.body.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let result = self.flow.ask(query).await?;
                Ok(Some(msg.reply(serde_json::to_value(&result)?)))
            }

            // `map`: run the flow and return the connectome graph.
            "map" => {
                let query = msg.body.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let (_result, graph) = self.flow.ask_with_map(query).await?;
                Ok(Some(msg.reply(serde_json::to_value(&graph)?)))
            }

            // `index`: ingest a batch of documents over a2a.
            // Body: {"docs": [{"id": "...", "text": "..."}, ...]}
            "index" => {
                let docs: Vec<Document> = msg
                    .body
                    .get("docs")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()?
                    .unwrap_or_default();
                let total = self.flow.index(docs).await?;
                Ok(Some(msg.reply(serde_json::json!({ "total": total }))))
            }

            _ => Ok(None),
        }
    }
}
