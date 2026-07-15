//! What a node does with an incoming message.

use async_trait::async_trait;
use rrf_core::Result;

use crate::message::Message;

/// A node's behaviour: answer (or ignore) inbound a2a messages.
///
/// The engine implements this to expose recall/classify over the network; a
/// bare node might implement `ping`. Returning `Ok(None)` means "no reply".
#[async_trait]
pub trait Handler: Send + Sync {
    /// Handle one message, optionally producing a reply.
    async fn handle(&self, msg: Message) -> Result<Option<Message>>;
}

/// A trivial handler that replies to `ping` with `pong` and ignores the rest.
/// Useful as a liveness endpoint and as a test double.
pub struct PingHandler {
    /// This node's id (used as the reply sender).
    pub me: crate::message::NodeId,
}

#[async_trait]
impl Handler for PingHandler {
    async fn handle(&self, msg: Message) -> Result<Option<Message>> {
        if msg.verb == "ping" {
            Ok(Some(msg.reply(serde_json::json!({ "pong": true, "node": self.me.as_str() }))))
        } else {
            Ok(None)
        }
    }
}
