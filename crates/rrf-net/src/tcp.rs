//! TCP a2a: the same [`Handler`] contract, over the wire.
//!
//! Framing is newline-delimited JSON — one [`Message`] per line. `Message`
//! serializes without embedded newlines, so this frames unambiguously while
//! staying trivial to inspect and to speak from any language.

use std::sync::Arc;

use rrf_core::{Result, RrfError};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};

use crate::handler::Handler;
use crate::message::Message;

/// Serve `handler` over TCP until the listener is dropped.
///
/// Returns the bound [`std::net::SocketAddr`] and the accept-loop
/// [`tokio::task::JoinHandle`]. Bind to `127.0.0.1:0` to get an OS-assigned
/// port (read it from the returned address).
pub async fn serve(
    addr: impl ToSocketAddrs,
    handler: Arc<dyn Handler>,
) -> Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| RrfError::Net(format!("bind: {e}")))?;
    let local = listener
        .local_addr()
        .map_err(|e| RrfError::Net(format!("local_addr: {e}")))?;

    let task = tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _peer)) => {
                    let h = handler.clone();
                    tokio::spawn(async move {
                        if let Err(e) = serve_conn(stream, h).await {
                            tracing::debug!("rrf-net conn ended: {e}");
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("rrf-net accept error: {e}");
                    break;
                }
            }
        }
    });
    Ok((local, task))
}

async fn serve_conn(stream: TcpStream, handler: Arc<dyn Handler>) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();
    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|e| RrfError::Net(format!("read: {e}")))?
    {
        if line.trim().is_empty() {
            continue;
        }
        let msg: Message = serde_json::from_str(&line)?;
        if let Some(reply) = handler.handle(msg).await? {
            let mut buf = serde_json::to_string(&reply)?;
            buf.push('\n');
            write_half
                .write_all(buf.as_bytes())
                .await
                .map_err(|e| RrfError::Net(format!("write: {e}")))?;
        }
    }
    Ok(())
}

/// Open a connection, send one request, and await a single reply.
pub async fn request(addr: impl ToSocketAddrs, msg: &Message) -> Result<Message> {
    let stream = TcpStream::connect(addr)
        .await
        .map_err(|e| RrfError::Net(format!("connect: {e}")))?;
    let (read_half, mut write_half) = stream.into_split();

    let mut buf = serde_json::to_string(msg)?;
    buf.push('\n');
    write_half
        .write_all(buf.as_bytes())
        .await
        .map_err(|e| RrfError::Net(format!("write: {e}")))?;

    let mut lines = BufReader::new(read_half).lines();
    let line = lines
        .next_line()
        .await
        .map_err(|e| RrfError::Net(format!("read: {e}")))?
        .ok_or_else(|| RrfError::Net("connection closed before reply".into()))?;
    Ok(serde_json::from_str(&line)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handler::PingHandler;
    use crate::message::NodeId;

    #[tokio::test]
    async fn tcp_ping_pong() {
        let handler = Arc::new(PingHandler {
            me: NodeId::new("server"),
        });
        let (addr, _task) = serve("127.0.0.1:0", handler).await.unwrap();
        let reply = request(
            addr,
            &Message::request("client", "server", "ping", serde_json::json!({})),
        )
        .await
        .unwrap();
        assert_eq!(reply.body["pong"], serde_json::json!(true));
    }
}
