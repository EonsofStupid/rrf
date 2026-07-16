//! Sprint 13 gate: push-stream changefeed over a2a. One long-lived `watch`
//! connection sees history drained and *live* writes arrive as pushed
//! frames — the client sends exactly one request (no polling). Resume by
//! seq is preserved across reconnects, and token gating covers the stream.

use std::sync::Arc;

use rrf_client::Client;
use rrf_core::{Embedding, Recall, VectorRecord};
use rrf_flow::{FlowNode, ReasonReadyFlow};
use rrf_net::tcp;

fn rec(id: &str, seed: f32) -> VectorRecord {
    VectorRecord::new(
        id,
        Embedding(vec![seed, 1.0 - seed, 0.5, 0.25]),
        format!("watch corpus {id}"),
    )
}

async fn watch_node(estate: Arc<connxism::Estate>, token: Option<&str>) -> std::net::SocketAddr {
    let flow = Arc::new(ReasonReadyFlow::default_engine());
    let mut node = FlowNode::new(flow, "watch-node").with_estate(estate);
    if let Some(t) = token {
        node = node.with_token(t);
    }
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    addr
}

#[tokio::test(flavor = "multi_thread")]
async fn push_stream_sees_live_changes_and_resumes_by_seq() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "w").unwrap());
    let recall = estate.recall();

    // History before anyone subscribes.
    recall
        .upsert(vec![rec("h1", 0.1), rec("h2", 0.2)])
        .await
        .unwrap();

    let addr = watch_node(estate.clone(), None).await;
    let client = Client::new(addr.to_string());

    // One long-lived subscription; every frame lands in the channel.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    let watcher = {
        let client = client.clone();
        tokio::spawn(async move {
            let mut n = 0usize;
            client
                .watch(0, move |change| {
                    let _ = tx.send(change);
                    n += 1;
                    n < 5 // stop after history (2) + live (3)
                })
                .await
        })
    };

    // The drain: history arrives first, in order.
    let mut seen: Vec<serde_json::Value> = Vec::new();
    for _ in 0..2 {
        let f = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("history frame within 5s")
            .expect("channel open");
        seen.push(f);
    }
    assert_eq!(seen[0]["doc_id"], "h1");
    assert_eq!(seen[1]["doc_id"], "h2");

    // LIVE writes: pushed to the already-open connection — the client never
    // sends another request.
    recall
        .upsert(vec![rec("l1", 0.3), rec("l2", 0.4)])
        .await
        .unwrap();
    recall.remove(&"h1".into()).await.unwrap();
    for _ in 0..3 {
        let f = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("live frame within 5s (push, not poll)")
            .expect("channel open");
        seen.push(f);
    }
    assert_eq!(seen[2]["doc_id"], "l1");
    assert_eq!(seen[3]["doc_id"], "l2");
    assert_eq!(seen[4]["doc_id"], "h1");
    assert_eq!(seen[4]["op"], "remove");

    // Seqs strictly increase across the whole stream.
    let seqs: Vec<u64> = seen.iter().map(|c| c["seq"].as_u64().unwrap()).collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "seqs: {seqs:?}");

    // The callback stopped the stream; the returned cursor resumes cleanly.
    let cursor = watcher.await.unwrap().unwrap();
    assert_eq!(cursor, seqs[4] + 1);

    // Writes while nobody watches…
    recall.upsert(vec![rec("r1", 0.6)]).await.unwrap();

    // …arrive exactly once on reconnect from the cursor (no replay).
    let resumed = {
        let client = client.clone();
        tokio::spawn(async move {
            let got = Arc::new(std::sync::Mutex::new(Vec::new()));
            let got2 = got.clone();
            client
                .watch(cursor, move |change| {
                    got2.lock().unwrap().push(change);
                    false // one frame is all we expect
                })
                .await
                .unwrap();
            Arc::try_unwrap(got).unwrap().into_inner().unwrap()
        })
    };
    let resumed = tokio::time::timeout(std::time::Duration::from_secs(5), resumed)
        .await
        .expect("resumed frame within 5s")
        .unwrap();
    assert_eq!(resumed.len(), 1);
    assert_eq!(resumed[0]["doc_id"], "r1");
}

#[tokio::test(flavor = "multi_thread")]
async fn watch_is_token_gated() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "wt").unwrap());
    estate.recall().upsert(vec![rec("d", 0.5)]).await.unwrap();
    let addr = watch_node(estate, Some("s3cret")).await;

    // No token: refused with an error frame → the client surfaces Err.
    let err = Client::new(addr.to_string())
        .watch(0, |_| true)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unauthorized"), "{err}");

    // Bearer: streams.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let client = Client::new(addr.to_string()).with_token("s3cret");
    tokio::spawn(async move {
        client
            .watch(0, move |c| {
                let _ = tx.send(c);
                false
            })
            .await
    });
    let frame = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("frame within 5s")
        .expect("channel open");
    assert_eq!(frame["doc_id"], "d");
}
