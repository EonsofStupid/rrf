//! `rrf-bench` — the measurement harness. Real runs, real numbers.
//!
//! Measures, end to end through the ingestion machine (embed → index →
//! persist) and the query path (hybrid search):
//!
//! - ingest throughput (docs/sec) and wall time,
//! - query latency p50 / p95 / p99 over a query mix.
//!
//! Stores: `mem` (in-memory) and `estate` (persistent kvs). External baselines
//! are run *outside* this tree with the same corpus/queries and compared on
//! the emitted numbers — this repo carries only its own engine.
//!
//! ```sh
//! cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store estate
//! ```

use std::sync::Arc;
use std::time::Instant;

use embedder::DeterministicEmbedder;
use recall::FlatRecall;
use rrf_core::{Document, Embedder, Recall};
use rrf_flow::{spawn_ingest, IngestConfig};

/// Base roots composed into a realistic synthetic vocabulary (~8k distinct
/// terms — real corpora are zipfian over 10⁴–10⁶ terms, not a handful).
const ROOTS: &[&str] = &[
    "estate",
    "vector",
    "recall",
    "storage",
    "upgrade",
    "migration",
    "tokio",
    "signal",
    "daemon",
    "reranker",
    "network",
    "graph",
    "memory",
    "agent",
    "mesh",
    "warp",
    "connector",
    "drive",
    "mailbox",
    "index",
    "shard",
    "batch",
    "trend",
    "shape",
    "tag",
    "readiness",
    "reason",
    "flow",
    "hybrid",
    "lexical",
    "cosine",
    "fusion",
    "ingest",
    "backpressure",
];
const VOCAB_SIZE: u64 = 8192;

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Draw one term: a root suffixed into one of `VOCAB_SIZE` distinct words,
/// with a zipf-ish skew (low suffixes are much more common).
fn synth_term(seed: &mut u64) -> String {
    let root = ROOTS[(xorshift(seed) % ROOTS.len() as u64) as usize];
    // Square a uniform draw to skew mass toward small suffixes.
    let u = (xorshift(seed) % 1000) as f64 / 1000.0;
    let suffix = ((u * u) * (VOCAB_SIZE as f64 / ROOTS.len() as f64)) as u64;
    format!("{root}{suffix}")
}

fn synth_doc(i: usize, seed: &mut u64) -> Document {
    let len = 24 + (xorshift(seed) % 40) as usize;
    let words: Vec<String> = (0..len).map(|_| synth_term(seed)).collect();
    Document::new(words.join(" ")).with_id(format!("doc-{i}"))
}

fn synth_query(seed: &mut u64) -> String {
    let len = 2 + (xorshift(seed) % 3) as usize;
    let words: Vec<String> = (0..len).map(|_| synth_term(seed)).collect();
    words.join(" ")
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_str(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn percentile(sorted_us: &[u128], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx] as f64 / 1000.0
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let docs = arg("--docs", 10_000);
    let queries = arg("--queries", 200);
    let batch = arg("--batch", 256);
    let concurrency = arg("--concurrency", 4);
    let top_k = arg("--top-k", 10);
    let store_kind = arg_str("--store", "mem");

    let embedder: Arc<dyn Embedder> = Arc::new(DeterministicEmbedder::new());

    // Keep the estate dir alive for the whole run.
    let tmp;
    let store: Arc<dyn Recall> = match store_kind.as_str() {
        "estate" => {
            tmp = tempfile::tempdir()?;
            let estate = connxism::Estate::open(tmp.path(), "bench")?;
            Arc::new(estate.recall())
        }
        _ => Arc::new(FlatRecall::new()),
    };

    println!(
        "# rrf-bench — store={store_kind} docs={docs} batch={batch} concurrency={concurrency}\n"
    );

    // ---- ingest: full machine (embed → index → persist) ----
    let handle = spawn_ingest(
        embedder.clone(),
        store.clone(),
        IngestConfig {
            batch_size: batch,
            concurrency,
            ..IngestConfig::default()
        },
    );
    let mut seed = 0x5EED_u64;
    let t0 = Instant::now();
    for i in 0..docs {
        handle.submit(synth_doc(i, &mut seed)).await?;
    }
    let stats = handle.finish().await?;
    let ingest_secs = t0.elapsed().as_secs_f64();

    assert_eq!(stats.indexed as usize, docs, "all docs must index");
    assert_eq!(store.len().await? as usize, docs);

    println!("## ingest");
    println!("| metric | value |");
    println!("|---|---|");
    println!("| documents | {docs} |");
    println!("| wall time | {ingest_secs:.2} s |");
    println!("| throughput | {:.0} docs/sec |", stats.docs_per_sec);
    println!("| batches | {} |", stats.batches);
    println!("| errors | {} |", stats.errors);

    // ---- query: hybrid latency ----
    let mut lat_us: Vec<u128> = Vec::with_capacity(queries);
    let mut qseed = 0xFACADE_u64;
    for _ in 0..queries {
        let q = synth_query(&mut qseed);
        let emb = embedder.embed_one(&q).await?;
        let t = Instant::now();
        let hits = store.hybrid_search(&q, &emb, top_k).await?;
        lat_us.push(t.elapsed().as_micros());
        assert!(!hits.is_empty(), "queries over a populated store must hit");
    }
    lat_us.sort_unstable();

    println!("\n## query (hybrid, top-{top_k}, {queries} queries)");
    println!("| percentile | latency |");
    println!("|---|---|");
    println!("| p50 | {:.2} ms |", percentile(&lat_us, 0.50));
    println!("| p95 | {:.2} ms |", percentile(&lat_us, 0.95));
    println!("| p99 | {:.2} ms |", percentile(&lat_us, 0.99));
    println!(
        "| throughput | {:.0} qps (sequential) |",
        1000.0 / percentile(&lat_us, 0.50).max(1e-9)
    );

    Ok(())
}
