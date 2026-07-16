//! `rro-eval` — the honest number. Real IR benchmarks, real models, ablations.
//!
//! Every accuracy figure RRO has ever published came from the deterministic
//! hash embedder scoring synthetic vectors against synthetic vectors. This
//! binary exists to replace those with numbers that mean something, on public
//! benchmarks with third-party relevance judgments nobody here wrote.
//!
//! ## The ablation ladder
//!
//! Results are dismissible unless you can see what each layer bought. So every
//! run reports the same queries through progressively more of the engine:
//!
//! | arm | path |
//! |---|---|
//! | `bm25`   | lexical only — the floor. No model, no vectors. |
//! | `dense`  | ANN over embeddings only. |
//! | `hybrid` | dense + BM25, reciprocal-rank-fused. |
//! | `rro`    | hybrid + cross-encoder rerank (the full object). |
//!
//! If `hybrid` doesn't beat `dense`, fusion isn't earning its cost. If `rro`
//! doesn't beat `hybrid`, the reranker isn't earning its latency. Either is a
//! real finding and gets printed, not buried.
//!
//! ## Metrics
//!
//! nDCG@10 (the BEIR/BRIGHT standard, and graded — nfcorpus qrels score 0..2),
//! plus Recall@10 and MRR@10. nDCG is the headline because it is what published
//! baselines report, which is the only way a claim here is checkable.
//!
//! ## Usage
//!
//! ```sh
//! RRO_EMBEDDER=llamacpp \
//! RRO_EVAL_DATA=eval-data/nfcorpus \
//!   cargo run --release --bin rro-eval
//!
//! # add the reranker arm
//! RRO_RERANKER=vllm RRO_EMBEDDER=llamacpp RRO_EVAL_DATA=eval-data/nfcorpus \
//!   cargo run --release --bin rro-eval
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use model_registry::{build_embedder, build_reranker, EmbedderConfig, RerankerConfig};
use rro_core::{Candidate, Document, Recall, Reranker, VectorRecord};

/// One benchmark query with its graded judgments.
struct EvalQuery {
    id: String,
    text: String,
    /// doc id -> relevance grade (>0 is relevant; nfcorpus uses 0..2).
    rels: HashMap<String, u8>,
}

fn main() -> anyhow::Result<()> {
    rro_engine::init_tracing();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::var("RRO_EVAL_DATA")
        .unwrap_or_else(|_| "eval-data/nfcorpus".to_string())
        .into();
    let top_k: usize = env_usize("RRO_EVAL_K", 10);
    let recall_k: usize = env_usize("RRO_EVAL_RECALL_K", 100);
    let max_queries: usize = env_usize("RRO_EVAL_MAX_QUERIES", 0);
    let max_docs: usize = env_usize("RRO_EVAL_MAX_DOCS", 0);

    // ---- load real data ---------------------------------------------------
    let docs = load_corpus(&dir, max_docs)?;
    let queries = load_queries(&dir, max_queries)?;
    println!("corpus: {} docs   queries: {}", docs.len(), queries.len());
    if queries.is_empty() || docs.is_empty() {
        anyhow::bail!("no data at {} — is RRO_EVAL_DATA right?", dir.display());
    }

    // ---- models -----------------------------------------------------------
    let ecfg = EmbedderConfig::from_env()?;
    let embedder = build_embedder(&ecfg).await?;
    println!(
        "embedder: {} ({}) dim={}",
        ecfg.kind.as_str(),
        embedder.model_name(),
        embedder.dim()
    );

    // The reranker arm only runs when explicitly asked for; the default lexical
    // reranker would make `rro` identical to `hybrid` and imply a lift that
    // isn't there.
    let rcfg = RerankerConfig::from_env()?;
    let reranker: Option<Arc<dyn Reranker>> = if std::env::var("RRO_RERANKER").is_ok() {
        let r = build_reranker(&rcfg).await?;
        println!("reranker: {} ({})", rcfg.kind.as_str(), r.model_name());
        Some(r)
    } else {
        println!("reranker: none (set RRO_RERANKER to add the `rro` arm)");
        None
    };

    // ---- index ------------------------------------------------------------
    let estate_dir = tempfile::tempdir()?;
    let estate = Arc::new(connxism::Estate::open(
        estate_dir.path().to_str().unwrap(),
        "eval",
    )?);
    let recall = estate.recall();

    let t = Instant::now();
    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let embeddings = embedder.embed_documents(&texts).await?;
    let embed_secs = t.elapsed().as_secs_f64();
    println!(
        "embedded {} docs in {embed_secs:.1}s ({:.0} docs/sec)",
        docs.len(),
        docs.len() as f64 / embed_secs
    );

    let records: Vec<VectorRecord> = docs
        .iter()
        .zip(embeddings)
        .map(|(d, e)| VectorRecord::new(d.id.clone(), e, d.text.clone()))
        .collect();
    let t = Instant::now();
    recall.upsert(records).await?;
    println!("indexed in {:.1}s", t.elapsed().as_secs_f64());

    // ---- run the ladder ---------------------------------------------------
    let mut arms: Vec<(&str, Vec<f64>, Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let names: Vec<&str> = if reranker.is_some() {
        vec!["bm25", "dense", "hybrid", "rro"]
    } else {
        vec!["bm25", "dense", "hybrid"]
    };

    for arm in names {
        let mut ndcg = Vec::new();
        let mut rec = Vec::new();
        let mut mrr = Vec::new();
        let mut per_query: Vec<(String, f64)> = Vec::new();
        let t = Instant::now();

        for q in &queries {
            // Queries take the instruction-prefixed path; documents did not.
            let qv = embedder.embed_query_one(&q.text).await?;
            let hits: Vec<Candidate> = match arm {
                // TRUE lexical-only. The first version of this faked BM25 by
                // handing hybrid_search a zero vector — but fusion still ran,
                // blending a degenerate dense ranking into the lexical one, and
                // scored 0.159 against a published BM25 baseline of ~0.325.
                // Half the real number. A baseline that is broken LOW makes
                // every arm above it look good, which is the most flattering
                // possible bug and therefore the one to be most suspicious of.
                "bm25" => recall.lexical_search(&q.text, recall_k).await?,
                "dense" => recall.search(&qv, recall_k).await?,
                "hybrid" => recall.hybrid_search(&q.text, &qv, recall_k).await?,
                "rro" => {
                    let c = recall.hybrid_search(&q.text, &qv, recall_k).await?;
                    reranker.as_ref().unwrap().rerank(&q.text, c, top_k).await?
                }
                _ => unreachable!(),
            };
            let ids: Vec<&str> = hits.iter().take(top_k).map(|c| c.id.as_str()).collect();
            let n = ndcg_at_k(&ids, &q.rels, top_k);
            per_query.push((q.id.clone(), n));
            ndcg.push(n);
            rec.push(recall_at_k(&ids, &q.rels));
            mrr.push(mrr_at_k(&ids, &q.rels));
        }
        let secs = t.elapsed().as_secs_f64();
        // The mean hides everything. Show where this arm actually fails, so a
        // headline number can be argued with rather than taken on faith.
        per_query.sort_by(|a, b| a.1.total_cmp(&b.1));
        let worst: Vec<String> = per_query
            .iter()
            .take(3)
            .map(|(id, n)| format!("{id}={n:.3}"))
            .collect();
        println!("  {arm:<8} worst queries: {}", worst.join("  "));
        arms.push((arm, ndcg, rec, mrr, secs));
    }

    // ---- report -----------------------------------------------------------
    println!("\n=== {} | {} queries | top_k={top_k} recall_k={recall_k} ===", dir.display(), queries.len());
    println!("{:<8} {:>9} {:>10} {:>9} {:>12}", "arm", "nDCG@10", "Recall@10", "MRR@10", "ms/query");
    let mut prev_ndcg = 0.0;
    for (name, ndcg, rec, mrr, secs) in &arms {
        let n = mean(ndcg);
        let delta = if prev_ndcg > 0.0 {
            format!("  ({:+.1}% vs prev)", (n / prev_ndcg - 1.0) * 100.0)
        } else {
            String::new()
        };
        println!(
            "{:<8} {:>9.4} {:>10.4} {:>9.4} {:>12.1}{delta}",
            name,
            n,
            mean(rec),
            mean(mrr),
            secs * 1000.0 / queries.len() as f64
        );
        prev_ndcg = n;
    }
    println!("\nPublished BEIR nDCG@10 for reference — BM25 ~0.325, and strong dense");
    println!("models land ~0.32-0.38 on nfcorpus. A number far above that band means");
    println!("the harness is wrong, not that the engine is magic.");
    Ok(())
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// nDCG@k with graded relevance (the BEIR standard).
fn ndcg_at_k(ids: &[&str], rels: &HashMap<String, u8>, k: usize) -> f64 {
    let dcg: f64 = ids
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let g = *rels.get(*id).unwrap_or(&0) as f64;
            (2f64.powf(g) - 1.0) / ((i + 2) as f64).log2()
        })
        .sum();
    // Ideal: the best k grades this query could possibly have returned.
    let mut ideal: Vec<u8> = rels.values().copied().filter(|g| *g > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, g)| (2f64.powf(*g as f64) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn recall_at_k(ids: &[&str], rels: &HashMap<String, u8>) -> f64 {
    let total = rels.values().filter(|g| **g > 0).count();
    if total == 0 {
        return 0.0;
    }
    let found = ids
        .iter()
        .filter(|id| rels.get(**id).is_some_and(|g| *g > 0))
        .count();
    found as f64 / total as f64
}

fn mrr_at_k(ids: &[&str], rels: &HashMap<String, u8>) -> f64 {
    for (i, id) in ids.iter().enumerate() {
        if rels.get(*id).is_some_and(|g| *g > 0) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// BEIR/MTEB `corpus.jsonl`: `{_id, title, text}`. Title is prepended — the
/// standard BEIR convention, and dropping it measurably hurts every model
/// equally, so it would flatter nothing but would not be comparable to
/// published numbers.
fn load_corpus(dir: &Path, limit: usize) -> anyhow::Result<Vec<Document>> {
    let f = std::fs::read_to_string(dir.join("corpus.jsonl"))?;
    let mut out = Vec::new();
    for line in f.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let title = v["title"].as_str().unwrap_or_default();
        let text = v["text"].as_str().unwrap_or_default();
        let full = if title.is_empty() {
            text.to_string()
        } else {
            format!("{title}. {text}")
        };
        out.push(Document::new(full).with_id(id));
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

/// `queries.jsonl` + `qrels/test.tsv`. Only queries WITH judgments are scored:
/// an unjudged query has no ground truth, so including it would score 0 and
/// silently drag every arm down by the same amount — noise, not signal.
fn load_queries(dir: &Path, limit: usize) -> anyhow::Result<Vec<EvalQuery>> {
    let mut qrels: HashMap<String, HashMap<String, u8>> = HashMap::new();
    let tsv = std::fs::read_to_string(dir.join("qrels/test.tsv"))?;
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue; // header
        }
        let mut it = line.split('\t');
        let (Some(q), Some(d), Some(s)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let score: u8 = s.trim().parse().unwrap_or(0);
        if score > 0 {
            qrels.entry(q.to_string()).or_default().insert(d.to_string(), score);
        }
    }

    let f = std::fs::read_to_string(dir.join("queries.jsonl"))?;
    let mut out = Vec::new();
    for line in f.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let Some(rels) = qrels.remove(&id) else {
            continue;
        };
        out.push(EvalQuery {
            id,
            text: v["text"].as_str().unwrap_or_default().to_string(),
            rels,
        });
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    Ok(out)
}
