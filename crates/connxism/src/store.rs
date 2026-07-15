//! The persistent recall store: vectors + BM25 postings + payloads in one
//! estate, hybrid-searchable.
//!
//! [`ConnXRecall`] implements [`rrf_core::Recall`]. `search` is dense cosine;
//! `hybrid_search` fuses dense and lexical rankings with reciprocal rank
//! fusion. All RocksDB work runs on the blocking pool so the tokio runtime
//! never stalls. Postings writes are blind puts (one row per (term, doc)),
//! but the estate counters (doc count, token totals, shape census) are
//! read-modify-write, so writers serialize behind an async mutex.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use rrf_core::text::content_tokens;
use rrf_core::{Candidate, Embedding, Id, Recall, Result, RrfError, VectorRecord};
use tokio::sync::Mutex;

use crate::estate::{rocks_err, Db, Estate};
use crate::index::{bm25_scores, reciprocal_rank_fusion, Bm25Params, Posting, Postings};
use crate::keys::{
    self, CF_DOCS, CF_META, CF_TERMS, CF_VECS, META_DOC_COUNT, META_ESTATE, META_SHAPES,
    META_TOTAL_TOKENS,
};
use crate::model::{EstateInfo, Shape, StoredDoc};

/// How much of each ranking feeds the fusion stage.
const FUSION_DEPTH_FACTOR: usize = 4;
/// The standard reciprocal-rank-fusion constant.
const RRF_K: f32 = 60.0;

/// Persistent, hybrid (dense + lexical) recall over an estate.
#[derive(Clone)]
pub struct ConnXRecall {
    db: Db,
    writer: Arc<Mutex<()>>,
    params: Bm25Params,
}

impl Estate {
    /// The estate's recall store (shares this estate's database).
    pub fn recall(&self) -> ConnXRecall {
        ConnXRecall {
            db: self.db.clone(),
            writer: Arc::new(Mutex::new(())),
            params: Bm25Params::default(),
        }
    }
}

impl ConnXRecall {
    /// Fetch a stored document by id.
    pub async fn doc(&self, id: &str) -> Result<Option<StoredDoc>> {
        let db = self.db.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes()))
            .await
            .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }

    /// Lexical (BM25) search over the persistent inverted index.
    pub async fn lexical_search(&self, query: &str, top_k: usize) -> Result<Vec<Candidate>> {
        let db = self.db.clone();
        let params = self.params;
        let terms = content_tokens(query);
        if terms.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        tokio::task::spawn_blocking(move || lexical_blocking(&db, params, &terms, top_k))
            .await
            .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }
}

#[async_trait]
impl Recall for ConnXRecall {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        // Serialize writers: postings updates are read-modify-write.
        let _guard = self.writer.lock().await;
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || upsert_blocking(&db, records))
            .await
            .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }

    async fn search(&self, query: &Embedding, top_k: usize) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let q = query.clone();
        tokio::task::spawn_blocking(move || dense_blocking(&db, &q, top_k, true))
            .await
            .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }

    async fn hybrid_search(
        &self,
        query_text: &str,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let params = self.params;
        let q = query.clone();
        let terms = content_tokens(query_text);
        let depth = top_k.saturating_mul(FUSION_DEPTH_FACTOR).max(top_k);

        tokio::task::spawn_blocking(move || {
            // Two rankings over the same estate…
            let dense = dense_blocking(&db, &q, depth, false)?;
            let lexical = if terms.is_empty() {
                Vec::new()
            } else {
                lexical_blocking(&db, params, &terms, depth)?
            };

            // …fused by reciprocal rank fusion.
            let lists = [
                dense
                    .iter()
                    .map(|c| c.id.as_str().to_string())
                    .collect::<Vec<_>>(),
                lexical
                    .iter()
                    .map(|c| c.id.as_str().to_string())
                    .collect::<Vec<_>>(),
            ];
            let fused = reciprocal_rank_fusion(&lists, RRF_K);

            let mut out = Vec::with_capacity(top_k);
            for (doc_id, score) in fused.into_iter().take(top_k) {
                if let Some(doc) = db.get_json::<StoredDoc>(CF_DOCS, doc_id.as_bytes())? {
                    let mut c = Candidate::new(doc.id, doc.text, score);
                    c.metadata = doc.metadata;
                    out.push(c);
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }

    async fn len(&self) -> Result<usize> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || db.get_u64(META_DOC_COUNT).map(|n| n as usize))
            .await
            .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }

    async fn remove(&self, id: &Id) -> Result<()> {
        let _guard = self.writer.lock().await;
        let db = self.db.clone();
        let id = id.as_str().to_string();
        tokio::task::spawn_blocking(move || remove_blocking(&db, &id))
            .await
            .map_err(|e| RrfError::Recall(format!("join: {e}")))?
    }
}

// ---- blocking internals (run on the blocking pool) ----------------------------

fn upsert_blocking(db: &Db, records: Vec<VectorRecord>) -> Result<()> {
    // Dimension guard: fixed by the first upsert, enforced forever after.
    let mut info: EstateInfo = db
        .get_json(CF_META, META_ESTATE)?
        .ok_or_else(|| RrfError::Recall("estate not initialized".into()))?;
    let dim = records[0].embedding.dim();
    match info.dim {
        None => {
            info.dim = Some(dim);
            db.put_json(CF_META, META_ESTATE, &info)?;
        }
        Some(expected) if expected != dim => {
            return Err(RrfError::DimMismatch { expected, got: dim });
        }
        _ => {}
    }
    for r in &records {
        if r.embedding.dim() != dim {
            return Err(RrfError::DimMismatch {
                expected: dim,
                got: r.embedding.dim(),
            });
        }
    }

    let mut doc_count = db.get_u64(META_DOC_COUNT)?;
    let mut total_tokens = db.get_u64(META_TOTAL_TOKENS)?;
    let mut shapes: BTreeMap<String, u64> = db.get_json(CF_META, META_SHAPES)?.unwrap_or_default();

    // Postings are one row per (term, doc): every index write below is a
    // blind put/delete — no read-modify-write, flat cost as terms grow.
    let mut batch = rocksdb::WriteBatch::default();
    let docs_cf = db.cf(CF_DOCS)?;
    let vecs_cf = db.cf(CF_VECS)?;
    let terms_cf = db.cf(CF_TERMS)?;

    for r in records {
        let id = r.id.as_str().to_string();

        // Overwrite semantics: retract the old version's postings and counters.
        if let Some(old) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? {
            for term in content_tokens(&old.text) {
                batch.delete_cf(terms_cf, keys::term_key(&term, &id));
            }
            total_tokens = total_tokens.saturating_sub(old.token_len as u64);
            if let Some(n) = shapes.get_mut(&old.shape.key()) {
                *n = n.saturating_sub(1);
            }
            doc_count = doc_count.saturating_sub(1);
        }

        let tokens = content_tokens(&r.text);
        let token_len = tokens.len() as u32;
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *tf.entry(t).or_insert(0) += 1;
        }
        for (term, f) in tf {
            let posting = Posting {
                tf: f,
                len: token_len,
            };
            batch.put_cf(
                terms_cf,
                keys::term_key(&term, &id),
                serde_json::to_vec(&posting)?,
            );
        }

        let shape = Shape::of(&r.metadata);
        *shapes.entry(shape.key()).or_insert(0) += 1;
        doc_count += 1;
        total_tokens += token_len as u64;

        let doc = StoredDoc {
            id: id.clone(),
            text: r.text,
            metadata: r.metadata,
            tags: Vec::new(),
            shape,
            token_len,
            connector_id: None,
        };
        batch.put_cf(docs_cf, id.as_bytes(), serde_json::to_vec(&doc)?);
        batch.put_cf(
            vecs_cf,
            id.as_bytes(),
            keys::encode_vec(r.embedding.as_slice()),
        );
    }

    let meta_cf = db.cf(CF_META)?;
    batch.put_cf(meta_cf, META_DOC_COUNT, doc_count.to_le_bytes());
    batch.put_cf(meta_cf, META_TOTAL_TOKENS, total_tokens.to_le_bytes());
    batch.put_cf(meta_cf, META_SHAPES, serde_json::to_vec(&shapes)?);

    db.0.write(batch).map_err(rocks_err)
}

/// Prefix-scan a term's postings rows.
fn scan_postings(db: &Db, term: &str) -> Result<Postings> {
    let terms_cf = db.cf(CF_TERMS)?;
    let prefix = keys::term_prefix(term);
    let mut out = Postings::new();
    for item in db.0.iterator_cf(
        terms_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    ) {
        let (k, v) = item.map_err(rocks_err)?;
        if !k.starts_with(&prefix) {
            break;
        }
        let doc_id = String::from_utf8_lossy(&k[prefix.len()..]).into_owned();
        let posting: Posting = serde_json::from_slice(&v)?;
        out.push((doc_id, posting));
    }
    Ok(out)
}

/// Dense cosine scan. When `fetch_payload` is false, candidates carry ids and
/// scores only (fusion fetches winners' payloads afterwards).
fn dense_blocking(
    db: &Db,
    query: &Embedding,
    top_k: usize,
    fetch_payload: bool,
) -> Result<Vec<Candidate>> {
    let vecs_cf = db.cf(CF_VECS)?;
    let mut scored: Vec<(String, f32)> = Vec::new();
    for item in db.0.iterator_cf(vecs_cf, rocksdb::IteratorMode::Start) {
        let (k, v) = item.map_err(rocks_err)?;
        let emb = Embedding(keys::decode_vec(&v));
        scored.push((String::from_utf8_lossy(&k).into_owned(), query.cosine(&emb)));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(top_k);

    let mut out = Vec::with_capacity(scored.len());
    for (id, score) in scored {
        if fetch_payload {
            if let Some(doc) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? {
                let mut c = Candidate::new(doc.id, doc.text, score);
                c.metadata = doc.metadata;
                out.push(c);
                continue;
            }
        }
        out.push(Candidate::new(id, String::new(), score));
    }
    Ok(out)
}

fn lexical_blocking(
    db: &Db,
    params: Bm25Params,
    terms: &[String],
    top_k: usize,
) -> Result<Vec<Candidate>> {
    let n_docs = db.get_u64(META_DOC_COUNT)?;
    let total_tokens = db.get_u64(META_TOTAL_TOKENS)?;
    let avgdl = if n_docs == 0 {
        1.0
    } else {
        total_tokens as f32 / n_docs as f32
    };

    let mut term_postings: Vec<(String, Postings)> = Vec::with_capacity(terms.len());
    for t in terms {
        term_postings.push((t.clone(), scan_postings(db, t)?));
    }

    let scores = bm25_scores(params, n_docs, avgdl, &term_postings);
    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);

    let mut out = Vec::with_capacity(ranked.len());
    for (id, score) in ranked {
        out.push(Candidate::new(id, String::new(), score));
    }
    Ok(out)
}

fn remove_blocking(db: &Db, id: &str) -> Result<()> {
    let Some(old) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? else {
        return Ok(());
    };

    let mut batch = rocksdb::WriteBatch::default();
    let terms_cf = db.cf(CF_TERMS)?;
    for term in content_tokens(&old.text) {
        batch.delete_cf(terms_cf, keys::term_key(&term, id));
    }
    batch.delete_cf(db.cf(CF_DOCS)?, id.as_bytes());
    batch.delete_cf(db.cf(CF_VECS)?, id.as_bytes());

    let meta_cf = db.cf(CF_META)?;
    let doc_count = db.get_u64(META_DOC_COUNT)?.saturating_sub(1);
    let total_tokens = db
        .get_u64(META_TOTAL_TOKENS)?
        .saturating_sub(old.token_len as u64);
    let mut shapes: BTreeMap<String, u64> = db.get_json(CF_META, META_SHAPES)?.unwrap_or_default();
    if let Some(n) = shapes.get_mut(&old.shape.key()) {
        *n = n.saturating_sub(1);
    }
    batch.put_cf(meta_cf, META_DOC_COUNT, doc_count.to_le_bytes());
    batch.put_cf(meta_cf, META_TOTAL_TOKENS, total_tokens.to_le_bytes());
    batch.put_cf(meta_cf, META_SHAPES, serde_json::to_vec(&shapes)?);

    db.0.write(batch).map_err(rocks_err)
}
