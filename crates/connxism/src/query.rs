//! Query execution: one spec ([`rrf_core::EstateQuery`], pure data in the
//! core contract), every retrieval capability of the estate.
//!
//! Filter execution is two-strategy: **filter-first** (exact id-set from
//! payload secondary indexes, then exact scoring inside it) when every
//! referenced field is indexed and the set is small enough; **post-filter**
//! (over-fetch + hydrate + retain) otherwise. Facets, filtered counts, and
//! cursor-paged **scroll** live beside it on the estate.

use rrf_core::{Candidate, Embedding, EstateQuery, Metadata, Recall as _, Result};

use crate::estate::{rocks_err, Estate};
use crate::keys::CF_DOCS;
use crate::model::StoredDoc;
use crate::store::ConnXRecall;

/// How hard filtered searches over-fetch before post-filtering.
const FILTER_OVERFETCH: usize = 8;

/// Above this many index-matched ids, exact scoring over the set costs more
/// than over-fetch + post-filter; fall back.
const INDEXED_SCOPE_MAX: usize = 4096;

/// The standard reciprocal-rank-fusion constant (same as the hybrid path).
const FUSION_RRF_K: f32 = 60.0;

impl ConnXRecall {
    /// Execute a typed query. Strategy, in order: explicit scope wins;
    /// otherwise a fully-indexed filter resolves its exact id-set first and
    /// scores inside it; otherwise hybrid (ANN + BM25, fused) with
    /// over-fetch + post-filter.
    pub async fn query(&self, q: EstateQuery) -> Result<Vec<Candidate>> {
        let top_k = q.top_k;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let text = q.text.clone().unwrap_or_default();
        let vector = match &q.vector {
            Some(v) => v.clone(),
            None => Embedding(Vec::new()),
        };

        let dsl = q.effective_filter();
        let fetch = if dsl.is_empty() {
            top_k
        } else {
            top_k.saturating_mul(FILTER_OVERFETCH)
        };

        // `prefiltered` = the candidate set already satisfies the filter
        // exactly (resolved from payload indexes); no post-filter needed.
        // `allowed` = the id universe a scope or prefilter restricts to —
        // any extra ranking fused in below must respect it too.
        let mut prefiltered = false;
        let mut allowed: Option<std::collections::HashSet<String>> = None;
        let mut results = match &q.scope {
            Some(scope) => {
                allowed = Some(scope.iter().cloned().collect());
                self.scoped_search(&text, &vector, fetch, scope.clone())
                    .await?
            }
            None if !dsl.is_empty() => match crate::filter::ids_where(&self.db, &dsl)? {
                Some(ids) if ids.len() <= INDEXED_SCOPE_MAX => {
                    prefiltered = true;
                    allowed = Some(ids.iter().cloned().collect());
                    if ids.is_empty() {
                        Vec::new()
                    } else {
                        self.scoped_search(&text, &vector, top_k, ids).await?
                    }
                }
                _ => {
                    self.unscoped(&text, &vector, q.vector.is_some(), fetch)
                        .await?
                }
            },
            None => {
                self.unscoped(&text, &vector, q.vector.is_some(), fetch)
                    .await?
            }
        };

        // Weighted sparse: a third ranking, fused by reciprocal rank fusion
        // with whatever the dense/lexical strategy produced.
        if let Some(sv) = &q.sparse {
            if !sv.is_empty() {
                let mut sparse = self.sparse_search(sv, fetch).await?;
                if let Some(allowed) = &allowed {
                    sparse.retain(|c| allowed.contains(c.id.as_str()));
                }
                if !sparse.is_empty() {
                    let lists = [
                        results
                            .iter()
                            .map(|c| c.id.as_str().to_string())
                            .collect::<Vec<_>>(),
                        sparse
                            .iter()
                            .map(|c| c.id.as_str().to_string())
                            .collect::<Vec<_>>(),
                    ];
                    let fused = crate::index::reciprocal_rank_fusion(&lists, FUSION_RRF_K);
                    let mut out = Vec::with_capacity(fetch.min(fused.len()));
                    for (id, score) in fused.into_iter().take(fetch) {
                        if let Some(doc) = self.doc(&id).await? {
                            let mut c = Candidate::new(doc.id, doc.text, score);
                            c.metadata = doc.metadata;
                            out.push(c);
                        }
                    }
                    results = out;
                }
            }
        }

        if !dsl.is_empty() && !prefiltered {
            // Lexical-path candidates may carry empty payloads; hydrate before
            // filtering so the clauses see real metadata.
            for c in results.iter_mut() {
                if c.metadata.is_empty() {
                    if let Some(doc) = self.doc(c.id.as_str()).await? {
                        c.metadata = doc.metadata;
                        if c.text.is_empty() {
                            c.text = doc.text;
                        }
                    }
                }
            }
            results.retain(|c| dsl.matches(&c.metadata));
        }
        if let Some(t) = q.score_threshold {
            results.retain(|c| c.score >= t);
        }
        results.truncate(top_k);
        if !q.with_payload {
            for c in results.iter_mut() {
                c.text.clear();
                c.metadata = Metadata::new();
            }
        }
        Ok(results)
    }

    async fn unscoped(
        &self,
        text: &str,
        vector: &Embedding,
        has_vector: bool,
        fetch: usize,
    ) -> Result<Vec<Candidate>> {
        if has_vector {
            self.hybrid_search(text, vector, fetch).await
        } else {
            self.lexical_search(text, fetch).await
        }
    }
}

fn matches_filter(metadata: &Metadata, filter: &Metadata) -> bool {
    filter.iter().all(|(k, v)| metadata.get(k) == Some(v))
}

impl Estate {
    /// Facet: value → document count for a metadata field. v1 scans the doc
    /// column family (secondary indexes take over in the P3 tail — this is
    /// exact today, and honest about its cost).
    pub fn facet(&self, field: &str) -> Result<std::collections::BTreeMap<String, u64>> {
        let handle = self.db.cf(CF_DOCS)?;
        let mut out = std::collections::BTreeMap::new();
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: StoredDoc = serde_json::from_slice(&v)?;
            if let Some(value) = doc.metadata.get(field) {
                let key = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                *out.entry(key).or_insert(0) += 1;
            }
        }
        Ok(out)
    }

    /// Count documents matching a metadata equality filter (empty filter =
    /// exact total from the counter, free).
    pub fn count(&self, filter: &Metadata) -> Result<u64> {
        if filter.is_empty() {
            return self.db.get_u64(crate::keys::META_DOC_COUNT);
        }
        let handle = self.db.cf(CF_DOCS)?;
        let mut n = 0u64;
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: StoredDoc = serde_json::from_slice(&v)?;
            if matches_filter(&doc.metadata, filter) {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Scroll: cursor-paged document listing (id order). Pass the last id of
    /// the previous page as `after`; returns up to `limit` docs.
    pub fn scroll(&self, after: Option<&str>, limit: usize) -> Result<Vec<StoredDoc>> {
        let handle = self.db.cf(CF_DOCS)?;
        let mode = match after {
            Some(a) => rocksdb::IteratorMode::From(a.as_bytes(), rocksdb::Direction::Forward),
            None => rocksdb::IteratorMode::Start,
        };
        let mut out = Vec::new();
        for item in self.db.0.iterator_cf(handle, mode) {
            let (k, v) = item.map_err(rocks_err)?;
            if let Some(a) = after {
                if k.as_ref() == a.as_bytes() {
                    continue; // strictly after the cursor
                }
            }
            out.push(serde_json::from_slice(&v)?);
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }
}
