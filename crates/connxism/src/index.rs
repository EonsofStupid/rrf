//! The persistent BM25 inverted index.
//!
//! Postings live in the `terms` column family as **one row per
//! (term, document)**: key `term \x00 doc_id`, value a single [`Posting`].
//! Writes are blind puts — no read-modify-write — and reads are sorted prefix
//! scans, so indexing cost stays flat as hot terms grow. Entries carry the
//! document token length so lexical scoring never fetches payloads.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One document's entry in a term's postings list.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Posting {
    /// Term frequency in the document.
    pub tf: u32,
    /// The document's content-token length (BM25 `dl`).
    pub len: u32,
}

/// A term's fetched postings: `(doc id, posting)` rows, unique by doc id.
pub type Postings = Vec<(String, Posting)>;

/// Okapi BM25 parameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term-frequency saturation.
    pub k1: f32,
    /// Length normalization.
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Bm25Params { k1: 1.2, b: 0.75 }
    }
}

/// Score every document matching any query term. `n_docs` and `avgdl` come
/// from the estate counters; `term_postings` is the fetched postings per term.
pub fn bm25_scores(
    params: Bm25Params,
    n_docs: u64,
    avgdl: f32,
    term_postings: &[(String, Postings)],
) -> HashMap<String, f32> {
    let n = n_docs.max(1) as f32;
    let avgdl = avgdl.max(1.0);
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (_term, postings) in term_postings {
        let df = postings.len() as f32;
        if df == 0.0 {
            continue;
        }
        // BM25 idf with +0.5 smoothing, clamped non-negative.
        let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln().max(0.0);
        for (doc_id, p) in postings.iter() {
            let f = p.tf as f32;
            let dl = p.len as f32;
            let denom = f + params.k1 * (1.0 - params.b + params.b * dl / avgdl);
            let s = idf * (f * (params.k1 + 1.0)) / denom;
            *scores.entry(doc_id.clone()).or_insert(0.0) += s;
        }
    }
    scores
}

/// Reciprocal rank fusion: fuse ranked lists into one ranking.
///
/// `score(d) = Σ_lists 1 / (k + rank_of_d_in_list)` with ranks starting at 1;
/// documents absent from a list contribute nothing for it. The standard
/// constant is `k = 60`.
pub fn reciprocal_rank_fusion(lists: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut fused: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (i, id) in list.iter().enumerate() {
            *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (k + (i as f32 + 1.0));
        }
    }
    let mut out: Vec<(String, f32)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_prefers_agreement() {
        // "b" is ranked well by both lists; "a" and "c" only by one each.
        let lists = vec![vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]];
        let fused = reciprocal_rank_fusion(&lists, 60.0);
        assert_eq!(fused[0].0, "b");
    }

    #[test]
    fn bm25_scores_rarer_terms_higher() {
        let common: Postings = (0..50)
            .map(|i| (format!("d{i}"), Posting { tf: 1, len: 10 }))
            .collect();
        let rare: Postings = vec![("d0".into(), Posting { tf: 1, len: 10 })];

        let scores = bm25_scores(
            Bm25Params::default(),
            100,
            10.0,
            &[("common".into(), common), ("rare".into(), rare)],
        );
        // d0 matched both terms; its score must beat any common-only doc.
        let d0 = scores["d0"];
        let d1 = scores["d1"];
        assert!(d0 > d1);
    }
}
