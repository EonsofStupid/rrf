//! An in-memory, brute-force cosine store.
//!
//! Exact nearest-neighbour by full scan. For the corpus sizes the engine is
//! born on this is correct and fast; when the working set outgrows a linear
//! scan, swap in an ANN index behind the same [`rrf_core::Recall`] trait — the
//! flow never notices.

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;
use rrf_core::{Candidate, Embedding, Id, Recall, Result, RrfError, VectorRecord};

/// A flat (exhaustive) in-memory vector store.
#[derive(Default)]
pub struct FlatRecall {
    inner: RwLock<HashMap<Id, VectorRecord>>,
    dim: RwLock<Option<usize>>,
}

impl FlatRecall {
    /// A fresh, empty store.
    pub fn new() -> Self {
        Self::default()
    }

    fn check_dim(&self, emb: &Embedding) -> Result<()> {
        let mut d = self
            .dim
            .write()
            .map_err(|_| RrfError::Recall("dim lock poisoned".into()))?;
        match *d {
            None => {
                *d = Some(emb.dim());
                Ok(())
            }
            Some(expected) if expected == emb.dim() => Ok(()),
            Some(expected) => Err(RrfError::DimMismatch {
                expected,
                got: emb.dim(),
            }),
        }
    }
}

#[async_trait]
impl Recall for FlatRecall {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<()> {
        for r in &records {
            self.check_dim(&r.embedding)?;
        }
        let mut map = self
            .inner
            .write()
            .map_err(|_| RrfError::Recall("store lock poisoned".into()))?;
        for r in records {
            map.insert(r.id.clone(), r);
        }
        Ok(())
    }

    async fn search(&self, query: &Embedding, top_k: usize) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        // Query dim must match the store (once anything is indexed).
        if let Some(expected) = *self
            .dim
            .read()
            .map_err(|_| RrfError::Recall("dim lock poisoned".into()))?
        {
            if expected != query.dim() {
                return Err(RrfError::DimMismatch {
                    expected,
                    got: query.dim(),
                });
            }
        }

        let map = self
            .inner
            .read()
            .map_err(|_| RrfError::Recall("store lock poisoned".into()))?;

        let mut scored: Vec<Candidate> = map
            .values()
            .map(|r| {
                let mut c = Candidate::new(r.id.clone(), r.text.clone(), query.cosine(&r.embedding));
                c.metadata = r.metadata.clone();
                c
            })
            .collect();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn len(&self) -> Result<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| RrfError::Recall("store lock poisoned".into()))?
            .len())
    }

    async fn remove(&self, id: &Id) -> Result<()> {
        self.inner
            .write()
            .map_err(|_| RrfError::Recall("store lock poisoned".into()))?
            .remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(v: &[f32]) -> Embedding {
        Embedding(v.to_vec())
    }

    #[tokio::test]
    async fn upsert_search_roundtrip() {
        let store = FlatRecall::new();
        store
            .upsert(vec![
                VectorRecord::new("a", emb(&[1.0, 0.0, 0.0]), "apple"),
                VectorRecord::new("b", emb(&[0.0, 1.0, 0.0]), "banana"),
                VectorRecord::new("c", emb(&[0.9, 0.1, 0.0]), "apricot"),
            ])
            .await
            .unwrap();

        assert_eq!(store.len().await.unwrap(), 3);
        let hits = store.search(&emb(&[1.0, 0.0, 0.0]), 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id.as_str(), "a");
        assert_eq!(hits[1].id.as_str(), "c");
    }

    #[tokio::test]
    async fn dim_mismatch_is_rejected() {
        let store = FlatRecall::new();
        store
            .upsert(vec![VectorRecord::new("a", emb(&[1.0, 0.0]), "x")])
            .await
            .unwrap();
        let err = store
            .upsert(vec![VectorRecord::new("b", emb(&[1.0, 0.0, 0.0]), "y")])
            .await;
        assert!(matches!(err, Err(RrfError::DimMismatch { .. })));
    }
}
