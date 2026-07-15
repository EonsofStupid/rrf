//! Out-of-band graph apply: the two-phase pattern's second phase.
//!
//! Upserts commit durably and **enqueue**; a dedicated applier thread drains
//! the queue into the ANN graph. Ingest is never blocked by graph
//! construction (until the backpressure cap), and searches stay correct via
//! the **pending overlay**: not-yet-applied vectors are scored exactly and
//! merged over the graph's results, and pending removals mask stale graph
//! hits — read-your-writes by construction. Crash-safe trivially: pendings
//! are already durable in the `vecs` column family, and reopening an estate
//! rebuilds the graph from it.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock as StdRwLock};

use recall::AnnIndex;
use rrf_core::{Embedding, Id};

/// Backpressure cap: above this many queued entries, producers block.
const PENDING_CAP: usize = 200_000;
/// Applier batch size per graph write-lock acquisition.
const APPLY_BATCH: usize = 512;

#[derive(Default)]
struct State {
    /// Apply order (ids may repeat; the map holds the latest op).
    queue: VecDeque<Id>,
    /// Latest pending op per id: `Some` = upsert, `None` = remove.
    latest: HashMap<Id, Option<Embedding>>,
    /// The applier is mid-batch (queue may be empty while work is in flight).
    applying: bool,
    /// Shutdown flag.
    stopped: bool,
}

/// Shared pending set + signaling.
pub(crate) struct Pending {
    state: Mutex<State>,
    /// Wakes the applier (work arrived / shutdown).
    work: Condvar,
    /// Wakes waiters (space available / drained).
    settled: Condvar,
}

impl Pending {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Pending {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            settled: Condvar::new(),
        })
    }

    /// Enqueue an upsert (blocks at the backpressure cap).
    pub(crate) fn push_upsert(&self, id: Id, embedding: Embedding) {
        let mut s = self.state.lock().expect("pending lock");
        while s.queue.len() >= PENDING_CAP && !s.stopped {
            s = self.settled.wait(s).expect("pending wait");
        }
        s.queue.push_back(id.clone());
        s.latest.insert(id, Some(embedding));
        drop(s);
        self.work.notify_one();
    }

    /// Enqueue a removal.
    pub(crate) fn push_remove(&self, id: Id) {
        let mut s = self.state.lock().expect("pending lock");
        s.queue.push_back(id.clone());
        s.latest.insert(id, None);
        drop(s);
        self.work.notify_one();
    }

    /// Snapshot the overlay for a search: pending upserts scored exactly by
    /// the caller, pending removals masked. Cheap when drained (the steady
    /// state); proportional to backlog when not.
    pub(crate) fn overlay(&self, query: &Embedding) -> (Vec<(Id, f32)>, Vec<Id>) {
        let s = self.state.lock().expect("pending lock");
        let q = query.normalized();
        let mut ups = Vec::new();
        let mut dels = Vec::new();
        for (id, op) in &s.latest {
            match op {
                Some(emb) => ups.push((id.clone(), q.cosine(&emb.normalized()))),
                None => dels.push(id.clone()),
            }
        }
        (ups, dels)
    }

    /// Block until every queued op has been applied to the graph.
    pub(crate) fn quiesce(&self) {
        let mut s = self.state.lock().expect("pending lock");
        while (!s.queue.is_empty() || s.applying) && !s.stopped {
            s = self.settled.wait(s).expect("pending wait");
        }
    }

    /// Signal shutdown and wake everyone.
    pub(crate) fn stop(&self) {
        let mut s = self.state.lock().expect("pending lock");
        s.stopped = true;
        drop(s);
        self.work.notify_all();
        self.settled.notify_all();
    }

    /// Spawn the applier thread for `ann`. Runs until [`Pending::stop`].
    pub(crate) fn spawn_applier(
        self: &Arc<Self>,
        ann: Arc<StdRwLock<AnnIndex>>,
    ) -> std::thread::JoinHandle<()> {
        let pending = Arc::clone(self);
        std::thread::Builder::new()
            .name("rrf-ann-applier".into())
            .spawn(move || loop {
                // Collect one batch under the lock.
                let mut batch: Vec<(Id, Option<Embedding>)> = Vec::with_capacity(APPLY_BATCH);
                {
                    let mut s = pending.state.lock().expect("pending lock");
                    while s.queue.is_empty() && !s.stopped {
                        s = pending.work.wait(s).expect("pending wait");
                    }
                    if s.stopped {
                        return;
                    }
                    while batch.len() < APPLY_BATCH {
                        let Some(id) = s.queue.pop_front() else { break };
                        // Duplicate queue entries: only the first pop finds
                        // the op; later pops skip.
                        if let Some(op) = s.latest.remove(&id) {
                            batch.push((id, op));
                        }
                    }
                    s.applying = true;
                }

                // Apply outside the pending lock (graph lock only).
                {
                    let mut graph = ann.write().expect("ann lock");
                    for (id, op) in batch {
                        match op {
                            Some(emb) => graph.insert(id, &emb),
                            None => graph.remove(&id),
                        }
                    }
                }

                let mut s = pending.state.lock().expect("pending lock");
                s.applying = false;
                drop(s);
                pending.settled.notify_all();
            })
            .expect("spawn applier")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall::AnnConfig;

    #[test]
    fn applier_drains_and_quiesce_waits() {
        let ann = Arc::new(StdRwLock::new(AnnIndex::new(AnnConfig::default())));
        let pending = Pending::new();
        let handle = pending.spawn_applier(ann.clone());

        for i in 0..1000 {
            pending.push_upsert(
                Id::new(format!("p{i}")),
                Embedding(vec![i as f32, 1.0, 0.5]),
            );
        }
        pending.quiesce();
        assert_eq!(ann.read().unwrap().len(), 1000);

        // Overlay is empty once drained.
        let (ups, dels) = pending.overlay(&Embedding(vec![1.0, 0.0, 0.0]));
        assert!(ups.is_empty() && dels.is_empty());

        pending.stop();
        handle.join().unwrap();
    }

    #[test]
    fn overlay_sees_unapplied_upserts_and_removes() {
        // No applier: everything stays pending.
        let pending = Pending::new();
        pending.push_upsert(Id::new("a"), Embedding(vec![1.0, 0.0]));
        pending.push_remove(Id::new("b"));

        let (ups, dels) = pending.overlay(&Embedding(vec![1.0, 0.0]));
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].0.as_str(), "a");
        assert!(ups[0].1 > 0.99);
        assert_eq!(dels, vec![Id::new("b")]);
        pending.stop();
    }
}
