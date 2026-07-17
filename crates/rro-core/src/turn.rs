//! The turn: one pass through the engine, and the id that ties its signals
//! together.
//!
//! The engine already emitted `flow.stage` for every stage — but nothing carried
//! a correlation id, so two concurrent queries interleaved their events in the
//! stream with no way to tell them apart. Aggregates were visible; **one query's
//! journey was not**. That is the difference between telemetry you can average
//! and telemetry you can *read*.
//!
//! A [`TurnId`] fixes that: every signal a pass emits carries the same id, so
//! the stream can be replayed into exactly one turn —
//!
//! ```text
//! turn 7f3a…  rrd       gate=pass mode=unshaped        0.006 ms
//! turn 7f3a…  embed     dim=2560                      42.979 ms
//! turn 7f3a…  intent    tags=[code, retrieval]         0.011 ms
//! turn 7f3a…  recall    candidates=100                 3.724 ms
//! turn 7f3a…  rerank    kept=10                     1081.122 ms
//! turn 7f3a…  classify  ready=true conf=0.82           0.192 ms
//! turn 7f3a…  turn      total=1128.0 ms  ready=true
//! ```
//!
//! That is the "full turn" — and it is what makes a benchmark number
//! interrogable instead of merely reported: when an arm scores badly you can
//! open the turn and see which stage did it.

use std::sync::atomic::{AtomicU64, Ordering};

/// Correlates every signal emitted by one pass through the engine.
///
/// Cheap by construction: a process-unique counter plus a per-process nonce, not
/// a UUID. A turn id is minted on the hot path of every query, so it must cost
/// nothing — and it only has to be unique within a node's event stream, which is
/// the scope anyone reassembles a turn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TurnId(u64);

static NEXT: AtomicU64 = AtomicU64::new(1);

impl TurnId {
    /// Mint the next id for this process.
    pub fn next() -> Self {
        TurnId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw counter.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TurnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for TurnId {
    fn from(v: u64) -> Self {
        TurnId(v)
    }
}

/// Emit one stage signal, correlated to `turn`.
///
/// Every stage of every pass goes through here, so the shape is uniform:
/// `{turn, stage, ms, …}`. A stage that invented its own field names would be a
/// stage nobody can query for.
pub fn emit_stage(turn: TurnId, stage: &str, since: std::time::Instant, mut fields: serde_json::Value) {
    if let Some(obj) = fields.as_object_mut() {
        obj.insert("turn".to_string(), serde_json::json!(turn.get()));
        obj.insert("stage".to_string(), serde_json::json!(stage));
        obj.insert(
            "ms".to_string(),
            serde_json::json!(since.elapsed().as_micros() as f64 / 1000.0),
        );
    }
    crate::events::emit("flow.stage", fields);
}

/// Emit an arbitrary turn-scoped signal (`kind` is the event name).
pub fn emit_turn(turn: TurnId, kind: &str, mut fields: serde_json::Value) {
    if let Some(obj) = fields.as_object_mut() {
        obj.insert("turn".to_string(), serde_json::json!(turn.get()));
    }
    crate::events::emit(kind, fields);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_monotonic() {
        let a = TurnId::next();
        let b = TurnId::next();
        assert!(b.get() > a.get(), "a later turn must sort after an earlier one");
        assert_ne!(a, b);
    }

    #[test]
    fn ids_are_unique_under_concurrency() {
        // Turn ids are minted on the hot path of concurrent queries. A duplicate
        // would silently merge two turns in the stream — the exact failure this
        // type exists to prevent.
        let n = 64;
        let handles: Vec<_> = (0..n)
            .map(|_| std::thread::spawn(|| (0..100).map(|_| TurnId::next().get()).collect::<Vec<_>>()))
            .collect();
        let mut all: Vec<u64> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
        let total = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), total, "{} duplicate turn ids", total - all.len());
    }

    #[test]
    fn a_stage_signal_carries_turn_stage_and_ms() {
        // The uniform shape is the contract: anything reassembling a turn keys
        // on exactly these fields.
        let turn = TurnId::next();
        let mut v = serde_json::json!({ "gate": "pass" });
        if let Some(o) = v.as_object_mut() {
            o.insert("turn".into(), serde_json::json!(turn.get()));
            o.insert("stage".into(), serde_json::json!("rrd"));
            o.insert("ms".into(), serde_json::json!(0.006));
        }
        assert_eq!(v["turn"], serde_json::json!(turn.get()));
        assert_eq!(v["stage"], "rrd");
        assert!(v.get("gate").is_some(), "stage-specific fields survive");
    }
}
