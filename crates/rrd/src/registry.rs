//! The shape registry: the sliver lattice.
//!
//! Modes are the base shapes; every observed shape attaches beneath its mode
//! as a **sliver** — a thin specialization. Shapes *evolve*: a drifted
//! payload (field added/removed) is a new sliver beside its sibling, never a
//! silent reuse of the old plan.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mode::Mode;
use crate::shape::ShapeFingerprint;

/// A registered sliver: one observed shape under a mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sliver {
    /// Stable id (registry-scoped, dense).
    pub id: u64,
    /// Canonical shape key.
    pub key: String,
    /// The mode this sliver specializes.
    pub mode: Mode,
    /// Documents observed with this shape.
    pub count: u64,
}

/// The lattice: canonical shape key → sliver.
#[derive(Debug, Default)]
pub struct ShapeRegistry {
    slivers: HashMap<String, Sliver>,
    next_id: u64,
}

impl ShapeRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a shape: returns its sliver id and whether it is new (a JIT
    /// compile moment). Re-observation only bumps the count.
    pub fn observe(&mut self, shape: &ShapeFingerprint, mode: Mode) -> (u64, bool) {
        let key = shape.key();
        if let Some(s) = self.slivers.get_mut(&key) {
            s.count += 1;
            return (s.id, false);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.slivers.insert(
            key.clone(),
            Sliver {
                id,
                key,
                mode,
                count: 1,
            },
        );
        (id, true)
    }

    /// All slivers under a mode (the mode's slice of the lattice).
    pub fn slivers_of(&self, mode: Mode) -> Vec<&Sliver> {
        let mut v: Vec<&Sliver> = self.slivers.values().filter(|s| s.mode == mode).collect();
        v.sort_by_key(|s| s.id);
        v
    }

    /// Number of distinct slivers observed.
    pub fn len(&self) -> usize {
        self.slivers.len()
    }

    /// Whether nothing has been observed yet.
    pub fn is_empty(&self) -> bool {
        self.slivers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rro_core::Metadata;

    fn shape(keys: &[&str]) -> ShapeFingerprint {
        let m: Metadata = keys
            .iter()
            .map(|k| (k.to_string(), serde_json::Value::from("x")))
            .collect();
        ShapeFingerprint::of(&m)
    }

    #[test]
    fn same_shape_registers_once_drift_registers_new() {
        let mut reg = ShapeRegistry::new();
        let (id1, new1) = reg.observe(&shape(&["from", "subject"]), Mode::Mail);
        let (id2, new2) = reg.observe(&shape(&["from", "subject"]), Mode::Mail);
        assert!(new1 && !new2);
        assert_eq!(id1, id2);

        // Drift: an extra field is a NEW sliver, never silent reuse.
        let (id3, new3) = reg.observe(&shape(&["from", "subject", "thread"]), Mode::Mail);
        assert!(new3);
        assert_ne!(id1, id3);
        assert_eq!(reg.slivers_of(Mode::Mail).len(), 2);
    }
}
