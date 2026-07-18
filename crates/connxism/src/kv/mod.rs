//! The KV storage seam: exactly one backend, chosen at compile time.
//!
//! `connxism` speaks to one key/value backend, chosen by cargo feature —
//! `kvs-rocks` (the default) or `kvs-fjall`. Exactly one backend module compiles
//! per build; the two never coexist. Everything above this seam (`estate`,
//! `store`, `txn`, `query`, `filter`, `rels`) uses the re-exported
//! `Db`/`Batch`/`KvItem` and never names a concrete backend.
//!
//! **Selection is by precedence, not exclusion**, so any feature combination —
//! including `--all-features`, which CI uses everywhere — resolves to a single
//! backend rather than erroring:
//!   - `kvs-rocks` on (the default, and whenever both are enabled) → RocksDB.
//!     The shipping default stays the one CI's `--all-features` jobs exercise.
//!   - `kvs-fjall` on **and `kvs-rocks` off** → Fjall. This is how the Fjall
//!     build is produced: `--no-default-features --features kvs-fjall`.
//!
//! Comparing RocksDB against Fjall is a *clyffy-level* concern — build clyffy
//! against `connxism` each way and compare there. The engine carries no
//! dual-backend or differential machinery.

#[cfg(not(any(feature = "kvs-rocks", feature = "kvs-fjall")))]
compile_error!("connxism: enable a storage backend — `kvs-rocks` (default) or `kvs-fjall`");

#[cfg(feature = "kvs-rocks")]
mod rocks;
#[cfg(feature = "kvs-rocks")]
pub(crate) use rocks::{Batch, Db, KvItem};

// Fjall only when RocksDB is *not* selected, so both features together (e.g.
// `--all-features`) compile as RocksDB rather than double-defining the seam.
#[cfg(all(feature = "kvs-fjall", not(feature = "kvs-rocks")))]
mod fjall;
#[cfg(all(feature = "kvs-fjall", not(feature = "kvs-rocks")))]
pub(crate) use fjall::{Batch, Db, KvItem};
