//! The Fjall storage backend (`kvs-fjall`).
//!
//! One of the two mutually-exclusive backends behind the KV seam (see
//! [`crate::kv`]). It maps connxism's RocksDB usage onto Fjall 3.1.7:
//!   - **column families → keyspaces** (`Database::keyspace`, one per
//!     [`COLUMN_FAMILIES`] entry),
//!   - **atomic cross-CF `WriteBatch` → `db.batch()`** committed through the
//!     shared journal,
//!   - **`IteratorMode::From`/`Start` → `Keyspace::range`/`iter`** (Fjall
//!     iterators are owned snapshots yielding [`fjall::Guard`]s),
//!   - **BlobDB on the vector CFs → KV separation** per keyspace,
//!   - and the one thing Fjall has no equivalent for — RocksDB's associative
//!     merge operator on `tdf` — is replaced by a **transaction-scoped
//!     read-modify-write** accumulated in the [`Batch`] and applied at
//!     [`Db::write`]. The on-disk `i64 LE` format is unchanged, so `tdf` needs
//!     no data migration and the two backends stay byte-compatible.
//!
//! Correctness of the RMW relies on connxism serialising every write through one
//! writer lock (see `store.rs`/`txn.rs`), so a `tdf` counter never races itself
//! between the read and the batch commit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fjall::config::{BloomConstructionPolicy, CompressionPolicy, FilterPolicy, FilterPolicyEntry};
use fjall::{
    CompressionType, Database, Keyspace, KeyspaceCreateOptions, KvSeparationOptions, PersistMode,
};
use rro_core::{Result, RroError};

use crate::estate::EstateConfig;
use crate::keys::{self, CF_META, COLUMN_FAMILIES};

/// Values at or above this size land in the keyspace's value log (KV
/// separation) instead of the LSM data blocks — the Fjall analogue of RocksDB's
/// BlobDB `min_blob_size`. A single dense vector (≥ 2 KiB) clears it; the small
/// keys around it do not, so compaction moves pointers, not vectors.
const KV_SEPARATION_THRESHOLD: u32 = 4 * 1024;

/// Map a Fjall error into the engine error type.
fn fjall_err(e: fjall::Error) -> RroError {
    RroError::Recall(format!("kvs: {e}"))
}

/// Per-keyspace options mirroring the RocksDB backend's per-CF tuning
/// (`kv/rocks.rs`): a point-lookup bloom, compression, per-CF memtable size, and
/// KV separation on the vector CFs.
fn keyspace_options(name: &str, memtable_bytes: u64) -> KeyspaceCreateOptions {
    // Point-lookup CFs get a bloom (10 bits/key ≈ 1% false positives) and are
    // told to expect hits; range-scanned CFs skip the bloom — a whole-key filter
    // cannot answer a prefix scan, so it would be pure space/cache waste.
    let point_lookup = matches!(
        name,
        keys::CF_DOCS
            | keys::CF_VECS
            | keys::CF_NVECS
            | keys::CF_MVECS
            | keys::CF_META
            | keys::CF_NODES
            | keys::CF_CONNS
            | keys::CF_COLL
            | keys::CF_TDF
    );
    let is_vector = matches!(name, keys::CF_VECS | keys::CF_NVECS | keys::CF_MVECS);

    let filter = if point_lookup {
        FilterPolicy::all(FilterPolicyEntry::Bloom(
            BloomConstructionPolicy::BitsPerKey(10.0),
        ))
    } else {
        FilterPolicy::all(FilterPolicyEntry::None)
    };
    // Dense f32 vectors do not compress meaningfully; paying CPU to not shrink
    // them is a straight loss on the hot read path. Everything else gets LZ4.
    let compression = if is_vector {
        CompressionPolicy::all(CompressionType::None)
    } else {
        CompressionPolicy::all(CompressionType::Lz4)
    };

    let mut opts = KeyspaceCreateOptions::default()
        .max_memtable_size(memtable_bytes)
        .expect_point_read_hits(point_lookup)
        .filter_policy(filter)
        .data_block_compression_policy(compression);

    // BlobDB → KV separation on the vector CFs: values above the threshold live
    // in a value log the LSM only references, so compaction moves pointers, not
    // 10 KiB vectors. Same intent as the RocksDB BlobDB knob.
    if is_vector {
        opts = opts.with_kv_separation(Some(
            KvSeparationOptions::default().separation_threshold(KV_SEPARATION_THRESHOLD),
        ));
    }
    opts
}

/// Decode an `i64 LE` counter value (tolerant of short/absent buffers).
fn read_i64(b: impl AsRef<[u8]>) -> i64 {
    let b = b.as_ref();
    let mut a = [0u8; 8];
    a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
    i64::from_le_bytes(a)
}

/// A column-family handle: a borrow of one open keyspace, carrying its canonical
/// name so batched merges can key their accumulator by CF. Same shape as the
/// RocksDB backend's `Cf` so every call site stays backend-agnostic.
#[derive(Clone, Copy)]
pub(crate) struct Cf<'a> {
    ks: &'a Keyspace,
    name: &'static str,
}

/// One raw key/value entry yielded by a store iterator (`iter_from`/`iter_all`).
pub(crate) type KvItem = Result<(Box<[u8]>, Box<[u8]>)>;

/// One staged write in a [`Batch`].
enum Op {
    Put(Keyspace, Vec<u8>, Vec<u8>),
    Delete(Keyspace, Vec<u8>),
    /// A document-frequency delta against `tdf` — applied as a read-modify-write
    /// at commit (Fjall has no merge operator). Carries the CF name so deltas to
    /// the same counter compose within one batch.
    MergeDf(Keyspace, &'static str, Vec<u8>, i64),
}

/// An accumulating atomic write, committed by [`Db::write`]. Unlike RocksDB's
/// `WriteBatch` (which is created from the DB), this is a backend-neutral op log
/// so the recall paths can build one with [`Batch::new`] and no DB handle, then
/// hand it to `Db::write` which translates it into a Fjall batch — folding the
/// `merge_df` deltas into read-modify-writes on the way.
#[derive(Default)]
pub(crate) struct Batch(Vec<Op>);

impl Batch {
    pub(crate) fn new() -> Self {
        Batch(Vec::new())
    }

    pub(crate) fn put_cf(&mut self, cf: Cf, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) {
        self.0.push(Op::Put(
            cf.ks.clone(),
            key.as_ref().to_vec(),
            value.as_ref().to_vec(),
        ));
    }

    pub(crate) fn delete_cf(&mut self, cf: Cf, key: impl AsRef<[u8]>) {
        self.0
            .push(Op::Delete(cf.ks.clone(), key.as_ref().to_vec()));
    }

    /// Stage a document-frequency delta (`±1`) against `cf`'s counter for `key`
    /// (see the module docs — this becomes an RMW at commit).
    pub(crate) fn merge_df(&mut self, cf: Cf, key: impl AsRef<[u8]>, delta: i64) {
        self.0.push(Op::MergeDf(
            cf.ks.clone(),
            cf.name,
            key.as_ref().to_vec(),
            delta,
        ));
    }
}

/// The open database: the Fjall handle, one keyspace per column family, and the
/// fsync-on-write choice. `Arc`-wrapped so clones are cheap (every
/// `ConnXRecall` holds one).
struct Inner {
    db: Database,
    parts: std::collections::HashMap<&'static str, Keyspace>,
    fsync: bool,
    path: PathBuf,
}

/// Shared handle to the open database — the KV seam, the single place that names
/// Fjall. Cloneable; all clones see one DB.
#[derive(Clone)]
pub(crate) struct Db(Arc<Inner>);

impl Db {
    /// Open (or create) the estate's Fjall database at `path`, applying the same
    /// per-CF tuning the RocksDB backend does: a shared block/blob cache, a
    /// global memtable ceiling, background workers, per-keyspace point-lookup
    /// blooms, compression, and KV separation on the vector CFs.
    pub(crate) fn open(path: &Path, config: &EstateConfig) -> Result<Db> {
        // Each keyspace's memtable is capped below (`max_memtable_size`), so the
        // total write memory is bounded by that × the live keyspaces — the same
        // budget the RocksDB backend states explicitly via `db_write_buffer_size`.
        let db = Database::builder(path)
            // One shared block/blob cache across keyspaces (RocksDB's shared LRU).
            .cache_size(config.block_cache_bytes as u64)
            // Background compaction/flush workers (RocksDB's `background_jobs`).
            .worker_threads(config.background_jobs.max(1))
            .open()
            .map_err(fjall_err)?;

        let write_buffer_bytes = config.write_buffer_bytes as u64;
        let mut parts = std::collections::HashMap::with_capacity(COLUMN_FAMILIES.len());
        for name in COLUMN_FAMILIES {
            let name = *name;
            let ks = db
                .keyspace(name, || keyspace_options(name, write_buffer_bytes))
                .map_err(fjall_err)?;
            parts.insert(name, ks);
        }
        Ok(Db(Arc::new(Inner {
            db,
            parts,
            fsync: config.fsync_writes,
            path: path.to_path_buf(),
        })))
    }

    pub(crate) fn cf(&self, name: &str) -> Result<Cf<'_>> {
        self.0
            .parts
            .get_key_value(name)
            .map(|(n, ks)| Cf { ks, name: n })
            .ok_or_else(|| RroError::Recall(format!("missing column family `{name}`")))
    }

    /// Raw value read from `cf`.
    pub(crate) fn get_cf(&self, cf: Cf, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        Ok(cf
            .ks
            .get(key)
            .map_err(fjall_err)?
            .map(|v| v.as_ref().to_vec()))
    }

    /// Raw single-key write into `cf` (bypassing a batch).
    pub(crate) fn put_cf(
        &self,
        cf: Cf,
        key: impl AsRef<[u8]>,
        value: impl AsRef<[u8]>,
    ) -> Result<()> {
        cf.ks
            .insert(key.as_ref().to_vec(), value.as_ref().to_vec())
            .map_err(fjall_err)
    }

    pub(crate) fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        cf: &str,
        key: &[u8],
    ) -> Result<Option<T>> {
        let handle = self.cf(cf)?;
        match self.get_cf(handle, key)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    pub(crate) fn put_json<T: serde::Serialize>(
        &self,
        cf: &str,
        key: &[u8],
        value: &T,
    ) -> Result<()> {
        let handle = self.cf(cf)?;
        self.put_cf(handle, key, serde_json::to_vec(value)?)
    }

    /// Iterate `cf` from `start` forward (a prefix or range seek). Callers apply
    /// their own `starts_with`/range break, matching `IteratorMode::From`.
    pub(crate) fn iter_from<'a>(
        &'a self,
        cf: Cf<'a>,
        start: &[u8],
    ) -> impl Iterator<Item = KvItem> + 'a {
        cf.ks.range(start.to_vec()..).map(guard_to_item)
    }

    /// Iterate every entry of `cf` in key order.
    pub(crate) fn iter_all<'a>(&'a self, cf: Cf<'a>) -> impl Iterator<Item = KvItem> + 'a {
        cf.ks.iter().map(guard_to_item)
    }

    /// Commit a batch atomically, honoring the estate's fsync-on-write choice.
    ///
    /// The `merge_df` deltas are folded into the same batch as read-modify-
    /// writes: composed per counter, read once against the committed store, and
    /// re-inserted. Correct because connxism serialises writers.
    pub(crate) fn write(&self, batch: Batch) -> Result<()> {
        let mut wb = self.0.db.batch();
        // Accumulate df deltas per (CF, key) so repeated merges compose before
        // the single read-modify-write.
        let mut df: BTreeMap<(&'static str, Vec<u8>), (Keyspace, i64)> = BTreeMap::new();
        for op in batch.0 {
            match op {
                Op::Put(ks, k, v) => wb.insert(&ks, k, v),
                Op::Delete(ks, k) => wb.remove(&ks, k),
                Op::MergeDf(ks, name, k, delta) => {
                    let entry = df.entry((name, k)).or_insert((ks, 0));
                    entry.1 += delta;
                }
            }
        }
        for ((_, key), (ks, delta)) in df {
            let current = ks.get(&key).map_err(fjall_err)?.map(read_i64).unwrap_or(0);
            wb.insert(&ks, key, (current + delta).to_le_bytes().to_vec());
        }
        wb.commit().map_err(fjall_err)?;
        if self.0.fsync {
            self.0.db.persist(PersistMode::SyncAll).map_err(fjall_err)?;
        }
        Ok(())
    }

    /// Make `cf`'s writes durable. Fjall persists at the database (journal)
    /// level, so this syncs the whole database — the callers that flush a single
    /// CF (e.g. the graph blob) want exactly that durability.
    pub(crate) fn flush_cf(&self, _cf: Cf) -> Result<()> {
        self.0.db.persist(PersistMode::SyncAll).map_err(fjall_err)
    }

    /// Sync the journal (Fjall's write-ahead log).
    pub(crate) fn flush_wal(&self, _sync: bool) -> Result<()> {
        self.0.db.persist(PersistMode::SyncAll).map_err(fjall_err)
    }

    /// Force a full compaction of `cf` — the operator-invoked optimizer pass,
    /// the same as RocksDB's `compact_range_cf`. Best-effort like that endpoint
    /// (which cannot fail the caller); a failure is logged, not propagated.
    pub(crate) fn compact_cf(&self, cf: Cf) {
        if let Err(e) = cf.ks.major_compact() {
            tracing::warn!("fjall major_compact of `{}` failed: {e}", cf.name);
        }
    }

    /// Live on-disk bytes held by `cf`.
    pub(crate) fn cf_sst_bytes(&self, cf: Cf) -> Result<u64> {
        Ok(cf.ks.disk_space())
    }

    /// Take a consistent on-disk snapshot into `path`. Fjall has no checkpoint
    /// primitive, so this persists the journal and copies the database
    /// directory — callers snapshot at quiescent points (see `Estate`).
    pub(crate) fn snapshot_to(&self, path: &Path) -> Result<()> {
        self.0.db.persist(PersistMode::SyncAll).map_err(fjall_err)?;
        copy_dir_all(&self.0.path, path).map_err(|e| RroError::Recall(format!("snapshot: {e}")))
    }

    pub(crate) fn get_u64(&self, key: &[u8]) -> Result<u64> {
        let handle = self.cf(CF_META)?;
        Ok(self
            .get_cf(handle, key)?
            .map(|b| {
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[..8.min(b.len())]);
                u64::from_le_bytes(a)
            })
            .unwrap_or(0))
    }
}

/// Resolve one iterator [`fjall::Guard`] into an owned key/value pair.
fn guard_to_item(guard: fjall::Guard) -> KvItem {
    let (k, v) = guard.into_inner().map_err(fjall_err)?;
    Ok((k.as_ref().into(), v.as_ref().into()))
}

/// Recursively copy a directory tree (the snapshot's file copy).
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &to)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}
