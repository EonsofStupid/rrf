//! The RocksDB storage backend (`kvs-rocks`, the default).
//!
//! This is one of the two mutually-exclusive backends behind the KV seam (see
//! [`crate::kv`]). It owns the RocksDB configuration — the per-column-family
//! options, the shared block cache, BlobDB for the vector CFs, the prefix bloom
//! on the postings CF, and the `tdf` associative merge operator — and exposes
//! the seam surface (`Db`/`Cf`/`Batch`/`KvItem`) that the rest of `connxism`
//! speaks to. Nothing outside this file names RocksDB.

use std::path::Path;
use std::sync::Arc;

use rocksdb::{ColumnFamily, Options, DB};
use rro_core::{Result, RroError};

use crate::estate::EstateConfig;
use crate::keys::{self, CF_META, COLUMN_FAMILIES};

/// A column-family handle: a borrow of the open database, tied to it for the
/// duration of a read/write. This is the KV seam's opaque CF token — call sites
/// obtain one from [`Db::cf`] and hand it to the `Db`/[`Batch`] methods rather
/// than reaching for a backend-specific handle type. RocksDB's `cf_handle`
/// returns a borrow (`&ColumnFamily`) under single-threaded mode, so the seam
/// carries that lifetime; the Fjall backend wraps a borrowed `Keyspace` the same
/// shape, keeping every call site backend-agnostic.
#[derive(Clone, Copy)]
pub(crate) struct Cf<'a>(&'a ColumnFamily);

/// One raw key/value entry yielded by a store iterator (`iter_from`/`iter_all`).
/// Aliased so the iterator return types stay readable across the seam.
pub(crate) type KvItem = Result<(Box<[u8]>, Box<[u8]>)>;

/// An accumulating atomic write, committed by [`Db::write`]. Wraps the backend's
/// native batch (RocksDB `WriteBatch`) so the recall paths never name it: they
/// build one with [`Batch::new`], stage `put`/`delete`/`merge_df` against a
/// [`Cf`], and hand it back to `Db::write`, which commits it atomically.
#[derive(Default)]
pub(crate) struct Batch(rocksdb::WriteBatch);

impl Batch {
    /// A fresh, empty batch.
    pub(crate) fn new() -> Self {
        Batch(rocksdb::WriteBatch::default())
    }

    /// Stage a key/value write into `cf`.
    pub(crate) fn put_cf(&mut self, cf: Cf, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) {
        self.0.put_cf(cf.0, key, value);
    }

    /// Stage a delete of `key` in `cf`.
    pub(crate) fn delete_cf(&mut self, cf: Cf, key: impl AsRef<[u8]>) {
        self.0.delete_cf(cf.0, key);
    }

    /// Stage a document-frequency delta (`±1`) against `cf`'s counter for `key`.
    ///
    /// On RocksDB this is a blind associative merge (`merge_i64_add` composes the
    /// operands at read/compaction time). The Fjall backend has no merge
    /// operator, so this same call becomes a transaction-scoped read-modify-write
    /// accumulated and applied at commit — the on-disk `i64 LE` format is
    /// identical, so the two backends are byte-compatible on `tdf`.
    pub(crate) fn merge_df(&mut self, cf: Cf, key: impl AsRef<[u8]>, delta: i64) {
        self.0.merge_cf(cf.0, key, delta.to_le_bytes());
    }
}

/// Shared handle to the open database. Cloneable; all clones see one DB.
///
/// This is the KV seam — the single place that names RocksDB. Every recall path
/// reaches the store through these methods (`cf`, `get_cf`/`put_cf`,
/// `get_json`/`put_json`, `iter_from`/`iter_all`, `write`, the flush/compact
/// housekeeping), never through a raw backend handle.
#[derive(Clone)]
pub(crate) struct Db(Arc<DB>, bool);

impl Db {
    /// Open (or create) the estate's RocksDB at `path`, applying the per-CF
    /// options that make the estate's recall sub-millisecond.
    pub(crate) fn open(path: &Path, config: &EstateConfig) -> Result<Db> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // ---- RocksDB, actually configured -------------------------------
        //
        // Everything below was RocksDB defaults until 2026-07-16, which for a
        // 16-CF estate serving point lookups meant: an 8 MiB block cache shared
        // by every CF, and NO bloom filters — so each point lookup (a doc, a
        // vector, a posting) touched every SST at every level before answering.
        // For an engine whose whole claim is sub-ms recall, that is not a
        // detail; it is the difference between a memory hit and a disk walk.
        //
        // Sized from the config so a laptop and the GB10 are the same code with
        // a different number, not two paths.
        opts.increase_parallelism(config.background_jobs as i32);
        opts.set_max_background_jobs(config.background_jobs as i32);

        // The memtable budget is the sum nobody was computing. `write_buffer_size`
        // is set PER CF (below), and RocksDB keeps up to `max_write_buffer_number`
        // (default 2) live per CF — so the real ceiling is
        // `write_buffer_bytes × max_write_buffer_number × COLUMN_FAMILY_COUNT`.
        // At the defaults that is 64 MiB × 2 × 16 = **2 GiB** of memtables, a
        // number that just fell out of a per-CF knob nobody multiplied out. Cap
        // it explicitly with `db_write_buffer_size`, a hard ceiling across all
        // CFs, so the estate's write memory is a stated budget rather than an
        // accident of the CF count.
        let max_write_buffers = 2u64;
        let memtable_budget =
            (config.write_buffer_bytes as u64) * max_write_buffers * (COLUMN_FAMILIES.len() as u64);
        opts.set_db_write_buffer_size(memtable_budget as usize);

        // One shared block cache across CFs: a per-CF cache partitions memory by
        // guesswork, and the hot set moves with the workload.
        let cache = rocksdb::Cache::new_lru_cache(config.block_cache_bytes);

        let block_opts = |bloom: bool| {
            let mut b = rocksdb::BlockBasedOptions::default();
            b.set_block_cache(&cache);
            b.set_block_size(16 * 1024);
            // Cache index/filter blocks WITH the data, and pin the top level:
            // otherwise the filters get evicted under load and the bloom stops
            // helping exactly when it matters.
            b.set_cache_index_and_filter_blocks(true);
            b.set_pin_l0_filter_and_index_blocks_in_cache(true);
            if bloom {
                // 10 bits/key ~= 1% false positives — the standard point-lookup
                // trade. Only on CFs actually read by exact key.
                b.set_bloom_filter(10.0, false);
            }
            b
        };

        // Per-CF options, matched to how each CF is actually read.
        let descriptors: Vec<rocksdb::ColumnFamilyDescriptor> = COLUMN_FAMILIES
            .iter()
            .map(|cf| {
                let mut cf_opts = Options::default();

                // Point-lookup CFs get a bloom; range-scanned CFs do not (a
                // filter cannot answer a prefix scan, so it would be pure write
                // amplification and cache pressure).
                let point_lookup = matches!(
                    *cf,
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
                cf_opts.set_block_based_table_factory(&block_opts(point_lookup));

                // Vectors are dense f32 that do not compress meaningfully;
                // paying CPU to not shrink them is a straight loss on the hot
                // read path. Everything else (text, postings, JSON) does.
                if matches!(*cf, keys::CF_VECS | keys::CF_NVECS | keys::CF_MVECS) {
                    cf_opts.set_compression_type(rocksdb::DBCompressionType::None);

                    // BlobDB for the vector CFs. A 2560-d f32 vector is ~10 KiB,
                    // and in a plain LSM every value is rewritten by every
                    // compaction that touches its key — so the vectors, which
                    // never change, get copied over and over for the sake of
                    // compacting the small keys around them. BlobDB stores values
                    // above `min_blob_size` in separate blob files that the LSM
                    // only references, so compaction moves 8-byte pointers instead
                    // of 10 KiB payloads. `min_blob_size` is set below a single
                    // vector so every vector lands in a blob; nothing smaller
                    // (there is nothing smaller in these CFs) pays the indirection.
                    cf_opts.set_enable_blob_files(true);
                    cf_opts.set_min_blob_size(4 * 1024);
                    cf_opts.set_enable_blob_gc(true);
                } else {
                    cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                }

                // The BM25 postings CF is read by **prefix scan**: keys are
                // `term \x00 doc_id` and a lexical lookup seeks `term \x00` then
                // iterates. A whole-key bloom cannot help a scan (it answers "is
                // this exact key present", not "does this prefix exist"), so
                // `CF_TERMS` was left with no filter at all — every posting-list
                // read paid full index descent. A **prefix** extractor + memtable
                // prefix bloom fixes exactly that: the bloom answers "could this
                // term have any postings" and skips SSTables and memtables that
                // hold none. The extractor is custom because terms are
                // variable-length — the prefix is everything up to and including
                // the first NUL, not a fixed byte count.
                if *cf == keys::CF_TERMS {
                    cf_opts.set_prefix_extractor(rocksdb::SliceTransform::create(
                        "term_prefix",
                        |key: &[u8]| match key.iter().position(|&b| b == 0) {
                            Some(nul) => &key[..=nul],
                            None => key,
                        },
                        Some(|key: &[u8]| key.contains(&0)),
                    ));
                    cf_opts.set_memtable_prefix_bloom_ratio(0.1);
                }

                cf_opts.set_write_buffer_size(config.write_buffer_bytes);

                // `tdf` carries an associative merge operator so document-
                // frequency counters update as blind merge writes.
                if *cf == keys::CF_TDF {
                    cf_opts.set_merge_operator_associative("i64_add", merge_i64_add);
                }
                rocksdb::ColumnFamilyDescriptor::new(*cf, cf_opts)
            })
            .collect();
        let db = DB::open_cf_descriptors(&opts, path, descriptors).map_err(rocks_err)?;
        Ok(Db(Arc::new(db), config.fsync_writes))
    }

    pub(crate) fn cf(&self, name: &str) -> Result<Cf<'_>> {
        self.0
            .cf_handle(name)
            .map(Cf)
            .ok_or_else(|| RroError::Recall(format!("missing column family `{name}`")))
    }

    /// Raw value read from `cf`.
    pub(crate) fn get_cf(&self, cf: Cf, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        self.0.get_cf(cf.0, key).map_err(rocks_err)
    }

    /// Raw single-key write into `cf` (bypassing a batch).
    pub(crate) fn put_cf(
        &self,
        cf: Cf,
        key: impl AsRef<[u8]>,
        value: impl AsRef<[u8]>,
    ) -> Result<()> {
        self.0.put_cf(cf.0, key, value).map_err(rocks_err)
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
    /// their own `starts_with`/range break, matching RocksDB `IteratorMode::From`.
    pub(crate) fn iter_from<'a>(
        &'a self,
        cf: Cf<'a>,
        start: &[u8],
    ) -> impl Iterator<Item = KvItem> + 'a {
        self.0
            .iterator_cf(
                cf.0,
                rocksdb::IteratorMode::From(start, rocksdb::Direction::Forward),
            )
            .map(|r| r.map_err(rocks_err))
    }

    /// Iterate every entry of `cf` in key order (RocksDB `IteratorMode::Start`).
    pub(crate) fn iter_all<'a>(&'a self, cf: Cf<'a>) -> impl Iterator<Item = KvItem> + 'a {
        self.0
            .iterator_cf(cf.0, rocksdb::IteratorMode::Start)
            .map(|r| r.map_err(rocks_err))
    }

    /// Commit a batch, honoring the estate's fsync-on-write choice.
    pub(crate) fn write(&self, batch: Batch) -> Result<()> {
        if self.1 {
            let mut wo = rocksdb::WriteOptions::default();
            wo.set_sync(true);
            self.0.write_opt(batch.0, &wo).map_err(rocks_err)
        } else {
            self.0.write(batch.0).map_err(rocks_err)
        }
    }

    /// Flush `cf`'s memtable to an SST.
    pub(crate) fn flush_cf(&self, cf: Cf) -> Result<()> {
        self.0.flush_cf(cf.0).map_err(rocks_err)
    }

    /// Sync the write-ahead log.
    pub(crate) fn flush_wal(&self, sync: bool) -> Result<()> {
        self.0.flush_wal(sync).map_err(rocks_err)
    }

    /// Force a full-range compaction of `cf` (operator-invoked optimizer pass).
    pub(crate) fn compact_cf(&self, cf: Cf) {
        self.0.compact_range_cf(cf.0, None::<&[u8]>, None::<&[u8]>);
    }

    /// Live SST bytes held by `cf`.
    pub(crate) fn cf_sst_bytes(&self, cf: Cf) -> Result<u64> {
        Ok(self
            .0
            .property_int_value_cf(cf.0, "rocksdb.total-sst-files-size")
            .map_err(rocks_err)?
            .unwrap_or(0))
    }

    /// Take a consistent on-disk snapshot into `path` (a fresh directory).
    pub(crate) fn snapshot_to(&self, path: &Path) -> Result<()> {
        let checkpoint = rocksdb::checkpoint::Checkpoint::new(&self.0).map_err(rocks_err)?;
        checkpoint.create_checkpoint(path).map_err(rocks_err)
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

/// Associative merge: value = existing + Σ operand (i64 LE deltas).
fn merge_i64_add(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &rocksdb::MergeOperands,
) -> Option<Vec<u8>> {
    let read = |b: &[u8]| -> i64 {
        let mut a = [0u8; 8];
        a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
        i64::from_le_bytes(a)
    };
    let mut acc = existing.map(read).unwrap_or(0);
    for op in operands {
        acc += read(op);
    }
    Some(acc.to_le_bytes().to_vec())
}

/// Map RocksDB errors into the engine error type.
fn rocks_err(e: rocksdb::Error) -> RroError {
    RroError::Recall(format!("kvs: {e}"))
}
