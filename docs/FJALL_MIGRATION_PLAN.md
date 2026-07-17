# RocksDB → Fjall migration plan

> Branch: `fork/rocksdb-to-fjall`. Status: **feasibility proven** (`crates/connxism/tests/fjall_spike.rs`, 4/4 green against Fjall 3.1.7). Not yet started on production `connxism`.
> Precondition (met): the engine is fully operational on RocksDB — 12 PRs, `main` green, 80 test binaries, zero `unsafe`, headline gates reproduced live.

## COSTAR

- **Context.** connxism runs on the C++ RocksDB via `bindgen` (needs clang/libclang at build). It is the only non-Rust dependency in a workspace where all 13 crates are `#![forbid(unsafe_code)]`. Fjall is a pure-Rust LSM (its own crate is `#![deny(unsafe_code)]`), so the fork removes the C++/bindgen toolchain requirement and makes the whole storage stack safe Rust — the "one clean engine, not maintained forks of C++ libs" direction.
- **Objective.** Replace RocksDB with Fjall behind connxism's existing `Db` wrapper, with **zero behavior change**: the full connxism test suite passes on the Fjall backend.
- **Scope.** Entirely inside `connxism` (~4 files: `estate.rs` `Db`, `store.rs`, `txn.rs`, `keys.rs`). Nothing else in the tree touches RocksDB. `recall`, the paged sidecar, `CF_GRAPH` are unaffected.
- **Tactics.** A storage-backend seam (a small `Kv` trait over the current `Db` methods) so RocksDB and Fjall **coexist behind a feature flag** during migration — not a rip-and-replace. Implement `FjallDb`, run the suite on both, then flip the default and remove RocksDB.
- **Audience.** clyffy-01 (this box) and any future puller; the payoff is a Rust-only build.
- **Result.** `connxism` on Fjall, suite green, RocksDB + bindgen gone.

## The seam (measured)

Every RocksDB call in connxism, and its Fjall equivalent (all proven in the spike except where noted):

| RocksDB (count in tree) | Fjall 3.x | proven |
|---|---|---|
| Column family (`ColumnFamilyDescriptor`, `cf_handle`) | `db.keyspace(name, opts)` → `Keyspace` | ✅ |
| `WriteBatch` + cross-CF atomic `write` (13 sites) | `db.batch()` → `insert(&ks, k, v)` / `commit()` | ✅ (atomic across keyspaces) |
| `iterator_cf(From/Start, Forward)` prefix/range scan (31 sites) | `ks.prefix(p)` / `ks.range(r)` (DoubleEndedIterator) | ✅ |
| `get_cf` / `put_cf` / `delete_cf` | `ks.get` / `ks.insert` / `ks.remove` | ✅ |
| WAL + `flush_wal(sync)` / fsync `WriteOptions` | `db.persist(PersistMode::SyncAll)` | — (API present) |
| Compression (Lz4/None per CF), block cache, write buffer, bg jobs | keyspace/db config knobs | — (API present) |
| **Merge operator** (`merge_operator_associative` "i64_add", `merge_cf`) for `tdf` df counters | **no equivalent** → RMW under the writer lock | ✅ (spike: nets correctly) |
| `checkpoint::Checkpoint` (hardlink SST backup) for `snapshot_to` | MVCC read snapshot (`db.snapshot()`); filesystem backup = copy keyspace dir | — (different mechanism) |
| `set_prefix_extractor` + memtable prefix bloom on `CF_TERMS` | Fjall bloom + prefix iter; tuning knob differs | — |
| BlobDB (`set_enable_blob_files`) on vector CFs | **moot** — 6b moved vectors to the paged `graph.vectors` sidecar, out of the LSM | ✅ (already done) |

## The three real design items

1. **Merge operators → read-modify-write.** connxism's `tdf` document-frequency counter uses RocksDB's associative merge for blind `+1/−1` (the no-read-modify-write law). Fjall has none. Replacement: RMW (`get` → add → `insert`), correct because connxism **already serializes every write through one writer lock**, so a term's counter never races itself. Proven in `df_counter_replacement_without_merge_operators`. (Alternative if the write lock is ever removed: derive df on demand from the postings keyspace.)
2. **Checkpoint/backup.** `snapshot_to` hardlinks SSTs for a crash-consistent copy. Fjall snapshots are MVCC *read* snapshots, not filesystem backups. Replacement: copy the keyspace directory under a persist barrier, or adopt Fjall's own backup if one lands. Lower priority (snapshots are an operator op).
3. **Prefix bloom on `CF_TERMS`.** The BM25 postings CF uses a custom prefix extractor + memtable prefix bloom. Fjall has bloom filters and prefix iteration; the exact tuning maps differently. Functional parity is free; perf tuning is a follow-up.

## Order (methodical, `main` never breaks)

1. **Seam.** Introduce a `Kv` trait mirroring the current `Db` methods (`get/put/delete_json`, `write(batch)`, `iterator`, `get_u64`, merge). Refactor connxism to it against the *existing* RocksDB impl — no behavior change, suite stays green. (PR 1)
2. **`FjallDb`.** Implement `Kv` on Fjall behind a `fjall` feature; RocksDB stays default. RMW for the counter. (PR 2)
3. **Parity gate.** Run the entire connxism suite on the Fjall backend in CI (feature-matrix). Fix divergences. (PR 3)
4. **Flip + remove.** Default to Fjall; delete the RocksDB path + `bindgen` dep once green on both a full cycle. (PR 4)

**Gate for the fork:** `cargo test -p connxism --features fjall` is green — same tests, same assertions, no RocksDB.

## Not in scope

Vector storage (paged sidecar), the ANN graph, recall, quantizers — none touch the LSM backend. This fork is the map/document/postings substrate only.
