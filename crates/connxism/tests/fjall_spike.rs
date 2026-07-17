//! Feasibility spike for the RocksDB→Fjall fork (branch `fork/rocksdb-to-fjall`).
//!
//! Proves the storage seam maps. connxism leans on RocksDB for **keyspaces**
//! (column families), **atomic cross-keyspace batches** (WriteBatch), and
//! **prefix/range iteration** (IteratorMode::From). Fjall — a pure-Rust LSM whose
//! own crate is `#![deny(unsafe_code)]` — covers all three, and the one real gap
//! (merge operators) is documented here with its replacement, in a test, not a
//! wiki.
//!
//! Dev-only: never touches the production build. It grounds the migration in the
//! actual Fjall 3.x API before a line of connxism is rewritten.

use fjall::{Database, KeyspaceCreateOptions};

#[test]
fn keyspaces_map_to_column_families_and_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();

    // A keyspace per RocksDB column family.
    let nodes = db
        .keyspace("nodes", KeyspaceCreateOptions::default)
        .unwrap();
    let vecs = db.keyspace("vecs", KeyspaceCreateOptions::default).unwrap();

    nodes.insert("n1", "alpha").unwrap();
    vecs.insert("n1", [1u8, 2, 3]).unwrap();

    assert_eq!(nodes.get("n1").unwrap().as_deref(), Some(&b"alpha"[..]));
    assert_eq!(vecs.get("n1").unwrap().as_deref(), Some(&[1u8, 2, 3][..]));

    // Reopen: durable across restart (WAL recovery), like RocksDB.
    drop((nodes, vecs, db));
    let db = Database::builder(dir.path()).open().unwrap();
    let nodes = db
        .keyspace("nodes", KeyspaceCreateOptions::default)
        .unwrap();
    assert_eq!(nodes.get("n1").unwrap().as_deref(), Some(&b"alpha"[..]));
}

#[test]
fn cross_keyspace_batch_is_atomic() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();
    let docs = db.keyspace("docs", KeyspaceCreateOptions::default).unwrap();
    let terms = db
        .keyspace("terms", KeyspaceCreateOptions::default)
        .unwrap();

    // The commit-path invariant connxism relies on: a doc and its postings land
    // together or not at all. Fjall's batch spans keyspaces and commits atomically.
    let mut batch = db.batch();
    batch.insert(&docs, "d1", "hello world");
    batch.insert(&terms, "hello\x00d1", "");
    batch.insert(&terms, "world\x00d1", "");
    batch.commit().unwrap();

    assert!(docs.get("d1").unwrap().is_some());
    assert!(terms.get("hello\x00d1").unwrap().is_some());
    assert!(terms.get("world\x00d1").unwrap().is_some());
}

#[test]
fn prefix_scan_walks_one_terms_postings_list() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();
    let terms = db
        .keyspace("terms", KeyspaceCreateOptions::default)
        .unwrap();

    // `term \x00 doc_id` rows — the BM25 postings layout. A prefix scan over
    // `term \x00` walks exactly that term's list, same as IteratorMode::From.
    for doc in ["d1", "d2", "d9"] {
        terms.insert(format!("rust\x00{doc}"), "").unwrap();
    }
    terms.insert("go\x00d1", "").unwrap();

    let hits: Vec<String> = terms
        .prefix(b"rust\x00")
        .map(|guard| {
            // Fjall yields a `Guard`; `.key()` resolves it (fallible read).
            let k = guard.key().unwrap();
            String::from_utf8_lossy(&k).into_owned()
        })
        .collect();
    assert_eq!(
        hits.len(),
        3,
        "prefix scan returns only `rust` postings: {hits:?}"
    );
    assert!(hits.iter().all(|k| k.starts_with("rust\x00")));
}

/// The ONE real gap: RocksDB's associative merge operator (connxism's `tdf`
/// document-frequency counter does blind `+1/-1` merges, no read-modify-write).
/// Fjall has no merge-operator API. This documents the replacement the migration
/// must adopt — a read-modify-write, correct because connxism already serializes
/// every write through one writer lock, so a term's counter never races itself.
#[test]
fn df_counter_replacement_without_merge_operators() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();
    let tdf = db.keyspace("tdf", KeyspaceCreateOptions::default).unwrap();

    let bump = |delta: i64| {
        let cur = tdf
            .get("rust")
            .unwrap()
            .map(|b| i64::from_le_bytes(b.as_ref().try_into().unwrap()))
            .unwrap_or(0);
        tdf.insert("rust", (cur + delta).to_le_bytes()).unwrap();
    };
    bump(1);
    bump(1);
    bump(1);
    bump(-1);
    let df = i64::from_le_bytes(
        tdf.get("rust")
            .unwrap()
            .unwrap()
            .as_ref()
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        df, 2,
        "df counter nets correctly via RMW — no merge operator needed"
    );
}
