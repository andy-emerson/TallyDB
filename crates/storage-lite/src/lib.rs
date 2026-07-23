//! `storage-lite` — append-optimized, ordered, columnar storage.
//!
//! This is TallyDB's most original piece of engineering: no existing
//! mature project shares this exact combination of assumptions, so there
//! is no differential oracle to test against here (unlike `query-lite`,
//! which can diff against DuckDB/DataFusion). Correctness rests entirely
//! on this crate's own test suite. Take that seriously — write the tests
//! before or alongside the implementation, not after.
//!
//! ## The three assumptions, as they apply to this crate
//! 1. **Append-optimized.** The write path is a cheap, low-latency append
//!    of one row at a time. Whether the in-memory write buffer is
//!    literally row-major or a set of column arrays appended at the tail
//!    is an implementation detail — either is fine, the goal is cheap
//!    per-event append, not a specific memory layout at this stage.
//! 2. **Ordered.** Segments are partitioned on a declared **ordering key**;
//!    data is expected to arrive roughly sorted on it. The ordering key is
//!    usually a timestamp but need not be — any monotonic-on-ingest key
//!    (sequence id, event id, ledger offset) works, so partition on the
//!    *declared* key, don't hardcode "time." Zone maps (min/max per column
//!    per segment) exploit this ordering for query pruning, and delta /
//!    delta-of-delta compression exploits it to shrink ordered columns. This
//!    is why "ordered" is load-bearing: lose the clustering and both pruning
//!    and compression collapse.
//! 3. **Numeric-or-key.** Every column here is an `arrow-lite` column —
//!    numeric (`f64` or `i64`) or key, nothing else. This crate should never
//!    need to know about a third type; if it looks like it does, that's a
//!    signal something is wrong upstream, not a reason to add one here.
//!
//! ## Mutation model — one mechanism, reused everywhere
//! Segments are immutable once flushed. There is no in-place update.
//! Every case that needs to change existing data — an out-of-order
//! correction, a SQL `UPDATE`, a SQL `DELETE` — goes through
//! **tombstone + reinsert**: mark the old row deleted, append a new row
//! if there is one, resolve at the next compaction. Do not build a
//! second mutation path for any of these cases; if something seems to
//! need one, that's a design smell worth raising, not implementing around.
//!
//! **Open decision that gates the tombstone format (tracked in issues):**
//! what makes "newest version wins" well-defined — the row-identity rule.
//! Two candidates: an InfluxDB-style `(key-set, ordering-key)` primary key
//! with overwrite-on-collision (identity is closed, but two distinct events
//! sharing keys+ordering-key can't coexist without a disambiguating key), or
//! a kdb+-style pure-append model with an internal monotonic row id and
//! predicate-scoped deletes (duplicates allowed, deletes address rows by
//! predicate). These produce different tombstone records — decide before
//! fixing the segment/tombstone format, don't hardcode one meanwhile.
//!
//! **Second open decision (tracked in issues): dictionary scope.** Key
//! columns intern strings into a dictionary — per-table (global) or
//! per-segment? Global keeps codes stable across segments (no remapping in
//! cross-segment GROUP BY/joins) but adds a shared mutable structure
//! alongside immutable segments. Per-segment keeps segments fully
//! self-contained (clean immutability and compaction, maps 1:1 onto Arrow's
//! per-batch dictionary export) but requires code remapping at query time.
//! Interacts with the row-identity decision and compaction design — decide
//! when the segment format is fixed, not before, and don't hardcode either.
//!
//! ## Backend split (native today, WASM later)
//! I/O should sit behind a trait from the start. The native implementation
//! (mmap'd files) is the only one being built right now — but the trait
//! boundary is what lets an OPFS-backed WASM implementation slot in later
//! without touching anything above this crate. Don't let native-specific
//! assumptions (real filesystem paths, blocking I/O) leak past the trait.
//!
//! ## Scope for this crate
//! - Write buffer (append path)
//! - Flush: write buffer -> immutable columnar segment
//! - Compaction: merge segments, resolve tombstones
//! - Zone maps: min/max per column per segment, used for query pruning
//! - Compression: delta/delta-of-delta for ordered numeric columns
//!   (timestamps in particular), a documented scheme for general f64
//!   columns (e.g. Gorilla/XOR) — pick one, document the tradeoff, don't
//!   gold-plate this before the rest of the crate works.
//!
//! ## Explicitly NOT in scope for this crate
//! No SQL, no query planning — that's `query-lite`. No schema-level
//! type decisions — that's `engine`. No BLAS/Lua — that's the compute
//! crates.

pub mod mem;

pub use mem::{RowValue, Segment, StorageError, WriteBuffer};

// TODO: I/O backend trait (native mmap implementation first — designed
//       together with the on-disk segment format, not before; see mem.rs)
// TODO: on-disk segment format (header, per-column buffers, zone map)
// TODO: tombstone record + "newest version wins" read resolution
// TODO: compaction: merge segments, drop resolved tombstones
// TODO: correctness test suite (this crate has no external oracle —
//       these tests ARE the spec, treat them accordingly)
