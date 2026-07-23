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
//! **Row identity (issue #1, decided 2026-07-23): kdb+-style pure
//! append.** Every row carries an internal monotonic row id; duplicates —
//! including distinct events sharing keys and ordering-key value — are
//! first-class. `UPDATE`/`DELETE` address rows by predicate; a correction
//! supersedes by ingest sequence, which is what "newest version wins"
//! means. Tombstones therefore reference row ids (or carry predicates),
//! never key tuples. The rejected InfluxDB-style `(key-set, ordering-key)`
//! primary key silently collapses same-tuple events; if user-visible
//! overwrite semantics are ever needed, the path is an opt-in declared
//! uniqueness constraint layered on top — additive, not a reversal.
//!
//! **Dictionary scope (issue #6, decided 2026-07-23): per-segment.** Each
//! segment carries its own interning table, so segments are fully
//! self-contained: immutability holds everywhere, crash consistency is
//! trivial, compaction merges dictionaries as part of merging segments,
//! and the layout maps 1:1 onto Arrow's per-batch dictionary export. With
//! identity resolved by row id, compaction never compares key values
//! across segments — which was the workload a global dictionary would
//! have served. Cross-segment grouping/joins remap codes at query time,
//! bounded by the low-cardinality assumption; the recorded extension is a
//! process-lifetime remap cache, added only when profiling produces a
//! number that asks for it.
//!
//! ## Backend split (native today, WASM later)
//! I/O sits behind the [`io::StorageBackend`] trait — a flat namespace
//! of named byte objects with atomic publish, nothing filesystem-shaped
//! in its contract. The native implementation is a directory of files
//! ([`io::FsBackend`]); an OPFS-backed WASM implementation slots in
//! later without touching anything above the trait. mmap and ranged
//! reads are recorded follow-ups for when query-time pruning or a
//! profiling number asks for them (v1 opens decode into memory, so mmap
//! would buy nothing today).
//!
//! ## Scope for this crate
//! Built: the write buffer and append path ([`mem`]), the multi-segment
//! per-table [`store::Store`] with internal row ids, the deterministic
//! golden-locked on-disk format with zone maps and per-column codec
//! tags ([`mod@format`]), delta-of-delta for the ordered ordering key
//! ([`codec`], measurement cited there), and persistence with
//! reopen-and-verify behind the backend trait ([`io`]). Durability
//! boundary: [`store::Store::flush`].
//!
//! Still ahead: tombstones and "newest version wins" resolution,
//! compaction (both M2.3), zone-map pruning at query time (with WHERE,
//! M2.4), and the general-`f64` codec (#30, deferred by ruling —
//! uncompressed behind the tag is the shipped interim answer).
//!
//! ## Explicitly NOT in scope for this crate
//! No SQL, no query planning — that's `query-lite`. No schema-level
//! type decisions — that's `engine`. No BLAS/Lua — that's the compute
//! crates.

pub mod codec;
pub mod format;
pub mod io;
pub mod mem;
pub mod store;

pub use codec::{Codec, CodecError};
pub use format::{decode_segment, encode_segment, FormatError};
pub use io::{FsBackend, IoError, MemBackend, StorageBackend};
pub use mem::{RowValue, Segment, StorageError, WriteBuffer};
pub use store::{Store, DEFAULT_SEGMENT_ROWS};

// TODO: tombstone record + "newest version wins" read resolution
// TODO: compaction: merge segments, drop resolved tombstones
