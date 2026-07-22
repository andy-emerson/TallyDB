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
//! 2. **Ordered.** Segments are time-partitioned; data is expected to
//!    arrive roughly in order. Zone maps (min/max per column per segment)
//!    exist to exploit this for query pruning.
//! 3. **Numeric-or-key.** Every column here is an `arrow-lite` column —
//!    numeric or key, nothing else. This crate should never need to know
//!    about a third type; if it looks like it does, that's a signal
//!    something is wrong upstream, not a reason to add one here.
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

// TODO: I/O backend trait (native mmap implementation first)
// TODO: write buffer / append path
// TODO: segment format (header, per-column buffers, zone map)
// TODO: flush: write buffer -> segment
// TODO: tombstone record + "newest version wins" read resolution
// TODO: compaction: merge segments, drop resolved tombstones
// TODO: correctness test suite (this crate has no external oracle —
//       these tests ARE the spec, treat them accordingly)
