//! The store's behavioral tests — persistence, the on-disk format and
//! its golden bytes, the write-ahead log under crash shapes, compaction,
//! residency, and read-only opens. They drive [`Store`](super::Store)
//! directly, which the crate's public API does not expose, so they live
//! inside the module rather than in `tests/`.

mod compaction;
mod format_v1;
mod persistence;
mod read_only;
mod residency;
mod wal;
