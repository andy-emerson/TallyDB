//! A counting allocator, for the tests whose claim is about memory.
//!
//! Two of them measure peak bytes live during a query — the top-k
//! bound (#80) and the join's column pushdown (#81). The mechanism is
//! identical, so it lives here; what cannot be shared is the
//! `#[global_allocator]` attribute itself, which each test binary
//! declares over [`Counting`], because a global allocator is
//! process-wide and integration tests are separate processes.
//!
//! That process-wide reach is also why each of those binaries holds
//! exactly one measuring test: a second one running concurrently would
//! see the first one's allocations in its peak.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// The system allocator, counting live and peak bytes on the way past.
pub struct Counting;

// SAFETY: every method forwards to `System` unchanged; the counters are
// bookkeeping around it and never affect the pointers handed out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
}

/// Peak bytes live *above the level at entry* while `body` runs.
pub fn peak_of(body: impl FnOnce()) -> usize {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    body();
    PEAK.load(Ordering::Relaxed).saturating_sub(baseline)
}
