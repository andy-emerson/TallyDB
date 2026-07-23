//! Contiguous, 64-byte-aligned element buffers — the physical storage
//! behind every column.
//!
//! ## Why not `Vec<T>`
//!
//! `Vec` aligns to the element type (8 bytes for `f64`/`i64`); this crate's
//! contract promises 64-byte alignment so the same buffer is cache-line
//! clean for BLAS/LAPACK and matches Arrow's recommended allocation
//! alignment. [`Buffer`] is the one place that owns that promise — a small
//! `Vec` work-alike over a 64-byte-aligned allocation. All other unsafe in
//! this crate should stay confined to this module and the C Data Interface.
//!
//! ## Sharing: clone is O(1), mutation copies on write
//!
//! Handles share the allocation: cloning a buffer clones an `Arc`, not the
//! bytes. Mutation (`push`, `extend`, `as_mut_slice`) first ensures the
//! handle is the allocation's only owner, copying it if not — so clones
//! behave exactly like independent buffers while reads stay zero-copy.
//! This is what lets a query result carry a stored column, and the C Data
//! export hand it out, without duplicating row data.
//!
//! ## Alignment guarantee
//!
//! [`Buffer::as_ptr`] is 64-byte aligned for every buffer, including an
//! empty one (which uses a non-null, 64-aligned placeholder address and
//! performs no allocation).

use crate::bitmap::Bitmap;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

/// Alignment, in bytes, of every buffer allocation.
pub const BUFFER_ALIGN: usize = 64;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for i64 {}
    impl Sealed for u32 {}
}

/// The element types a [`Buffer`] may hold.
///
/// Sealed to the physical types the format defines: `f64` and `i64` value
/// columns, and `u32` dictionary codes for key columns. A new element type
/// (e.g. `f32`, issue #3) is an additive `impl` here plus a subtype-enum
/// variant — not a format migration.
pub trait Element: sealed::Sealed + Copy + PartialEq + fmt::Debug + Send + Sync + 'static {}

impl Element for f64 {}
impl Element for i64 {}
impl Element for u32 {}

/// One owned, 64-byte-aligned allocation; deallocates on drop. Buffers
/// share these through an `Arc`.
struct RawBuf<T: Element> {
    /// 64-byte aligned; a placeholder address (no allocation) when
    /// `cap == 0`.
    ptr: NonNull<T>,
    /// Elements allocated.
    cap: usize,
    _owns: PhantomData<T>,
}

// SAFETY: RawBuf owns its allocation outright and Element requires
// Send + Sync, so moving or sharing it across threads moves or shares
// plain numeric data.
unsafe impl<T: Element> Send for RawBuf<T> {}
unsafe impl<T: Element> Sync for RawBuf<T> {}

impl<T: Element> RawBuf<T> {
    /// The 64-aligned, non-null placeholder pointer used while unallocated.
    fn placeholder() -> NonNull<T> {
        // SAFETY: BUFFER_ALIGN is non-zero, so the pointer is non-null; it
        // carries no provenance and is never read or written — `cap == 0`
        // guards every access.
        unsafe { NonNull::new_unchecked(std::ptr::without_provenance_mut(BUFFER_ALIGN)) }
    }

    /// No allocation, zero capacity.
    fn empty() -> Self {
        RawBuf {
            ptr: Self::placeholder(),
            cap: 0,
            _owns: PhantomData,
        }
    }

    /// The allocation layout for `cap` elements.
    ///
    /// # Panics
    /// If the byte size overflows (unreachable for real row counts).
    fn layout(cap: usize) -> Layout {
        let bytes = cap.checked_mul(size_of::<T>()).expect("capacity overflow");
        Layout::from_size_align(bytes, BUFFER_ALIGN).expect("layout overflow")
    }

    /// A fresh allocation of `cap > 0` elements, uninitialized.
    fn allocate(cap: usize) -> Self {
        debug_assert!(cap > 0);
        let layout = Self::layout(cap);
        // SAFETY: layout has non-zero size (cap > 0, T never zero-sized).
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw.cast::<T>()) else {
            handle_alloc_error(layout);
        };
        RawBuf {
            ptr,
            cap,
            _owns: PhantomData,
        }
    }
}

impl<T: Element> Drop for RawBuf<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            // SAFETY: pointer came from `alloc` with this exact layout;
            // elements are Copy, so no per-element drop is needed.
            unsafe { dealloc(self.ptr.as_ptr().cast(), Self::layout(self.cap)) };
        }
    }
}

/// A growable, contiguous, 64-byte-aligned buffer of one [`Element`] type.
/// Clones share the allocation; mutation copies on write.
///
/// ```
/// use arrow_lite::buffer::{Buffer, BUFFER_ALIGN};
///
/// let mut buf = Buffer::<f64>::new();
/// buf.extend_from_slice(&[1.0, 2.0, 3.0]);
/// buf.push(4.0);
/// assert_eq!(&buf[..], &[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(buf.as_ptr() as usize % BUFFER_ALIGN, 0);
///
/// // A clone is a second handle to the same bytes — no copy...
/// let shared = buf.clone();
/// assert_eq!(shared.as_ptr(), buf.as_ptr());
/// // ...until one side mutates, which copies first (the other survives).
/// buf.push(5.0);
/// assert_eq!(&shared[..], &[1.0, 2.0, 3.0, 4.0]);
/// ```
pub struct Buffer<T: Element> {
    raw: Arc<RawBuf<T>>,
    /// Elements in use by *this handle* (a shared allocation may hold
    /// more capacity than any one handle uses).
    len: usize,
}

impl<T: Element> Buffer<T> {
    /// An empty buffer. Does not allocate.
    pub fn new() -> Self {
        Buffer {
            raw: Arc::new(RawBuf::empty()),
            len: 0,
        }
    }

    /// An empty buffer with room for `cap` elements.
    pub fn with_capacity(cap: usize) -> Self {
        Buffer {
            raw: Arc::new(if cap == 0 {
                RawBuf::empty()
            } else {
                RawBuf::allocate(cap)
            }),
            len: 0,
        }
    }

    /// A buffer holding a copy of `values`.
    pub fn from_slice(values: &[T]) -> Self {
        let mut buf = Self::with_capacity(values.len());
        buf.extend_from_slice(values);
        buf
    }

    /// Ensures this handle solely owns an allocation with room for
    /// `additional` more elements, copying the current contents into a
    /// fresh allocation when shared or full.
    fn make_mut(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).expect("capacity overflow");
        let unique = Arc::get_mut(&mut self.raw).is_some();
        if unique && needed <= self.raw.cap {
            return;
        }
        // Geometric growth keeps amortized push O(1); a shared allocation
        // is copied at the same policy.
        let new_cap = needed.max(self.raw.cap * 2).max(8);
        let new = RawBuf::allocate(new_cap);
        if self.len > 0 {
            // SAFETY: source is valid for len elements; destination is a
            // fresh allocation of new_cap >= len; the regions are
            // distinct.
            unsafe {
                std::ptr::copy_nonoverlapping(self.raw.ptr.as_ptr(), new.ptr.as_ptr(), self.len);
            }
        }
        self.raw = Arc::new(new);
    }

    /// Appends one element.
    pub fn push(&mut self, value: T) {
        self.make_mut(1);
        // SAFETY: make_mut guarantees sole ownership and len < cap, so the
        // write is in bounds and unaliased.
        unsafe { self.raw.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    /// Appends a slice of elements.
    pub fn extend_from_slice(&mut self, values: &[T]) {
        // SAFETY: make_mut (inside extend_from_raw) guarantees room and
        // sole ownership; a slice is always valid for its own length.
        unsafe { self.extend_from_raw(values.as_ptr(), values.len()) };
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The elements as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is valid for `len` initialized elements (or a
        // well-aligned placeholder when len == 0, valid for an empty
        // slice); shared handles only ever read below their own len.
        unsafe { std::slice::from_raw_parts(self.raw.ptr.as_ptr(), self.len) }
    }

    /// The elements as a mutable slice (copies first if the allocation is
    /// shared).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.make_mut(0);
        // SAFETY: as for `as_slice`, plus make_mut guarantees sole
        // ownership and &mut self gives handle exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.raw.ptr.as_ptr(), self.len) }
    }

    /// The base pointer — 64-byte aligned, non-null even when empty.
    pub fn as_ptr(&self) -> *const T {
        self.raw.ptr.as_ptr()
    }

    /// Appends `count` elements copied byte-wise from `src`.
    ///
    /// For the C Data Interface import path, where foreign buffers carry
    /// no alignment promise — the byte-wise copy is legal for any `src`.
    ///
    /// # Safety
    /// `src` must be valid for reading `count * size_of::<T>()` bytes that
    /// represent initialized `T` values (any bit pattern is a valid f64/
    /// i64/u32, so this reduces to the range being readable).
    pub(crate) unsafe fn extend_from_raw(&mut self, src: *const T, count: usize) {
        self.make_mut(count);
        // SAFETY: make_mut guarantees sole ownership and room past len;
        // caller guarantees src readable; a fresh or solely-owned
        // destination cannot overlap a live source slice.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.cast::<u8>(),
                self.raw.ptr.as_ptr().add(self.len).cast::<u8>(),
                count * size_of::<T>(),
            );
        }
        self.len += count;
    }
}

impl<T: Element> Default for Buffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Element> Clone for Buffer<T> {
    /// O(1): shares the allocation. The first mutation on either handle
    /// copies (see the module docs).
    fn clone(&self) -> Self {
        Buffer {
            raw: Arc::clone(&self.raw),
            len: self.len,
        }
    }
}

impl<T: Element> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Element> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Element> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Element> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buf = Self::new();
        for v in iter {
            buf.push(v);
        }
        buf
    }
}

/// A numeric column: an aligned value buffer plus validity, present only
/// when the schema declares the column nullable.
///
/// A `NOT NULL` column has **no** bitmap — its buffer is BLAS-ready by
/// construction, with no null check on the compute path. The ordering key
/// is always `NOT NULL`. For a nullable column, the value under a null slot
/// is unspecified but initialized (constructors take a concrete value for
/// every row).
///
/// ```
/// use arrow_lite::{Bitmap, NumericColumn};
///
/// let col = NumericColumn::new_non_null([1.5, 2.5].into_iter().collect());
/// assert_eq!(col.null_count(), 0);
/// assert!(col.validity().is_none());
///
/// let nullable = NumericColumn::new_nullable(
///     [1.5, 0.0].into_iter().collect(),
///     Bitmap::from_bools([true, false]),
/// );
/// assert_eq!(nullable.null_count(), 1);
/// assert!(!nullable.is_valid(1));
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct NumericColumn<T: Element> {
    values: Buffer<T>,
    validity: Option<Bitmap>,
}

impl<T: Element> NumericColumn<T> {
    /// A column the schema declares `NOT NULL`: values only, no bitmap.
    pub fn new_non_null(values: Buffer<T>) -> Self {
        NumericColumn {
            values,
            validity: None,
        }
    }

    /// A nullable column: values plus a validity bitmap of equal length.
    ///
    /// # Panics
    /// If the bitmap length differs from the value count.
    pub fn new_nullable(values: Buffer<T>, validity: Bitmap) -> Self {
        assert_eq!(
            values.len(),
            validity.len(),
            "validity length {} != value count {}",
            validity.len(),
            values.len()
        );
        NumericColumn {
            values,
            validity: Some(validity),
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The value buffer.
    pub fn values(&self) -> &Buffer<T> {
        &self.values
    }

    /// The validity bitmap — `None` exactly when the schema said
    /// `NOT NULL`.
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Whether the row at `index` holds a value (always true without a
    /// bitmap).
    ///
    /// # Panics
    /// If `index >= len`.
    pub fn is_valid(&self, index: usize) -> bool {
        assert!(
            index < self.len(),
            "row {index} out of range {}",
            self.len()
        );
        match &self.validity {
            Some(bm) => bm.get(index),
            None => true,
        }
    }

    /// Number of null rows (zero without a bitmap).
    pub fn null_count(&self) -> usize {
        match &self.validity {
            Some(bm) => bm.len() - bm.count_set(),
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn empty_buffer_is_aligned_and_allocation_free() {
        let buf = Buffer::<f64>::new();
        assert!(buf.is_empty());
        assert_eq!(buf.as_ptr() as usize % BUFFER_ALIGN, 0);
        assert_eq!(buf.as_slice(), &[] as &[f64]);
    }

    #[test]
    fn alignment_holds_across_growth() {
        // Push one element at a time so every reallocation size is hit.
        let mut buf = Buffer::<i64>::new();
        for i in 0..1000 {
            buf.push(i);
            assert_eq!(buf.as_ptr() as usize % BUFFER_ALIGN, 0);
        }
        assert_eq!(buf.len(), 1000);
        assert!(buf.iter().copied().eq(0..1000));
    }

    #[test]
    fn from_slice_and_clone_are_independent_copies() {
        let a = Buffer::from_slice(&[1.0, 2.0, 3.0]);
        let mut b = a.clone();
        b.as_mut_slice()[0] = 9.0;
        assert_eq!(a[0], 1.0);
        assert_eq!(b[0], 9.0);
        assert_ne!(a, b);
        assert_eq!(b.as_ptr() as usize % BUFFER_ALIGN, 0);
    }

    #[test]
    fn clone_shares_until_mutation() {
        let mut a = Buffer::from_slice(&[1i64, 2, 3]);
        let b = a.clone();
        // Zero-copy: both handles read the same allocation.
        assert_eq!(a.as_ptr(), b.as_ptr());
        // Mutating one triggers copy-on-write; the other keeps the
        // original bytes at the original address.
        let b_ptr = b.as_ptr();
        a.push(4);
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert_eq!(b.as_ptr(), b_ptr);
        assert_eq!(b.as_slice(), &[1, 2, 3]);
        assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(a.as_ptr() as usize % BUFFER_ALIGN, 0);
    }

    #[test]
    fn with_capacity_preallocates_aligned() {
        let buf = Buffer::<u32>::with_capacity(17);
        assert!(buf.is_empty());
        assert_eq!(buf.as_ptr() as usize % BUFFER_ALIGN, 0);
    }

    #[test]
    #[should_panic(expected = "validity length")]
    fn nullable_column_length_mismatch_panics() {
        NumericColumn::new_nullable(Buffer::from_slice(&[1i64, 2]), Bitmap::new_set(3));
    }

    #[test]
    fn non_null_column_has_no_bitmap() {
        let col = NumericColumn::new_non_null(Buffer::from_slice(&[1i64, 2, 3]));
        assert!(col.validity().is_none());
        assert_eq!(col.null_count(), 0);
        assert!(col.is_valid(2));
    }

    proptest! {
        /// Reference model: push/extend agree with a plain Vec built the
        /// same way, and the result stays aligned.
        #[test]
        fn matches_vec_model(chunks in prop::collection::vec(
            prop::collection::vec(any::<i64>(), 0..20), 0..20)) {
            let mut buf = Buffer::<i64>::new();
            let mut model = Vec::new();
            for chunk in &chunks {
                if chunk.len() == 1 {
                    buf.push(chunk[0]);
                } else {
                    buf.extend_from_slice(chunk);
                }
                model.extend_from_slice(if chunk.len() == 1 { &chunk[..1] } else { chunk });
                prop_assert_eq!(buf.as_ptr() as usize % BUFFER_ALIGN, 0);
            }
            prop_assert_eq!(buf.as_slice(), model.as_slice());
        }

        /// Interleaved mutation of clone pairs stays independent — the
        /// copy-on-write is never observable except through pointers.
        #[test]
        fn cow_clones_match_independent_vecs(
            initial in prop::collection::vec(any::<i64>(), 0..20),
            ops in prop::collection::vec((any::<bool>(), any::<i64>()), 0..20),
        ) {
            let mut a = Buffer::from_slice(&initial);
            let mut b = a.clone();
            let mut model_a = initial.clone();
            let mut model_b = initial;
            for (to_a, value) in ops {
                if to_a {
                    a.push(value);
                    model_a.push(value);
                } else {
                    b.push(value);
                    model_b.push(value);
                }
            }
            prop_assert_eq!(a.as_slice(), model_a.as_slice());
            prop_assert_eq!(b.as_slice(), model_b.as_slice());
        }

        /// null_count + count_set partition the rows.
        #[test]
        fn null_count_partitions_rows(bools in prop::collection::vec(any::<bool>(), 0..100)) {
            let values: Buffer<f64> = (0..bools.len()).map(|i| i as f64).collect();
            let col = NumericColumn::new_nullable(values, Bitmap::from_bools(bools.iter().copied()));
            let valid = (0..col.len()).filter(|&i| col.is_valid(i)).count();
            prop_assert_eq!(valid + col.null_count(), col.len());
        }
    }
}
