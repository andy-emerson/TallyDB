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

/// A growable, contiguous, 64-byte-aligned buffer of one [`Element`] type.
///
/// ```
/// use arrow_lite::buffer::{Buffer, BUFFER_ALIGN};
///
/// let mut buf = Buffer::<f64>::new();
/// buf.extend_from_slice(&[1.0, 2.0, 3.0]);
/// buf.push(4.0);
/// assert_eq!(&buf[..], &[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(buf.as_ptr() as usize % BUFFER_ALIGN, 0);
/// ```
pub struct Buffer<T: Element> {
    /// 64-byte aligned; a placeholder address (no allocation) when
    /// `cap == 0`.
    ptr: NonNull<T>,
    /// Elements in use.
    len: usize,
    /// Elements allocated.
    cap: usize,
    _owns: PhantomData<T>,
}

// SAFETY: Buffer owns its allocation outright and Element requires
// Send + Sync, so moving or sharing the handle across threads moves or
// shares plain numeric data.
unsafe impl<T: Element> Send for Buffer<T> {}
unsafe impl<T: Element> Sync for Buffer<T> {}

impl<T: Element> Buffer<T> {
    /// The 64-aligned, non-null placeholder pointer used while unallocated.
    fn placeholder() -> NonNull<T> {
        // SAFETY: BUFFER_ALIGN is non-zero, so the pointer is non-null; it
        // carries no provenance and is never read or written — `cap == 0`
        // guards every access.
        unsafe { NonNull::new_unchecked(std::ptr::without_provenance_mut(BUFFER_ALIGN)) }
    }

    /// An empty buffer. Does not allocate.
    pub fn new() -> Self {
        Buffer {
            ptr: Self::placeholder(),
            len: 0,
            cap: 0,
            _owns: PhantomData,
        }
    }

    /// An empty buffer with room for `cap` elements.
    pub fn with_capacity(cap: usize) -> Self {
        let mut buf = Self::new();
        if cap > 0 {
            buf.grow_to(cap);
        }
        buf
    }

    /// A buffer holding a copy of `values`.
    pub fn from_slice(values: &[T]) -> Self {
        let mut buf = Self::with_capacity(values.len());
        buf.extend_from_slice(values);
        buf
    }

    /// The allocation layout for `cap` elements.
    ///
    /// # Panics
    /// If the byte size overflows (unreachable for real row counts).
    fn layout(cap: usize) -> Layout {
        let bytes = cap.checked_mul(size_of::<T>()).expect("capacity overflow");
        Layout::from_size_align(bytes, BUFFER_ALIGN).expect("layout overflow")
    }

    /// Reallocates to exactly `new_cap` elements, copying `len` elements.
    fn grow_to(&mut self, new_cap: usize) {
        debug_assert!(new_cap > self.cap);
        let new_layout = Self::layout(new_cap);
        // SAFETY: new_layout has non-zero size (new_cap > cap >= 0 and T is
        // never zero-sized for the sealed element types).
        let raw = unsafe { alloc(new_layout) };
        let Some(new_ptr) = NonNull::new(raw.cast::<T>()) else {
            handle_alloc_error(new_layout);
        };
        if self.cap > 0 {
            // SAFETY: both regions are valid for `len` elements and
            // distinct (fresh allocation); old pointer came from `alloc`
            // with `layout(self.cap)`.
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
                dealloc(self.ptr.as_ptr().cast(), Self::layout(self.cap));
            }
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
    }

    /// Ensures capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).expect("capacity overflow");
        if needed > self.cap {
            // Geometric growth keeps amortized push O(1).
            let new_cap = needed.max(self.cap * 2).max(8);
            self.grow_to(new_cap);
        }
    }

    /// Appends one element.
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        // SAFETY: reserve guarantees len < cap, so the write is in bounds
        // of the allocation.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    /// Appends a slice of elements.
    pub fn extend_from_slice(&mut self, values: &[T]) {
        self.reserve(values.len());
        // SAFETY: reserve guarantees room for `values.len()` elements past
        // `len`; source and destination cannot overlap because we own the
        // destination exclusively through &mut self.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.ptr.as_ptr().add(self.len),
                values.len(),
            );
        }
        self.len += values.len();
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
        // slice).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The elements as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as for `as_slice`, plus &mut self gives exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The base pointer — 64-byte aligned, non-null even when empty.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
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
        self.reserve(count);
        // SAFETY: reserve guarantees room past len; caller guarantees src
        // readable; regions cannot overlap (we own dst exclusively).
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.cast::<u8>(),
                self.ptr.as_ptr().add(self.len).cast::<u8>(),
                count * size_of::<T>(),
            );
        }
        self.len += count;
    }
}

impl<T: Element> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            // SAFETY: pointer came from `alloc` with this exact layout;
            // elements are Copy, so no per-element drop is needed.
            unsafe { dealloc(self.ptr.as_ptr().cast(), Self::layout(self.cap)) };
        }
    }
}

impl<T: Element> Default for Buffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Element> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
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
