//! The public column type — numeric or key, nothing else — and zero-copy
//! views over it.
//!
//! [`Column`] has exactly two variants. That is the numeric-or-key
//! invariant enforced by construction: code holding a `Column` can be
//! handed a number or a key, and the compiler makes any third case
//! unrepresentable. (Naming per issue #7, decided as numeric-or-key —
//! final since the 2026-07-23 ruling; a key is a label, not a primary
//! key.)
//!
//! ## The type-tag registry
//!
//! [`ColumnType`] is the serialization tag for a column's physical type.
//! Tags cross format boundaries **as these integers, never as strings**,
//! so a future rename of the Rust identifiers cannot corrupt stored data.
//! The registry is append-only and never renumbered:
//!
//! | tag | type |
//! |-----|------|
//! | 0 | `f64` numeric |
//! | 1 | `i64` numeric |
//! | 2 | key (`u32` dictionary codes) |
//!
//! `f32` (issue #3) will take tag 3 additively.
//!
//! ## Views
//!
//! A view is an offset + length over a column — Arrow-native slicing. The
//! window `rows[o .. o + n]` of an `f64` column is `&values[o .. o + n]`:
//! the same bytes the column owns, no copy. This is what lets window
//! functions feed BLAS/LAPACK with pointer arithmetic instead of a copy
//! per window.

use crate::bitmap::Bitmap;
use crate::buffer::{Element, NumericColumn};
use crate::key::{Dictionary, KeyColumn};

/// The physical column types, as their frozen serialization tags.
///
/// ```
/// use arrow_lite::ColumnType;
///
/// // The registry: these numbers are the format, frozen forever.
/// assert_eq!(ColumnType::F64 as u8, 0);
/// assert_eq!(ColumnType::I64 as u8, 1);
/// assert_eq!(ColumnType::Key as u8, 2);
/// ```
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColumnType {
    /// `f64` values.
    F64 = 0,
    /// `i64` values (also the physical type under `Timestamp`/`Decimal64`
    /// logical annotations).
    I64 = 1,
    /// `u32` dictionary codes plus an interning table.
    Key = 2,
}

impl ColumnType {
    /// The tag for a stored integer, if registered.
    ///
    /// The inverse of `as u8` — deserialization goes through here so an
    /// unknown tag is an explicit `None`, never a panic or a silent
    /// misread.
    pub fn from_tag(tag: u8) -> Option<ColumnType> {
        match tag {
            0 => Some(ColumnType::F64),
            1 => Some(ColumnType::I64),
            2 => Some(ColumnType::Key),
            _ => None,
        }
    }
}

/// The numeric half of a column: one value buffer, `f64` or `i64`.
///
/// An enum rather than a generic so `Column` stays a concrete type; the
/// variant is the numeric subtype tag made structural. `F32` (issue #3)
/// arrives as a new variant — additive, not a migration.
#[derive(Clone, PartialEq, Debug)]
pub enum NumericData {
    /// An `f64` column.
    F64(NumericColumn<f64>),
    /// An `i64` column.
    I64(NumericColumn<i64>),
}

impl NumericData {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            NumericData::F64(c) => c.len(),
            NumericData::I64(c) => c.len(),
        }
    }

    /// Whether the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of null rows.
    pub fn null_count(&self) -> usize {
        match self {
            NumericData::F64(c) => c.null_count(),
            NumericData::I64(c) => c.null_count(),
        }
    }

    /// The validity bitmap, if the column is nullable.
    pub fn validity(&self) -> Option<&Bitmap> {
        match self {
            NumericData::F64(c) => c.validity(),
            NumericData::I64(c) => c.validity(),
        }
    }
}

/// A column: numeric or key. There is no third variant, by design.
///
/// ```
/// use arrow_lite::{Buffer, Column, ColumnType, KeyColumn, NumericColumn, NumericData};
///
/// let price = Column::Numeric(NumericData::F64(NumericColumn::new_non_null(
///     Buffer::from_slice(&[101.5, 102.0]),
/// )));
/// let sym = Column::Key(KeyColumn::from_values(["AAPL", "MSFT"]));
///
/// // The invariant by construction: every column is one of exactly two
/// // shapes, and matching handles both.
/// for col in [&price, &sym] {
///     match col {
///         Column::Numeric(n) => assert_eq!(n.len(), 2),
///         Column::Key(k) => assert_eq!(k.len(), 2),
///     }
/// }
/// assert_eq!(price.column_type(), ColumnType::F64);
/// assert_eq!(sym.column_type(), ColumnType::Key);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub enum Column {
    /// A value column: `f64` or `i64`, used in arithmetic, aggregation,
    /// windows.
    Numeric(NumericData),
    /// A label column: dictionary codes, used for filtering, grouping,
    /// joining — never arithmetic.
    Key(KeyColumn),
}

impl Column {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Column::Numeric(n) => n.len(),
            Column::Key(k) => k.len(),
        }
    }

    /// Whether the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of null rows.
    pub fn null_count(&self) -> usize {
        match self {
            Column::Numeric(n) => n.null_count(),
            Column::Key(k) => k.null_count(),
        }
    }

    /// This column's frozen serialization tag.
    pub fn column_type(&self) -> ColumnType {
        match self {
            Column::Numeric(NumericData::F64(_)) => ColumnType::F64,
            Column::Numeric(NumericData::I64(_)) => ColumnType::I64,
            Column::Key(_) => ColumnType::Key,
        }
    }
}

/// Panics unless `offset + len` rows fit inside `outer` rows.
fn check_range(offset: usize, len: usize, outer: usize) {
    let end = offset.checked_add(len).expect("view range overflow");
    assert!(end <= outer, "view [{offset}, {end}) out of range {outer}");
}

/// A zero-copy window over a [`NumericColumn`]: an offset and a length,
/// no data.
///
/// ```
/// use arrow_lite::{Buffer, NumericColumn};
///
/// let col = NumericColumn::new_non_null(Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]));
/// let view = col.view(1, 3);
/// assert_eq!(view.values(), &[2.0, 3.0, 4.0]);
/// // Zero-copy: the view's data IS the column's buffer, offset by one.
/// assert!(std::ptr::eq(view.values().as_ptr(), &col.values()[1]));
/// // Views nest, with offsets relative to the view.
/// assert_eq!(view.slice(1, 2).values(), &[3.0, 4.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct NumericView<'a, T: Element> {
    column: &'a NumericColumn<T>,
    /// Absolute row offset into the column.
    offset: usize,
    len: usize,
}

impl<T: Element> NumericColumn<T> {
    /// A zero-copy view of `len` rows starting at `offset`.
    ///
    /// # Panics
    /// If `offset + len` exceeds the row count.
    pub fn view(&self, offset: usize, len: usize) -> NumericView<'_, T> {
        check_range(offset, len, self.len());
        NumericView {
            column: self,
            offset,
            len,
        }
    }
}

impl<'a, T: Element> NumericView<'a, T> {
    /// Number of rows in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view covers zero rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The viewed values — a plain slice of the column's own buffer.
    pub fn values(&self) -> &'a [T] {
        &self.column.values().as_slice()[self.offset..self.offset + self.len]
    }

    /// Whether view row `index` holds a value.
    ///
    /// # Panics
    /// If `index >= len`.
    pub fn is_valid(&self, index: usize) -> bool {
        check_range(index, 1, self.len);
        self.column.is_valid(self.offset + index)
    }

    /// Number of null rows inside the view (counted; a view stores no
    /// bitmap of its own).
    pub fn null_count(&self) -> usize {
        match self.column.validity() {
            Some(_) => (0..self.len).filter(|&i| !self.is_valid(i)).count(),
            None => 0,
        }
    }

    /// A sub-view; `offset` is relative to this view.
    ///
    /// # Panics
    /// If `offset + len` exceeds this view's length.
    pub fn slice(&self, offset: usize, len: usize) -> NumericView<'a, T> {
        check_range(offset, len, self.len);
        NumericView {
            column: self.column,
            offset: self.offset + offset,
            len,
        }
    }
}

/// A zero-copy window over a [`KeyColumn`].
///
/// ```
/// use arrow_lite::KeyColumn;
///
/// let col = KeyColumn::from_values(["a", "b", "c", "b"]);
/// let view = col.view(1, 2);
/// assert_eq!(view.codes(), &[1, 2]);
/// assert_eq!(view.slice(1, 1).codes(), &[2]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct KeyView<'a> {
    column: &'a KeyColumn,
    /// Absolute row offset into the column.
    offset: usize,
    len: usize,
}

impl KeyColumn {
    /// A zero-copy view of `len` rows starting at `offset`.
    ///
    /// # Panics
    /// If `offset + len` exceeds the row count.
    pub fn view(&self, offset: usize, len: usize) -> KeyView<'_> {
        check_range(offset, len, self.len());
        KeyView {
            column: self,
            offset,
            len,
        }
    }
}

impl<'a> KeyView<'a> {
    /// Number of rows in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view covers zero rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The viewed codes — a plain slice of the column's own buffer.
    pub fn codes(&self) -> &'a [u32] {
        &self.column.codes().as_slice()[self.offset..self.offset + self.len]
    }

    /// The dictionary the codes index into (views share the column's).
    pub fn dictionary(&self) -> &'a Dictionary {
        self.column.dictionary()
    }

    /// Whether view row `index` holds a value.
    ///
    /// # Panics
    /// If `index >= len`.
    pub fn is_valid(&self, index: usize) -> bool {
        check_range(index, 1, self.len);
        self.column.is_valid(self.offset + index)
    }

    /// A sub-view; `offset` is relative to this view.
    ///
    /// # Panics
    /// If `offset + len` exceeds this view's length.
    pub fn slice(&self, offset: usize, len: usize) -> KeyView<'a> {
        check_range(offset, len, self.len);
        KeyView {
            column: self.column,
            offset: self.offset + offset,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use proptest::prelude::*;

    fn f64_col(n: usize) -> NumericColumn<f64> {
        NumericColumn::new_non_null((0..n).map(|i| i as f64).collect())
    }

    #[test]
    fn type_tags_are_frozen() {
        // The registry regression test: renumbering these is a format
        // break and must fail loudly here.
        assert_eq!(ColumnType::F64 as u8, 0);
        assert_eq!(ColumnType::I64 as u8, 1);
        assert_eq!(ColumnType::Key as u8, 2);
        for t in [ColumnType::F64, ColumnType::I64, ColumnType::Key] {
            assert_eq!(ColumnType::from_tag(t as u8), Some(t));
        }
        assert_eq!(ColumnType::from_tag(3), None);
        assert_eq!(ColumnType::from_tag(u8::MAX), None);
    }

    #[test]
    fn column_type_matches_variant() {
        let f = Column::Numeric(NumericData::F64(f64_col(1)));
        let i = Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
            Buffer::from_slice(&[1i64]),
        )));
        let k = Column::Key(KeyColumn::from_values(["x"]));
        assert_eq!(f.column_type(), ColumnType::F64);
        assert_eq!(i.column_type(), ColumnType::I64);
        assert_eq!(k.column_type(), ColumnType::Key);
        for c in [&f, &i, &k] {
            assert_eq!(c.len(), 1);
            assert_eq!(c.null_count(), 0);
        }
    }

    #[test]
    fn view_is_zero_copy() {
        let col = f64_col(100);
        let view = col.view(10, 20);
        assert_eq!(view.len(), 20);
        // Pointer identity: the view's first element is the column's
        // element 10 — same allocation, no copy.
        assert!(std::ptr::eq(view.values().as_ptr(), &col.values()[10]));
    }

    #[test]
    fn empty_and_full_views() {
        let col = f64_col(5);
        assert!(col.view(0, 0).is_empty());
        assert!(col.view(5, 0).is_empty()); // offset == len is a valid empty view
        assert_eq!(col.view(0, 5).values(), col.values().as_slice());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn view_past_end_panics() {
        f64_col(5).view(3, 3);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn nested_slice_past_end_panics() {
        let col = f64_col(10);
        col.view(2, 4).slice(2, 3);
    }

    #[test]
    fn view_sees_validity_through_offset() {
        let col = NumericColumn::new_nullable(
            (0..8).map(|i| i as f64).collect(),
            Bitmap::from_bools([true, false, true, true, false, true, true, true]),
        );
        let view = col.view(1, 4); // rows 1..5: null, valid, valid, null
        assert!(!view.is_valid(0));
        assert!(view.is_valid(1));
        assert_eq!(view.null_count(), 2);
        assert_eq!(view.slice(1, 2).null_count(), 0);
    }

    #[test]
    fn key_view_shares_dictionary() {
        let col = KeyColumn::from_values(["a", "b", "c", "b", "a"]);
        let view = col.view(2, 3);
        assert_eq!(view.codes(), &[2, 1, 0]);
        assert!(std::ptr::eq(view.dictionary(), col.dictionary()));
        assert!(std::ptr::eq(view.codes().as_ptr(), &col.codes()[2]));
    }

    proptest! {
        /// Nested slicing composes like slicing the underlying data
        /// directly: view(a, b).slice(c, d) == data[a + c .. a + c + d].
        #[test]
        fn nested_views_match_direct_slices(
            n in 0usize..64,
            picks in prop::collection::vec((any::<prop::sample::Index>(), any::<prop::sample::Index>()), 1..8),
        ) {
            let col = NumericColumn::new_non_null((0..n as i64).collect());
            let data: Vec<i64> = (0..n as i64).collect();
            // Walk a random chain of nested slices, tracking the absolute
            // range it should correspond to.
            let (mut start, mut len) = (0, n);
            let mut view = col.view(0, n);
            for (o, l) in picks {
                let off = o.index(len + 1); // 0..=len
                let sub = l.index(len - off + 1); // 0..=len-off
                view = view.slice(off, sub);
                start += off;
                len = sub;
                prop_assert_eq!(view.values(), &data[start..start + len]);
            }
        }
    }
}
