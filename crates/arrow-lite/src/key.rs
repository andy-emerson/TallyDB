//! Key columns: `u32` dictionary codes per row plus the interning table of
//! distinct values.
//!
//! A key is a label — a symbol, a sensor id, a category — used for
//! filtering, grouping, and joining, never arithmetic. Each distinct value
//! is interned once into a [`Dictionary`] and every row stores only the
//! `u32` code (u32 only — no u64 variant, per issue #2). The dictionary is
//! the one variable-width structure in the system, and it is *reference
//! data, not row data*: sized by distinct values, touched at intern time
//! and once-per-distinct-value predicate evaluation, never on the per-row
//! scan/compute path.
//!
//! The dictionary stores its values directly in Arrow's Utf8 layout —
//! `i32` offsets plus a byte buffer — so C Data Interface export (issue
//! #14) hands these buffers over as the dictionary values array without
//! rebuilding them.
//!
//! Dictionary *scope* (per-column, per-table, or per-database) is an open
//! decision (issue #6). Nothing here entrenches it: [`Dictionary`] is a
//! standalone value type, and who owns one is the caller's business.

use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use std::collections::HashMap;

/// An interning table of distinct string values, in Arrow Utf8 layout.
///
/// Codes are assigned densely in first-seen order: the first distinct
/// value gets code 0, the next code 1, and so on. Once assigned, a code
/// never changes — later interning only appends.
///
/// ```
/// use arrow_lite::Dictionary;
///
/// let mut dict = Dictionary::new();
/// let a = dict.intern("AAPL");
/// let b = dict.intern("MSFT");
/// assert_eq!((a, b), (0, 1));
/// assert_eq!(dict.intern("AAPL"), a); // idempotent
/// assert_eq!(dict.value(b), "MSFT");
/// assert_eq!(dict.len(), 2);
/// ```
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Dictionary {
    /// Arrow Utf8 offsets: `len() + 1` entries, monotonically
    /// non-decreasing, starting at 0; value `code` spans
    /// `bytes[offsets[code] .. offsets[code + 1]]`.
    offsets: Vec<i32>,
    /// Concatenated UTF-8 bytes of all distinct values.
    bytes: Vec<u8>,
    /// Value → code lookup for interning. Duplicates the value bytes;
    /// acceptable for reference data sized by distinct values.
    index: HashMap<String, u32>,
}

impl Dictionary {
    /// An empty dictionary.
    pub fn new() -> Self {
        Dictionary {
            offsets: vec![0],
            bytes: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Returns the code for `value`, interning it if unseen.
    ///
    /// # Panics
    /// If distinct values would exceed `u32` codes, or total value bytes
    /// would exceed the `i32` offset range Arrow Utf8 allows.
    pub fn intern(&mut self, value: &str) -> u32 {
        if let Some(&code) = self.index.get(value) {
            return code;
        }
        let code = u32::try_from(self.len()).expect("dictionary full: u32 code space exhausted");
        let end = self
            .bytes
            .len()
            .checked_add(value.len())
            .and_then(|n| i32::try_from(n).ok())
            .expect("dictionary full: Utf8 offset range exhausted");
        self.bytes.extend_from_slice(value.as_bytes());
        self.offsets.push(end);
        self.index.insert(value.to_owned(), code);
        code
    }

    /// The code for `value`, if it has been interned.
    ///
    /// This is the hook string predicates use: resolve each matching
    /// distinct value to its code once, then filter rows by integer
    /// set-membership.
    pub fn code_of(&self, value: &str) -> Option<u32> {
        self.index.get(value).copied()
    }

    /// The value for `code`.
    ///
    /// # Panics
    /// If `code` was never assigned.
    pub fn value(&self, code: u32) -> &str {
        let i = code as usize;
        assert!(i < self.len(), "code {code} out of range {}", self.len());
        let (start, end) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        // Interning only ever appends whole &str values, so every offset
        // pair brackets valid UTF-8.
        std::str::from_utf8(&self.bytes[start..end]).expect("dictionary bytes are UTF-8")
    }

    /// Number of distinct values.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether no values have been interned.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The Arrow Utf8 offsets buffer (`len() + 1` entries).
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    /// The concatenated UTF-8 value bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A key column: `u32` codes per row, a validity bitmap when the schema
/// declares the column nullable, and the dictionary the codes index into.
///
/// Construction validates every code against the dictionary, so a
/// `KeyColumn` never holds a dangling code.
///
/// ```
/// use arrow_lite::KeyColumn;
///
/// let col = KeyColumn::from_values(["a", "b", "a", "a"]);
/// assert_eq!(col.codes().as_slice(), &[0, 1, 0, 0]);
/// assert_eq!(col.dictionary().len(), 2);
/// assert_eq!(col.value_at(2), Some("a"));
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct KeyColumn {
    codes: Buffer<u32>,
    validity: Option<Bitmap>,
    dictionary: Dictionary,
}

impl KeyColumn {
    /// A `NOT NULL` key column from codes and their dictionary.
    ///
    /// # Panics
    /// If any code is out of the dictionary's range.
    pub fn new_non_null(codes: Buffer<u32>, dictionary: Dictionary) -> Self {
        Self::validate_codes(&codes, &dictionary);
        KeyColumn {
            codes,
            validity: None,
            dictionary,
        }
    }

    /// A nullable key column. The code under a null slot is unspecified
    /// but must still be in range (keeps every read in bounds).
    ///
    /// # Panics
    /// If the bitmap length differs from the row count, or any code is out
    /// of the dictionary's range.
    pub fn new_nullable(codes: Buffer<u32>, validity: Bitmap, dictionary: Dictionary) -> Self {
        assert_eq!(
            codes.len(),
            validity.len(),
            "validity length {} != row count {}",
            validity.len(),
            codes.len()
        );
        Self::validate_codes(&codes, &dictionary);
        KeyColumn {
            codes,
            validity: Some(validity),
            dictionary,
        }
    }

    /// Interns each value in order, building a `NOT NULL` column and its
    /// dictionary together.
    pub fn from_values<'a, I: IntoIterator<Item = &'a str>>(values: I) -> Self {
        let mut dictionary = Dictionary::new();
        let codes = values
            .into_iter()
            .map(|v| dictionary.intern(v))
            .collect::<Buffer<u32>>();
        KeyColumn {
            codes,
            validity: None,
            dictionary,
        }
    }

    /// Every code — including any under a null slot — must be in range.
    fn validate_codes(codes: &Buffer<u32>, dict: &Dictionary) {
        if let Some(&bad) = codes.iter().find(|&&c| (c as usize) >= dict.len()) {
            panic!("code {bad} out of dictionary range {}", dict.len());
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The per-row code buffer.
    pub fn codes(&self) -> &Buffer<u32> {
        &self.codes
    }

    /// The validity bitmap — `None` exactly when the schema said
    /// `NOT NULL`.
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// The dictionary the codes index into.
    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    /// Whether the row at `index` holds a value.
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

    /// The rendered value at `index`, or `None` for a null row.
    ///
    /// Rendering is for tests and debugging — the query pipeline works on
    /// codes and hands rendering to the application.
    ///
    /// # Panics
    /// If `index >= len`.
    pub fn value_at(&self, index: usize) -> Option<&str> {
        if self.is_valid(index) {
            Some(self.dictionary.value(self.codes[index]))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn empty_dictionary() {
        let dict = Dictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.offsets(), &[0]);
        assert_eq!(dict.bytes(), &[] as &[u8]);
        assert_eq!(dict.code_of("anything"), None);
    }

    #[test]
    fn interning_is_idempotent_and_codes_are_stable() {
        let mut dict = Dictionary::new();
        let a = dict.intern("alpha");
        let b = dict.intern("beta");
        // Re-interning returns the same code and adds nothing.
        assert_eq!(dict.intern("alpha"), a);
        assert_eq!(dict.len(), 2);
        // Interning more values leaves earlier codes and values untouched.
        let c = dict.intern("gamma");
        assert_eq!((a, b, c), (0, 1, 2));
        assert_eq!(dict.value(a), "alpha");
        assert_eq!(dict.value(b), "beta");
    }

    #[test]
    fn empty_string_is_a_distinct_value() {
        let mut dict = Dictionary::new();
        let empty = dict.intern("");
        let x = dict.intern("x");
        assert_eq!(dict.value(empty), "");
        assert_eq!(dict.value(x), "x");
        assert_eq!(dict.offsets(), &[0, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn value_of_unassigned_code_panics() {
        Dictionary::new().value(0);
    }

    #[test]
    #[should_panic(expected = "out of dictionary range")]
    fn dangling_code_rejected_at_construction() {
        let mut dict = Dictionary::new();
        dict.intern("only");
        KeyColumn::new_non_null(Buffer::from_slice(&[0, 1]), dict);
    }

    #[test]
    #[should_panic(expected = "validity length")]
    fn nullable_length_mismatch_panics() {
        let mut dict = Dictionary::new();
        dict.intern("v");
        KeyColumn::new_nullable(Buffer::from_slice(&[0, 0]), Bitmap::new_set(3), dict);
    }

    #[test]
    fn nullable_key_column_reads_nulls() {
        let mut dict = Dictionary::new();
        dict.intern("v");
        let col = KeyColumn::new_nullable(
            Buffer::from_slice(&[0, 0]),
            Bitmap::from_bools([true, false]),
            dict,
        );
        assert_eq!(col.null_count(), 1);
        assert_eq!(col.value_at(0), Some("v"));
        assert_eq!(col.value_at(1), None);
    }

    proptest! {
        /// Reference model: interning a sequence of values agrees with a
        /// HashMap-based model, and offsets stay a valid Utf8 layout.
        #[test]
        fn intern_matches_model(values in prop::collection::vec("[a-z]{0,8}", 0..100)) {
            let mut dict = Dictionary::new();
            let mut model: Vec<String> = Vec::new();
            for v in &values {
                let code = dict.intern(v);
                let expect = match model.iter().position(|m| m == v) {
                    Some(i) => i,
                    None => {
                        model.push(v.clone());
                        model.len() - 1
                    }
                };
                prop_assert_eq!(code as usize, expect);
            }
            prop_assert_eq!(dict.len(), model.len());
            for (i, v) in model.iter().enumerate() {
                prop_assert_eq!(dict.value(i as u32), v);
                prop_assert_eq!(dict.code_of(v), Some(i as u32));
            }
            // Utf8 layout invariants: monotone offsets from 0 to bytes.len().
            prop_assert_eq!(dict.offsets()[0], 0);
            prop_assert!(dict.offsets().windows(2).all(|w| w[0] <= w[1]));
            prop_assert_eq!(*dict.offsets().last().unwrap() as usize, dict.bytes().len());
        }

        /// from_values round-trips every row through codes + dictionary.
        #[test]
        fn from_values_roundtrip(values in prop::collection::vec("[a-z]{0,4}", 0..100)) {
            let col = KeyColumn::from_values(values.iter().map(String::as_str));
            prop_assert_eq!(col.len(), values.len());
            prop_assert_eq!(col.null_count(), 0);
            for (i, v) in values.iter().enumerate() {
                prop_assert_eq!(col.value_at(i), Some(v.as_str()));
            }
            // Distinct count matches the input's distinct count.
            let distinct: std::collections::HashSet<_> = values.iter().collect();
            prop_assert_eq!(col.dictionary().len(), distinct.len());
        }
    }
}
