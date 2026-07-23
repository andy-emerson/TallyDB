//! The shared bitmap: one bit per row, LSB-ordered per Arrow's validity
//! spec.
//!
//! One implementation serves three consumers: validity sidecars on nullable
//! columns (here), row selections in `query-lite` (WHERE results,
//! dictionary-LIKE matches), and likely tombstone masks in `storage-lite`.
//!
//! ## Layout
//!
//! Bit `i` lives in byte `i / 8` at bit position `i % 8`, least-significant
//! bit first — exactly Arrow's validity-bitmap layout, so the byte buffer
//! can cross the C Data Interface without translation. The struct also
//! carries a length in bits, because the byte buffer alone cannot say
//! whether trailing bits are payload or padding.
//!
//! ## Canonical form
//!
//! Padding bits past `len` in the final byte are always zero. Every
//! constructor and operation maintains this, which is what lets equality,
//! [`Bitmap::count_set`], and the binary operations work whole bytes at a
//! time without masking on every call — only [`Bitmap::not`] has to re-mask
//! its tail. The invariant is checked by tests, not by callers.

/// A fixed-length sequence of bits in Arrow's validity-bitmap layout.
///
/// ```
/// use arrow_lite::Bitmap;
///
/// let mut bm = Bitmap::new_unset(10);
/// bm.set(3, true);
/// bm.set(9, true);
/// assert_eq!(bm.count_set(), 2);
/// assert_eq!(bm.iter_set().collect::<Vec<_>>(), vec![3, 9]);
/// // LSB order: bit 3 of the first byte is 0b0000_1000.
/// assert_eq!(bm.as_bytes(), &[0b0000_1000, 0b0000_0010]);
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Bitmap {
    /// `ceil(len / 8)` bytes; padding bits past `len` are zero.
    bytes: Vec<u8>,
    /// Length in bits.
    len: usize,
}

/// Number of bytes needed to hold `len` bits.
fn byte_len(len: usize) -> usize {
    len.div_ceil(8)
}

/// Mask selecting the payload bits of the final byte, or `0xFF` when the
/// length is a whole number of bytes (including zero).
fn tail_mask(len: usize) -> u8 {
    match len % 8 {
        0 => 0xFF,
        tail => (1u8 << tail) - 1,
    }
}

impl Bitmap {
    /// A bitmap of `len` bits, all set.
    pub fn new_set(len: usize) -> Self {
        let mut bytes = vec![0xFF; byte_len(len)];
        if let Some(last) = bytes.last_mut() {
            *last &= tail_mask(len);
        }
        Bitmap { bytes, len }
    }

    /// A bitmap of `len` bits, all unset.
    pub fn new_unset(len: usize) -> Self {
        Bitmap {
            bytes: vec![0; byte_len(len)],
            len,
        }
    }

    /// Builds a bitmap from one bool per bit.
    ///
    /// ```
    /// use arrow_lite::Bitmap;
    ///
    /// let bm = Bitmap::from_bools([true, false, true]);
    /// assert_eq!(bm.len(), 3);
    /// assert_eq!(bm.as_bytes(), &[0b0000_0101]);
    /// ```
    pub fn from_bools<I: IntoIterator<Item = bool>>(bools: I) -> Self {
        let iter = bools.into_iter();
        let (lower, _) = iter.size_hint();
        let mut bytes = Vec::with_capacity(byte_len(lower));
        let mut len = 0;
        for b in iter {
            if len % 8 == 0 {
                bytes.push(0);
            }
            if b {
                *bytes.last_mut().expect("just pushed") |= 1 << (len % 8);
            }
            len += 1;
        }
        Bitmap { bytes, len }
    }

    /// Length in bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the bitmap has zero bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bit at `index`.
    ///
    /// # Panics
    /// If `index >= len`.
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.len, "bit {index} out of range {}", self.len);
        self.bytes[index / 8] & (1 << (index % 8)) != 0
    }

    /// Sets the bit at `index` to `value`.
    ///
    /// # Panics
    /// If `index >= len`.
    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < self.len, "bit {index} out of range {}", self.len);
        let mask = 1 << (index % 8);
        if value {
            self.bytes[index / 8] |= mask;
        } else {
            self.bytes[index / 8] &= !mask;
        }
    }

    /// Number of set bits.
    ///
    /// For a validity bitmap this is the column's non-null count; for a row
    /// selection it is the number of selected rows.
    pub fn count_set(&self) -> usize {
        // Padding bits are zero (canonical form), so whole bytes are safe.
        self.bytes.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Bitwise AND with an equal-length bitmap.
    ///
    /// For validity, AND is "non-null in both" — the validity of a
    /// two-input arithmetic expression. For selections, it is predicate
    /// conjunction.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn and(&self, other: &Bitmap) -> Bitmap {
        self.zip_bytes(other, |a, b| a & b)
    }

    /// Bitwise OR with an equal-length bitmap.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn or(&self, other: &Bitmap) -> Bitmap {
        self.zip_bytes(other, |a, b| a | b)
    }

    /// Bitwise NOT.
    ///
    /// ```
    /// use arrow_lite::Bitmap;
    ///
    /// let bm = Bitmap::from_bools([true, false, true]);
    /// assert_eq!(bm.not(), Bitmap::from_bools([false, true, false]));
    /// ```
    pub fn not(&self) -> Bitmap {
        let mut bytes: Vec<u8> = self.bytes.iter().map(|b| !b).collect();
        // NOT is the one operation that turns padding ones — re-mask the
        // tail to keep canonical form.
        if let Some(last) = bytes.last_mut() {
            *last &= tail_mask(self.len);
        }
        Bitmap {
            bytes,
            len: self.len,
        }
    }

    /// Iterates the indices of set bits, in increasing order.
    ///
    /// For a row selection this is the scan order: each yielded index is a
    /// selected row. Skips unset bytes a byte at a time.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.bytes
            .iter()
            .enumerate()
            .filter(|(_, &byte)| byte != 0)
            .flat_map(|(byte_index, &byte)| {
                (0..8)
                    .filter(move |bit| byte & (1 << bit) != 0)
                    .map(move |bit| byte_index * 8 + bit)
            })
    }

    /// The underlying bytes, LSB-ordered, padding bits zero — exactly the
    /// buffer Arrow's C Data Interface expects for validity.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Byte-wise binary operation on two equal-length bitmaps.
    ///
    /// Sound for AND and OR because both preserve zero padding; NOT does
    /// not go through here.
    fn zip_bytes(&self, other: &Bitmap, op: impl Fn(u8, u8) -> u8) -> Bitmap {
        assert_eq!(
            self.len, other.len,
            "bitmap length mismatch: {} vs {}",
            self.len, other.len
        );
        Bitmap {
            bytes: self
                .bytes
                .iter()
                .zip(&other.bytes)
                .map(|(&a, &b)| op(a, b))
                .collect(),
            len: self.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Lengths that straddle the byte and u64-word boundaries where
    /// off-by-one bugs live.
    const EDGE_LENGTHS: &[usize] = &[0, 1, 7, 8, 9, 63, 64, 65];

    /// The canonical-form invariant: padding bits past `len` are zero.
    fn assert_canonical(bm: &Bitmap) {
        assert_eq!(bm.as_bytes().len(), bm.len().div_ceil(8));
        if let Some(&last) = bm.as_bytes().last() {
            assert_eq!(last & !tail_mask(bm.len()), 0, "padding bits set");
        }
    }

    #[test]
    fn constructors_at_edge_lengths() {
        for &len in EDGE_LENGTHS {
            let set = Bitmap::new_set(len);
            let unset = Bitmap::new_unset(len);
            assert_eq!(set.len(), len);
            assert_eq!(unset.len(), len);
            assert_eq!(set.count_set(), len);
            assert_eq!(unset.count_set(), 0);
            assert_canonical(&set);
            assert_canonical(&unset);
            for i in 0..len {
                assert!(set.get(i));
                assert!(!unset.get(i));
            }
        }
    }

    #[test]
    fn empty_bitmap() {
        let bm = Bitmap::new_set(0);
        assert!(bm.is_empty());
        assert_eq!(bm.as_bytes(), &[] as &[u8]);
        assert_eq!(bm.iter_set().count(), 0);
        assert_eq!(bm.not(), bm);
        assert_eq!(bm.and(&bm), bm);
    }

    #[test]
    fn set_and_clear_across_byte_boundary() {
        let mut bm = Bitmap::new_unset(16);
        bm.set(7, true);
        bm.set(8, true);
        assert_eq!(bm.as_bytes(), &[0b1000_0000, 0b0000_0001]);
        bm.set(7, false);
        assert_eq!(bm.as_bytes(), &[0b0000_0000, 0b0000_0001]);
        assert_eq!(bm.iter_set().collect::<Vec<_>>(), vec![8]);
    }

    #[test]
    fn op_identities_at_edge_lengths() {
        for &len in EDGE_LENGTHS {
            let ones = Bitmap::new_set(len);
            let zeros = Bitmap::new_unset(len);
            // Identity and annihilator elements.
            assert_eq!(ones.and(&zeros), zeros);
            assert_eq!(ones.or(&zeros), ones);
            assert_eq!(zeros.not(), ones);
            assert_eq!(ones.not(), zeros);
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn get_past_len_panics() {
        Bitmap::new_set(8).get(8);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn and_length_mismatch_panics() {
        Bitmap::new_set(8).and(&Bitmap::new_set(9));
    }

    proptest! {
        /// Reference model: every operation must agree with the same
        /// operation on a plain `Vec<bool>`.
        #[test]
        fn matches_bool_vec_model(a in prop::collection::vec(any::<bool>(), 0..200),
                                  b in prop::collection::vec(any::<bool>(), 0..200)) {
            let bm = Bitmap::from_bools(a.iter().copied());
            prop_assert_eq!(bm.len(), a.len());
            assert_canonical(&bm);
            for (i, &expect) in a.iter().enumerate() {
                prop_assert_eq!(bm.get(i), expect);
            }
            prop_assert_eq!(bm.count_set(), a.iter().filter(|&&x| x).count());
            let set_indices: Vec<usize> =
                a.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i).collect();
            prop_assert_eq!(bm.iter_set().collect::<Vec<_>>(), set_indices);

            let not = bm.not();
            assert_canonical(&not);
            for (i, &expect) in a.iter().enumerate() {
                prop_assert_eq!(not.get(i), !expect);
            }
            prop_assert_eq!(not.not(), bm.clone());

            // Binary ops need equal lengths: truncate to the shorter input.
            let n = a.len().min(b.len());
            let (ta, tb) = (&a[..n], &b[..n]);
            let (ba, bb) = (Bitmap::from_bools(ta.iter().copied()),
                            Bitmap::from_bools(tb.iter().copied()));
            let and = ba.and(&bb);
            let or = ba.or(&bb);
            assert_canonical(&and);
            assert_canonical(&or);
            for i in 0..n {
                prop_assert_eq!(and.get(i), ta[i] && tb[i]);
                prop_assert_eq!(or.get(i), ta[i] || tb[i]);
            }
            // De Morgan: !(a & b) == !a | !b.
            prop_assert_eq!(and.not(), ba.not().or(&bb.not()));
        }

        /// Set/clear round-trip at arbitrary positions.
        #[test]
        fn set_get_roundtrip(len in 1usize..200, tweaks in prop::collection::vec((any::<prop::sample::Index>(), any::<bool>()), 0..50)) {
            let mut bm = Bitmap::new_unset(len);
            let mut model = vec![false; len];
            for (index, value) in tweaks {
                let i = index.index(len);
                bm.set(i, value);
                model[i] = value;
                prop_assert_eq!(bm.get(i), value);
            }
            assert_canonical(&bm);
            prop_assert_eq!(Bitmap::from_bools(model), bm);
        }
    }
}
