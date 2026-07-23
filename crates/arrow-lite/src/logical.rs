//! Logical-type annotations: how an `i64` column asks to be *seen* at
//! export.
//!
//! In-engine there are only physical types — an `i64` is an `i64`, and
//! every operator treats it as one. But many `i64` columns *mean*
//! something richer: nanoseconds since the epoch, or a fixed-point money
//! amount. A [`LogicalType`] records that meaning so C Data Interface
//! export (issue #14) can present the same buffer as `timestamp[ns]` or
//! `decimal64(scale)`, and ecosystem consumers see datetimes and decimals
//! instead of opaque integers.
//!
//! The annotation is consulted **only at export**. It never changes the
//! in-engine representation, arithmetic, or comparison — which is why it
//! lives beside the column types rather than inside them.
//!
//! ## The tag registry
//!
//! Like [`ColumnType`], logical annotations cross
//! format boundaries as integers, never strings. The registry is
//! append-only and never renumbered; tag 0 is reserved to mean "no
//! annotation":
//!
//! | tag | annotation | payload |
//! |-----|------------|---------|
//! | 0 | (reserved: none) | — |
//! | 1 | `TimestampNs` | — |
//! | 2 | `Decimal64` | scale |

use crate::column::ColumnType;

/// Maximum decimal digits a 64-bit decimal can carry; also the precision
/// exported for every `Decimal64` column.
pub const DECIMAL64_PRECISION: u8 = 18;

/// An export-time annotation over a physical `i64` column.
///
/// ```
/// use arrow_lite::{ColumnType, LogicalType};
///
/// let ts = LogicalType::TimestampNs;
/// let money = LogicalType::Decimal64 { scale: 2 };
///
/// // Registry: frozen integers, never strings.
/// assert_eq!(ts.tag(), 1);
/// assert_eq!(money.tag(), 2);
///
/// // Annotations apply to i64 columns only.
/// assert!(ts.valid_for(ColumnType::I64));
/// assert!(!ts.valid_for(ColumnType::F64));
///
/// // What export will render, in Arrow C-Data format-string terms.
/// assert_eq!(ts.c_data_format(), "tsn:");
/// assert_eq!(money.c_data_format(), "d:18,2,64");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogicalType {
    /// Nanoseconds since the Unix epoch, no timezone — exported as
    /// Arrow `timestamp[ns]`.
    TimestampNs,
    /// A fixed-point value: the stored integer times `10^-scale` —
    /// exported as Arrow `decimal64(18, scale)`.
    Decimal64 {
        /// Digits after the decimal point; at most
        /// [`DECIMAL64_PRECISION`].
        scale: u8,
    },
}

impl LogicalType {
    /// This annotation's frozen serialization tag.
    pub fn tag(&self) -> u8 {
        match self {
            LogicalType::TimestampNs => 1,
            LogicalType::Decimal64 { .. } => 2,
        }
    }

    /// Rebuilds an annotation from its stored tag and payload.
    ///
    /// The payload byte carries `Decimal64`'s scale and is ignored by
    /// payload-free annotations. Returns `None` for unknown tags, tag 0
    /// (which means "no annotation" and has no `LogicalType` value), and
    /// out-of-range payloads — deserialization must treat all three as
    /// data errors, never guesses.
    pub fn from_parts(tag: u8, payload: u8) -> Option<LogicalType> {
        match tag {
            1 => Some(LogicalType::TimestampNs),
            2 if payload <= DECIMAL64_PRECISION => Some(LogicalType::Decimal64 { scale: payload }),
            _ => None,
        }
    }

    /// Whether this annotation may sit on a column of `column_type`.
    ///
    /// All current annotations reinterpret an `i64` buffer; nothing
    /// annotates floats or keys.
    pub fn valid_for(&self, column_type: ColumnType) -> bool {
        column_type == ColumnType::I64
    }

    /// The Arrow C Data Interface format string export renders for an
    /// `i64` column carrying this annotation.
    ///
    /// # Panics
    /// If a `Decimal64` scale exceeds [`DECIMAL64_PRECISION`] (constructed
    /// by hand around `from_parts`' validation).
    pub fn c_data_format(&self) -> String {
        match *self {
            LogicalType::TimestampNs => "tsn:".to_owned(),
            LogicalType::Decimal64 { scale } => {
                assert!(
                    scale <= DECIMAL64_PRECISION,
                    "decimal scale {scale} exceeds precision {DECIMAL64_PRECISION}"
                );
                format!("d:{DECIMAL64_PRECISION},{scale},64")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_are_frozen() {
        assert_eq!(LogicalType::TimestampNs.tag(), 1);
        assert_eq!(LogicalType::Decimal64 { scale: 0 }.tag(), 2);
    }

    #[test]
    fn from_parts_round_trips_and_rejects() {
        for lt in [
            LogicalType::TimestampNs,
            LogicalType::Decimal64 { scale: 0 },
            LogicalType::Decimal64 { scale: 2 },
            LogicalType::Decimal64 {
                scale: DECIMAL64_PRECISION,
            },
        ] {
            let payload = match lt {
                LogicalType::Decimal64 { scale } => scale,
                _ => 0,
            };
            assert_eq!(LogicalType::from_parts(lt.tag(), payload), Some(lt));
        }
        assert_eq!(LogicalType::from_parts(0, 0), None); // reserved: none
        assert_eq!(LogicalType::from_parts(3, 0), None); // unregistered
        assert_eq!(LogicalType::from_parts(2, DECIMAL64_PRECISION + 1), None);
    }

    #[test]
    fn annotations_apply_to_i64_only() {
        for lt in [
            LogicalType::TimestampNs,
            LogicalType::Decimal64 { scale: 4 },
        ] {
            assert!(lt.valid_for(ColumnType::I64));
            assert!(!lt.valid_for(ColumnType::F64));
            assert!(!lt.valid_for(ColumnType::Key));
        }
    }

    #[test]
    fn format_strings_match_arrow_c_data_spec() {
        assert_eq!(LogicalType::TimestampNs.c_data_format(), "tsn:");
        assert_eq!(
            LogicalType::Decimal64 { scale: 0 }.c_data_format(),
            "d:18,0,64"
        );
        assert_eq!(
            LogicalType::Decimal64 { scale: 18 }.c_data_format(),
            "d:18,18,64"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds precision")]
    fn oversized_scale_panics_at_format_time() {
        LogicalType::Decimal64 { scale: 19 }.c_data_format();
    }
}
