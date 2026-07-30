//! The vectorized vocabulary (M4, the Lua trial's option A): owned
//! numeric vectors, arithmetic operators over whole columns, and
//! rolling combinators — the ufunc half of the NumPy model the column
//! kernels were missing. A composed kernel enters the interpreter once
//! and loops in native code:
//!
//! ```text
//! .luascalar rel(a, b) return (a - b) / b
//! .luascalar rdot(a, b) return rolling_dot(a, b, 64)
//! ```
//!
//! ## The vector value
//!
//! A **vector** is an owned `f64` column living entirely inside a Lua
//! userdata block (header, values, then one validity byte per element),
//! so the interpreter's GC owns the memory and no destructor runs at
//! the C boundary (the module discipline: `lua_error` may `longjmp`
//! only over `Copy` locals). Unlike input views, vectors carry no
//! generation stamp — they own their data and remain valid for the
//! state's life; only *views* are poisoned when their call ends.
//!
//! ## The operators
//!
//! `+ - * /` and unary `-` work elementwise over any mix of input
//! views (`f64` or `i64` — integers convert exact-or-loud, F3), other
//! vectors, and plain numbers (which broadcast). Division is IEEE,
//! exactly like the native scalar-expression slot, so a promoted
//! kernel computes bit-identical answers. NULL propagates per element
//! (three-valued logic, F1): a NULL element in either operand makes
//! that result element NULL, and an operand that *is* the `NULL`
//! sentinel makes every element NULL. Sized operands must agree on
//! length — loudly.
//!
//! ## The rolling combinators
//!
//! `rolling_sum(x, w)`, `rolling_mean(x, w)`, `rolling_dot(x, y, w)`,
//! and — since M5.0 — `rolling_var(x, w)` / `rolling_std(x, w)` (names
//! Lua-side only — SQL spells these with window frames, so nothing new
//! enters the SQL surface). Frames are trailing, exactly the SQL shape
//! `ROWS BETWEEN w-1 PRECEDING AND CURRENT ROW`: row `i` covers the
//! last `min(i+1, w)` elements. Inputs must be dense (non-NULL) — the
//! same loud rule the curated ops enforce. The sums run
//! Neumaier-compensated and re-anchor with a fresh window recompute
//! every `w` steps, the engine's incremental-window discipline — never
//! the plain cumsum idiom, which is the catastrophic-cancellation form
//! the engine rejected. The dispersion pair additionally accumulates
//! about a **shift taken from the data**, for the same reason
//! `engine`'s pair statistics do.
//!
//! ## The series transforms (M5.0)
//!
//! `lag(x, k)`, `diff(x)`, `log_returns(x)`, `ewma(x, alpha)`. These
//! read a whole column and write a whole column — a different shape
//! from the registry statistics (`var_pop`, `covar_pop`, …), which
//! reduce one *frame* to one number. Both shapes are reachable from a
//! kernel; which one a name has is fixed by what it produces.
//!
//! None of the four bears a standard SQL name, which is why they live
//! here and not in the SQL registry (#77.1 = a, ruled 2026-07-29:
//! SQL exposes only operations with standard names). SQL's `LAG`/`LEAD`
//! are a *different* thing arriving separately — window functions over
//! a frame, not column transforms.
//!
//! The head rows a transform cannot define come back **NULL**, never
//! filled with a stand-in: `lag(x, k)` and the differencing pair have
//! no row to reference before the column's start. `ewma` defines every
//! row from `y[0] = x[0]`, so it has no NULL head; its recurrence is
//! the unadjusted `y[i] = α·x[i] + (1−α)·y[i−1]`, the form a live feed
//! carries in O(1) state.
//!
//! Compositions belong in the prelude, not here: simple returns are
//! `diff(px) / lag(px, 1)`, expanding aggregates are a rolling
//! combinator at `w = #x`, and a z-score is
//! `(x - rolling_mean(x, w)) / rolling_std(x, w)`. Each is one
//! readable line over natives, so nothing per-element runs in the
//! interpreter.

use crate::ffi;
use crate::values::{
    self, int_as_f64_exact, is_valid, raise, InputPayload, META_INPUT, TAG_F64, TAG_I64, TAG_KEY,
};
use arrow_lite::Bitmap;
use std::ffi::{c_int, CStr};

pub(crate) const META_VECTOR: &CStr = c"tallydb.vector";

/// The header at the start of a vector userdata block; `len` values
/// (`f64`) follow it, then `len` validity bytes (1 = valid).
#[repr(C)]
struct VectorHeader {
    len: usize,
}

const fn vector_size(len: usize) -> usize {
    std::mem::size_of::<VectorHeader>() + len * std::mem::size_of::<f64>() + len
}

/// A borrowed, `Copy` description of a vector userdata's parts.
#[derive(Clone, Copy)]
struct VectorParts {
    values: *mut f64,
    validity: *mut u8,
    len: usize,
}

unsafe fn parts(header: *mut VectorHeader) -> VectorParts {
    unsafe {
        let len = (*header).len;
        let values = header.add(1).cast::<f64>();
        VectorParts {
            values,
            validity: values.add(len).cast::<u8>(),
            len,
        }
    }
}

/// Allocates a vector of `len` elements, pushes it on the stack, and
/// returns its parts. Elements start as 0.0/valid; callers overwrite.
///
/// # Safety
/// `raw` must be a state [`install`] has prepared. Lua may raise on
/// allocation failure, so callers hold only `Copy` locals.
unsafe fn push_vector(raw: *mut ffi::lua_State, len: usize) -> VectorParts {
    unsafe {
        let header = ffi::lua_newuserdatauv(raw, vector_size(len), 0).cast::<VectorHeader>();
        (*header).len = len;
        let out = parts(header);
        std::ptr::write_bytes(out.values, 0, len); // count is in f64 units
        std::ptr::write_bytes(out.validity, 1, len);
        ffi::luaL_setmetatable(raw, META_VECTOR.as_ptr());
        out
    }
}

/// One operand of a vectorized operation, resolved from a stack slot.
#[derive(Clone, Copy)]
enum Operand {
    /// An `f64` input view (generation-checked at resolution).
    F64View {
        values: *const f64,
        validity: *const Bitmap,
        len: usize,
    },
    /// An `i64` input view; elements convert exact-or-loud (F3).
    I64View {
        values: *const i64,
        validity: *const Bitmap,
        len: usize,
    },
    /// An owned vector.
    Vector(VectorParts),
    /// A plain number, broadcast to every element.
    Number(f64),
    /// The NULL sentinel: every result element is NULL.
    Null,
}

impl Operand {
    fn len(self) -> Option<usize> {
        match self {
            Operand::F64View { len, .. }
            | Operand::I64View { len, .. }
            | Operand::Vector(VectorParts { len, .. }) => Some(len),
            Operand::Number(_) | Operand::Null => None,
        }
    }
}

/// Resolves the value at `idx` as an operand, or raises. Key views are
/// refused (codes are identities, not quantities); integer scalars
/// convert exact-or-loud like integer elements.
unsafe fn operand(state: *mut ffi::lua_State, idx: c_int) -> Result<Operand, ()> {
    unsafe {
        if ffi::lua_type(state, idx) == ffi::LUA_TNUMBER {
            if ffi::lua_isinteger(state, idx) == 1 {
                let mut ok = 0;
                let integer = ffi::lua_tointegerx(state, idx, &mut ok);
                return match int_as_f64_exact(integer) {
                    Some(value) => Ok(Operand::Number(value)),
                    None => {
                        raise(state, c"integer operand does not fit f64 exactly");
                        Err(())
                    }
                };
            }
            let mut ok = 0;
            return Ok(Operand::Number(ffi::lua_tonumberx(state, idx, &mut ok)));
        }
        if values::is_null_sentinel(state, idx) {
            return Ok(Operand::Null);
        }
        let vector = ffi::luaL_testudata(state, idx, META_VECTOR.as_ptr());
        if !vector.is_null() {
            return Ok(Operand::Vector(parts(vector.cast::<VectorHeader>())));
        }
        let payload = ffi::luaL_testudata(state, idx, META_INPUT.as_ptr()).cast::<InputPayload>();
        if !payload.is_null() {
            let view = *payload;
            if *view.generation != view.born {
                raise(state, c"view used outside its call");
                return Err(());
            }
            return match view.tag {
                TAG_F64 => Ok(Operand::F64View {
                    values: view.data.cast::<f64>(),
                    validity: view.validity,
                    len: view.len,
                }),
                TAG_I64 => Ok(Operand::I64View {
                    values: view.data.cast::<i64>(),
                    validity: view.validity,
                    len: view.len,
                }),
                _ => {
                    raise(state, c"keys are not arithmetic (codes are identities)");
                    Err(())
                }
            };
        }
        raise(
            state,
            c"vector arithmetic takes views, vectors, and numbers",
        );
        Err(())
    }
}

/// The element at `offset` of `operand`: `Ok(None)` for a NULL element,
/// `Err` (raised) for an inexact `i64`.
unsafe fn element(
    state: *mut ffi::lua_State,
    operand: Operand,
    offset: usize,
) -> Result<Option<f64>, ()> {
    unsafe {
        match operand {
            Operand::Number(value) => Ok(Some(value)),
            Operand::Null => Ok(None),
            Operand::F64View {
                values, validity, ..
            } => {
                if is_valid(validity, offset) {
                    Ok(Some(*values.add(offset)))
                } else {
                    Ok(None)
                }
            }
            Operand::I64View {
                values, validity, ..
            } => {
                if !is_valid(validity, offset) {
                    return Ok(None);
                }
                match int_as_f64_exact(*values.add(offset)) {
                    Some(value) => Ok(Some(value)),
                    None => {
                        raise(state, c"i64 element does not fit f64 exactly");
                        Err(())
                    }
                }
            }
            Operand::Vector(vector) => {
                if *vector.validity.add(offset) != 0 {
                    Ok(Some(*vector.values.add(offset)))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// The common length of two operands: sized operands must agree, and
/// at least one side must be sized (two scalars never dispatch here —
/// plain Lua arithmetic handles them).
unsafe fn common_length(
    state: *mut ffi::lua_State,
    lhs: Operand,
    rhs: Operand,
) -> Result<usize, ()> {
    match (lhs.len(), rhs.len()) {
        (Some(a), Some(b)) if a == b => Ok(a),
        (Some(_), Some(_)) => {
            unsafe { raise(state, c"vector operands have different lengths") };
            Err(())
        }
        (Some(len), None) | (None, Some(len)) => Ok(len),
        (None, None) => {
            unsafe { raise(state, c"vector operator on two scalars") };
            Err(())
        }
    }
}

/// A dense `f64` source: a raw slice or a broadcast scalar. The fast
/// paths dispatch on this — no per-element validity or exactness
/// machinery, so the loops carry no raise points and auto-vectorize.
#[derive(Clone, Copy)]
enum DenseF64 {
    Slice(*const f64),
    Scalar(f64),
}

impl DenseF64 {
    /// # Safety
    /// For `Slice`, `offset` must be in bounds of the source.
    unsafe fn at(self, offset: usize) -> f64 {
        match self {
            DenseF64::Slice(values) => unsafe { *values.add(offset) },
            DenseF64::Scalar(value) => value,
        }
    }
}

/// Resolves an operand as a dense `f64` source if it is one: a plain
/// number, an `f64` view with no NULLs, or an all-valid vector. `i64`
/// views never qualify — their per-element exactness check can raise,
/// which the fast paths must not.
unsafe fn dense_f64(operand: Operand) -> Option<DenseF64> {
    unsafe {
        match operand {
            Operand::Number(value) => Some(DenseF64::Scalar(value)),
            Operand::F64View {
                values,
                validity,
                len,
            } => (validity.is_null() || (*validity).count_set() == len)
                .then_some(DenseF64::Slice(values)),
            Operand::Vector(vector) => std::slice::from_raw_parts(vector.validity, vector.len)
                .iter()
                .all(|&valid| valid != 0)
                .then_some(DenseF64::Slice(vector.values)),
            Operand::I64View { .. } | Operand::Null => None,
        }
    }
}

const OP_ADD: i64 = 1;
const OP_SUB: i64 = 2;
const OP_MUL: i64 = 3;
const OP_DIV: i64 = 4;

/// The shared binary metamethod: op selected by upvalue. Division is
/// IEEE — exactly the native scalar-expression semantics.
unsafe extern "C" fn vector_binary(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let op = ffi::lua_tointegerx(state, ffi::lua_upvalueindex(1), std::ptr::null_mut());
        let Ok(lhs) = operand(state, 1) else { return 0 };
        let Ok(rhs) = operand(state, 2) else { return 0 };
        let Ok(len) = common_length(state, lhs, rhs) else {
            return 0;
        };
        // The dense fast path: both operands dense f64 — one tight,
        // raise-free loop per operator.
        if let (Some(a), Some(b)) = (dense_f64(lhs), dense_f64(rhs)) {
            let out = push_vector(state, len);
            match op {
                OP_ADD => (0..len).for_each(|i| *out.values.add(i) = a.at(i) + b.at(i)),
                OP_SUB => (0..len).for_each(|i| *out.values.add(i) = a.at(i) - b.at(i)),
                OP_MUL => (0..len).for_each(|i| *out.values.add(i) = a.at(i) * b.at(i)),
                _ => (0..len).for_each(|i| *out.values.add(i) = a.at(i) / b.at(i)),
            }
            return 1;
        }
        let out = push_vector(state, len);
        for offset in 0..len {
            let Ok(a) = element(state, lhs, offset) else {
                return 0;
            };
            let Ok(b) = element(state, rhs, offset) else {
                return 0;
            };
            match (a, b) {
                (Some(a), Some(b)) => {
                    *out.values.add(offset) = match op {
                        OP_ADD => a + b,
                        OP_SUB => a - b,
                        OP_MUL => a * b,
                        _ => a / b,
                    };
                }
                _ => *out.validity.add(offset) = 0,
            }
        }
        1
    }
}

/// Unary minus.
unsafe extern "C" fn vector_negate(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(source) = operand(state, 1) else {
            return 0;
        };
        let Some(len) = source.len() else {
            return raise(state, c"vector negation of a scalar");
        };
        let out = push_vector(state, len);
        for offset in 0..len {
            let Ok(value) = element(state, source, offset) else {
                return 0;
            };
            match value {
                Some(value) => *out.values.add(offset) = -value,
                None => *out.validity.add(offset) = 0,
            }
        }
        1
    }
}

/// `v[i]` — element read, exactly like an input view's.
unsafe extern "C" fn vector_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let header = ffi::luaL_testudata(state, 1, META_VECTOR.as_ptr()).cast::<VectorHeader>();
        if header.is_null() {
            return raise(state, c"vector accessor on a non-vector");
        }
        let vector = parts(header);
        let Ok(offset) = values::element_index(state, 2, vector.len) else {
            return 0;
        };
        if *vector.validity.add(offset) != 0 {
            ffi::lua_pushnumber(state, *vector.values.add(offset));
        } else {
            values::push_null(state);
        }
        1
    }
}

/// `#v`.
unsafe extern "C" fn vector_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let header = ffi::luaL_testudata(state, 1, META_VECTOR.as_ptr()).cast::<VectorHeader>();
        if header.is_null() {
            return raise(state, c"vector accessor on a non-vector");
        }
        ffi::lua_pushinteger(state, (*header).len as i64);
        1
    }
}

/// Vectors are immutable — results feed forward, like views.
unsafe extern "C" fn vector_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe { raise(state, c"vectors are read-only (compose a new one)") }
}

// ---------------------------------------------------------------------
// Rolling combinators
// ---------------------------------------------------------------------

/// A Neumaier-compensated sum — the accumulation discipline the
/// engine's incremental windows use; never the raw cumsum idiom.
#[derive(Clone, Copy, Default)]
struct Compensated {
    sum: f64,
    correction: f64,
}

impl Compensated {
    fn add(&mut self, value: f64) {
        let total = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - total) + value;
        } else {
            self.correction += (value - total) + self.sum;
        }
        self.sum = total;
    }

    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

/// A dense (non-NULL) numeric operand for the rolling combinators —
/// the same loud rule the curated ops enforce.
unsafe fn dense_operand(state: *mut ffi::lua_State, idx: c_int) -> Result<Operand, ()> {
    unsafe {
        let source = operand(state, idx)?;
        let dense = match source {
            Operand::F64View { validity, len, .. } | Operand::I64View { validity, len, .. } => {
                validity.is_null() || (*validity).count_set() == len
            }
            Operand::Vector(vector) => {
                (0..vector.len).all(|offset| *vector.validity.add(offset) != 0)
            }
            Operand::Number(_) | Operand::Null => {
                raise(state, c"rolling combinators take columns, not scalars");
                return Err(());
            }
        };
        if !dense {
            raise(
                state,
                c"rolling combinators take non-null input; this column carries NULLs",
            );
            return Err(());
        }
        Ok(source)
    }
}

/// The window-width argument: a positive integer.
unsafe fn window_argument(state: *mut ffi::lua_State, idx: c_int) -> Result<usize, ()> {
    unsafe {
        let mut is_integer = 0;
        let window = ffi::lua_tointegerx(state, idx, &mut is_integer);
        if is_integer == 0 || window < 1 {
            raise(state, c"the window width must be a positive integer");
            return Err(());
        }
        Ok(window as usize)
    }
}

const ROLL_SUM: i64 = 1;
const ROLL_MEAN: i64 = 2;
const ROLL_DOT: i64 = 3;
const ROLL_VAR: i64 = 4;
const ROLL_STD: i64 = 5;

/// The one-column series transforms (M5.0): each reads a whole column
/// and writes a whole column, unlike the registry statistics, which
/// reduce one *frame* to one number. `lag`/`diff`/`log_returns` leave
/// the head undefined (NULL) because the rows they would reference sit
/// before the column's first row.
const SERIES_LAG: i64 = 1;
const SERIES_DIFF: i64 = 2;
const SERIES_LOG_RETURNS: i64 = 3;
const SERIES_EWMA: i64 = 4;

/// A positive count argument (`lag`'s distance).
unsafe fn count_argument(state: *mut ffi::lua_State, idx: c_int) -> Result<usize, ()> {
    unsafe {
        let mut is_integer = 0;
        let count = ffi::lua_tointegerx(state, idx, &mut is_integer);
        if is_integer == 0 || count < 1 {
            raise(state, c"the lag distance must be a positive integer");
            return Err(());
        }
        Ok(count as usize)
    }
}

/// EWMA's smoothing factor: a number in (0, 1].
unsafe fn alpha_argument(state: *mut ffi::lua_State, idx: c_int) -> Result<f64, ()> {
    unsafe {
        let mut is_number = 0;
        let alpha = ffi::lua_tonumberx(state, idx, &mut is_number);
        if is_number == 0 || !(alpha > 0.0 && alpha <= 1.0) {
            raise(state, c"the EWMA smoothing factor must be in (0, 1]");
            return Err(());
        }
        Ok(alpha)
    }
}

/// The one-column series transforms. One native O(n) pass; the head
/// rows a transform cannot define are marked NULL rather than filled
/// with a stand-in, so a script cannot mistake "no prior row" for a
/// value.
unsafe extern "C" fn series(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let op = ffi::lua_tointegerx(state, ffi::lua_upvalueindex(1), std::ptr::null_mut());
        let Ok(x) = dense_operand(state, 1) else {
            return 0;
        };
        let Some(len) = x.len() else {
            raise(state, c"series transforms take columns, not scalars");
            return 0;
        };
        // `lag` takes a distance; `ewma` a smoothing factor; the rest
        // take nothing. The head each one leaves undefined follows.
        let (back, alpha) = match op {
            SERIES_LAG => {
                let Ok(count) = count_argument(state, 2) else {
                    return 0;
                };
                (count, 0.0)
            }
            SERIES_EWMA => {
                let Ok(alpha) = alpha_argument(state, 2) else {
                    return 0;
                };
                (0, alpha)
            }
            _ => (1, 0.0),
        };
        let out = push_vector(state, len);
        let dense = dense_f64(x);
        let value = |offset: usize| -> Result<f64, ()> {
            match dense {
                Some(source) => Ok(source.at(offset)),
                None => Ok(element(state, x, offset)?.expect("dense operand")),
            }
        };
        if op == SERIES_EWMA {
            // y[0] = x[0]; y[i] = α·x[i] + (1−α)·y[i−1] — the recursive
            // (unadjusted) form, the one a live feed can carry in O(1)
            // state. Every row is defined, so no NULL head.
            let mut previous = 0.0;
            for offset in 0..len {
                let Ok(xi) = value(offset) else { return 0 };
                previous = if offset == 0 {
                    xi
                } else {
                    alpha * xi + (1.0 - alpha) * previous
                };
                *out.values.add(offset) = previous;
            }
            return 1;
        }
        for offset in 0..len {
            if offset < back {
                *out.validity.add(offset) = 0; // no row to reference
                continue;
            }
            let (Ok(current), Ok(prior)) = (value(offset), value(offset - back)) else {
                return 0;
            };
            *out.values.add(offset) = match op {
                SERIES_LAG => prior,
                SERIES_DIFF => current - prior,
                // ln(current / prior): the ratio first, so a scale
                // shared by both rows cancels before the logarithm.
                _ => (current / prior).ln(),
            };
        }
        1
    }
}

/// The rolling core: one native O(n) sweep per call. Trailing frames
/// (`min(i+1, w)` elements — SQL's `ROWS BETWEEN w-1 PRECEDING AND
/// CURRENT ROW`), compensated accumulation, and a fresh window
/// recompute every `w` steps so rounding cannot accumulate.
unsafe extern "C" fn rolling(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let op = ffi::lua_tointegerx(state, ffi::lua_upvalueindex(1), std::ptr::null_mut());
        let Ok(x) = dense_operand(state, 1) else {
            return 0;
        };
        let (y, window_idx) = if op == ROLL_DOT {
            let Ok(y) = dense_operand(state, 2) else {
                return 0;
            };
            (y, 3)
        } else {
            (x, 2)
        };
        let Ok(len) = common_length(state, x, y) else {
            return 0;
        };
        let Ok(window) = window_argument(state, window_idx) else {
            return 0;
        };
        let out = push_vector(state, len);
        // The dense fast path: raw slices, no raise points in the sweep.
        if let (Some(xs), Some(ys)) = (dense_f64(x), dense_f64(y)) {
            let term = |offset: usize| -> Result<f64, ()> {
                let a = xs.at(offset);
                Ok(if op == ROLL_DOT { a * ys.at(offset) } else { a })
            };
            let _ = if op == ROLL_VAR || op == ROLL_STD {
                rolling_spread_sweep(op, out, len, window, term)
            } else {
                rolling_sweep(op, out, len, window, term)
            };
            return 1;
        }
        // The general path — i64 exactness raises per element inside
        // `term` (NULLs were already refused).
        let term = |offset: usize| -> Result<f64, ()> {
            let a = element(state, x, offset)?.expect("dense operand");
            if op == ROLL_DOT {
                let b = element(state, y, offset)?.expect("dense operand");
                Ok(a * b)
            } else {
                Ok(a)
            }
        };
        let swept = if op == ROLL_VAR || op == ROLL_STD {
            rolling_spread_sweep(op, out, len, window, term)
        } else {
            rolling_sweep(op, out, len, window, term)
        };
        match swept {
            Ok(()) => 1,
            Err(()) => 0,
        }
    }
}

/// The rolling dispersion sweep: `rolling_var` / `rolling_std`, the
/// column-shaped twins of SQL's `var_pop`/`stddev_pop` window
/// functions (M5.0). O(n), one add and one remove per step.
///
/// Deviations are accumulated **about a shift taken from the data**,
/// exactly as `engine`'s `ShiftedMoments` does for the pair
/// statistics, and for the same reason: `E[x²] − E[x]²` over raw
/// values is the catastrophic form (it cancels away the answer when
/// the data sits at a large offset), while the same expression over
/// deviations from a nearby shift keeps every accumulated term at the
/// window's own scale. The shift is re-taken every `window` steps,
/// which also bounds add/remove drift to one period — the same
/// re-anchoring cadence `rolling_sweep` uses above. The two sweeps
/// stay separate because this one carries second moments and a shift
/// that the sum/mean/dot sweep has no use for.
unsafe fn rolling_spread_sweep<F: Fn(usize) -> Result<f64, ()> + Copy>(
    op: i64,
    out: VectorParts,
    len: usize,
    window: usize,
    term: F,
) -> Result<(), ()> {
    unsafe {
        let mut shift = 0.0;
        let mut count = 0.0f64;
        let mut sum = Compensated::default();
        let mut squares = Compensated::default();
        let mut anchored: Option<usize> = None;
        for offset in 0..len {
            let lo = (offset + 1).saturating_sub(window);
            let stale = anchored.is_none_or(|at| offset - at >= window);
            if stale {
                // Re-anchor on this row's own value and rebuild the
                // live window about it.
                shift = term(offset)?;
                sum = Compensated::default();
                squares = Compensated::default();
                count = 0.0;
                for inner in lo..=offset {
                    let d = term(inner)? - shift;
                    sum.add(d);
                    squares.add(d * d);
                    count += 1.0;
                }
                anchored = Some(offset);
            } else {
                let d = term(offset)? - shift;
                sum.add(d);
                squares.add(d * d);
                count += 1.0;
                if offset >= window {
                    let gone = term(offset - window)? - shift;
                    sum.add(-gone);
                    squares.add(-(gone * gone));
                    count -= 1.0;
                }
            }
            let total = sum.value();
            let variance = (squares.value() - total * total / count) / count;
            // Rounding can push a non-negative variance just below
            // zero, and `sqrt` of that is NaN. Clamp — but test NaN
            // first, because `f64::max` discards a NaN operand and
            // would report a confident zero for an undefined window.
            let variance = if variance.is_nan() {
                variance
            } else {
                variance.max(0.0)
            };
            *out.values.add(offset) = if op == ROLL_STD {
                variance.sqrt()
            } else {
                variance
            };
        }
        Ok(())
    }
}

/// The rolling sweep body, shared by the fast and general term
/// sources. `term` may raise (longjmp) on the general path, so this
/// frame holds only `Copy` state.
unsafe fn rolling_sweep<F: Fn(usize) -> Result<f64, ()> + Copy>(
    op: i64,
    out: VectorParts,
    len: usize,
    window: usize,
    term: F,
) -> Result<(), ()> {
    unsafe {
        let mut sum = Compensated::default();
        for offset in 0..len {
            sum.add(term(offset)?);
            if offset >= window {
                sum.add(-term(offset - window)?);
            }
            // Re-anchor: recompute the live window from scratch so the
            // add/subtract drift is bounded by one period.
            if (offset + 1) % window == 0 {
                let mut fresh = Compensated::default();
                let start = (offset + 1).saturating_sub(window);
                for inner in start..=offset {
                    fresh.add(term(inner)?);
                }
                sum = fresh;
            }
            let count = (offset + 1).min(window) as f64;
            *out.values.add(offset) = match op {
                ROLL_MEAN => sum.value() / count,
                _ => sum.value(),
            };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Reading a returned column (the `return <vector>` kernel shape)
// ---------------------------------------------------------------------

/// A column-shaped script result, read out of the interpreter: the
/// elements of a returned vector or numeric view, or nothing at all
/// (the script wrote `out[i]` instead). Anything else is an error.
/// Dense results skip the per-element `Option`.
pub(crate) enum ColumnResult {
    /// The script returned nothing; its `out[i]` writes stand.
    None,
    /// One element of the returned column per call — `None` is NULL.
    Elements(Vec<Option<f64>>),
    /// A column with no NULLs, read in bulk.
    Dense(Vec<f64>),
}

/// Reads the value at the top of the stack as a column result. Runs
/// outside any Lua frame (after `lua_pcall` returned), so plain `Err`
/// strings are fine here.
///
/// # Safety
/// `raw` must be a state `install` has prepared with at least one
/// value on the stack.
pub(crate) unsafe fn read_column_result(raw: *mut ffi::lua_State) -> Result<ColumnResult, String> {
    unsafe {
        if ffi::lua_type(raw, -1) == ffi::LUA_TNIL {
            return Ok(ColumnResult::None);
        }
        let vector = ffi::luaL_testudata(raw, -1, META_VECTOR.as_ptr());
        if !vector.is_null() {
            let vector = parts(vector.cast::<VectorHeader>());
            if let Some(DenseF64::Slice(values)) = dense_f64(Operand::Vector(vector)) {
                return Ok(ColumnResult::Dense(
                    std::slice::from_raw_parts(values, vector.len).to_vec(),
                ));
            }
            let mut elements = Vec::with_capacity(vector.len);
            for offset in 0..vector.len {
                elements
                    .push((*vector.validity.add(offset) != 0).then(|| *vector.values.add(offset)));
            }
            return Ok(ColumnResult::Elements(elements));
        }
        let payload = ffi::luaL_testudata(raw, -1, META_INPUT.as_ptr()).cast::<InputPayload>();
        if !payload.is_null() {
            let view = *payload;
            if *view.generation != view.born {
                return Err("result: view used outside its call".to_owned());
            }
            if view.tag == TAG_F64
                && (view.validity.is_null() || (*view.validity).count_set() == view.len)
            {
                return Ok(ColumnResult::Dense(
                    std::slice::from_raw_parts(view.data.cast::<f64>(), view.len).to_vec(),
                ));
            }
            let mut elements = Vec::with_capacity(view.len);
            for offset in 0..view.len {
                let value = match view.tag {
                    _ if !is_valid(view.validity, offset) => None,
                    TAG_F64 => Some(*view.data.cast::<f64>().add(offset)),
                    TAG_I64 => match int_as_f64_exact(*view.data.cast::<i64>().add(offset)) {
                        Some(value) => Some(value),
                        None => {
                            return Err("result: i64 element does not fit f64 exactly".to_owned())
                        }
                    },
                    _ => return Err("result: a key view is not a numeric column".to_owned()),
                };
                elements.push(value);
            }
            return Ok(ColumnResult::Elements(elements));
        }
        Err(
            "result: a column kernel returns a vector, a view, or nothing (writing out[i])"
                .to_owned(),
        )
    }
}

/// The all-NULL vector the `NULL` sentinel's arithmetic produces when
/// its other operand is sized — so `NULL * v` and `v * NULL` agree
/// (three-valued logic, elementwise). Returns false when neither
/// operand is sized (the caller pushes the plain sentinel).
///
/// # Safety
/// `raw` must be a state `install` has prepared, inside a metamethod
/// call with the two operands at stack slots 1 and 2.
pub(crate) unsafe fn push_null_arith_result(raw: *mut ffi::lua_State) -> bool {
    unsafe {
        for idx in [1, 2] {
            let len = {
                let vector = ffi::luaL_testudata(raw, idx, META_VECTOR.as_ptr());
                if !vector.is_null() {
                    Some((*vector.cast::<VectorHeader>()).len)
                } else {
                    let payload =
                        ffi::luaL_testudata(raw, idx, META_INPUT.as_ptr()).cast::<InputPayload>();
                    if !payload.is_null() && (*payload).tag != TAG_KEY {
                        Some((*payload).len)
                    } else {
                        None
                    }
                }
            };
            if let Some(len) = len {
                let out = push_vector(raw, len);
                std::ptr::write_bytes(out.validity, 0, len);
                return true;
            }
        }
        false
    }
}

/// Installs the vector metatable, the arithmetic metamethods on both
/// vectors and input views, and the rolling combinators.
///
/// # Safety
/// `raw` must be a valid state whose input-view metatable
/// [`values::install`] has already created; empty stack.
pub(crate) unsafe fn install(raw: *mut ffi::lua_State) {
    unsafe {
        ffi::luaL_newmetatable(raw, META_VECTOR.as_ptr());
        ffi::lua_pushcclosure(raw, vector_index, 0);
        ffi::lua_setfield(raw, -2, c"__index".as_ptr());
        ffi::lua_pushcclosure(raw, vector_len, 0);
        ffi::lua_setfield(raw, -2, c"__len".as_ptr());
        ffi::lua_pushcclosure(raw, vector_newindex, 0);
        ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
        ffi::lua_settop(raw, 0);

        // The operators, on vectors and input views alike.
        for meta in [META_VECTOR, META_INPUT] {
            // Metatables live in the registry under their names.
            ffi::lua_getfield(raw, ffi::LUA_REGISTRYINDEX, meta.as_ptr());
            for (name, op) in [
                (c"__add".as_ptr(), OP_ADD),
                (c"__sub".as_ptr(), OP_SUB),
                (c"__mul".as_ptr(), OP_MUL),
                (c"__div".as_ptr(), OP_DIV),
            ] {
                ffi::lua_pushinteger(raw, op);
                ffi::lua_pushcclosure(raw, vector_binary, 1);
                ffi::lua_setfield(raw, -2, name);
            }
            ffi::lua_pushcclosure(raw, vector_negate, 0);
            ffi::lua_setfield(raw, -2, c"__unm".as_ptr());
            ffi::lua_settop(raw, 0);
        }

        // The rolling combinators (Lua-side names only; SQL spells
        // these with window frames).
        for (name, op) in [
            (c"rolling_sum".as_ptr(), ROLL_SUM),
            (c"rolling_mean".as_ptr(), ROLL_MEAN),
            (c"rolling_dot".as_ptr(), ROLL_DOT),
            (c"rolling_var".as_ptr(), ROLL_VAR),
            (c"rolling_std".as_ptr(), ROLL_STD),
        ] {
            ffi::lua_pushinteger(raw, op);
            ffi::lua_pushcclosure(raw, rolling, 1);
            ffi::lua_setglobal(raw, name);
        }

        // The one-column series transforms (M5.0). These carry no
        // standard SQL name, so by the #77.1 (a) ruling they live here
        // rather than in the SQL registry.
        for (name, op) in [
            (c"lag".as_ptr(), SERIES_LAG),
            (c"diff".as_ptr(), SERIES_DIFF),
            (c"log_returns".as_ptr(), SERIES_LOG_RETURNS),
            (c"ewma".as_ptr(), SERIES_EWMA),
        ] {
            ffi::lua_pushinteger(raw, op);
            ffi::lua_pushcclosure(raw, series, 1);
            ffi::lua_setglobal(raw, name);
        }
    }
}
