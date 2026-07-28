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
//! `rolling_sum(x, w)`, `rolling_mean(x, w)`, `rolling_dot(x, y, w)`
//! (names Lua-side only — SQL spells these with window frames, so
//! nothing new enters the SQL surface). Frames are trailing, exactly
//! the SQL shape `ROWS BETWEEN w-1 PRECEDING AND CURRENT ROW`: row `i`
//! covers the last `min(i+1, w)` elements. Inputs must be dense
//! (non-NULL) — the same loud rule the curated ops enforce. The sums
//! run Neumaier-compensated and re-anchor with a fresh window
//! recompute every `w` steps, the engine's incremental-window
//! discipline — never the plain cumsum idiom, which is the
//! catastrophic-cancellation form the engine rejected.

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
        // Dense elements: NULLs were refused, and i64 exactness raises
        // per element inside `term`.
        let term = |state: *mut ffi::lua_State, offset: usize| -> Result<f64, ()> {
            let a = element(state, x, offset)?.expect("dense operand");
            if op == ROLL_DOT {
                let b = element(state, y, offset)?.expect("dense operand");
                Ok(a * b)
            } else {
                Ok(a)
            }
        };
        let out = push_vector(state, len);
        let mut sum = Compensated::default();
        for offset in 0..len {
            let Ok(new) = term(state, offset) else {
                return 0;
            };
            sum.add(new);
            if offset >= window {
                let Ok(old) = term(state, offset - window) else {
                    return 0;
                };
                sum.add(-old);
            }
            // Re-anchor: recompute the live window from scratch so the
            // add/subtract drift is bounded by one period.
            if (offset + 1) % window == 0 {
                let mut fresh = Compensated::default();
                let start = (offset + 1).saturating_sub(window);
                for inner in start..=offset {
                    let Ok(value) = term(state, inner) else {
                        return 0;
                    };
                    fresh.add(value);
                }
                sum = fresh;
            }
            let count = (offset + 1).min(window) as f64;
            *out.values.add(offset) = match op {
                ROLL_MEAN => sum.value() / count,
                _ => sum.value(),
            };
        }
        1
    }
}

// ---------------------------------------------------------------------
// Reading a returned column (the `return <vector>` kernel shape)
// ---------------------------------------------------------------------

/// A `Copy` description of a column-shaped script result: a vector or
/// a numeric input view at `idx`, or `None` for nil (the script wrote
/// `out[i]` instead). Anything else is an error.
pub(crate) enum ColumnResult {
    /// The script returned nothing; its `out[i]` writes stand.
    None,
    /// One element of the returned column per call — `None` is NULL.
    Elements(Vec<Option<f64>>),
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
        ] {
            ffi::lua_pushinteger(raw, op);
            ffi::lua_pushcclosure(raw, rolling, 1);
            ffi::lua_setglobal(raw, name);
        }
    }
}
