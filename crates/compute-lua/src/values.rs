//! The value map — how engine values and script values convert at the
//! Lua boundary, implementing the frozen contract (DESIGN.md, *The Lua
//! layer*, decision records of 2026-07-26):
//!
//! - **F1 — NULL is a sentinel.** SQL NULL crosses to Lua as the global
//!   `NULL`, a pd.NA-style singleton — not `nil` (which deletes table
//!   slots) and not NaN (a computed value). Arithmetic touching `NULL`
//!   propagates to `NULL` (three-valued logic); `x == NULL` is the guard
//!   idiom (identity equality); relational `<`/`<=` against it is a loud
//!   error, because Lua forces those operators to a boolean.
//! - **F2 — the result type is declared, never inferred.** A script's
//!   output column or scalar result has its type ([`ReturnType`]) fixed
//!   by the caller before the script runs, so a query yields the same
//!   Arrow schema on every run regardless of what a call happens to
//!   return.
//! - **F3 — coercion is exact-or-loud.** A Lua integer fills `i64` and a
//!   float fills `f64` as-is; either fills the other type only when the
//!   conversion is lossless, otherwise it is a loud error — never a
//!   silent truncation or rounding. A boolean maps to `i64` `{0, 1}`
//!   (booleans are transient values, never a column type). `nil` and
//!   `NULL` both mark the slot NULL. A string is interned into the
//!   output key dictionary — the only way a script produces a key.
//! - **F4 — keys read as codes, with lazy text.** A key element reads as
//!   its integer dictionary code; `v:text(i)` decodes on demand and
//!   `v:code_of(literal)` resolves a literal once, so key comparison
//!   stays integer-cheap and element reads stay zero-copy.
//!
//! On top of the contract sit the additive ergonomics the NULL decision
//! record names as the common path: `v:sum()` (a null-skipping batch
//! reduction computed engine-side — the sentinel is never materialized)
//! and `v:mask()` (an out-of-band validity view reading booleans, so the
//! value stream stays purely numeric).
//!
//! ## Lifetime discipline: generations, not dangling pointers
//!
//! Every view userdata records the interpreter's generation counter at
//! bind time. Each `eval_*` call bumps the counter on every exit path,
//! so a view (or mask) smuggled past its call — stashed in a global, a
//! closure, anywhere — fails loudly on next use instead of dereferencing
//! a dead borrow.
//!
//! ## C-boundary discipline (shared with `state`)
//!
//! Accessors called from Lua keep only `Copy` locals and raise via
//! [`raise`] in tail position, so `lua_error`'s `longjmp` unwinds no
//! Rust destructor; nothing here panics — every fallible condition is
//! checked and raised as a Lua error inside the surrounding `lua_pcall`.

use crate::ffi;
use arrow_lite::{Bitmap, Dictionary};
use std::ffi::{c_char, c_int, CStr};

/// A borrowed input column for one script call, in the engine's own
/// representation: a value buffer, an optional validity bitmap (absent
/// means no nulls), and for keys the dictionary the codes index into.
pub enum ColumnView<'a> {
    /// An `f64` column; elements read as Lua floats.
    F64 {
        /// The value buffer.
        values: &'a [f64],
        /// One bit per element; unset means NULL. `None` = all valid.
        validity: Option<&'a Bitmap>,
    },
    /// An `i64` column; elements read as Lua integers — exactly.
    I64 {
        /// The value buffer.
        values: &'a [i64],
        /// One bit per element; unset means NULL. `None` = all valid.
        validity: Option<&'a Bitmap>,
    },
    /// A key column; elements read as integer dictionary codes (F4).
    Key {
        /// The per-row codes.
        codes: &'a [u32],
        /// One bit per element; unset means NULL. `None` = all valid.
        validity: Option<&'a Bitmap>,
        /// The dictionary the codes index into.
        dictionary: &'a Dictionary,
    },
}

/// The preallocated output column a script writes through the `out`
/// view. Validity is reset to all-NULL at bind time, so a slot the
/// script never writes comes back NULL — there is no uninitialized
/// state a caller can observe.
pub enum OutputColumn<'a> {
    /// An `f64` output column.
    F64 {
        /// The value buffer to fill.
        values: &'a mut [f64],
        /// Rewritten to `values.len()` bits, initially all unset.
        validity: &'a mut Bitmap,
    },
    /// An `i64` output column.
    I64 {
        /// The value buffer to fill.
        values: &'a mut [i64],
        /// Rewritten to `values.len()` bits, initially all unset.
        validity: &'a mut Bitmap,
    },
    /// A key output column; scripts fill it with strings, which are
    /// interned here — the only way a script produces a key (F3).
    Key {
        /// The code buffer to fill.
        codes: &'a mut [u32],
        /// Rewritten to `codes.len()` bits, initially all unset.
        validity: &'a mut Bitmap,
        /// The dictionary output strings are interned into.
        dictionary: &'a mut Dictionary,
    },
}

/// The declared type of a script's scalar result (F2). Declared by the
/// caller at registration time and enforced exact-or-loud (F3) — never
/// inferred from the value a call happens to return.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReturnType {
    /// The result fills an `f64` slot.
    F64,
    /// The result fills an `i64` slot.
    I64,
    /// The result is a key. Scalar key results have no consumer yet and
    /// no dictionary to intern into; `eval_scalar` refuses this loudly.
    Key,
}

/// A script's scalar result, coerced to its declared [`ReturnType`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ScalarValue {
    /// An `f64` result.
    F64(f64),
    /// An `i64` result.
    I64(i64),
    /// SQL NULL: the script returned `NULL` (or `nil`).
    Null,
}

const TAG_F64: u8 = 0;
const TAG_I64: u8 = 1;
const TAG_KEY: u8 = 2;

/// Payload of an input-view userdata: borrowed engine buffers plus the
/// generation stamp that bounds their lifetime.
#[repr(C)]
#[derive(Clone, Copy)]
struct InputPayload {
    data: *const u8,
    /// Null means every element is valid.
    validity: *const Bitmap,
    /// Null unless `tag == TAG_KEY`.
    dictionary: *const Dictionary,
    len: usize,
    tag: u8,
    /// The interpreter generation this view was bound in.
    born: u64,
    /// The live generation counter; a mismatch with `born` means the
    /// view outlived its call.
    generation: *const u64,
}

/// Payload of the `out` output-view userdata.
#[repr(C)]
#[derive(Clone, Copy)]
struct OutputPayload {
    data: *mut u8,
    validity: *mut Bitmap,
    /// Null unless `tag == TAG_KEY`.
    dictionary: *mut Dictionary,
    len: usize,
    tag: u8,
    born: u64,
    generation: *const u64,
}

/// Payload of a `v:mask()` userdata: the parent view's validity, read
/// out-of-band as booleans.
#[repr(C)]
#[derive(Clone, Copy)]
struct MaskPayload {
    /// Null means every element is valid.
    validity: *const Bitmap,
    len: usize,
    born: u64,
    generation: *const u64,
}

const META_INPUT: &CStr = c"tallydb.column";
const META_OUTPUT: &CStr = c"tallydb.output";
const META_MASK: &CStr = c"tallydb.mask";
const META_NULL: &CStr = c"tallydb.null";
/// Registry key of the NULL singleton (distinct from the metatable's
/// registry entry under [`META_NULL`]).
const REG_NULL: &CStr = c"tallydb.null.value";
/// Registry key of the generation cell — a `u64` in a userdata block,
/// anchored for the state's whole life so payload pointers to it never
/// dangle.
const REG_GENERATION: &CStr = c"tallydb.generation";

/// Installs the view metatables, the NULL sentinel, and the generation
/// cell into a fresh state; returns the pointer to the generation
/// counter.
///
/// # Safety
/// `raw` must be a valid, just-created Lua state with an empty stack.
pub(crate) unsafe fn install(raw: *mut ffi::lua_State) -> *mut u64 {
    unsafe {
        // Input views: element reads and method dispatch via __index.
        ffi::luaL_newmetatable(raw, META_INPUT.as_ptr());
        ffi::lua_pushcclosure(raw, input_index, 0);
        ffi::lua_setfield(raw, -2, c"__index".as_ptr());
        ffi::lua_pushcclosure(raw, input_len, 0);
        ffi::lua_setfield(raw, -2, c"__len".as_ptr());
        ffi::lua_pushcclosure(raw, input_newindex, 0);
        ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
        ffi::lua_settop(raw, 0);

        // The output view: coercing writes, read-back, length.
        ffi::luaL_newmetatable(raw, META_OUTPUT.as_ptr());
        ffi::lua_pushcclosure(raw, output_index, 0);
        ffi::lua_setfield(raw, -2, c"__index".as_ptr());
        ffi::lua_pushcclosure(raw, output_len, 0);
        ffi::lua_setfield(raw, -2, c"__len".as_ptr());
        ffi::lua_pushcclosure(raw, output_newindex, 0);
        ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
        ffi::lua_settop(raw, 0);

        // Masks: boolean element reads only.
        ffi::luaL_newmetatable(raw, META_MASK.as_ptr());
        ffi::lua_pushcclosure(raw, mask_index, 0);
        ffi::lua_setfield(raw, -2, c"__index".as_ptr());
        ffi::lua_pushcclosure(raw, mask_len, 0);
        ffi::lua_setfield(raw, -2, c"__len".as_ptr());
        ffi::lua_pushcclosure(raw, mask_newindex, 0);
        ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
        ffi::lua_settop(raw, 0);

        // The NULL sentinel's metatable: every arithmetic, bitwise, and
        // concatenation metamethod propagates to NULL (three-valued
        // logic). No __eq (identity equality is the guard idiom), no
        // __lt/__le (comparison is a loud error — Lua forces those to a
        // boolean, so 3VL cannot propagate through them).
        ffi::luaL_newmetatable(raw, META_NULL.as_ptr());
        for op in [
            c"__add".as_ptr(),
            c"__sub".as_ptr(),
            c"__mul".as_ptr(),
            c"__div".as_ptr(),
            c"__mod".as_ptr(),
            c"__pow".as_ptr(),
            c"__unm".as_ptr(),
            c"__idiv".as_ptr(),
            c"__band".as_ptr(),
            c"__bor".as_ptr(),
            c"__bxor".as_ptr(),
            c"__bnot".as_ptr(),
            c"__shl".as_ptr(),
            c"__shr".as_ptr(),
            c"__concat".as_ptr(),
        ] {
            ffi::lua_pushcclosure(raw, null_propagate, 0);
            ffi::lua_setfield(raw, -2, op);
        }
        ffi::lua_pushcclosure(raw, null_tostring, 0);
        ffi::lua_setfield(raw, -2, c"__tostring".as_ptr());
        ffi::lua_settop(raw, 0);

        // The singleton itself: a one-byte userdata under that
        // metatable, stashed in the registry and exposed as global NULL.
        ffi::lua_newuserdatauv(raw, 1, 0);
        ffi::luaL_setmetatable(raw, META_NULL.as_ptr());
        ffi::lua_setfield(raw, ffi::LUA_REGISTRYINDEX, REG_NULL.as_ptr());
        ffi::lua_getfield(raw, ffi::LUA_REGISTRYINDEX, REG_NULL.as_ptr());
        ffi::lua_setglobal(raw, c"NULL".as_ptr());

        // The generation cell: registry-anchored, so it outlives every
        // view that points at it.
        let generation = ffi::lua_newuserdatauv(raw, std::mem::size_of::<u64>(), 0).cast::<u64>();
        generation.write(0);
        ffi::lua_setfield(raw, ffi::LUA_REGISTRYINDEX, REG_GENERATION.as_ptr());
        ffi::lua_settop(raw, 0);
        generation
    }
}

/// Binds `view` as the global `name`. The borrow is bounded by the
/// generation stamp: the caller bumps the counter when its call ends.
///
/// # Safety
/// `raw` must be a state [`install`] has prepared and `generation` its
/// generation cell; the borrows inside `view` must stay live until the
/// caller bumps the generation.
pub(crate) unsafe fn bind_input(
    raw: *mut ffi::lua_State,
    generation: *const u64,
    name: &CStr,
    view: &ColumnView<'_>,
) -> Result<(), String> {
    let payload = match *view {
        ColumnView::F64 { values, validity } => InputPayload {
            data: values.as_ptr().cast(),
            validity: validity_ptr(validity, values.len(), name)?,
            dictionary: std::ptr::null(),
            len: values.len(),
            tag: TAG_F64,
            born: unsafe { *generation },
            generation,
        },
        ColumnView::I64 { values, validity } => InputPayload {
            data: values.as_ptr().cast(),
            validity: validity_ptr(validity, values.len(), name)?,
            dictionary: std::ptr::null(),
            len: values.len(),
            tag: TAG_I64,
            born: unsafe { *generation },
            generation,
        },
        ColumnView::Key {
            codes,
            validity,
            dictionary,
        } => InputPayload {
            data: codes.as_ptr().cast(),
            validity: validity_ptr(validity, codes.len(), name)?,
            dictionary,
            len: codes.len(),
            tag: TAG_KEY,
            born: unsafe { *generation },
            generation,
        },
    };
    unsafe {
        let slot = ffi::lua_newuserdatauv(raw, std::mem::size_of::<InputPayload>(), 0)
            .cast::<InputPayload>();
        slot.write(payload);
        ffi::luaL_setmetatable(raw, META_INPUT.as_ptr());
        ffi::lua_setglobal(raw, name.as_ptr());
    }
    Ok(())
}

/// Checks a validity bitmap's length against its buffer and returns the
/// payload pointer (null when the column has no nulls).
fn validity_ptr(
    validity: Option<&Bitmap>,
    len: usize,
    name: &CStr,
) -> Result<*const Bitmap, String> {
    match validity {
        None => Ok(std::ptr::null()),
        Some(bitmap) if bitmap.len() == len => Ok(bitmap),
        Some(bitmap) => Err(format!(
            "column {:?}: validity has {} bits for {len} values",
            name,
            bitmap.len()
        )),
    }
}

/// Binds `output` as the global `out`, resetting its validity to
/// all-NULL so unwritten slots come back NULL.
///
/// # Safety
/// Same contract as [`bind_input`]; additionally the `&mut` borrows
/// inside `output` are written through raw pointers until the caller
/// bumps the generation, so the caller must not touch them meanwhile.
pub(crate) unsafe fn bind_output(
    raw: *mut ffi::lua_State,
    generation: *const u64,
    output: &mut OutputColumn<'_>,
) {
    let payload = match output {
        OutputColumn::F64 { values, validity } => {
            **validity = Bitmap::new_unset(values.len());
            OutputPayload {
                data: values.as_mut_ptr().cast(),
                validity: *validity,
                dictionary: std::ptr::null_mut(),
                len: values.len(),
                tag: TAG_F64,
                born: unsafe { *generation },
                generation,
            }
        }
        OutputColumn::I64 { values, validity } => {
            **validity = Bitmap::new_unset(values.len());
            OutputPayload {
                data: values.as_mut_ptr().cast(),
                validity: *validity,
                dictionary: std::ptr::null_mut(),
                len: values.len(),
                tag: TAG_I64,
                born: unsafe { *generation },
                generation,
            }
        }
        OutputColumn::Key {
            codes,
            validity,
            dictionary,
        } => {
            **validity = Bitmap::new_unset(codes.len());
            OutputPayload {
                data: codes.as_mut_ptr().cast(),
                validity: *validity,
                dictionary: *dictionary,
                len: codes.len(),
                tag: TAG_KEY,
                born: unsafe { *generation },
                generation,
            }
        }
    };
    unsafe {
        let slot = ffi::lua_newuserdatauv(raw, std::mem::size_of::<OutputPayload>(), 0)
            .cast::<OutputPayload>();
        slot.write(payload);
        ffi::luaL_setmetatable(raw, META_OUTPUT.as_ptr());
        ffi::lua_setglobal(raw, c"out".as_ptr());
    }
}

/// Reads the value at the top of the stack as a typed scalar result,
/// applying the exact-or-loud coercion (F3) for the declared type (F2).
/// The stack is left unchanged.
///
/// # Safety
/// `raw` must be a state [`install`] has prepared with at least one
/// value on the stack.
pub(crate) unsafe fn read_result(
    raw: *mut ffi::lua_State,
    declared: ReturnType,
) -> Result<ScalarValue, String> {
    let tag = match declared {
        ReturnType::F64 => TAG_F64,
        ReturnType::I64 => TAG_I64,
        // No scalar-key consumer exists; refuse rather than invent a
        // dictionary for it (see ReturnType::Key).
        ReturnType::Key => {
            return Err("a key-typed scalar result has no output dictionary; \
                 produce keys through an output column"
                .to_owned())
        }
    };
    match unsafe { coerce(raw, -1, tag, std::ptr::null_mut()) } {
        Ok(Coerced::F64(value)) => Ok(ScalarValue::F64(value)),
        Ok(Coerced::I64(value)) => Ok(ScalarValue::I64(value)),
        Ok(Coerced::Null) => Ok(ScalarValue::Null),
        // coerce only interns for TAG_KEY, which is refused above.
        Ok(Coerced::KeyCode(_)) => unreachable!("key result refused before coercion"),
        Err(message) => Err(format!("result: {}", message.to_string_lossy())),
    }
}

/// A Lua value coerced to a typed slot.
enum Coerced {
    F64(f64),
    I64(i64),
    KeyCode(u32),
    Null,
}

/// The F3 coercion: reads the Lua value at `idx` for a slot of type
/// `tag`. Exact-or-loud; every rejection is a static message suitable
/// for raising from a C accessor. `dictionary` must be non-null exactly
/// when `tag` is [`TAG_KEY`].
///
/// # Safety
/// `raw` must be a state [`install`] has prepared and `idx` a valid
/// stack index.
unsafe fn coerce(
    raw: *mut ffi::lua_State,
    idx: c_int,
    tag: u8,
    dictionary: *mut Dictionary,
) -> Result<Coerced, &'static CStr> {
    unsafe {
        match ffi::lua_type(raw, idx) {
            // nil and the NULL sentinel both mark the slot NULL (F1/F3).
            ffi::LUA_TNIL => return Ok(Coerced::Null),
            _ if !ffi::luaL_testudata(raw, idx, META_NULL.as_ptr()).is_null() => {
                return Ok(Coerced::Null)
            }
            _ => {}
        }
        let value_type = ffi::lua_type(raw, idx);
        match tag {
            TAG_F64 => match value_type {
                ffi::LUA_TNUMBER if ffi::lua_isinteger(raw, idx) == 1 => {
                    let mut ok = 0;
                    let integer = ffi::lua_tointegerx(raw, idx, &mut ok);
                    match int_as_f64_exact(integer) {
                        Some(value) => Ok(Coerced::F64(value)),
                        None => Err(c"integer does not fit f64 exactly"),
                    }
                }
                ffi::LUA_TNUMBER => {
                    let mut ok = 0;
                    Ok(Coerced::F64(ffi::lua_tonumberx(raw, idx, &mut ok)))
                }
                ffi::LUA_TBOOLEAN => Err(c"a boolean maps to i64 {0, 1}, not f64"),
                ffi::LUA_TSTRING => Err(c"a string produces a key, not f64"),
                _ => Err(c"f64 slot expects a number or NULL"),
            },
            TAG_I64 => match value_type {
                ffi::LUA_TNUMBER if ffi::lua_isinteger(raw, idx) == 1 => {
                    let mut ok = 0;
                    Ok(Coerced::I64(ffi::lua_tointegerx(raw, idx, &mut ok)))
                }
                ffi::LUA_TNUMBER => {
                    let mut ok = 0;
                    let float = ffi::lua_tonumberx(raw, idx, &mut ok);
                    match float_as_i64_exact(float) {
                        Some(value) => Ok(Coerced::I64(value)),
                        None => Err(c"float does not fit i64 exactly"),
                    }
                }
                ffi::LUA_TBOOLEAN => Ok(Coerced::I64(i64::from(ffi::lua_toboolean(raw, idx) != 0))),
                ffi::LUA_TSTRING => Err(c"a string produces a key, not i64"),
                _ => Err(c"i64 slot expects a number, boolean, or NULL"),
            },
            _ => match value_type {
                ffi::LUA_TSTRING => {
                    let mut len = 0usize;
                    let text = ffi::lua_tolstring(raw, idx, &mut len);
                    let bytes = std::slice::from_raw_parts(text.cast::<u8>(), len);
                    let Ok(value) = std::str::from_utf8(bytes) else {
                        return Err(c"key text must be valid UTF-8");
                    };
                    // Pre-check what Dictionary::intern would panic on,
                    // so the failure is a Lua error, not an abort.
                    let dict = &mut *dictionary;
                    if dict.code_of(value).is_none()
                        && (dict.len() >= u32::MAX as usize
                            || dict.bytes().len() + value.len() > i32::MAX as usize)
                    {
                        return Err(c"output key dictionary is full");
                    }
                    Ok(Coerced::KeyCode(dict.intern(value)))
                }
                _ => Err(c"only a string produces a key (codes are per-segment)"),
            },
        }
    }
}

/// The exact `i64` → `f64` conversion, or `None` where rounding would
/// occur (integers beyond ±2⁵³ are not all representable).
fn int_as_f64_exact(integer: i64) -> Option<f64> {
    let float = integer as f64;
    // Compare in i128: `float` can be 2⁶³ exactly, which i64 cannot hold.
    (float as i128 == i128::from(integer)).then_some(float)
}

/// The exact `f64` → `i64` conversion: losslessly integral and in
/// range, or `None`. NaN and the infinities fail the fract test.
fn float_as_i64_exact(float: f64) -> Option<i64> {
    const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
    if float.fract() != 0.0 {
        return None;
    }
    (-TWO_POW_63..TWO_POW_63)
        .contains(&float)
        .then_some(float as i64)
}

/// Pushes `message` and raises it — tail position only, `Copy` state
/// only (the module discipline).
unsafe fn raise(state: *mut ffi::lua_State, message: &CStr) -> c_int {
    unsafe {
        let bytes = message.to_bytes();
        ffi::lua_pushlstring(state, bytes.as_ptr().cast::<c_char>(), bytes.len());
        ffi::lua_error(state)
    }
}

/// Pushes the NULL sentinel.
unsafe fn push_null(state: *mut ffi::lua_State) {
    unsafe {
        ffi::lua_getfield(state, ffi::LUA_REGISTRYINDEX, REG_NULL.as_ptr());
    }
}

/// Reads the integer at `idx` as a 1-based element index into a view of
/// `len` elements, or raises.
unsafe fn element_index(state: *mut ffi::lua_State, idx: c_int, len: usize) -> Result<usize, ()> {
    unsafe {
        let mut is_integer = 0;
        let index = ffi::lua_tointegerx(state, idx, &mut is_integer);
        if is_integer == 0 {
            raise(state, c"view index must be an integer");
            return Err(());
        }
        if index < 1 || index as usize > len {
            raise(state, c"view index out of range");
            return Err(());
        }
        Ok((index - 1) as usize)
    }
}

/// Whether element `offset` is valid under a possibly-absent bitmap.
unsafe fn is_valid(validity: *const Bitmap, offset: usize) -> bool {
    unsafe { validity.is_null() || (*validity).get(offset) }
}

// ---------------------------------------------------------------------
// Input-view accessors
// ---------------------------------------------------------------------

/// Fetches the checked input payload at argument 1, raising on a
/// non-view or a view that outlived its call.
unsafe fn input_arg(state: *mut ffi::lua_State) -> Result<InputPayload, ()> {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_INPUT.as_ptr()).cast::<InputPayload>();
        if payload.is_null() {
            raise(state, c"view accessor on a non-view");
            return Err(());
        }
        let view = *payload;
        if *view.generation != view.born {
            raise(state, c"view used outside its call");
            return Err(());
        }
        Ok(view)
    }
}

/// `v[i]` for an integer `i`; `v:method` for a string key (F4 methods
/// and the batch ergonomics).
unsafe extern "C" fn input_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = input_arg(state) else {
            return 0;
        };
        if ffi::lua_type(state, 2) == ffi::LUA_TSTRING {
            return input_method(state, view);
        }
        let Ok(offset) = element_index(state, 2, view.len) else {
            return 0;
        };
        if !is_valid(view.validity, offset) {
            push_null(state); // NULL crosses as the sentinel (F1)
        } else {
            match view.tag {
                TAG_F64 => ffi::lua_pushnumber(state, *view.data.cast::<f64>().add(offset)),
                TAG_I64 => ffi::lua_pushinteger(state, *view.data.cast::<i64>().add(offset)),
                // Keys read as their dictionary codes (F4).
                _ => ffi::lua_pushinteger(state, i64::from(*view.data.cast::<u32>().add(offset))),
            }
        }
        1
    }
}

/// Dispatches `v:mask()`, `v:sum()`, `v:text(i)`, `v:code_of(s)` — with
/// the type check at dispatch, so a wrong method is loud at the name.
unsafe fn input_method(state: *mut ffi::lua_State, view: InputPayload) -> c_int {
    unsafe {
        let mut len = 0usize;
        let name = ffi::lua_tolstring(state, 2, &mut len);
        match std::slice::from_raw_parts(name.cast::<u8>(), len) {
            b"mask" => {
                ffi::lua_pushcclosure(state, view_mask, 0);
                1
            }
            b"sum" if view.tag != TAG_KEY => {
                ffi::lua_pushcclosure(state, view_sum, 0);
                1
            }
            b"sum" => raise(
                state,
                c"sum() is a numeric-view method; keys are not arithmetic",
            ),
            b"text" if view.tag == TAG_KEY => {
                ffi::lua_pushcclosure(state, key_text, 0);
                1
            }
            b"code_of" if view.tag == TAG_KEY => {
                ffi::lua_pushcclosure(state, key_code_of, 0);
                1
            }
            b"text" | b"code_of" => raise(state, c"text() and code_of() are key-view methods"),
            _ => raise(state, c"no such view method"),
        }
    }
}

/// `#v`.
unsafe extern "C" fn input_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = input_arg(state) else {
            return 0;
        };
        ffi::lua_pushinteger(state, view.len as i64);
        1
    }
}

/// Input views are read-only.
unsafe extern "C" fn input_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe { raise(state, c"input views are read-only") }
}

/// `v:mask()` — the out-of-band validity view (elements read as
/// booleans), zero-copy over the same bitmap.
unsafe extern "C" fn view_mask(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = input_arg(state) else {
            return 0;
        };
        let slot = ffi::lua_newuserdatauv(state, std::mem::size_of::<MaskPayload>(), 0)
            .cast::<MaskPayload>();
        slot.write(MaskPayload {
            validity: view.validity,
            len: view.len,
            born: view.born,
            generation: view.generation,
        });
        ffi::luaL_setmetatable(state, META_MASK.as_ptr());
        1
    }
}

/// `v:sum()` — the null-skipping batch reduction, computed engine-side
/// so the sentinel is never materialized. Matches SQL SUM: all-NULL (or
/// empty) sums to NULL, and an `i64` sum overflows loudly rather than
/// silently widening.
unsafe extern "C" fn view_sum(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = input_arg(state) else {
            return 0;
        };
        // Method dispatch checks the tag, but a closure is detachable
        // (`local f = v.sum; f(k)`), so the check must live here too —
        // summing a key view would read u32 codes as i64s, past the
        // buffer.
        if view.tag == TAG_KEY {
            return raise(
                state,
                c"sum() is a numeric-view method; keys are not arithmetic",
            );
        }
        let mut seen = false;
        if view.tag == TAG_F64 {
            let mut sum = 0.0f64;
            for offset in 0..view.len {
                if is_valid(view.validity, offset) {
                    sum += *view.data.cast::<f64>().add(offset);
                    seen = true;
                }
            }
            if seen {
                ffi::lua_pushnumber(state, sum);
            } else {
                push_null(state);
            }
        } else {
            let mut sum = 0i64;
            for offset in 0..view.len {
                if is_valid(view.validity, offset) {
                    match sum.checked_add(*view.data.cast::<i64>().add(offset)) {
                        Some(next) => sum = next,
                        None => return raise(state, c"i64 sum overflows"),
                    }
                    seen = true;
                }
            }
            if seen {
                ffi::lua_pushinteger(state, sum);
            } else {
                push_null(state);
            }
        }
        1
    }
}

/// `v:text(i)` — decodes one key element to its string on demand (F4's
/// lazy text). A NULL element decodes to NULL.
unsafe extern "C" fn key_text(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = input_arg(state) else {
            return 0;
        };
        if view.tag != TAG_KEY {
            return raise(state, c"text() and code_of() are key-view methods");
        }
        let Ok(offset) = element_index(state, 2, view.len) else {
            return 0;
        };
        if !is_valid(view.validity, offset) {
            push_null(state);
            return 1;
        }
        let code = *view.data.cast::<u32>().add(offset);
        let dictionary = &*view.dictionary;
        if code as usize >= dictionary.len() {
            return raise(state, c"key code out of dictionary range");
        }
        let value = dictionary.value(code);
        ffi::lua_pushlstring(state, value.as_ptr().cast::<c_char>(), value.len());
        1
    }
}

/// `v:code_of(literal)` — resolves a literal to its code once (the
/// once-per-distinct-value pattern), or `nil` when the literal was
/// never interned (absence, not SQL NULL).
unsafe extern "C" fn key_code_of(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = input_arg(state) else {
            return 0;
        };
        if view.tag != TAG_KEY {
            return raise(state, c"text() and code_of() are key-view methods");
        }
        if ffi::lua_type(state, 2) != ffi::LUA_TSTRING {
            return raise(state, c"code_of expects a string literal");
        }
        let mut len = 0usize;
        let text = ffi::lua_tolstring(state, 2, &mut len);
        let bytes = std::slice::from_raw_parts(text.cast::<u8>(), len);
        // The dictionary holds only UTF-8, so a non-UTF-8 literal is
        // simply absent.
        match std::str::from_utf8(bytes)
            .ok()
            .and_then(|literal| (*view.dictionary).code_of(literal))
        {
            Some(code) => ffi::lua_pushinteger(state, i64::from(code)),
            None => ffi::lua_pushnil(state),
        }
        1
    }
}

// ---------------------------------------------------------------------
// Output-view accessors
// ---------------------------------------------------------------------

/// Fetches the checked output payload at argument 1.
unsafe fn output_arg(state: *mut ffi::lua_State) -> Result<OutputPayload, ()> {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_OUTPUT.as_ptr()).cast::<OutputPayload>();
        if payload.is_null() {
            raise(state, c"output accessor on a non-output");
            return Err(());
        }
        let view = *payload;
        if *view.generation != view.born {
            raise(state, c"view used outside its call");
            return Err(());
        }
        Ok(view)
    }
}

/// `out[i] = x` — the F3 coercion into the declared output type; `nil`
/// or `NULL` marks the slot NULL (the write a Lua table cannot hold).
unsafe extern "C" fn output_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = output_arg(state) else {
            return 0;
        };
        if ffi::lua_type(state, 2) != ffi::LUA_TNUMBER {
            return raise(state, c"output index must be an integer");
        }
        let Ok(offset) = element_index(state, 2, view.len) else {
            return 0;
        };
        match coerce(state, 3, view.tag, view.dictionary) {
            Ok(Coerced::F64(value)) => {
                *view.data.cast::<f64>().add(offset) = value;
                (*view.validity).set(offset, true);
            }
            Ok(Coerced::I64(value)) => {
                *view.data.cast::<i64>().add(offset) = value;
                (*view.validity).set(offset, true);
            }
            Ok(Coerced::KeyCode(code)) => {
                *view.data.cast::<u32>().add(offset) = code;
                (*view.validity).set(offset, true);
            }
            Ok(Coerced::Null) => (*view.validity).set(offset, false),
            Err(message) => return raise(state, message),
        }
        0
    }
}

/// `out[i]` — reads a written slot back (a key slot reads as its code;
/// a NULL or unwritten slot reads as NULL), so running kernels like
/// cumulative sums can consult their own output.
unsafe extern "C" fn output_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = output_arg(state) else {
            return 0;
        };
        let Ok(offset) = element_index(state, 2, view.len) else {
            return 0;
        };
        if !(*view.validity).get(offset) {
            push_null(state);
        } else {
            match view.tag {
                TAG_F64 => ffi::lua_pushnumber(state, *view.data.cast::<f64>().add(offset)),
                TAG_I64 => ffi::lua_pushinteger(state, *view.data.cast::<i64>().add(offset)),
                _ => ffi::lua_pushinteger(state, i64::from(*view.data.cast::<u32>().add(offset))),
            }
        }
        1
    }
}

/// `#out`.
unsafe extern "C" fn output_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(view) = output_arg(state) else {
            return 0;
        };
        ffi::lua_pushinteger(state, view.len as i64);
        1
    }
}

// ---------------------------------------------------------------------
// Mask accessors
// ---------------------------------------------------------------------

/// `m[i]` — `true` where the element is valid, `false` where NULL.
unsafe extern "C" fn mask_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_MASK.as_ptr()).cast::<MaskPayload>();
        if payload.is_null() {
            return raise(state, c"mask accessor on a non-mask");
        }
        let mask = *payload;
        if *mask.generation != mask.born {
            return raise(state, c"view used outside its call");
        }
        let Ok(offset) = element_index(state, 2, mask.len) else {
            return 0;
        };
        ffi::lua_pushboolean(state, c_int::from(is_valid(mask.validity, offset)));
        1
    }
}

/// `#m`.
unsafe extern "C" fn mask_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_MASK.as_ptr()).cast::<MaskPayload>();
        if payload.is_null() {
            return raise(state, c"mask accessor on a non-mask");
        }
        let mask = *payload;
        if *mask.generation != mask.born {
            return raise(state, c"view used outside its call");
        }
        ffi::lua_pushinteger(state, mask.len as i64);
        1
    }
}

/// Masks are read-only.
unsafe extern "C" fn mask_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe { raise(state, c"masks are read-only") }
}

// ---------------------------------------------------------------------
// NULL-sentinel metamethods
// ---------------------------------------------------------------------

/// Every arithmetic/bitwise/concat metamethod: the result is NULL (3VL).
unsafe extern "C" fn null_propagate(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        push_null(state);
        1
    }
}

/// `tostring(NULL)` — diagnostics only; the text never becomes a value.
unsafe extern "C" fn null_tostring(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let text = b"NULL";
        ffi::lua_pushlstring(state, text.as_ptr().cast::<c_char>(), text.len());
        1
    }
}

/// The data pointer a view userdata (input or output) carries — the
/// zero-copy proof hook, compared against the source buffer's pointer
/// in tests exactly like the engine's passthrough pointer checks.
///
/// # Safety
/// `raw` must be a state [`install`] has prepared.
pub(crate) unsafe fn view_data_pointer(
    raw: *mut ffi::lua_State,
    view_global: &CStr,
) -> Option<*const u8> {
    unsafe {
        ffi::lua_getglobal(raw, view_global.as_ptr());
        let input = ffi::luaL_testudata(raw, -1, META_INPUT.as_ptr());
        let pointer = if !input.is_null() {
            Some((*input.cast::<InputPayload>()).data)
        } else {
            let output = ffi::luaL_testudata(raw, -1, META_OUTPUT.as_ptr());
            (!output.is_null()).then(|| (*output.cast::<OutputPayload>()).data.cast_const())
        };
        ffi::lua_settop(raw, -2);
        pointer
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::{float_as_i64_exact, int_as_f64_exact};

    #[test]
    fn int_to_f64_edges() {
        assert_eq!(int_as_f64_exact(0), Some(0.0));
        assert_eq!(int_as_f64_exact(-1), Some(-1.0));
        // 2^53 is the last contiguous exact integer; 2^53 + 1 rounds.
        assert_eq!(int_as_f64_exact(1 << 53), Some(9_007_199_254_740_992.0));
        assert_eq!(int_as_f64_exact((1 << 53) + 1), None);
        // i64::MIN is a power of two — exact; i64::MAX rounds up to 2^63.
        assert_eq!(
            int_as_f64_exact(i64::MIN),
            Some(-9_223_372_036_854_775_808.0)
        );
        assert_eq!(int_as_f64_exact(i64::MAX), None);
    }

    #[test]
    fn float_to_i64_edges() {
        assert_eq!(float_as_i64_exact(0.0), Some(0));
        assert_eq!(float_as_i64_exact(-0.0), Some(0));
        assert_eq!(float_as_i64_exact(2.5), None);
        assert_eq!(float_as_i64_exact(-3.0), Some(-3));
        // The boundary: -2^63 is exactly i64::MIN; +2^63 is one past MAX.
        assert_eq!(
            float_as_i64_exact(-9_223_372_036_854_775_808.0),
            Some(i64::MIN)
        );
        assert_eq!(float_as_i64_exact(9_223_372_036_854_775_808.0), None);
        assert_eq!(float_as_i64_exact(f64::NAN), None);
        assert_eq!(float_as_i64_exact(f64::INFINITY), None);
        assert_eq!(float_as_i64_exact(f64::NEG_INFINITY), None);
    }
}
