//! SQL-in-Lua (Role 2, #70): scripts driving the embedder. The other
//! half of the bidirectional embed — Role 1 (Lua-in-SQL) runs kernels
//! *inside* queries; this seam lets a script *issue* them and feed
//! derived data back, completing the loop:
//!
//! ```text
//! local r, n = query("SELECT ts, px FROM ticks WHERE sym = 'ES'")
//! local rel = (r.px - rolling_mean(r.px, 64)) / rolling_mean(r.px, 64)
//! for i = 1, n do append("signals", { ts = r.ts[i], s = rel[i] }) end
//! ```
//!
//! ## The seam
//!
//! This crate stays backend-agnostic: it defines [`ScriptHost`] — the
//! two operations a driver script may ask of its embedder — and the
//! trampolines that expose them as the globals `query` and `append`.
//! *What* SQL means is the embedder's business (`engine` implements the
//! trait over its `Database`).
//!
//! ## The script surface
//!
//! - `query(sql)` — one SQL statement. A SELECT returns two values: a
//!   table mapping column names to **input views** (the same zero-copy
//!   views window kernels consume — f64/i64 elements, keys with
//!   `text()`/`code_of()`, `NULL` for null slots) and the row count.
//!   INSERT/UPDATE/DELETE return the affected-row count; statements
//!   with no result return `true`.
//! - `append(table, row)` — one row, exactly: the row table maps column
//!   names to Lua numbers (integers cross to `i64` columns exactly),
//!   strings (key columns), or the `NULL` sentinel. Every schema column
//!   must be present — `NULL` is spelled, never implied by absence.
//!   Returns the row's internal row id.
//!
//! ## Lifetime and re-entrancy
//!
//! Result views live exactly as long as the driving call: the embedder
//! holds each SELECT's buffers until the script returns, and the same
//! generation stamp that poisons kernel views poisons result views
//! after it. The globals exist in every state but are **refused with a
//! loud error outside [`LuaState::run_driver`]** — a window or scalar
//! kernel cannot re-enter the engine mid-query.
//!
//! [`LuaState::run_driver`]: crate::LuaState::run_driver

use crate::ffi;
use crate::values::{self, ColumnView};
use std::ffi::{c_char, c_int, c_void, CStr};

/// One value crossing from a script's row table into an append.
#[derive(Debug, Clone, PartialEq)]
pub enum ScriptValue {
    /// A Lua float — for `f64` columns.
    F64(f64),
    /// A Lua integer — exact, for `i64` columns.
    I64(i64),
    /// A Lua string — for key columns (interned on ingest).
    Text(String),
    /// The `NULL` sentinel.
    Null,
}

/// A SELECT's held result: owns the result buffers and lends them out
/// as views for the rest of the driving call.
pub trait ResultColumns: Send {
    /// Total rows.
    fn rows(&self) -> usize;
    /// Column names with views over the held buffers, in result order.
    fn columns(&self) -> Vec<(String, ColumnView<'_>)>;
}

/// What one driver statement produced.
pub enum SqlOutcome {
    /// A SELECT: held columns, alive until the script returns.
    Rows(Box<dyn ResultColumns>),
    /// INSERT/UPDATE/DELETE: rows affected.
    Affected(u64),
    /// A statement with no result (DDL an embedder chooses to allow).
    Done,
}

/// The embedder half of the driver seam. Implementations decide what
/// SQL runs and which statements are allowed; errors return as `Err`
/// strings, which scripts see as ordinary Lua errors.
pub trait ScriptHost {
    /// Runs one SQL statement.
    fn statement(&mut self, sql: &str) -> Result<SqlOutcome, String>;
    /// Appends one row (name–value pairs in arbitrary order) to a
    /// table, returning its internal row id.
    fn append(&mut self, table: &str, row: &[(String, ScriptValue)]) -> Result<u64, String>;
}

/// The active driving call the trampolines reach through the slot:
/// the host, the held SELECT results, and the generation cell views
/// are stamped with. Lives on `run_driver`'s stack; the raw host
/// pointer is valid exactly that long.
pub(crate) struct DriverCall {
    pub(crate) host: *mut dyn ScriptHost,
    pub(crate) held: Vec<Box<dyn ResultColumns>>,
    pub(crate) generation: *const u64,
}

/// The per-state slot: null except inside [`LuaState::run_driver`]
/// (which is what makes `query` in a kernel a loud error, not a
/// re-entrant call).
///
/// [`LuaState::run_driver`]: crate::LuaState::run_driver
pub(crate) struct DriverSlot(pub(crate) *mut DriverCall);

/// Installs the `query` and `append` globals, both reaching `slot`.
///
/// # Safety
/// `raw` must be a valid state with an empty stack; `slot` must stay
/// valid for the state's whole life.
pub(crate) unsafe fn install(raw: *mut ffi::lua_State, slot: *mut DriverSlot) {
    unsafe {
        for (name, function) in [
            (c"query".as_ptr(), driver_query as ffi::lua_CFunction),
            (c"append".as_ptr(), driver_append as ffi::lua_CFunction),
        ] {
            ffi::lua_pushlightuserdata(raw, slot.cast::<c_void>());
            ffi::lua_pushcclosure(raw, function, 1);
            ffi::lua_setglobal(raw, name);
        }
    }
}

/// The active call, or a raised refusal (kernels, and scripts run
/// through the kernel entry points, land here).
unsafe fn active_call(state: *mut ffi::lua_State) -> Result<*mut DriverCall, ()> {
    unsafe {
        let slot = ffi::lua_touserdata(state, ffi::lua_upvalueindex(1)).cast::<DriverSlot>();
        let call = (*slot).0;
        if call.is_null() {
            values::raise(
                state,
                c"query/append drive the engine from a script (run_driver); \
                  kernels cannot re-enter the engine",
            );
            return Err(());
        }
        Ok(call)
    }
}

/// The string at `idx`, without conversion (numbers are not silently
/// stringified). Returns a raised error for non-strings.
unsafe fn string_argument<'a>(
    state: *mut ffi::lua_State,
    idx: c_int,
    what: &'static CStr,
) -> Result<&'a str, ()> {
    unsafe {
        if ffi::lua_type(state, idx) != ffi::LUA_TSTRING {
            values::raise(state, what);
            return Err(());
        }
        let mut len = 0usize;
        let text = ffi::lua_tolstring(state, idx, &mut len);
        match std::str::from_utf8(std::slice::from_raw_parts(text.cast::<u8>(), len)) {
            Ok(text) => Ok(text),
            Err(_) => {
                values::raise(state, c"argument is not valid UTF-8");
                Err(())
            }
        }
    }
}

/// Pushes a dynamic error message, drops it, and raises — the
/// `host_call` discipline: the `longjmp` crosses no live destructor.
/// (The push's only failure mode is OOM, which would leak the String —
/// bounded, never unsound.)
unsafe fn raise_message(state: *mut ffi::lua_State, message: String) -> c_int {
    unsafe {
        ffi::lua_pushlstring(state, message.as_ptr().cast::<c_char>(), message.len());
        drop(message);
        ffi::lua_error(state)
    }
}

/// `query(sql)` — see the module docs for the result shapes.
unsafe extern "C" fn driver_query(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(call) = active_call(state) else {
            return 0;
        };
        if ffi::lua_gettop(state) != 1 {
            return values::raise(state, c"query takes one SQL string");
        }
        let Ok(sql) = string_argument(state, 1, c"query takes one SQL string") else {
            return 0;
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (*(*call).host).statement(sql)
        }));
        match outcome {
            Ok(Ok(SqlOutcome::Affected(count))) => {
                ffi::lua_pushinteger(state, count as i64);
                1
            }
            Ok(Ok(SqlOutcome::Done)) => {
                ffi::lua_pushboolean(state, 1);
                1
            }
            Ok(Ok(SqlOutcome::Rows(result))) => {
                // Table + one string and one userdata per column, plus
                // the row count. Reserved BEFORE the `columns` Vec below
                // exists, so this refusal's raise crosses no destructor.
                if ffi::lua_checkstack(state, 6) == 0 {
                    return values::raise(state, c"query result overflows the Lua stack");
                }
                // Park the result: the Box gives its buffers a stable
                // address while the Vec grows, and `run_driver` drops
                // them only after the generation bump poisons every
                // view handed out below.
                (*call).held.push(result);
                let result = (*call).held.last().expect("just pushed").as_ref();
                // `rows`/`columns` are embedder code behind a public
                // trait, so they run contained like every other embedder
                // entry point — a panicking implementation must not
                // unwind into C (which aborts the process).
                let described = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (result.rows(), result.columns())
                }));
                let Ok((rows, columns)) = described else {
                    return values::raise(state, c"query result columns panicked");
                };
                ffi::lua_createtable(state, 0, columns.len() as c_int);
                for (name, view) in &columns {
                    // The column's own name, so a malformed result says
                    // which column is malformed.
                    let named = std::ffi::CString::new(name.as_str())
                        .unwrap_or_else(|_| c"result".to_owned());
                    ffi::lua_pushlstring(state, name.as_ptr().cast::<c_char>(), name.len());
                    if let Err(message) =
                        values::push_input(state, (*call).generation, &named, view)
                    {
                        drop(named);
                        drop(columns);
                        return raise_message(state, message);
                    }
                    drop(named);
                    ffi::lua_settable(state, -3);
                }
                drop(columns);
                ffi::lua_pushinteger(state, rows as i64);
                2
            }
            Ok(Err(message)) => raise_message(state, message),
            Err(payload) => {
                drop(payload);
                values::raise(state, c"query host panicked")
            }
        }
    }
}

/// `append(table, row)` — see the module docs.
unsafe extern "C" fn driver_append(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let Ok(call) = active_call(state) else {
            return 0;
        };
        if ffi::lua_gettop(state) != 2 || ffi::lua_type(state, 2) != ffi::LUA_TTABLE {
            return values::raise(state, c"append takes a table name and a row table");
        }
        let Ok(table) = string_argument(state, 1, c"append takes a table name and a row table")
        else {
            return 0;
        };
        // Collect the row before calling out. Traversal itself raises
        // only on OOM (the bounded-leak note above); malformed entries
        // are collected as errors and raised after the Vec drops.
        let mut row: Vec<(String, ScriptValue)> = Vec::new();
        let mut malformed: Option<&'static CStr> = None;
        ffi::lua_pushnil(state);
        while ffi::lua_next(state, 2) != 0 {
            if ffi::lua_type(state, -2) != ffi::LUA_TSTRING {
                malformed = Some(c"append: row keys are column names (strings)");
                ffi::lua_settop(state, -3);
                break;
            }
            let mut len = 0usize;
            let text = ffi::lua_tolstring(state, -2, &mut len);
            let Ok(name) = std::str::from_utf8(std::slice::from_raw_parts(text.cast::<u8>(), len))
            else {
                malformed = Some(c"append: column name is not valid UTF-8");
                ffi::lua_settop(state, -3);
                break;
            };
            let value = if values::is_null_sentinel(state, -1) {
                Some(ScriptValue::Null)
            } else {
                match ffi::lua_type(state, -1) {
                    ffi::LUA_TNUMBER => {
                        if ffi::lua_isinteger(state, -1) == 1 {
                            let mut ok = 0;
                            Some(ScriptValue::I64(ffi::lua_tointegerx(state, -1, &mut ok)))
                        } else {
                            let mut ok = 0;
                            Some(ScriptValue::F64(ffi::lua_tonumberx(state, -1, &mut ok)))
                        }
                    }
                    ffi::LUA_TSTRING => {
                        let mut len = 0usize;
                        let text = ffi::lua_tolstring(state, -1, &mut len);
                        std::str::from_utf8(std::slice::from_raw_parts(text.cast::<u8>(), len))
                            .ok()
                            .map(|text| ScriptValue::Text(text.to_owned()))
                    }
                    _ => None,
                }
            };
            match value {
                Some(value) => row.push((name.to_owned(), value)),
                None => {
                    malformed = Some(
                        c"append: row values are numbers, strings, or NULL \
                          (views and vectors do not append whole; index one element)",
                    );
                    ffi::lua_settop(state, -3);
                    break;
                }
            }
            ffi::lua_settop(state, -2); // pop the value, keep the key
        }
        if let Some(message) = malformed {
            drop(row);
            return values::raise(state, message);
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (*(*call).host).append(table, &row)
        }));
        drop(row);
        match outcome {
            Ok(Ok(row_id)) => {
                ffi::lua_pushinteger(state, row_id as i64);
                1
            }
            Ok(Err(message)) => raise_message(state, message),
            Err(payload) => {
                drop(payload);
                values::raise(state, c"append host panicked")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The driver seam, unit-proven against a fake host: the result
    //! shapes, exactness across the boundary in both directions, the
    //! kernel refusal, and result-view poisoning.

    use super::*;
    use crate::values::ReturnType;
    use crate::LuaState;
    use arrow_lite::Dictionary;
    use std::sync::{Arc, Mutex};

    /// Everything `append` received: `(table, row)` per call.
    type Appended = Vec<(String, Vec<(String, ScriptValue)>)>;

    /// A fixed SELECT result: one f64, one i64 (beyond 2^53), one key.
    struct FakeResult {
        px: Vec<f64>,
        ts: Vec<i64>,
        codes: Vec<u32>,
        dictionary: Dictionary,
    }

    impl ResultColumns for FakeResult {
        fn rows(&self) -> usize {
            self.px.len()
        }
        fn columns(&self) -> Vec<(String, ColumnView<'_>)> {
            vec![
                (
                    "px".to_owned(),
                    ColumnView::F64 {
                        values: &self.px,
                        validity: None,
                    },
                ),
                (
                    "ts".to_owned(),
                    ColumnView::I64 {
                        values: &self.ts,
                        validity: None,
                    },
                ),
                (
                    "sym".to_owned(),
                    ColumnView::Key {
                        codes: &self.codes,
                        validity: None,
                        dictionary: &self.dictionary,
                    },
                ),
            ]
        }
    }

    /// Records every statement and append; SELECTs return the fixture.
    struct FakeHost {
        appended: Arc<Mutex<Appended>>,
    }

    impl ScriptHost for FakeHost {
        fn statement(&mut self, sql: &str) -> Result<SqlOutcome, String> {
            if sql.starts_with("SELECT") {
                let mut dictionary = Dictionary::new();
                let codes = vec![dictionary.intern("ES"), dictionary.intern("NQ")];
                Ok(SqlOutcome::Rows(Box::new(FakeResult {
                    px: vec![101.5, 99.25],
                    ts: vec![(1 << 53) + 1, (1 << 53) + 2],
                    codes,
                    dictionary,
                })))
            } else if sql.starts_with("DELETE") {
                Ok(SqlOutcome::Affected(3))
            } else if sql.starts_with("CREATE") {
                Ok(SqlOutcome::Done)
            } else {
                Err(format!("unsupported statement: {sql}"))
            }
        }

        fn append(&mut self, table: &str, row: &[(String, ScriptValue)]) -> Result<u64, String> {
            let mut appended = self.appended.lock().unwrap();
            appended.push((table.to_owned(), row.to_vec()));
            Ok(41 + appended.len() as u64)
        }
    }

    fn drive(state: &mut LuaState, host: &mut FakeHost, source: &str) -> Result<(), String> {
        let chunk = state.compile(source)?;
        state.run_driver(&chunk, host)
    }

    #[test]
    fn a_script_queries_computes_and_feeds_back_exactly() {
        let appended = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            appended: Arc::clone(&appended),
        };
        let mut state = LuaState::new().unwrap();
        drive(
            &mut state,
            &mut host,
            r#"
                local r, n = query("SELECT px, ts, sym FROM ticks")
                assert(n == 2)
                assert(#r.px == 2 and r.px[1] == 101.5)
                assert(r.ts[2] == (1 << 53) + 2)  -- i64 exact through the view
                assert(r.sym:text(1) == "ES" and r.sym:text(2) == "NQ")
                local double = r.px + r.px        -- the vectorized vocabulary
                for i = 1, n do
                    append("signals", { ts = r.ts[i], v = double[i], sym = r.sym:text(i), gap = NULL })
                end
                assert(query("DELETE FROM ticks WHERE px < 0") == 3)
                assert(query("CREATE TABLE x (ts BIGINT ORDERING KEY)") == true)
            "#,
        )
        .unwrap();
        let appended = appended.lock().unwrap();
        assert_eq!(appended.len(), 2);
        let (table, row) = &appended[0];
        assert_eq!(table, "signals");
        let field = |name: &str| {
            row.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .unwrap()
        };
        assert_eq!(field("ts"), ScriptValue::I64((1 << 53) + 1)); // exact
        assert_eq!(field("v"), ScriptValue::F64(203.0));
        assert_eq!(field("sym"), ScriptValue::Text("ES".to_owned()));
        assert_eq!(field("gap"), ScriptValue::Null);
    }

    #[test]
    fn kernels_cannot_reenter_the_engine() {
        let mut state = LuaState::new().unwrap();
        let chunk = state.compile("return query('SELECT 1')").unwrap();
        let error = state.eval_scalar(&chunk, &[], ReturnType::F64).unwrap_err();
        assert!(
            error.contains("kernels cannot re-enter the engine"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn result_views_poison_when_the_driving_call_ends() {
        let appended = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            appended: Arc::clone(&appended),
        };
        let mut state = LuaState::new().unwrap();
        drive(
            &mut state,
            &mut host,
            "local r = query('SELECT px FROM t'); stash = r.px",
        )
        .unwrap();
        let error = drive(&mut state, &mut host, "return stash[1]").unwrap_err();
        assert!(
            error.contains("outside its call"),
            "stale result view must be poisoned, got: {error}"
        );
    }

    #[test]
    fn host_errors_and_misuse_are_loud_lua_errors() {
        let appended = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            appended: Arc::clone(&appended),
        };
        let mut state = LuaState::new().unwrap();
        let error = drive(&mut state, &mut host, "query('UPDATE nope')").unwrap_err();
        assert!(error.contains("unsupported statement"), "got: {error}");
        let error = drive(&mut state, &mut host, "query(42)").unwrap_err();
        assert!(error.contains("one SQL string"), "got: {error}");
        let error = drive(
            &mut state,
            &mut host,
            "local r = query('SELECT px FROM t'); append('t', { px = r.px })",
        )
        .unwrap_err();
        assert!(
            error.contains("index one element"),
            "whole views must not append silently, got: {error}"
        );
        // The state survives all of it.
        drive(&mut state, &mut host, "assert(1 + 1 == 2)").unwrap();
    }
}
