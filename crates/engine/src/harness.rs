//! `extern "C"` hooks for the M1 end-to-end oracle script (dev/CI only).
//!
//! Compiled only under the `oracle-harness` feature so
//! `tests/m1_slice_oracle.py` can drive the whole vertical slice from
//! Python: build with `cargo build -p engine --features oracle-harness`,
//! then run the script against `target/debug/libengine.so`.
//!
//! Both hooks build the *same* deterministic table (a fixed
//! linear-congruential generator — no ambient randomness, so every run
//! and both hooks see identical data), append its rows one at a time
//! through the real ingest path, run real SQL, and export through the
//! real `ArrowArrayStream` doorway. The script then recomputes every
//! window with `np.linalg.lstsq` (and DuckDB's `regr_slope`, when
//! available) and diffs. That external recomputation — not this crate's
//! own tests — is what earns M1's "compute proven" cross-check.

use crate::database::Database;
use crate::table::Table;
use arrow_lite::{ArrowArrayStream, ColumnType, Field, Schema};
use storage_lite::RowValue;

/// Rows in the fixture.
const ROWS: i64 = 240;
/// The window: 19 preceding + current = 20 rows.
const PRECEDING: usize = 19;
/// Segment-row threshold: small enough that the fixture spans several
/// frozen segments plus a live write-buffer tail (240 rows → 3 × 64
/// frozen + 48 live), so the oracle exercises the multi-segment,
/// multi-batch path — windows spanning segment boundaries, per-segment
/// dictionaries — not just the M1 single-segment shape.
const SEGMENT_ROWS: usize = 64;

/// A fixed LCG (numerical recipes constants) so the fixture is identical
/// everywhere.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 53 bits → [0, 1).
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The fixture: three symbols with different underlying lines plus
/// deterministic noise, interleaved on an increasing ordering key —
/// ingested into a persistent table, flushed, closed, and **reopened
/// from disk**, so the oracle's cross-check covers the full storage
/// round trip (encode → backend → decode), not just the in-memory path.
fn fixture_table() -> Table {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
        Field::new("y", ColumnType::F64, false),
    ]);
    let dir = std::env::temp_dir().join(format!("tallydb-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut table =
        Table::persistent_with_segment_rows("trades", schema.clone(), "ts", &dir, SEGMENT_ROWS)
            .expect("fixture schema is valid");
    let mut rng = Lcg(0x5EED_1234_5678_9ABC);
    let symbols = [
        ("AAPL", 2.0, 5.0),
        ("MSFT", -0.75, 12.0),
        ("TSLA", 0.1, -3.0),
    ];
    for i in 0..ROWS {
        let (sym, slope, intercept) = symbols[(i % 3) as usize];
        let x = rng.next_f64() * 10.0;
        let noise = (rng.next_f64() - 0.5) * 0.2;
        let y = slope * x + intercept + noise;
        table
            .append(&[
                RowValue::I64(i),
                RowValue::Key(sym),
                RowValue::F64(x),
                RowValue::F64(y),
            ])
            .expect("fixture rows are valid");
    }
    table.flush().expect("fixture flush succeeds");
    drop(table);
    Table::persistent_with_segment_rows("trades", schema, "ts", &dir, SEGMENT_ROWS)
        .expect("fixture reopens from disk")
}

fn export(sql: &str, out: *mut ArrowArrayStream) {
    let table = fixture_table();
    match table.query_stream(sql) {
        // SAFETY: the caller (the oracle script) provides a valid,
        // writable destination struct.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("fixture query failed: {error}"),
    }
}

/// Exports the fixture's raw inputs (`ts, sym, x, y`).
///
/// # Safety
/// `out` must be valid, writable, and not hold a live export.
#[no_mangle]
pub unsafe extern "C" fn tallydb_m1_inputs_stream(out: *mut ArrowArrayStream) {
    export("SELECT ts, sym, x, y FROM trades", out);
}

/// Exports the rolling regression the engine computed: per-symbol
/// `regr_slope` / `regr_intercept` over `ROWS BETWEEN 19 PRECEDING AND
/// CURRENT ROW`.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_m1_regression_stream(out: *mut ArrowArrayStream) {
    export(
        "SELECT ts, sym, \
         regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS slope, \
         regr_intercept(y, x) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS intercept \
         FROM trades",
        out,
    );
}

/// The window size, so the script never hard-codes it out of sync.
#[no_mangle]
pub extern "C" fn tallydb_m1_window_preceding() -> u64 {
    PRECEDING as u64
}

/// The corpus fixture: 5,000 rows of the telemetry family (seed 24) —
/// jittered 1s cadence, 8 sensors, ~1% late arrivals, ~2% nulls in
/// `y` — ingested through the real append path and **compacted**, so
/// the ~1% disorder is resolved the way the design resolves it and
/// every query shape (windows included) runs. `ts` values are unique by
/// construction (cadence far exceeds jitter), which is what lets the
/// differential compare under `ORDER BY ts` as a total order.
fn corpus_table() -> Table {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
        Field::new("y", ColumnType::F64, true),
    ]);
    let mut table =
        Table::with_segment_rows("corpus", schema, "ts", 512).expect("corpus schema is valid");
    for row in corpus::Spec::telemetry(5_000, 24).generate() {
        let label = corpus::key_label(row.key);
        table
            .append(&[
                RowValue::I64(row.ts),
                RowValue::Key(&label),
                RowValue::F64(row.value),
                row.aux.map_or(RowValue::Null, RowValue::F64),
            ])
            .expect("corpus rows are valid");
    }
    table.compact().expect("corpus compaction succeeds");
    table
}

/// The corpus database: the fact table plus a `sensors` dimension —
/// seven of the eight sensors (K007 is deliberately missing, so INNER
/// and LEFT joins differ), each with a site label and a calibration
/// factor, split across segments so dictionary codes differ per side.
fn corpus_database() -> Database {
    let mut database = Database::new();
    database
        .add_table(corpus_table())
        .expect("fact table registers");
    let schema = Schema::new(vec![
        Field::new("id", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("site", ColumnType::Key, false),
        Field::new("calib", ColumnType::F64, false),
    ]);
    let mut sensors =
        Table::with_segment_rows("sensors", schema, "id", 3).expect("dimension schema is valid");
    for sensor in 0..7u32 {
        let label = corpus::key_label(sensor);
        sensors
            .append(&[
                RowValue::I64(i64::from(sensor)),
                RowValue::Key(&label),
                RowValue::Key(["north", "south", "east"][sensor as usize % 3]),
                RowValue::F64(0.5 + f64::from(sensor) * 0.25),
            ])
            .expect("dimension rows are valid");
    }
    database.add_table(sensors).expect("dimension registers");
    database
}

/// Exports the dimension table's rows, for the differential script to
/// replicate into DuckDB.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_corpus_dimension_stream(out: *mut ArrowArrayStream) {
    let database = corpus_database();
    match database.query_stream("SELECT id, sym, site, calib FROM sensors") {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("dimension export failed: {error}"),
    }
}

/// Exports the corpus fixture's raw rows (`ts, sym, x, y`), for the
/// differential script to replicate into DuckDB.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_corpus_inputs_stream(out: *mut ArrowArrayStream) {
    let table = corpus_table();
    match table.query_stream("SELECT ts, sym, x, y FROM corpus") {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("corpus export failed: {error}"),
    }
}

/// Runs one SQL statement (NUL-terminated UTF-8) against the corpus
/// fixture and exports the result. Returns 0 on success; on failure
/// prints the error to stderr and returns 1 with `out` untouched — the
/// differential script treats that as a failed query, loudly.
///
/// # Safety
/// `sql` must be a valid NUL-terminated string and `out` a valid,
/// writable destination not holding a live export.
#[no_mangle]
pub unsafe extern "C" fn tallydb_corpus_query_stream(
    sql: *const std::os::raw::c_char,
    out: *mut ArrowArrayStream,
) -> i32 {
    // SAFETY: caller contract — a valid NUL-terminated string.
    let sql = match unsafe { std::ffi::CStr::from_ptr(sql) }.to_str() {
        Ok(sql) => sql,
        Err(_) => {
            eprintln!("tallydb_corpus_query_stream: SQL is not UTF-8");
            return 1;
        }
    };
    let database = corpus_database();
    match database.query_stream(sql) {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => {
            unsafe { out.write(stream) };
            0
        }
        Err(error) => {
            eprintln!("tallydb_corpus_query_stream: {sql}: {error}");
            1
        }
    }
}

/// The Lua-window family's frame: 9 preceding + current = 10 rows
/// (deliberately different from the M1 regression window, so the two
/// oracles cannot mask each other).
const LUA_PRECEDING: usize = 9;

/// The Lua-window family's frame size, so the oracle script never
/// hard-codes it out of sync.
#[no_mangle]
pub extern "C" fn tallydb_lua_window_preceding() -> u64 {
    LUA_PRECEDING as u64
}

/// The Lua-window oracle family (M2.7): four application-registered
/// kernels — a pure-Lua MAD, a kernel calling the BLAS `dot` host
/// function, an `I64`-declared count, and a kernel that returns `NULL`
/// on short windows — run as partitioned SQL windows over the M1
/// fixture (persistent, reopened from disk, multi-segment), exporting
/// inputs and outputs together. The oracle script re-derives every
/// window with NumPy and diffs.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_lua_window_stream(out: *mut ArrowArrayStream) {
    let mut table = fixture_table();
    table
        .register_lua_window(
            "lua_mad",
            &["x"],
            "local n = #x\n\
             local mean = 0.0\n\
             for i = 1, n do mean = mean + x[i] end\n\
             mean = mean / n\n\
             local mad = 0.0\n\
             for i = 1, n do mad = mad + math.abs(x[i] - mean) end\n\
             return mad / n",
            ColumnType::F64,
        )
        .expect("lua_mad registers");
    table
        .register_lua_window("lua_wdot", &["y", "x"], "return dot(y, x)", ColumnType::F64)
        .expect("lua_wdot registers");
    table
        .register_lua_window(
            "lua_npos",
            &["x"],
            "local n = 0\nfor i = 1, #x do if x[i] > 5.0 then n = n + 1 end end\nreturn n",
            ColumnType::I64,
        )
        .expect("lua_npos registers");
    table
        .register_lua_window(
            "lua_spread",
            &["x"],
            "if #x < 3 then return NULL end\n\
             local lo, hi = x[1], x[1]\n\
             for i = 2, #x do\n\
               local v = x[i]\n\
               if v < lo then lo = v end\n\
               if v > hi then hi = v end\n\
             end\n\
             return hi - lo",
            ColumnType::F64,
        )
        .expect("lua_spread registers");
    let frame = format!(
        "OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN {LUA_PRECEDING} PRECEDING \
         AND CURRENT ROW)"
    );
    let sql = format!(
        "SELECT ts, sym, x, y, \
         lua_mad(x) {frame} AS mad, \
         lua_wdot(y, x) {frame} AS wdot, \
         lua_npos(x) {frame} AS npos, \
         lua_spread(x) {frame} AS spread \
         FROM trades"
    );
    match table.query_stream(&sql) {
        // SAFETY: the caller (the oracle script) provides a valid,
        // writable destination struct.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("lua-window fixture query failed: {error}"),
    }
}

/// An open benchmark context: one prebuilt table, so the timed calls
/// measure query + export only — never fixture construction.
pub struct BenchContext {
    table: Table,
}

/// The benchmark fixture's segment threshold — production-shaped (many
/// windows per segment), unlike the oracle fixtures' deliberately tiny
/// segments.
const BENCH_SEGMENT_ROWS: usize = 4096;

/// The pure-Lua mean-absolute-deviation kernel the bench registers —
/// the interpreter-only case (no native op involved).
const BENCH_MAD: &str = "local n = #x\n\
                         local mean = 0.0\n\
                         for i = 1, n do mean = mean + x[i] end\n\
                         mean = mean / n\n\
                         local mad = 0.0\n\
                         for i = 1, n do mad = mad + math.abs(x[i] - mean) end\n\
                         return mad / n";

/// Builds the benchmark table — `rows` rows, 8 symbols, strictly
/// increasing `ts`, non-null LCG-generated `x`/`y` — and registers the
/// Lua kernels the benchmark queries: `lua_dot(y, x)` (a kernel calling
/// the BLAS `dot` host function) and `lua_mad(x)` (pure interpreter).
/// Returns an owned context; release it with [`tallydb_bench_close`].
#[no_mangle]
pub extern "C" fn tallydb_bench_open(rows: u64) -> *mut BenchContext {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
        Field::new("y", ColumnType::F64, false),
    ]);
    let mut table = Table::with_segment_rows("bench", schema, "ts", BENCH_SEGMENT_ROWS)
        .expect("bench schema is valid");
    let mut rng = Lcg(0xBE9C_11FE_D0D0_5EED);
    for i in 0..rows as i64 {
        let label = format!("K{:03}", i % 8);
        let x = rng.next_f64() * 10.0;
        let y = 2.0 * x + (rng.next_f64() - 0.5);
        table
            .append(&[
                RowValue::I64(i),
                RowValue::Key(&label),
                RowValue::F64(x),
                RowValue::F64(y),
            ])
            .expect("bench rows are valid");
    }
    table
        .register_lua_window("lua_dot", &["y", "x"], "return dot(y, x)", ColumnType::F64)
        .expect("lua_dot registers");
    table
        .register_lua_window("lua_mad", &["x"], BENCH_MAD, ColumnType::F64)
        .expect("lua_mad registers");
    Box::into_raw(Box::new(BenchContext { table }))
}

/// Runs one SQL statement against the context's table and exports the
/// result. Returns 0 on success; on failure prints to stderr and
/// returns 1 with `out` untouched.
///
/// # Safety
/// `context` must come from [`tallydb_bench_open`] and not be closed;
/// `sql` must be a valid NUL-terminated string; `out` a valid, writable
/// destination not holding a live export.
#[no_mangle]
pub unsafe extern "C" fn tallydb_bench_query(
    context: *mut BenchContext,
    sql: *const std::os::raw::c_char,
    out: *mut ArrowArrayStream,
) -> i32 {
    // SAFETY: caller contract — a valid NUL-terminated string.
    let sql = match unsafe { std::ffi::CStr::from_ptr(sql) }.to_str() {
        Ok(sql) => sql,
        Err(_) => {
            eprintln!("tallydb_bench_query: SQL is not UTF-8");
            return 1;
        }
    };
    // SAFETY: caller contract — a live context from tallydb_bench_open.
    let table = unsafe { &(*context).table };
    match table.query_stream(sql) {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => {
            unsafe { out.write(stream) };
            0
        }
        Err(error) => {
            eprintln!("tallydb_bench_query: {sql}: {error}");
            1
        }
    }
}

/// Releases a benchmark context.
///
/// # Safety
/// `context` must come from [`tallydb_bench_open`] and not have been
/// closed already.
#[no_mangle]
pub unsafe extern "C" fn tallydb_bench_close(context: *mut BenchContext) {
    // SAFETY: caller contract — exactly one close per open.
    drop(unsafe { Box::from_raw(context) });
}

/// The mutation sequence the differential oracle replays in DuckDB.
/// KEEP IN SYNC with `MUTATIONS` in `tests/m2_mutation_oracle.py` — a
/// mismatch fails the oracle loudly, it cannot pass silently.
const MUTATIONS: &[&str] = &[
    "DELETE FROM trades WHERE sym = 'TSLA'",
    "DELETE FROM trades WHERE ts >= 220",
    "UPDATE trades SET y = 0 WHERE x < 2 AND sym IN ('AAPL', 'MSFT')",
    "UPDATE trades SET x = 5.5 WHERE ts < 30 AND sym <> 'MSFT'",
];

/// Exports the fixture after the scripted `UPDATE`/`DELETE` sequence and
/// a compaction — the end state the DuckDB differential diffs.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_m2_mutated_stream(out: *mut ArrowArrayStream) {
    let mut table = fixture_table();
    for statement in MUTATIONS {
        table
            .mutate(statement)
            .unwrap_or_else(|error| panic!("fixture mutation '{statement}' failed: {error}"));
    }
    table.compact().expect("fixture compaction succeeds");
    match table.query_stream("SELECT ts, sym, x, y FROM trades") {
        // SAFETY: the caller (the oracle script) provides a valid,
        // writable destination struct.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("mutated fixture query failed: {error}"),
    }
}
