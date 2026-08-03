//! `extern "C"` hooks for the Python oracle scripts (dev/CI only).
//!
//! Compiled only under the `oracle-harness` feature so the scripts in
//! `tests/` can drive the engine from Python: build with `cargo build
//! -p engine --features oracle-harness`, then run a script against
//! `target/debug/libengine.so`. Every hook here is called by name from
//! one of them — `m1_slice_oracle`, `m2_mutation_oracle`,
//! `m2_differential_oracle`, `m2_lua_window_oracle` (which also covers
//! the SQL-in-Lua driver pipeline), `m4_asof_oracle`,
//! `m5_view_oracle`, and the latency benchmark — so nothing in this
//! module is reachable from Rust.
//!
//! The fixtures are deterministic (a fixed linear-congruential
//! generator — no ambient randomness, so every run and every hook sees
//! identical data). Rows are appended one at a time through the real
//! ingest path; queries run real SQL; results leave through the real
//! `ArrowArrayStream` doorway; and the persistent fixtures are closed
//! and reopened from disk first, so a cross-check covers the storage
//! round trip too. The scripts then recompute independently — NumPy,
//! DuckDB — and diff. That external recomputation, not this crate's own
//! tests, is what earns the cross-check claims.

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

/// The corpus's quote history: the as-of join's dimension side (#65).
///
/// A second telemetry draw (seed 91, half the rows) over the same eight
/// sensor labels, so it interleaves with the corpus in time without
/// lining up with it: most corpus rows fall strictly between two
/// quotes, which is the case an as-of join exists for. Its time column
/// is `qts`, not `ts` — a dimension attribute sharing a fact column's
/// name is refused, and renaming is how a desk gets past that today.
///
/// Two shapes are injected, both **per symbol**, because that is the
/// only place either one means anything — the generator interleaves
/// eight keys, so its own 1s-scale disorder almost never displaces a
/// row past another row of the same key.
///
/// - Every 17th row repeats its symbol's previous timestamp, so the
///   tie rule (the last of them wins) is exercised rather than avoided.
/// - Every 29th row arrives 30s stale, well behind several of its own
///   symbol's earlier quotes, so the history reaches the join out of
///   order and the sort inside the index does real work.
///
/// The oracle script counts the ties before it trusts the families, so
/// the first claim re-earns itself every run; the second is covered by
/// the differential itself, which fails if the sort is removed.
fn quotes_table() -> Table {
    let schema = Schema::new(vec![
        Field::new("qts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("q", ColumnType::F64, false),
    ]);
    let mut table =
        Table::with_segment_rows("quotes", schema, "qts", 256).expect("quote schema is valid");
    // How stale a late quote is: ~30s, several of its own symbol's
    // quotes back at this family's ~8s per-symbol spacing.
    const LATE: i64 = 30_000_000_000;
    let mut latest: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
    for (index, row) in corpus::Spec::telemetry(2_500, 91)
        .generate()
        .into_iter()
        .enumerate()
    {
        let label = corpus::key_label(row.key);
        let previous = latest.get(&row.key).copied();
        let qts = match previous {
            Some(earlier) if index % 17 == 0 => earlier,
            Some(_) if index % 29 == 0 => row.ts - LATE,
            _ => row.ts,
        };
        // A late arrival does not move the symbol's clock forward.
        latest.insert(row.key, previous.map_or(qts, |earlier| earlier.max(qts)));
        table
            .append(&[
                RowValue::I64(qts),
                RowValue::Key(&label),
                RowValue::F64(row.value),
            ])
            .expect("quote rows are valid");
    }
    table
}

/// The corpus database: the fact table, a `sensors` dimension —
/// seven of the eight sensors (K007 is deliberately missing, so INNER
/// and LEFT joins differ), each with a site label and a calibration
/// factor, split across segments so dictionary codes differ per side —
/// and the `quotes` history the as-of families join against.
fn corpus_database() -> Database {
    let mut database = Database::new();
    database
        .add_table(corpus_table())
        .expect("fact table registers");
    database
        .add_table(quotes_table())
        .expect("quote history registers");
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

/// Exports the quote history's rows **in storage order**, for the
/// differential script to replicate into DuckDB.
///
/// Storage order is load-bearing here, not incidental: it is what
/// breaks a tie between quotes sharing a timestamp, so the referee has
/// to number the rows the same way the engine sees them.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_corpus_quotes_stream(out: *mut ArrowArrayStream) {
    let database = corpus_database();
    match database.query_stream("SELECT qts, sym, q FROM quotes") {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("quote export failed: {error}"),
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
/// kernels — a pure-Lua MAD, a kernel calling the `dot` host
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
    // The composed column kernels (option A): operators and rolling
    // combinators vectorized in native code, one interpreter entry per
    // query — including across segment boundaries, which the fixture's
    // 64-row segments exercise.
    table
        .register_lua_scalar("lua_rel", &["a", "b"], "return (a - b) / b")
        .expect("lua_rel registers");
    table
        .register_lua_scalar(
            "lua_rdot",
            &["a", "b"],
            &format!("return rolling_dot(a, b, {})", LUA_PRECEDING + 1),
        )
        .expect("lua_rdot registers");
    // M5.0's streaming primitives and one prelude composition, over
    // the same reopened multi-segment fixture: NumPy re-derives each.
    for (name, chunk) in [
        (
            "lua_rvar",
            format!("return rolling_var(a, {})", LUA_PRECEDING + 1),
        ),
        (
            "lua_rstd",
            format!("return rolling_std(a, {})", LUA_PRECEDING + 1),
        ),
        ("lua_lag", "return lag(a, 3)".to_owned()),
        ("lua_diff", "return diff(a)".to_owned()),
        ("lua_logret", "return log_returns(b)".to_owned()),
        ("lua_ewma", "return ewma(a, 0.25)".to_owned()),
        // The prelude, exercised through the same path as the natives.
        (
            "lua_zscore",
            format!("return zscore(a, {})", LUA_PRECEDING + 1),
        ),
    ] {
        table
            .register_lua_scalar(name, &["a", "b"], &chunk)
            .unwrap_or_else(|error| panic!("{name} registers: {error}"));
    }
    let frame = format!(
        "OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN {LUA_PRECEDING} PRECEDING \
         AND CURRENT ROW)"
    );
    let sql = format!(
        "SELECT ts, sym, x, y, \
         lua_mad(x) {frame} AS mad, \
         lua_wdot(y, x) {frame} AS wdot, \
         lua_npos(x) {frame} AS npos, \
         lua_spread(x) {frame} AS spread, \
         lua_rel(x, y) AS rel, \
         lua_rdot(x, y) AS rdot, \
         lua_rvar(x, y) AS rvar, \
         lua_rstd(x, y) AS rstd, \
         lua_lag(x, y) AS lagged, \
         lua_diff(x, y) AS differenced, \
         lua_logret(x, y) AS logret, \
         lua_ewma(x, y) AS ewma, \
         lua_zscore(x, y) AS zscore \
         FROM trades"
    );
    match table.query_stream(&sql) {
        // SAFETY: the caller (the oracle script) provides a valid,
        // writable destination struct.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("lua-window fixture query failed: {error}"),
    }
}

/// The driver-pipeline family (SQL-in-Lua, #70): a script drives the
/// engine end to end over the persistent, reopened, multi-segment M1
/// fixture — SELECT out, the vectorized vocabulary over the result
/// views (the contiguous result crosses segment boundaries and merges
/// per-segment key dictionaries), exact row-by-row feed-back into a
/// scratch table, SELECT back in. Exports the derived table (`ts, sym,
/// x, y, rel, rdot`) for the oracle script to re-derive with NumPy.
///
/// # Safety
/// As for [`tallydb_m1_inputs_stream`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_driver_pipeline_stream(out: *mut ArrowArrayStream) {
    let mut database = Database::new();
    database.add_table(fixture_table()).expect("fixture adds");
    let script = format!(
        "query(\"CREATE TABLE derived (ts BIGINT ORDERING KEY, sym SYMBOL NOT NULL, \
         x DOUBLE, y DOUBLE, rel DOUBLE, rdot DOUBLE)\")\n\
         local r, n = query(\"SELECT ts, sym, x, y FROM trades\")\n\
         local rel = (r.x - r.y) / r.y\n\
         local rdot = rolling_dot(r.x, r.y, {window})\n\
         for i = 1, n do\n\
           append(\"derived\", {{ ts = r.ts[i], sym = r.sym:text(i), x = r.x[i], \
           y = r.y[i], rel = rel[i], rdot = rdot[i] }})\n\
         end\n\
         local d, m = query(\"SELECT ts FROM derived\")\n\
         assert(m == n, \"feed-back kept every row\")",
        window = LUA_PRECEDING + 1
    );
    database.run_script(&script).expect("driver pipeline runs");
    match database.query_stream("SELECT ts, sym, x, y, rel, rdot FROM derived") {
        // SAFETY: the caller (the oracle script) provides a valid,
        // writable destination struct.
        Ok(stream) => unsafe { out.write(stream) },
        Err(error) => panic!("derived-table export failed: {error}"),
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
/// the `dot` host function) and `lua_mad(x)` (pure interpreter).
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
    table
        .register_lua_scalar(
            "lua_spread",
            &["a", "b"],
            "for i = 1, #a do out[i] = (a[i] - b[i]) / b[i] end",
        )
        .expect("lua_spread registers");
    table
        .register_lua_scalar("lua_rel", &["a", "b"], "return (a - b) / b")
        .expect("lua_rel registers");
    table
        .register_lua_scalar("lua_rdot", &["a", "b"], "return rolling_dot(a, b, 64)")
        .expect("lua_rdot registers");
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

// ------------------------------------------------- the as-of oracle

/// An open as-of oracle context (M4.4, issue #75): the M1 fixture in
/// its own persistent directory, driven statement by statement from
/// Python — mutations, compactions, reopens, and `ASOF` queries — so
/// the script can read the ingest-sequence watermark around every
/// mutation and build the explicit version table DuckDB re-derives
/// every cut from. The emulation lives in the referee only.
pub struct AsOfContext {
    /// `None` transiently during [`tallydb_asof_reopen`] — the old
    /// handle must drop before the directory is opened again.
    table: Option<Table>,
    dir: std::path::PathBuf,
}

impl AsOfContext {
    fn table(&self) -> &Table {
        self.table.as_ref().expect("context holds an open table")
    }
    fn table_mut(&mut self) -> &mut Table {
        self.table.as_mut().expect("context holds an open table")
    }
}

fn asof_schema() -> Schema {
    Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
        Field::new("y", ColumnType::F64, false),
    ])
}

/// Builds the as-of fixture — the M1 fixture's generator (same LCG,
/// same symbols) in a dedicated directory — and returns an owned
/// context. Release with [`tallydb_asof_close`].
#[no_mangle]
pub extern "C" fn tallydb_asof_open() -> *mut AsOfContext {
    let dir = std::env::temp_dir().join(format!("tallydb-asof-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut table =
        Table::persistent_with_segment_rows("trades", asof_schema(), "ts", &dir, SEGMENT_ROWS)
            .expect("as-of fixture schema is valid");
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
            .expect("as-of fixture rows are valid");
    }
    table.flush().expect("as-of fixture flush succeeds");
    Box::into_raw(Box::new(AsOfContext {
        table: Some(table),
        dir,
    }))
}

/// The context table's ingest-sequence watermark (the sequence the next
/// appended row will receive) — what the oracle script records around
/// every mutation to place version intervals.
///
/// # Safety
/// `context` must come from [`tallydb_asof_open`] and not be closed.
#[no_mangle]
pub unsafe extern "C" fn tallydb_asof_next_sequence(context: *mut AsOfContext) -> u64 {
    // SAFETY: caller contract — a live context.
    unsafe { &*context }.table().next_sequence()
}

/// Runs one mutation statement. Returns the rows changed, or -1 on
/// failure (printed to stderr).
///
/// # Safety
/// As for [`tallydb_asof_next_sequence`]; `sql` must be a valid
/// NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tallydb_asof_mutate(
    context: *mut AsOfContext,
    sql: *const std::os::raw::c_char,
) -> i64 {
    // SAFETY: caller contract — a valid NUL-terminated string.
    let sql = match unsafe { std::ffi::CStr::from_ptr(sql) }.to_str() {
        Ok(sql) => sql,
        Err(_) => {
            eprintln!("tallydb_asof_mutate: SQL is not UTF-8");
            return -1;
        }
    };
    // SAFETY: caller contract — a live context.
    match unsafe { &mut *context }.table_mut().mutate(sql) {
        Ok(changed) => i64::try_from(changed).unwrap_or(i64::MAX),
        Err(error) => {
            eprintln!("tallydb_asof_mutate: {sql}: {error}");
            -1
        }
    }
}

/// Compacts the context's table (superseded rows move to history).
/// Returns 0 on success, 1 on failure.
///
/// # Safety
/// As for [`tallydb_asof_next_sequence`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_asof_compact(context: *mut AsOfContext) -> i32 {
    // SAFETY: caller contract — a live context.
    match unsafe { &mut *context }.table_mut().compact() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("tallydb_asof_compact: {error}");
            1
        }
    }
}

/// Closes and reopens the table from its directory — the storage round
/// trip: manifest, segments, history, delete logs, and WAL all come
/// back from bytes. Returns 0 on success, 1 on failure.
///
/// # Safety
/// As for [`tallydb_asof_next_sequence`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_asof_reopen(context: *mut AsOfContext) -> i32 {
    // SAFETY: caller contract — a live context.
    let context = unsafe { &mut *context };
    context.table = None; // drop the old handle before reopening
    match Table::persistent_with_segment_rows(
        "trades",
        asof_schema(),
        "ts",
        &context.dir,
        SEGMENT_ROWS,
    ) {
        Ok(table) => {
            context.table = Some(table);
            0
        }
        Err(error) => {
            eprintln!("tallydb_asof_reopen: {error}");
            1
        }
    }
}

/// Runs one SQL query (`ASOF` included) against the context's table and
/// exports the result. Returns 0 on success; 1 on failure with `out`
/// untouched.
///
/// # Safety
/// As for [`tallydb_asof_next_sequence`]; `sql` must be a valid
/// NUL-terminated string; `out` a valid, writable destination not
/// holding a live export.
#[no_mangle]
pub unsafe extern "C" fn tallydb_asof_query(
    context: *mut AsOfContext,
    sql: *const std::os::raw::c_char,
    out: *mut ArrowArrayStream,
) -> i32 {
    // SAFETY: caller contract — a valid NUL-terminated string.
    let sql = match unsafe { std::ffi::CStr::from_ptr(sql) }.to_str() {
        Ok(sql) => sql,
        Err(_) => {
            eprintln!("tallydb_asof_query: SQL is not UTF-8");
            return 1;
        }
    };
    // SAFETY: caller contract — a live context.
    match unsafe { &*context }.table().query_stream(sql) {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => {
            unsafe { out.write(stream) };
            0
        }
        Err(error) => {
            eprintln!("tallydb_asof_query: {sql}: {error}");
            1
        }
    }
}

/// Releases an as-of context and its temporary directory.
///
/// # Safety
/// `context` must come from [`tallydb_asof_open`] and not have been
/// closed already.
#[no_mangle]
pub unsafe extern "C" fn tallydb_asof_close(context: *mut AsOfContext) {
    // SAFETY: caller contract — exactly one close per open.
    let context = unsafe { Box::from_raw(context) };
    let dir = context.dir.clone();
    drop(context);
    let _ = std::fs::remove_dir_all(dir);
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

// ---------------------------------------------------------------------
// The maintained-view family (#83, tranches 1 and 2): a database
// holding one source table and three views over it — bucketed
// (tranche 1), running, and cumulative (tranche 2) — driven statement
// by statement from the oracle script, which mirrors every statement
// into DuckDB and diffs each VIEW's answer — through the union read,
// so the oracle covers stale, refreshed, corrected, compacted, and
// reopened states alike — against DuckDB running the definition from
// scratch.

/// The view family's fixture: the source schema and the definitions the
/// context creates, exported so the script cannot drift from them.
pub const VIEW_DEFINITION: &str = "SELECT sym, ts / 5 AS bar, count(*) AS n, \
     sum(x) AS s, min(x) AS lo, max(x) AS hi, first(x) AS o, last(x) AS c \
     FROM trades GROUP BY sym, ts / 5";

/// The running view the fixture maintains beside the bucketed one —
/// per-symbol totals, no bucket, served from hidden-bucket partials.
pub const VIEW_RUNNING_DEFINITION: &str = "SELECT sym, count(*) AS n, sum(x) AS s, \
     avg(x) AS a, min(x) AS lo, max(x) AS hi, first(x) AS o, last(x) AS c \
     FROM trades GROUP BY sym";

/// The cumulative view the fixture maintains — one row per source row,
/// every admitted expanding window. Plain SQL in DuckDB too, which is
/// what makes the mirror exact.
pub const VIEW_CUMULATIVE_DEFINITION: &str = "SELECT ts, sym, \
     sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cs, \
     count(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cn, \
     avg(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ca, \
     min(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS clo, \
     max(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS chi \
     FROM trades";

/// The context behind the maintained-view oracle family.
pub struct ViewContext {
    /// `None` transiently during [`tallydb_view_reopen`].
    db: Option<Database>,
    dir: std::path::PathBuf,
}

impl ViewContext {
    fn db(&self) -> &Database {
        self.db.as_ref().expect("context holds an open database")
    }
    fn db_mut(&mut self) -> &mut Database {
        self.db.as_mut().expect("context holds an open database")
    }
    fn open_at(dir: &std::path::Path) -> Result<Database, crate::EngineError> {
        let source_dir = dir.join("trades");
        let views = [
            ("bars", VIEW_DEFINITION),
            ("totals", VIEW_RUNNING_DEFINITION),
            ("cum", VIEW_CUMULATIVE_DEFINITION),
        ];
        let existing = source_dir.join(storage_lite::store::MANIFEST).is_file();
        let mut db = Database::new();
        let table = if existing {
            Table::open("trades", &source_dir, storage_lite::StoreOptions::default())?
        } else {
            Table::persistent_with_segment_rows(
                "trades",
                asof_schema(),
                "ts",
                &source_dir,
                SEGMENT_ROWS,
            )?
        };
        let opened = views
            .into_iter()
            .map(|(name, definition)| {
                if existing {
                    crate::MaterializedView::open(
                        name,
                        dir.join(name),
                        &table,
                        storage_lite::StoreOptions::default(),
                    )
                } else {
                    crate::MaterializedView::persistent(name, definition, &table, dir.join(name))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        db.add_table(table)?;
        for view in opened {
            db.add_view(view)?;
        }
        Ok(db)
    }
}

/// Opens the view-family fixture: an empty persistent source table and
/// the three maintained views over it. The script drives every row in
/// via SQL, so its DuckDB mirror is exact by construction.
#[no_mangle]
pub extern "C" fn tallydb_view_open() -> *mut ViewContext {
    let dir = std::env::temp_dir().join(format!("tallydb-view-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for sub in ["trades", "bars", "totals", "cum"] {
        std::fs::create_dir_all(dir.join(sub)).expect("fixture directories");
    }
    let db = ViewContext::open_at(&dir).expect("view fixture opens");
    Box::into_raw(Box::new(ViewContext { db: Some(db), dir }))
}

/// The definition the context's bucketed view maintains, for the
/// script's DuckDB mirror. Returns a NUL-terminated static string.
#[no_mangle]
pub extern "C" fn tallydb_view_definition() -> *const std::os::raw::c_char {
    static DEFINITION: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    DEFINITION
        .get_or_init(|| std::ffi::CString::new(VIEW_DEFINITION).expect("no NULs"))
        .as_ptr()
}

/// As [`tallydb_view_definition`], for the running view.
#[no_mangle]
pub extern "C" fn tallydb_view_running_definition() -> *const std::os::raw::c_char {
    static DEFINITION: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    DEFINITION
        .get_or_init(|| std::ffi::CString::new(VIEW_RUNNING_DEFINITION).expect("no NULs"))
        .as_ptr()
}

/// As [`tallydb_view_definition`], for the cumulative view.
#[no_mangle]
pub extern "C" fn tallydb_view_cumulative_definition() -> *const std::os::raw::c_char {
    static DEFINITION: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    DEFINITION
        .get_or_init(|| std::ffi::CString::new(VIEW_CUMULATIVE_DEFINITION).expect("no NULs"))
        .as_ptr()
}

/// Runs one INSERT / UPDATE / DELETE against the source table. Returns
/// rows changed, or -1 on failure (printed to stderr).
///
/// # Safety
/// `context` must come from [`tallydb_view_open`] and not be closed;
/// `sql` must be a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tallydb_view_statement(
    context: *mut ViewContext,
    sql: *const std::os::raw::c_char,
) -> i64 {
    // SAFETY: caller contract — a valid NUL-terminated string.
    let sql = match unsafe { std::ffi::CStr::from_ptr(sql) }.to_str() {
        Ok(sql) => sql,
        Err(_) => {
            eprintln!("tallydb_view_statement: SQL is not UTF-8");
            return -1;
        }
    };
    // SAFETY: caller contract — a live context.
    match unsafe { &mut *context }.db_mut().mutate(sql) {
        Ok(changed) => i64::try_from(changed).unwrap_or(i64::MAX),
        Err(error) => {
            eprintln!("tallydb_view_statement: {sql}: {error}");
            -1
        }
    }
}

/// Refreshes every maintained view. Returns buckets re-folded, summed
/// saturating across the three (a rebuild floor reports `u64::MAX` and
/// saturates the sum), or -1 on failure.
///
/// # Safety
/// As for [`tallydb_view_statement`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_view_refresh(context: *mut ViewContext) -> i64 {
    // SAFETY: caller contract — a live context.
    let db = unsafe { &mut *context }.db_mut();
    let mut folded = 0u64;
    for name in ["bars", "totals", "cum"] {
        match db.refresh_view(name) {
            Ok(count) => folded = folded.saturating_add(count),
            Err(error) => {
                eprintln!("tallydb_view_refresh: {name}: {error}");
                return -1;
            }
        }
    }
    i64::try_from(folded).unwrap_or(i64::MAX)
}

/// Runs one query — against the view or the source — and exports the
/// result as an `ArrowArrayStream`. Returns 0 on success.
///
/// # Safety
/// As for [`tallydb_view_statement`]; `out` must be valid and writable.
#[no_mangle]
pub unsafe extern "C" fn tallydb_view_query_stream(
    context: *mut ViewContext,
    sql: *const std::os::raw::c_char,
    out: *mut ArrowArrayStream,
) -> i32 {
    // SAFETY: caller contract — a valid NUL-terminated string.
    let sql = match unsafe { std::ffi::CStr::from_ptr(sql) }.to_str() {
        Ok(sql) => sql,
        Err(_) => {
            eprintln!("tallydb_view_query_stream: SQL is not UTF-8");
            return 1;
        }
    };
    // SAFETY: caller contract — a live context.
    match unsafe { &*context }.db().query_stream(sql) {
        // SAFETY: the caller provides a valid, writable destination.
        Ok(stream) => {
            unsafe { out.write(stream) };
            0
        }
        Err(error) => {
            eprintln!("tallydb_view_query_stream: {sql}: {error}");
            1
        }
    }
}

/// Compacts the source table (kills move to history — the branch of
/// the touched-bucket derivation nothing else exercises end-to-end).
/// Returns 0 on success.
///
/// # Safety
/// As for [`tallydb_view_statement`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_view_compact(context: *mut ViewContext) -> i32 {
    // SAFETY: caller contract — a live context.
    match unsafe { &mut *context }.db_mut().compact("trades") {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("tallydb_view_compact: {error}");
            1
        }
    }
}

/// Closes and reopens the whole fixture from its directory — the
/// storage round trip for the source AND all three views: manifest,
/// segments, WAL, and each view's definition record with its stamp.
/// Returns 0 on success.
///
/// # Safety
/// As for [`tallydb_view_statement`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_view_reopen(context: *mut ViewContext) -> i32 {
    // SAFETY: caller contract — a live context.
    let context = unsafe { &mut *context };
    context.db = None; // drop both handles before the directories reopen
    match ViewContext::open_at(&context.dir) {
        Ok(db) => {
            context.db = Some(db);
            0
        }
        Err(error) => {
            eprintln!("tallydb_view_reopen: {error}");
            1
        }
    }
}

/// Closes the context and removes its directory.
///
/// # Safety
/// `context` must come from [`tallydb_view_open`]; exactly one close.
#[no_mangle]
pub unsafe extern "C" fn tallydb_view_close(context: *mut ViewContext) {
    // SAFETY: caller contract — exactly one close per open.
    let context = unsafe { Box::from_raw(context) };
    let dir = context.dir.clone();
    drop(context);
    let _ = std::fs::remove_dir_all(dir);
}
