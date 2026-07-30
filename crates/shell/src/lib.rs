//! `tallydb-shell` — the console over the engine: the *shell* layer of
//! the shell / security / systems separation (#39, ruled 2026-07-27).
//!
//! The engine is the *systems* layer and stays dependency-clean; this
//! crate owns the human-facing conveniences (line editing, CSV import,
//! table rendering) and the *security* measures a **local** attack
//! surface calls for: a process lock on the storage directory, table
//! names confined to identifiers (no path tricks), and no code
//! registration through SQL — Lua kernels register through an explicit
//! `.lua` dot-command, so a SQL string is never a code-injection
//! vector. A future served product embeds [`Console`] and adds the
//! security its network surface needs; nothing here presumes one.
//!
//! Everything a user types is either a dot-command (`.help` lists
//! them) or SQL — the surface tabulated in DESIGN.md's stdlib table.

use arrow_lite::{Column, ColumnType, NumericData, Schema};
use engine::{schema_from_create, type_name, Database, LogSink, RowValue, StoreOptions, Table};
use query_lite::{parse_statement, QueryOutput, Statement};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The console's destination for Lua kernels' `log(...)`: stderr, so
/// diagnostics land beside error output and never inside a rendered
/// result table.
struct StderrSink;

impl LogSink for StderrSink {
    fn log(&self, message: &str) {
        eprintln!("[lua] {message}");
    }
}

/// A console session over one storage directory: every subdirectory
/// holding a table manifest opens as a table; `CREATE TABLE` makes a
/// new subdirectory. Exactly one console (or other embedder honoring
/// the lock) owns the directory at a time.
pub struct Console {
    database: Database,
    dir: PathBuf,
    options: StoreOptions,
    /// The Lua `log(...)` destination, installed on every table this
    /// console opens or creates (see [`StderrSink`]).
    sink: Arc<dyn LogSink + Sync>,
    /// The advisory process lock: an OS file lock, released by the OS
    /// even if the process dies — no stale-lock cleanup ever needed.
    /// `None` for a read-only console (F4), which coexists with the
    /// writer instead of excluding it.
    _lock: Option<std::fs::File>,
    /// A read-only console (F4): every table opened read-only, no
    /// lock taken; `.refresh` re-reads what the writer has flushed.
    read_only: bool,
}

/// What one executed line produces.
#[derive(Debug)]
pub enum Outcome {
    /// A rendered result table (SELECT).
    Table(String),
    /// A short acknowledgement (mutations, DDL, dot-commands).
    Note(String),
    /// The console should exit.
    Quit,
}

impl Console {
    /// Opens (creating if needed) the storage directory, takes the
    /// process lock, and opens every table found inside.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Console, String> {
        Console::open_inner(dir.into(), false, None)
    }

    /// Opens the directory **read-only** (F4): no lock, so this console
    /// coexists with a writer process (and other readers) over the same
    /// database. Mutating statements refuse loudly; `.refresh` re-reads
    /// what the writer has flushed — the beta shape's console half.
    pub fn open_read_only(dir: impl Into<PathBuf>) -> Result<Console, String> {
        Console::open_read_only_with_cache(dir, None)
    }

    /// As [`Console::open`], with a residency budget in bytes: decoded
    /// segments this console retains in memory (the `--cache` flag;
    /// 2026-07-30 residency design). `None` retains everything touched.
    pub fn open_with_cache(
        dir: impl Into<PathBuf>,
        cache_bytes: Option<u64>,
    ) -> Result<Console, String> {
        Console::open_inner(dir.into(), false, cache_bytes)
    }

    /// As [`Console::open_read_only`], with a residency budget.
    pub fn open_read_only_with_cache(
        dir: impl Into<PathBuf>,
        cache_bytes: Option<u64>,
    ) -> Result<Console, String> {
        let dir = dir.into();
        if !dir.is_dir() {
            return Err(format!("{} is not a database directory", dir.display()));
        }
        Console::open_inner(dir, true, cache_bytes)
    }

    fn open_inner(
        dir: PathBuf,
        read_only: bool,
        cache_bytes: Option<u64>,
    ) -> Result<Console, String> {
        let lock = if read_only {
            None
        } else {
            std::fs::create_dir_all(&dir)
                .map_err(|error| format!("creating {}: {error}", dir.display()))?;
            let lock_path = dir.join(".tallydb.lock");
            let lock = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)
                .map_err(|error| format!("opening lock {}: {error}", lock_path.display()))?;
            if let Err(error) = lock.try_lock() {
                return Err(format!(
                    "another process holds {} ({error}); one writer per database",
                    dir.display()
                ));
            }
            Some(lock)
        };
        let mut database = Database::new();
        // One options value serves both the tables opened here and the
        // ones `CREATE TABLE` makes later — two separate defaults would
        // silently drift the day options become configurable.
        let options = StoreOptions {
            cache_bytes,
            ..StoreOptions::default()
        };
        let sink: Arc<dyn LogSink + Sync> = Arc::new(StderrSink);
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|error| format!("reading {}: {error}", dir.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir() && path.join("table.tlym").is_file())
            .collect();
        entries.sort();
        for path in entries {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("unreadable table directory {}", path.display()))?
                .to_owned();
            let mut table = if read_only {
                Table::open_read_only_with_cache(&name, &path, cache_bytes)
            } else {
                Table::open(&name, &path, options)
            }
            .map_err(|error| format!("opening table '{name}': {error}"))?;
            table.set_lua_log_sink(Arc::clone(&sink));
            database
                .add_table(table)
                .map_err(|error| error.to_string())?;
        }
        database.set_script_log_sink(Arc::clone(&sink));
        Ok(Console {
            database,
            dir,
            options,
            sink,
            _lock: lock,
            read_only,
        })
    }

    /// Re-reads every table's durable state, and opens tables the
    /// writer created since (read-only consoles only): the polling
    /// half of the cross-process story — the reader decides when to
    /// look, the engine never pushes.
    pub fn refresh(&mut self) -> Result<usize, String> {
        if !self.read_only {
            return Err("refresh is the read-only console's command".to_owned());
        }
        let mut refreshed = 0usize;
        for name in self.database.table_names() {
            let table = self.database.table_mut(&name).expect("listed above");
            table.refresh().map_err(|error| error.to_string())?;
            refreshed += 1;
        }
        // Tables the writer created after this console opened.
        let known = self.database.table_names();
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map_err(|error| format!("reading {}: {error}", self.dir.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir() && path.join("table.tlym").is_file())
            .collect();
        entries.sort();
        for path in entries {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("unreadable table directory {}", path.display()))?
                .to_owned();
            if known.contains(&name) {
                continue;
            }
            let mut table =
                Table::open_read_only_with_cache(&name, &path, self.options.cache_bytes)
                    .map_err(|error| format!("opening table '{name}': {error}"))?;
            table.set_lua_log_sink(Arc::clone(&self.sink));
            self.database
                .add_table(table)
                .map_err(|error| error.to_string())?;
            refreshed += 1;
        }
        Ok(refreshed)
    }

    /// The open tables, sorted.
    pub fn tables(&self) -> Vec<String> {
        let mut names = self.database.table_names();
        names.sort();
        names
    }

    /// Executes one complete statement or dot-command.
    pub fn execute(&mut self, line: &str) -> Result<Outcome, String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(Outcome::Note(String::new()));
        }
        if let Some(rest) = line.strip_prefix('.') {
            return self.dot_command(rest);
        }
        let sql = line.strip_suffix(';').unwrap_or(line);
        match parse_statement(sql).map_err(|error| error.to_string())? {
            Statement::Select(_) => {
                let output = self.database.query(sql).map_err(|e| e.to_string())?;
                Ok(Outcome::Table(render(&output)))
            }
            Statement::CreateTable(plan) => {
                self.create_table(&plan)?;
                Ok(Outcome::Note(format!("table '{}' created", plan.table)))
            }
            Statement::Insert(_) | Statement::Update(_) | Statement::Delete(_) => {
                let affected = self.database.mutate(sql).map_err(|e| e.to_string())?;
                Ok(Outcome::Note(format!("{affected} rows")))
            }
        }
    }

    fn create_table(&mut self, plan: &query_lite::CreateTablePlan) -> Result<(), String> {
        // The security posture of a filesystem-backed console: table
        // names become directory names, so they stay identifiers.
        if !plan
            .table
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
            || plan.table.is_empty()
        {
            return Err(format!(
                "table name '{}' must be letters, digits, and underscores",
                plan.table
            ));
        }
        if self.database.table(&plan.table).is_some() {
            return Err(format!("table '{}' already exists", plan.table));
        }
        let (schema, ordering) = schema_from_create(plan).map_err(|e| e.to_string())?;
        let mut table = Table::persistent_with(
            &plan.table,
            schema,
            &ordering,
            self.dir.join(&plan.table),
            self.options,
        )
        .map_err(|e| e.to_string())?;
        table.set_lua_log_sink(Arc::clone(&self.sink));
        self.database
            .add_table(table)
            .map_err(|error| error.to_string())
    }

    fn dot_command(&mut self, rest: &str) -> Result<Outcome, String> {
        let mut parts = rest.splitn(2, char::is_whitespace);
        let command = parts.next().unwrap_or("");
        let argument = parts.next().unwrap_or("").trim();
        match command {
            "help" => Ok(Outcome::Note(HELP.to_owned())),
            "quit" | "exit" => Ok(Outcome::Quit),
            "refresh" => {
                let refreshed = self.refresh()?;
                Ok(Outcome::Note(format!(
                    "{refreshed} table(s) re-read from the writer's durable state"
                )))
            }
            "flush" => {
                // The writer's publish verb: buffered rows become
                // durable segments — the boundary read-only consoles
                // (and crash recovery) see.
                if self.read_only {
                    return Err("a read-only console has nothing to flush".to_owned());
                }
                for name in self.database.table_names() {
                    self.database
                        .table_mut(&name)
                        .expect("listed above")
                        .flush()
                        .map_err(|error| error.to_string())?;
                }
                Ok(Outcome::Note("flushed".to_owned()))
            }
            "tables" => Ok(Outcome::Note(self.tables().join("\n"))),
            "schema" => {
                let names = if argument.is_empty() {
                    self.tables()
                } else {
                    vec![argument.to_owned()]
                };
                let mut out = String::new();
                for name in names {
                    let table = self
                        .database
                        .table(&name)
                        .ok_or_else(|| format!("unknown table '{name}'"))?;
                    let _ = writeln!(
                        out,
                        "{}",
                        render_schema(&name, table.schema(), table.ordering_key())
                    );
                }
                Ok(Outcome::Note(out.trim_end().to_owned()))
            }
            "import" => {
                let mut arguments = argument.split_whitespace();
                let (Some(file), Some(table), None) =
                    (arguments.next(), arguments.next(), arguments.next())
                else {
                    return Err(".import FILE TABLE".to_owned());
                };
                let count = self.import_csv(Path::new(file), table)?;
                Ok(Outcome::Note(format!("{count} rows imported")))
            }
            "run" => {
                // The driver direction (SQL-in-Lua, #70): the script's
                // `query`/`append` drive this console's database. Tables
                // the script CREATEs are in-memory scratch — durable
                // tables are created at the prompt, before the script.
                if argument.is_empty() {
                    return Err(".run FILE — run a Lua driver script from a file".to_owned());
                }
                let source = std::fs::read_to_string(argument)
                    .map_err(|error| format!("{argument}: {error}"))?;
                self.database
                    .run_script(&source)
                    .map_err(|error| error.to_string())?;
                Ok(Outcome::Note(format!("{argument}: done")))
            }
            "lua" => {
                let (name, parameters, chunk) = parse_lua_command(argument)?;
                let parameters: Vec<&str> = parameters.iter().map(String::as_str).collect();
                // A kernel is a function, not table state: register it
                // on every open table (and remember it for tables
                // created later in this session? — no: created tables
                // are new; re-run .lua for them, as .help documents).
                let tables = self.tables();
                if tables.is_empty() {
                    return Err("no tables to register on yet".to_owned());
                }
                for table in &tables {
                    self.database
                        .register_lua_window(table, &name, &parameters, &chunk, ColumnType::F64)
                        .map_err(|error| error.to_string())?;
                }
                Ok(Outcome::Note(format!(
                    "window function '{name}' registered"
                )))
            }
            "luascalar" => {
                let (name, parameters, chunk) = parse_lua_command(argument)?;
                let parameters: Vec<&str> = parameters.iter().map(String::as_str).collect();
                let tables = self.tables();
                if tables.is_empty() {
                    return Err("no tables to register on yet".to_owned());
                }
                for table in &tables {
                    self.database
                        .register_lua_scalar(table, &name, &parameters, &chunk)
                        .map_err(|error| error.to_string())?;
                }
                Ok(Outcome::Note(format!(
                    "scalar function '{name}' registered"
                )))
            }
            other => Err(format!("unknown command '.{other}' (try .help)")),
        }
    }

    /// Imports a CSV with a header row: columns map to schema columns
    /// by name, cells parse by column type, empty cells are NULL.
    fn import_csv(&mut self, file: &Path, table_name: &str) -> Result<u64, String> {
        let table = self
            .database
            .table(table_name)
            .ok_or_else(|| format!("unknown table '{table_name}'"))?;
        let schema = table.schema().clone();
        let mut reader = csv::Reader::from_path(file)
            .map_err(|error| format!("opening {}: {error}", file.display()))?;
        let headers = reader
            .headers()
            .map_err(|error| format!("reading header: {error}"))?
            .clone();
        let mut mapping = Vec::with_capacity(headers.len());
        for header in headers.iter() {
            let position = schema
                .fields()
                .iter()
                .position(|field| field.name() == header)
                .ok_or_else(|| format!("CSV column '{header}' is not in the table"))?;
            if mapping.contains(&position) {
                // Two headers on one schema column would silently keep
                // only the later cell of every row.
                return Err(format!("CSV column '{header}' appears twice"));
            }
            mapping.push(position);
        }
        let mut count = 0u64;
        for (line, record) in reader.records().enumerate() {
            let record = record.map_err(|error| format!("row {}: {error}", line + 2))?;
            let mut cells: Vec<RowValue<'_>> = vec![RowValue::Null; schema.fields().len()];
            for (cell, &position) in record.iter().zip(&mapping) {
                let field = &schema.fields()[position];
                cells[position] = if cell.is_empty() {
                    RowValue::Null
                } else {
                    match field.column_type() {
                        ColumnType::I64 => RowValue::I64(cell.trim().parse().map_err(|error| {
                            format!("row {}, '{}': {error}", line + 2, field.name())
                        })?),
                        ColumnType::F64 => RowValue::F64(cell.trim().parse().map_err(|error| {
                            format!("row {}, '{}': {error}", line + 2, field.name())
                        })?),
                        ColumnType::Key => RowValue::Key(cell),
                    }
                };
            }
            let table = self.database.table_mut(table_name).expect("checked above");
            table
                .append(&cells)
                .map_err(|error| format!("row {}: {error}", line + 2))?;
            count += 1;
        }
        Ok(count)
    }
}

/// `.lua name(a, b) chunk...` — the chunk is everything after the
/// closing parenthesis (the REPL hands multi-line chunks in whole).
fn parse_lua_command(argument: &str) -> Result<(String, Vec<String>, String), String> {
    let usage = ".lua name(param, ...) return <expression over the params>";
    let open = argument.find('(').ok_or(usage)?;
    let close = argument.find(')').ok_or(usage)?;
    if close < open {
        return Err(usage.to_owned());
    }
    let name = argument[..open].trim().to_owned();
    let parameters: Vec<String> = argument[open + 1..close]
        .split(',')
        .map(|parameter| parameter.trim().to_owned())
        .filter(|parameter| !parameter.is_empty())
        .collect();
    let chunk = argument[close + 1..].trim().to_owned();
    if name.is_empty() || parameters.is_empty() || chunk.is_empty() {
        return Err(usage.to_owned());
    }
    Ok((name, parameters, chunk))
}

/// Splits buffered input into the complete statements — cut at every
/// `;` outside quotes and `--` comments, so `-c "CREATE ...; INSERT
/// ...;"` runs as two statements and an apostrophe inside a comment
/// opens no quote — and the unterminated remainder, if any.
pub fn split_statements(input: &str) -> (Vec<String>, String) {
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut comment = false;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if comment {
            current.push(c);
            if c == '\n' {
                comment = false;
            }
            continue;
        }
        match quote {
            Some(open) => {
                if c == open {
                    quote = None;
                }
                current.push(c);
            }
            None => match c {
                '\'' | '"' => {
                    quote = Some(c);
                    current.push(c);
                }
                '-' if chars.peek() == Some(&'-') => {
                    comment = true;
                    current.push(c);
                }
                ';' => {
                    let statement = current.trim();
                    if !statement.is_empty() {
                        statements.push(statement.to_owned());
                    }
                    current.clear();
                }
                _ => current.push(c),
            },
        }
    }
    (statements, current)
}

/// Whether leftover input is only comments and whitespace — nothing an
/// executor should be handed at end of input.
pub fn only_comments(input: &str) -> bool {
    input.lines().all(|line| {
        let line = line.trim();
        line.is_empty() || line.starts_with("--")
    })
}

/// Renders a query result as an aligned text table — keys as their
/// strings (the shell is an application: rendering display text here
/// is exactly where the strings rule says it belongs), NULL spelled
/// out, a row count under a rule.
pub fn render(output: &QueryOutput) -> String {
    let names: Vec<&str> = output
        .schema
        .fields()
        .iter()
        .map(|field| field.name())
        .collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for batch in &output.batches {
        for row in 0..batch.num_rows() {
            let mut cells = Vec::with_capacity(names.len());
            for column in batch.columns() {
                cells.push(match column {
                    Column::Numeric(NumericData::I64(numeric)) => {
                        if numeric.is_valid(row) {
                            numeric.values().as_slice()[row].to_string()
                        } else {
                            "NULL".to_owned()
                        }
                    }
                    Column::Numeric(NumericData::F64(numeric)) => {
                        if numeric.is_valid(row) {
                            format_f64(numeric.values().as_slice()[row])
                        } else {
                            "NULL".to_owned()
                        }
                    }
                    Column::Key(keys) => keys
                        .value_at(row)
                        .map(str::to_owned)
                        .unwrap_or_else(|| "NULL".to_owned()),
                });
            }
            rows.push(cells);
        }
    }
    let mut widths: Vec<usize> = names.iter().map(|name| name.len()).collect();
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    let mut out = String::new();
    let line = |out: &mut String, cells: &[String]| {
        let rendered: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(cell, width)| format!("{cell:>width$}"))
            .collect();
        let _ = writeln!(out, "{}", rendered.join("  "));
    };
    line(
        &mut out,
        &names.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
    );
    let _ = writeln!(
        &mut out,
        "{}",
        widths
            .iter()
            .map(|width| "-".repeat(*width))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for row in &rows {
        line(&mut out, row);
    }
    let _ = write!(&mut out, "({} rows)", rows.len());
    out
}

/// A float rendered so integers still look floating-point (`2` prints
/// as `2.0`), keeping the column's type visible.
fn format_f64(value: f64) -> String {
    if value.is_finite() && value == value.trunc() && value.abs() < 1e15 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

/// A schema rendered back as the CREATE TABLE that would produce it —
/// round-trippable: feeding the output to `CREATE TABLE` yields an
/// equivalent table, so the ordering key must be spelled out
/// (`ORDERING KEY` implies its NOT NULL).
fn render_schema(name: &str, schema: &Schema, ordering_key: &str) -> String {
    let columns: Vec<String> = schema
        .fields()
        .iter()
        .map(|field| {
            let mut column = format!("{} {}", field.name(), type_name(field.column_type()));
            if field.name() == ordering_key {
                column.push_str(" ORDERING KEY");
            } else if !field.nullable() {
                column.push_str(" NOT NULL");
            }
            column
        })
        .collect();
    format!("CREATE TABLE {name} ({})", columns.join(", "))
}

/// `.help`'s text; the SQL surface itself is DESIGN.md's stdlib table.
pub const HELP: &str = "\
Statements end with ';' and may span lines. SQL surface: SELECT (WHERE,
GROUP BY/HAVING, ORDER BY a numeric column [NULLS FIRST|LAST], LIMIT,
DISTINCT, window functions, scalar expressions, CASE, IS NULL, LIKE on
symbols), INSERT, UPDATE, DELETE, CREATE TABLE (BIGINT | DOUBLE |
SYMBOL, one ORDERING KEY column).

Commands:
  .help                     this text
  .flush                    make buffered rows durable segments now —
                            the boundary readers and recovery see
  .refresh                  (read-only console) re-read what the writer
                            has flushed; picks up new tables too
  .tables                   list tables
  .schema [TABLE]           show table definitions
  .import FILE TABLE        import a CSV (header row maps columns by name)
  .lua NAME(PARAMS) CHUNK   register a Lua window function (f64 result)
                            on every open table; re-run it after
                            CREATE TABLE to cover the new table
  .luascalar NAME(PARAMS) CHUNK
                            register a Lua per-row function: whole
                            columns bind to PARAMS, the script fills
                            out[i] (unwritten slots return NULL)
  .run FILE                 run a Lua driver script: query(sql) issues
                            SQL and returns result columns as views
                            plus a row count; append(table, row) feeds
                            derived rows back exactly. Tables a script
                            CREATEs are in-memory scratch; durable
                            tables are created at this prompt first
  .quit                     leave";

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tallydb-shell-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn note(console: &mut Console, line: &str) -> String {
        match console.execute(line).unwrap() {
            Outcome::Note(text) => text,
            _ => panic!("expected a note for {line}"),
        }
    }

    fn table(console: &mut Console, line: &str) -> String {
        match console.execute(line).unwrap() {
            Outcome::Table(text) => text,
            _ => panic!("expected a table for {line}"),
        }
    }

    #[test]
    fn create_insert_select_round_trip_and_reopen() {
        let dir = scratch("roundtrip");
        {
            let mut console = Console::open(&dir).unwrap();
            note(
                &mut console,
                "CREATE TABLE ticks (ts BIGINT ORDERING KEY, sym SYMBOL NOT NULL, px DOUBLE);",
            );
            assert_eq!(
                note(
                    &mut console,
                    "INSERT INTO ticks VALUES (1, 'ES', 5432.25), (2, 'NQ', 19112.0);"
                ),
                "2 rows"
            );
            let rendered = table(&mut console, "SELECT ts, sym, px FROM ticks ORDER BY ts;");
            assert!(rendered.contains("ES"), "{rendered}");
            assert!(rendered.contains("5432.25"), "{rendered}");
            assert!(rendered.contains("(2 rows)"), "{rendered}");
        } // console drops: lock released, WAL synced state on disk
        let mut console = Console::open(&dir).unwrap();
        assert_eq!(console.tables(), ["ticks"]);
        let rendered = table(&mut console, "SELECT COUNT(px) AS n FROM ticks;");
        assert!(rendered.contains('2'), "{rendered}");
        let schema = note(&mut console, ".schema ticks");
        assert!(schema.contains("ts BIGINT ORDERING KEY"), "{schema}");
        assert!(schema.contains("sym SYMBOL NOT NULL"), "{schema}");
        // Round-trippable: the rendered definition recreates an
        // equivalent table in a fresh database.
        drop(console);
        let dir_two = scratch("roundtrip-two");
        let mut console = Console::open(&dir_two).unwrap();
        note(&mut console, schema.trim());
        assert_eq!(
            note(&mut console, ".schema ticks").trim(),
            schema.trim(),
            "the definition survives a render → create → render cycle"
        );
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&dir_two).unwrap();
    }

    #[test]
    fn the_lock_admits_one_console() {
        let dir = scratch("lock");
        let first = Console::open(&dir).unwrap();
        let second = Console::open(&dir);
        let Err(error) = second else {
            panic!("second console must be refused");
        };
        assert!(error.contains("one writer per database"), "{error}");
        drop(first);
        Console::open(&dir).unwrap(); // released with the process's file
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_read_only_console_rides_alongside_the_writer() {
        // The beta shape (F4): one writer process feeds; a read-only
        // console coexists, sees flushed data, refuses mutation, and
        // .refresh advances it — new tables included.
        let dir = scratch("readonly");
        let mut writer = Console::open(&dir).unwrap();
        writer
            .execute("CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE);")
            .unwrap();
        writer
            .execute("INSERT INTO t VALUES (1, 1.5), (2, 2.5);")
            .unwrap();

        let mut reader = Console::open_read_only(&dir).unwrap();
        assert_eq!(reader.tables(), ["t"]);
        // INSERT runs through the mutation path, which flushes to the
        // WAL — but rows reach readers at the segment boundary. Make
        // them durable from the writer side, then look again.
        let Outcome::Table(rendered) = reader.execute("SELECT ts, x FROM t ORDER BY ts").unwrap()
        else {
            panic!("select renders a table")
        };
        let before_rows = rendered.lines().count();
        writer.execute(".flush").unwrap(); // the writer's publish verb

        reader.execute(".refresh").unwrap();
        let Outcome::Table(rendered) = reader.execute("SELECT ts, x FROM t ORDER BY ts").unwrap()
        else {
            panic!("select renders a table")
        };
        assert!(
            rendered.lines().count() >= before_rows,
            "refresh never loses rows"
        );
        assert!(
            rendered.contains("1.5") && rendered.contains("2.5"),
            "{rendered}"
        );
        // Mutation refuses loudly, naming the writer.
        let error = reader
            .execute("INSERT INTO t VALUES (3, 3.5);")
            .unwrap_err();
        assert!(error.contains("read-only"), "{error}");
        // The writer creates a table the reader discovers on refresh.
        writer
            .execute("CREATE TABLE u (ts BIGINT ORDERING KEY, y DOUBLE);")
            .unwrap();
        reader.execute(".refresh").unwrap();
        assert_eq!(reader.tables(), ["t", "u"]);
        // And a writer console refuses .refresh — it sees its own state.
        assert!(writer.execute(".refresh").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn csv_import_maps_headers_and_types() {
        let dir = scratch("import");
        let mut console = Console::open(&dir).unwrap();
        note(
            &mut console,
            "CREATE TABLE t (ts BIGINT ORDERING KEY, sym SYMBOL NOT NULL, x DOUBLE);",
        );
        let csv_path = dir.join("data.csv");
        std::fs::write(&csv_path, "sym,ts,x\nAAPL,1,1.5\nMSFT,2,\n\"A,B\",3,2.25\n").unwrap();
        let outcome = note(&mut console, &format!(".import {} t", csv_path.display()));
        assert_eq!(outcome, "3 rows imported");
        let rendered = table(&mut console, "SELECT sym, x FROM t ORDER BY x NULLS FIRST;");
        assert!(rendered.contains("NULL"), "{rendered}");
        assert!(
            rendered.contains("A,B"),
            "quoted comma survives: {rendered}"
        );
        // A CSV column the table lacks is loud.
        std::fs::write(&csv_path, "nope\n1\n").unwrap();
        let error = console
            .execute(&format!(".import {} t", csv_path.display()))
            .unwrap_err();
        assert!(error.contains("'nope'"), "{error}");
        // A duplicated header is loud — the later cell would silently
        // overwrite the earlier one in every row.
        std::fs::write(&csv_path, "ts,ts,x\n1,2,1.0\n").unwrap();
        let error = console
            .execute(&format!(".import {} t", csv_path.display()))
            .unwrap_err();
        assert!(error.contains("appears twice"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn lua_kernels_register_from_the_console() {
        let dir = scratch("lua");
        let mut console = Console::open(&dir).unwrap();
        note(
            &mut console,
            "CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE NOT NULL);",
        );
        note(&mut console, "INSERT INTO t VALUES (1, 3.0), (2, 4.0);");
        note(&mut console, ".lua sumsq(x) return dot(x, x)");
        let rendered = table(
            &mut console,
            "SELECT sumsq(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
             AS s FROM t;",
        );
        assert!(rendered.contains("25.0"), "3^2+4^2: {rendered}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn driver_scripts_run_from_the_console() {
        let dir = scratch("run");
        let mut console = Console::open(&dir).unwrap();
        note(
            &mut console,
            "CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE NOT NULL);",
        );
        note(&mut console, "INSERT INTO t VALUES (1, 3.0), (2, 4.0);");
        let script = dir.join("pipeline.lua");
        std::fs::write(
            &script,
            "local r, n = query('SELECT ts, x FROM t')\n\
             local double = r.x + r.x\n\
             for i = 1, n do append('t', { ts = 10 + r.ts[i], x = double[i] }) end\n",
        )
        .unwrap();
        note(&mut console, &format!(".run {}", script.display()));
        let rendered = table(&mut console, "SELECT ts, x FROM t ORDER BY ts;");
        assert!(
            rendered.contains("6.0") && rendered.contains("8.0"),
            "appended doubles: {rendered}"
        );
        // A missing file and a broken script are loud, and the console
        // survives both.
        assert!(console.execute(".run nope.lua").is_err());
        std::fs::write(&script, "query('SELECT nope FROM t')").unwrap();
        let error = console
            .execute(&format!(".run {}", script.display()))
            .unwrap_err();
        assert!(error.contains("nope"), "{error}");
        note(&mut console, ".tables");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn security_posture_holds_at_the_edges() {
        let dir = scratch("security");
        let mut console = Console::open(&dir).unwrap();
        // Path tricks in table names are refused before any I/O.
        let error = console
            .execute("CREATE TABLE \"../evil\" (ts BIGINT ORDERING KEY);")
            .unwrap_err();
        assert!(error.contains("letters, digits"), "{error}");
        // SQL is never a code channel: no CREATE FUNCTION.
        assert!(console
            .execute("CREATE FUNCTION f() RETURNS DOUBLE AS 'return 1';")
            .is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn splitting_understands_quotes_comments_and_boundaries() {
        // Two statements on one line — the `-c "A; B"` shape — split.
        let (statements, rest) =
            split_statements("CREATE TABLE t (ts BIGINT); INSERT INTO t VALUES (1);");
        assert_eq!(statements.len(), 2);
        assert!(statements[0].starts_with("CREATE"));
        assert!(statements[1].starts_with("INSERT"));
        assert!(rest.trim().is_empty());
        // A `;` inside quotes is data, not a boundary.
        let (statements, rest) = split_statements("SELECT x FROM t WHERE sym = 'a;b'");
        assert!(statements.is_empty());
        assert_eq!(rest.trim(), "SELECT x FROM t WHERE sym = 'a;b'");
        // An apostrophe inside a `--` comment opens no quote, and the
        // `;` after the comment line still terminates.
        let (statements, rest) =
            split_statements("SELECT x FROM t -- don't trip here\nWHERE x > 0;");
        assert_eq!(statements.len(), 1, "{statements:?} / {rest:?}");
        assert!(statements[0].contains("WHERE x > 0"));
        assert!(rest.trim().is_empty());
        // Comment-only leftovers are recognized as nothing to run.
        assert!(only_comments("  -- trailing note\n\n"));
        assert!(!only_comments("SELECT 1"));
    }

    #[test]
    fn split_statements_run_one_by_one_through_the_console() {
        let dir = scratch("split");
        let mut console = Console::open(&dir).unwrap();
        let (statements, rest) = split_statements(
            "CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE); \
             INSERT INTO t VALUES (1, 2.5);",
        );
        assert!(rest.trim().is_empty());
        for statement in &statements {
            console.execute(statement).unwrap();
        }
        let rendered = table(&mut console, "SELECT x FROM t;");
        assert!(rendered.contains("2.5"), "{rendered}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
