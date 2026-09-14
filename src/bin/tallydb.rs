//! `tallydb` — the standalone console binary: the thin skin over
//! [`tallydb::Console`], which a future served product embeds
//! the same way. Interactive with line editing and history; batch via
//! `-c "sql"` or piped stdin.

use std::io::{IsTerminal, Read};
use tallydb::{only_comments, split_statements, Console, Outcome};

const USAGE: &str = "usage: tallydb DIR [--read-only] [--cache MiB] [-c \"sql\"]\n\
  DIR         the database directory (created if absent)\n\
  --read-only open alongside a writer process: queries only, .refresh\n\
              re-reads what the writer has flushed\n\
  --cache MiB the residency budget: decoded segments retained in\n\
              memory (default: unbounded — everything touched stays)\n\
  -c \"sql\"    run statements and exit (repeatable); also reads piped stdin";

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mut dir = None;
    let mut batch: Vec<String> = Vec::new();
    let mut read_only = false;
    let mut cache_bytes: Option<u64> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-c" => match arguments.next() {
                Some(sql) => batch.push(sql),
                None => exit_usage("-c needs a SQL argument"),
            },
            "--read-only" => read_only = true,
            "--cache" => match arguments.next().and_then(|mib| mib.parse::<u64>().ok()) {
                Some(mib) => cache_bytes = Some(mib.saturating_mul(1024 * 1024)),
                None => exit_usage("--cache needs a size in MiB"),
            },
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            other if dir.is_none() => dir = Some(other.to_owned()),
            other => exit_usage(&format!("unexpected argument '{other}'")),
        }
    }
    let Some(dir) = dir else {
        exit_usage("missing DIR");
    };
    let opened = if read_only {
        Console::open_read_only_with_cache(&dir, cache_bytes)
    } else {
        Console::open_with_cache(&dir, cache_bytes)
    };
    let mut console = match opened {
        Ok(console) => console,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    let piped = !std::io::stdin().is_terminal();
    if !batch.is_empty() || piped {
        let mut run = Run::default();
        for sql in &batch {
            if run.quit {
                break;
            }
            run_statements(&mut console, sql, &mut run);
        }
        // Only when no `-c` was given. An explicit statement means run
        // it and exit, the convention `sqlite3 db "..."` set and the
        // usage line above implies. Reading stdin as well hung any
        // script whose stdin was an open pipe — forever, holding the
        // writer lock the whole time (#117).
        if piped && batch.is_empty() && !run.quit {
            let mut input = String::new();
            if std::io::stdin().read_to_string(&mut input).is_ok() {
                run_statements(&mut console, &input, &mut run);
            }
        }
        // `.quit` stops the run but never launders an earlier error
        // into exit 0 — scripts rely on the code.
        std::process::exit(if run.failed { 1 } else { 0 });
    }

    interactive(&mut console, &dir);
}

fn exit_usage(reason: &str) -> ! {
    eprintln!("error: {reason}\n{USAGE}");
    std::process::exit(2)
}

/// What a batch run has seen so far.
#[derive(Default)]
struct Run {
    failed: bool,
    quit: bool,
}

/// One thing for the console to do, in the order the input asked.
#[derive(Debug, PartialEq, Eq)]
enum Step {
    /// A whole dot-command line, run as it stands.
    Dot(String),
    /// A complete SQL statement, its `;` already consumed.
    Sql(String),
}

/// Turns one chunk of input into the steps it asks for, leaving whatever
/// is still incomplete in `buffer`.
///
/// **Chunk-shaped, and decided per line inside.** A chunk is not a line:
/// rustyline hands a pasted block back as one string with embedded
/// newlines. Deciding dot-command-or-SQL once for the whole chunk let
/// the block's first line speak for all of it — a `CREATE TABLE` left a
/// following `.import` stranded in the SQL buffer, and a leading `.lua`
/// swallowed the `SELECT` beneath it into its own Lua chunk (#118). So
/// the decision happens per line, here, and both input paths come
/// through this function to get it.
fn plan_chunk(buffer: &mut String, chunk: &str) -> Vec<Step> {
    let mut steps = Vec::new();
    for line in chunk.lines() {
        if buffer.trim().is_empty() && line.trim_start().starts_with('.') {
            buffer.clear();
            steps.push(Step::Dot(line.to_owned()));
            continue;
        }
        buffer.push_str(line);
        buffer.push('\n');
        let (complete, rest) = split_statements(buffer);
        *buffer = rest;
        steps.extend(complete.into_iter().map(Step::Sql));
    }
    steps
}

/// Executes `input` statement by statement — `;` boundaries via
/// [`split_statements`] (several statements on one line included),
/// whole dot-command lines — recording errors and `.quit` in `run`.
fn run_statements(console: &mut Console, input: &str, run: &mut Run) {
    let mut buffer = String::new();
    for step in plan_chunk(&mut buffer, input) {
        if run.quit {
            return;
        }
        match step {
            Step::Dot(line) => execute(console, &line, run),
            Step::Sql(statement) => execute(console, &statement, run),
        }
    }
    // A trailing statement without its `;` still runs at end of input;
    // trailing comments and whitespace do not.
    if !run.quit && !only_comments(&buffer) {
        execute(console, buffer.trim(), run);
    }
}

/// Executes one statement, printing its outcome into `run`.
fn execute(console: &mut Console, statement: &str, run: &mut Run) {
    match console.execute(statement) {
        Ok(Outcome::Table(text)) | Ok(Outcome::Note(text)) => {
            if !text.is_empty() {
                println!("{text}");
            }
        }
        Ok(Outcome::Quit) => run.quit = true,
        Err(error) => {
            eprintln!("error: {error}");
            run.failed = true;
        }
    }
}

fn interactive(console: &mut Console, dir: &str) {
    println!(
        "TallyDB {} — {} table(s) open at {dir}\nStatements end with ';'.  .help for commands.",
        env!("CARGO_PKG_VERSION"),
        console.tables().len()
    );
    let mut editor = match rustyline::DefaultEditor::new() {
        Ok(editor) => editor,
        Err(error) => {
            eprintln!("error: line editor: {error}");
            std::process::exit(1);
        }
    };
    let history = std::path::Path::new(dir).join(".tallydb_history");
    let _ = editor.load_history(&history);
    let mut buffer = String::new();
    loop {
        // Trimmed, like the branch that consumes the buffer: after a
        // statement the buffer holds a leftover newline, and testing it
        // untrimmed showed a continuation prompt with nothing pending.
        let prompt = if buffer.trim().is_empty() {
            "tally> "
        } else {
            "  ...> "
        };
        match editor.readline(prompt) {
            Ok(line) => {
                let mut quit = false;
                for step in plan_chunk(&mut buffer, &line) {
                    let text = match &step {
                        Step::Dot(line) => line,
                        Step::Sql(statement) => statement,
                    };
                    let _ = editor.add_history_entry(text);
                    if execute_interactive(console, text) {
                        quit = true;
                        break;
                    }
                }
                if quit {
                    break;
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => buffer.clear(),
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(error) => {
                eprintln!("error: {error}");
                break;
            }
        }
    }
    let _ = editor.save_history(&history);
}

/// Executes and prints; returns true when the console should exit.
fn execute_interactive(console: &mut Console, statement: &str) -> bool {
    match console.execute(statement) {
        Ok(Outcome::Table(text)) | Ok(Outcome::Note(text)) => {
            if !text.is_empty() {
                println!("{text}");
            }
            false
        }
        Ok(Outcome::Quit) => true,
        Err(error) => {
            eprintln!("error: {error}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_chunk, Step};

    /// The steps a chunk asks for, as one string. Note that
    /// [`split_statements`] consumes the terminating `;`, so a
    /// planned statement does not carry it.
    fn steps(chunk: &str) -> Vec<Step> {
        let mut buffer = String::new();
        plan_chunk(&mut buffer, chunk)
    }

    /// The steps the same text asks for, delivered a line at a time.
    fn steps_by_line(chunk: &str) -> Vec<Step> {
        let mut buffer = String::new();
        let mut all = Vec::new();
        for line in chunk.lines() {
            all.extend(plan_chunk(&mut buffer, line));
        }
        all
    }

    /// #118: rustyline returns a pasted block as one string with
    /// embedded newlines. A paste must do what typing the same lines
    /// does — before the fix the block's first line decided for all of
    /// it, and everything under it was swallowed.
    #[test]
    fn a_pasted_block_does_what_the_same_lines_typed_do() {
        for chunk in [
            // A statement, then a dot command: the dot command was left
            // stranded in the SQL buffer and never ran.
            "CREATE TABLE t (ts BIGINT ORDERING KEY);\n.import t.csv t",
            // A dot command, then a statement: `.lua` takes the rest of
            // its line as the chunk, so the SELECT was compiled as Lua.
            ".lua f(r) return 1 end\nSELECT f(x) FROM t;",
            // Two dot commands.
            ".tables\n.schema t",
            // Two statements on separate lines.
            "SELECT 1 FROM t;\nSELECT 2 FROM t;",
        ] {
            assert_eq!(
                steps(chunk),
                steps_by_line(chunk),
                "pasted and typed must agree: {chunk:?}"
            );
        }
    }

    #[test]
    fn a_dot_command_after_a_statement_runs_as_a_dot_command() {
        let planned = steps("CREATE TABLE t (ts BIGINT ORDERING KEY);\n.import t.csv t");
        assert_eq!(
            planned,
            vec![
                Step::Sql("CREATE TABLE t (ts BIGINT ORDERING KEY)".to_owned()),
                Step::Dot(".import t.csv t".to_owned()),
            ]
        );
    }

    #[test]
    fn a_statement_under_a_dot_command_is_not_swallowed_by_it() {
        let planned = steps(".lua f(r) return 1 end\nSELECT f(x) FROM t;");
        assert_eq!(
            planned,
            vec![
                Step::Dot(".lua f(r) return 1 end".to_owned()),
                Step::Sql("SELECT f(x) FROM t".to_owned()),
            ]
        );
    }

    /// A statement spanning lines still accumulates; only a `;` ends it.
    #[test]
    fn a_statement_may_still_span_lines() {
        let mut buffer = String::new();
        assert!(plan_chunk(&mut buffer, "SELECT sym,").is_empty());
        assert!(plan_chunk(&mut buffer, "       count(*)").is_empty());
        assert_eq!(
            plan_chunk(&mut buffer, "FROM t;"),
            vec![Step::Sql("SELECT sym,\n       count(*)\nFROM t".to_owned())]
        );
        assert!(buffer.trim().is_empty(), "nothing left pending");
    }
}
