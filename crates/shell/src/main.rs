//! `tallydb` — the standalone console binary: the thin skin over
//! [`tallydb_shell::Console`], which a future served product embeds
//! the same way. Interactive with line editing and history; batch via
//! `-c "sql"` or piped stdin.

use std::io::{IsTerminal, Read};
use tallydb_shell::{only_comments, split_statements, Console, Outcome};

const USAGE: &str = "usage: tallydb DIR [--read-only] [-c \"sql\"]\n\
  DIR         the database directory (created if absent)\n\
  --read-only open alongside a writer process: queries only, .refresh\n\
              re-reads what the writer has flushed\n\
  -c \"sql\"    run statements and exit (repeatable); also reads piped stdin";

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mut dir = None;
    let mut batch: Vec<String> = Vec::new();
    let mut read_only = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-c" => match arguments.next() {
                Some(sql) => batch.push(sql),
                None => exit_usage("-c needs a SQL argument"),
            },
            "--read-only" => read_only = true,
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
        Console::open_read_only(&dir)
    } else {
        Console::open(&dir)
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
        if piped && !run.quit {
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

/// Executes `input` statement by statement — `;` boundaries via
/// [`split_statements`] (several statements on one line included),
/// whole dot-command lines — recording errors and `.quit` in `run`.
fn run_statements(console: &mut Console, input: &str, run: &mut Run) {
    let mut buffer = String::new();
    for line in input.lines() {
        if run.quit {
            return;
        }
        if buffer.trim().is_empty() && line.trim_start().starts_with('.') {
            buffer.clear();
            execute(console, line, run);
            continue;
        }
        buffer.push_str(line);
        buffer.push('\n');
        let (complete, rest) = split_statements(&buffer);
        buffer = rest;
        for statement in complete {
            if run.quit {
                return;
            }
            execute(console, &statement, run);
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
        let prompt = if buffer.is_empty() {
            "tally> "
        } else {
            "  ...> "
        };
        match editor.readline(prompt) {
            Ok(line) => {
                if buffer.trim().is_empty() && line.trim_start().starts_with('.') {
                    buffer.clear();
                    let _ = editor.add_history_entry(&line);
                    if execute_interactive(console, &line) {
                        break;
                    }
                    continue;
                }
                buffer.push_str(&line);
                buffer.push('\n');
                let (complete, rest) = split_statements(&buffer);
                buffer = rest;
                let mut quit = false;
                for statement in complete {
                    let _ = editor.add_history_entry(&statement);
                    if execute_interactive(console, &statement) {
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
