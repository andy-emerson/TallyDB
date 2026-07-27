//! `tallydb` — the standalone console binary: the thin skin over
//! [`tallydb_shell::Console`], which a future served product embeds
//! the same way. Interactive with line editing and history; batch via
//! `-c "sql"` or piped stdin.

use std::io::{IsTerminal, Read};
use tallydb_shell::{statement_complete, Console, Outcome};

const USAGE: &str = "usage: tallydb DIR [-c \"sql\"]\n\
  DIR         the database directory (created if absent)\n\
  -c \"sql\"    run statements and exit (repeatable); also reads piped stdin";

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mut dir = None;
    let mut batch: Vec<String> = Vec::new();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-c" => match arguments.next() {
                Some(sql) => batch.push(sql),
                None => exit_usage("-c needs a SQL argument"),
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
    let mut console = match Console::open(&dir) {
        Ok(console) => console,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    let piped = !std::io::stdin().is_terminal();
    if !batch.is_empty() || piped {
        let mut failed = false;
        for sql in &batch {
            failed |= run_statements(&mut console, sql);
        }
        if piped {
            let mut input = String::new();
            if std::io::stdin().read_to_string(&mut input).is_ok() {
                failed |= run_statements(&mut console, &input);
            }
        }
        std::process::exit(if failed { 1 } else { 0 });
    }

    interactive(&mut console, &dir);
}

fn exit_usage(reason: &str) -> ! {
    eprintln!("error: {reason}\n{USAGE}");
    std::process::exit(2)
}

/// Splits `input` on statement boundaries (`;` outside quotes; whole
/// dot-command lines) and executes each. Returns whether any failed.
fn run_statements(console: &mut Console, input: &str) -> bool {
    let mut failed = false;
    let mut buffer = String::new();
    for line in input.lines() {
        if buffer.is_empty() && line.trim_start().starts_with('.') {
            failed |= execute(console, line);
            continue;
        }
        buffer.push_str(line);
        buffer.push('\n');
        if statement_complete(&buffer) {
            failed |= execute(console, &buffer);
            buffer.clear();
        }
    }
    if !buffer.trim().is_empty() {
        failed |= execute(console, &buffer);
    }
    failed
}

/// Executes one statement, printing its outcome. Returns true on error.
fn execute(console: &mut Console, statement: &str) -> bool {
    match console.execute(statement) {
        Ok(Outcome::Table(text)) | Ok(Outcome::Note(text)) => {
            if !text.is_empty() {
                println!("{text}");
            }
            false
        }
        Ok(Outcome::Quit) => std::process::exit(0),
        Err(error) => {
            eprintln!("error: {error}");
            true
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
                if buffer.is_empty() && line.trim_start().starts_with('.') {
                    let _ = editor.add_history_entry(&line);
                    if execute_interactive(console, &line) {
                        break;
                    }
                    continue;
                }
                buffer.push_str(&line);
                buffer.push('\n');
                if statement_complete(&buffer) {
                    let _ = editor.add_history_entry(buffer.trim());
                    let statement = std::mem::take(&mut buffer);
                    if execute_interactive(console, &statement) {
                        break;
                    }
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
