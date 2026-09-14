//! The console binary's batch contract (#117): `-c` runs its statements
//! and exits, whatever stdin is doing.
//!
//! The bug this pins was invisible from a terminal — `is_terminal()` was
//! true, so the stdin branch never ran — and fatal from a script, where
//! an open pipe on stdin left the process waiting forever while holding
//! the database's writer lock. So the test has to be a real process with
//! a real pipe; nothing smaller reproduces it.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Waits up to `limit` for `child`, killing it and returning `None` on
/// timeout — `std::process` has no wait-with-deadline.
fn wait_for(child: &mut std::process::Child, limit: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => return Some(status),
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    }
}

#[test]
fn dash_c_exits_while_stdin_stays_open() {
    let dir = std::env::temp_dir().join(format!("tallydb-batch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut child = Command::new(env!("CARGO_BIN_EXE_tallydb"))
        .arg(&dir)
        .arg("-c")
        .arg("CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE);")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the console");

    // Hold the write end open and send nothing: stdin is neither a
    // terminal nor at EOF, which is exactly a script's pipe. Keeping
    // the handle bound is what holds it open — dropping it would send
    // the EOF that masks the bug.
    let stdin = child.stdin.take().expect("stdin pipe");

    let status = wait_for(&mut child, Duration::from_secs(20));
    drop(stdin);
    let _ = std::fs::remove_dir_all(&dir);

    let status = status.expect("`-c` must exit without waiting on stdin");
    assert!(status.success(), "`-c` exited {status}");
}
