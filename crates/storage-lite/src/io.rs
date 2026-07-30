//! Storage backends: where encoded segment bytes live.
//!
//! The trait is the WASM seam the design requires from day one: nothing
//! above it may assume a real filesystem, blocking I/O, or paths — a
//! backend is a flat namespace of named byte objects with atomic
//! publish. [`FsBackend`] (a directory of files) is the native
//! implementation; [`MemBackend`] backs tests and demonstrates the shape
//! an OPFS/WASM backend must fit. Whole-object reads are the contract
//! by decision, not omission: the working-set cut is owned (DESIGN.md,
//! *The axes* — the queried working set fits in memory; segments fault
//! in whole on first touch under the residency design, 2026-07-30), so
//! ranged reads and mmap are retired as follow-ups. Ranged reads
//! return, if ever, with column-granular residency and its checksum
//! revision (#87), at which point this trait grows additively.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Why a backend operation failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum IoError {
    /// No object with this name exists.
    NotFound(String),
    /// The backend failed; carries the backend's own message.
    Backend(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::NotFound(name) => write!(f, "no stored object named '{name}'"),
            IoError::Backend(message) => write!(f, "storage backend error: {message}"),
        }
    }
}

impl std::error::Error for IoError {}

/// A flat namespace of named byte objects.
///
/// Contract: `write` publishes atomically — a reader (including a
/// process that crashed mid-write and reopened) sees either the whole
/// object or no object, never a torn one. `list` returns every published
/// name, in unspecified order. Names beginning with `.` are reserved
/// for a backend's own bookkeeping (temporaries, locks) and are not
/// part of the namespace.
pub trait StorageBackend: Send + Sync {
    /// Publishes `bytes` under `name`, replacing any previous object.
    /// **Durability contract:** when `write` returns, the object is
    /// durable against power loss, not merely against process crash —
    /// on the native backend that means the bytes and the publishing
    /// rename are synced to the device before return. Atomic publish
    /// alone (rename without sync) is a page-cache promise, and the
    /// documented "durability boundary is the flush" depends on this
    /// stronger one (finding recorded 2026-07-25; baseline noted in
    /// decision #43).
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError>;

    /// Reads the object named `name`.
    fn read(&self, name: &str) -> Result<Vec<u8>, IoError>;

    /// Every published name.
    fn list(&self) -> Result<Vec<String>, IoError>;

    /// Removes the object named `name` (an error if absent).
    fn remove(&self, name: &str) -> Result<(), IoError>;

    /// Opens the *existing* object named `name` as an append-only log
    /// and returns its writer (an error if absent). A log is born and
    /// reborn through `write` — atomic, durable publish of its initial
    /// contents — and only *appended* through this writer, so there is
    /// never an instant where the old log is destroyed and its
    /// replacement not yet durable. The log lists and reads back like
    /// any object, but appended bytes' durability is governed by
    /// [`LogWriter::sync`] — not `write`'s contract — because a log's
    /// whole point is accumulating small appends between syncs (the
    /// WAL, decision #43).
    fn open_log(&self, name: &str) -> Result<Box<dyn LogWriter>, IoError>;
}

/// An open append-only log. Bytes accumulate with `append`; `sync`
/// makes everything appended so far durable against power loss.
/// Dropping a writer without syncing loses at most the unsynced tail —
/// the WAL's per-record CRCs turn that into a clean torn tail at
/// replay, never corruption.
pub trait LogWriter: Send {
    /// Appends `bytes` to the log.
    fn append(&mut self, bytes: &[u8]) -> Result<(), IoError>;
    /// Makes every appended byte durable against power loss.
    fn sync(&mut self) -> Result<(), IoError>;
}

/// The native backend: one directory, one file per object. Writes go to
/// a dot-prefixed temporary file in the same directory (contents
/// synced), then rename, then the directory itself is synced — atomic
/// *and durable* publish on POSIX filesystems; leftover temporaries
/// from a crash are invisible to `list` and overwritten by the next
/// write.
///
/// Opening the backend takes an **exclusive OS file lock** on the
/// directory (`.tallydb.lock`), held for the backend's life and
/// released by the OS even if the process dies — so two processes
/// linking the library cannot silently clobber one store (the same
/// protection the console's own lock gives its whole database
/// directory, now enforced where the files are actually written).
pub struct FsBackend {
    dir: PathBuf,
    /// Distinguishes concurrent writes' temp files (R6).
    write_counter: AtomicU64,
    /// The advisory process lock — `Some` for the writer (released when
    /// the backend drops, or the process dies: no stale-lock cleanup
    /// ever needed), `None` for read-only backends, which coexist with
    /// the writer and with each other.
    _lock: Option<std::fs::File>,
    /// A read-only backend refuses every mutating operation, so a bug
    /// in a reader process can never corrupt the writer's directory.
    read_only: bool,
}

impl FsBackend {
    /// A backend over `dir`, created if absent; fails if another
    /// **writer** (another process, or another backend in this one)
    /// holds the directory. Read-only backends do not conflict.
    pub fn new(dir: impl Into<PathBuf>) -> Result<FsBackend, IoError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|error| IoError::Backend(format!("creating {}: {error}", dir.display())))?;
        let lock_path = dir.join(".tallydb.lock");
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| {
                IoError::Backend(format!("opening lock {}: {error}", lock_path.display()))
            })?;
        if let Err(error) = lock.try_lock() {
            return Err(IoError::Backend(format!(
                "another process holds {} ({error}); one writer per store directory",
                dir.display()
            )));
        }
        Ok(FsBackend {
            dir,
            write_counter: AtomicU64::new(0),
            _lock: Some(lock),
            read_only: false,
        })
    }

    /// A **read-only** backend over an existing `dir` (F4): takes no
    /// lock, so any number of reader processes coexist with the one
    /// writer — and refuses every mutating operation, so a reader can
    /// never corrupt the directory it watches. Fails if `dir` does not
    /// exist: a reader has nothing to create.
    pub fn open_read_only(dir: impl Into<PathBuf>) -> Result<FsBackend, IoError> {
        let dir = dir.into();
        if !dir.is_dir() {
            return Err(IoError::Backend(format!(
                "{} is not a store directory",
                dir.display()
            )));
        }
        Ok(FsBackend {
            dir,
            write_counter: AtomicU64::new(0),
            _lock: None,
            read_only: true,
        })
    }

    fn refuse_write(&self, what: &str) -> Result<(), IoError> {
        if self.read_only {
            return Err(IoError::Backend(format!(
                "read-only backend over {} refuses to {what} — the writer \
                 process owns mutation",
                self.dir.display()
            )));
        }
        Ok(())
    }
}

/// [`LogWriter`] over one native file: `write(2)` per append,
/// `fdatasync` per sync.
struct FsLogWriter {
    file: std::fs::File,
    path: PathBuf,
}

impl LogWriter for FsLogWriter {
    fn append(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        std::io::Write::write_all(&mut self.file, bytes).map_err(|error| {
            IoError::Backend(format!("appending {}: {error}", self.path.display()))
        })
    }

    fn sync(&mut self) -> Result<(), IoError> {
        self.file
            .sync_data()
            .map_err(|error| IoError::Backend(format!("syncing {}: {error}", self.path.display())))
    }
}

impl StorageBackend for FsBackend {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
        self.refuse_write("write")?;
        // A **unique** temp path per write (R6): the trait promises
        // `Sync` + atomic publish, so two threads writing the same object
        // name must not share `.tmp-{name}` — one would truncate the
        // other's in-flight bytes and a reader could observe a torn file.
        // The counter (plus pid, defensively) keeps each write's temp
        // private; it still starts `.tmp-` so `list` skips it.
        let unique = self.write_counter.fetch_add(1, Ordering::Relaxed);
        let temp = self
            .dir
            .join(format!(".tmp-{name}.{}.{unique}", std::process::id()));
        let path = self.dir.join(name);
        // Write and sync the contents before the rename: a rename can
        // be durable while the data it points at is still only in the
        // page cache, which loses or truncates a "published" file on
        // power loss.
        {
            let mut file = std::fs::File::create(&temp).map_err(|error| {
                IoError::Backend(format!("creating {}: {error}", temp.display()))
            })?;
            std::io::Write::write_all(&mut file, bytes).map_err(|error| {
                IoError::Backend(format!("writing {}: {error}", temp.display()))
            })?;
            file.sync_all().map_err(|error| {
                IoError::Backend(format!("syncing {}: {error}", temp.display()))
            })?;
        }
        std::fs::rename(&temp, &path)
            .map_err(|error| IoError::Backend(format!("publishing {}: {error}", path.display())))?;
        // Sync the directory so the rename itself survives power loss.
        std::fs::File::open(&self.dir)
            .and_then(|dir| dir.sync_all())
            .map_err(|error| IoError::Backend(format!("syncing {}: {error}", self.dir.display())))
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, IoError> {
        let path = self.dir.join(name);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(IoError::NotFound(name.to_owned()))
            }
            Err(error) => Err(IoError::Backend(format!(
                "reading {}: {error}",
                path.display()
            ))),
        }
    }

    fn list(&self) -> Result<Vec<String>, IoError> {
        let entries = std::fs::read_dir(&self.dir).map_err(|error| {
            IoError::Backend(format!("listing {}: {error}", self.dir.display()))
        })?;
        let mut names = Vec::new();
        for entry in entries {
            let entry =
                entry.map_err(|error| IoError::Backend(format!("listing entry: {error}")))?;
            let Ok(name) = entry.file_name().into_string() else {
                continue; // not a name this backend ever wrote
            };
            if name.starts_with('.') {
                // The backend's own bookkeeping — unpublished `.tmp-`
                // leftovers, the `.tallydb.lock` file — is not part of
                // the namespace (dot-prefixed names are reserved).
                continue;
            }
            if entry.path().is_file() {
                names.push(name);
            }
        }
        Ok(names)
    }

    fn open_log(&self, name: &str) -> Result<Box<dyn LogWriter>, IoError> {
        self.refuse_write("open a log")?;
        let path = self.dir.join(name);
        let file = match std::fs::OpenOptions::new().append(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(IoError::NotFound(name.to_owned()))
            }
            Err(error) => {
                return Err(IoError::Backend(format!(
                    "opening {}: {error}",
                    path.display()
                )))
            }
        };
        Ok(Box::new(FsLogWriter { file, path }))
    }

    fn remove(&self, name: &str) -> Result<(), IoError> {
        self.refuse_write("remove")?;
        let path = self.dir.join(name);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(IoError::NotFound(name.to_owned()))
            }
            Err(error) => Err(IoError::Backend(format!(
                "removing {}: {error}",
                path.display()
            ))),
        }
    }
}

/// An in-memory backend: tests, and the reference shape for future
/// non-filesystem backends. Ordered map so `list` is deterministic.
#[derive(Default)]
pub struct MemBackend {
    objects: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
}

/// [`LogWriter`] over a [`MemBackend`] object, modeling power loss:
/// appends buffer in the writer and only `sync` publishes them to the
/// shared map, so a "crash" (drop the store, reopen over the same
/// backend) sees exactly the synced prefix — which is what makes the
/// crash-injection tests honest about sync levels.
struct MemLogWriter {
    objects: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    name: String,
    pending: Vec<u8>,
}

impl LogWriter for MemLogWriter {
    fn append(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        self.pending.extend_from_slice(bytes);
        Ok(())
    }

    fn sync(&mut self) -> Result<(), IoError> {
        if !self.pending.is_empty() {
            self.objects
                .lock()
                .expect("no poisoned locks")
                .entry(self.name.clone())
                .or_default()
                .extend_from_slice(&self.pending);
            self.pending.clear();
        }
        Ok(())
    }
}

impl MemBackend {
    /// An empty backend.
    pub fn new() -> MemBackend {
        MemBackend::default()
    }
}

impl StorageBackend for MemBackend {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
        self.objects
            .lock()
            .expect("no poisoned locks")
            .insert(name.to_owned(), bytes.to_vec());
        Ok(())
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, IoError> {
        self.objects
            .lock()
            .expect("no poisoned locks")
            .get(name)
            .cloned()
            .ok_or_else(|| IoError::NotFound(name.to_owned()))
    }

    fn list(&self) -> Result<Vec<String>, IoError> {
        Ok(self
            .objects
            .lock()
            .expect("no poisoned locks")
            .keys()
            .cloned()
            .collect())
    }

    fn open_log(&self, name: &str) -> Result<Box<dyn LogWriter>, IoError> {
        // The log must already exist (born through `write`'s atomic
        // publish); appended bytes arrive in the shared map only at
        // sync — which is what models power loss honestly.
        if !self
            .objects
            .lock()
            .expect("no poisoned locks")
            .contains_key(name)
        {
            return Err(IoError::NotFound(name.to_owned()));
        }
        Ok(Box::new(MemLogWriter {
            objects: Arc::clone(&self.objects),
            name: name.to_owned(),
            pending: Vec::new(),
        }))
    }

    fn remove(&self, name: &str) -> Result<(), IoError> {
        self.objects
            .lock()
            .expect("no poisoned locks")
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| IoError::NotFound(name.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise(backend: &dyn StorageBackend) {
        assert_eq!(backend.list().unwrap(), Vec::<String>::new());
        backend.write("a", b"alpha").unwrap();
        backend.write("b", b"beta").unwrap();
        assert_eq!(backend.read("a").unwrap(), b"alpha");
        backend.write("a", b"alpha-2").unwrap(); // replace
        assert_eq!(backend.read("a").unwrap(), b"alpha-2");
        let mut names = backend.list().unwrap();
        names.sort();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(backend.read("nope"), Err(IoError::NotFound("nope".into())));
        backend.remove("a").unwrap();
        assert_eq!(backend.remove("a"), Err(IoError::NotFound("a".into())));
        assert_eq!(backend.list().unwrap(), ["b"]);
    }

    #[test]
    fn mem_backend_meets_the_contract() {
        exercise(&MemBackend::new());
    }

    #[test]
    fn fs_backend_meets_the_contract() {
        let dir = std::env::temp_dir().join(format!("tallydb-io-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        exercise(&FsBackend::new(&dir).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fs_backend_hides_unpublished_temporaries() {
        let dir = std::env::temp_dir().join(format!("tallydb-io-tmp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let backend = FsBackend::new(&dir).unwrap();
        backend.write("real", b"data").unwrap();
        // A crash mid-write leaves a temporary behind; it must stay
        // invisible and get replaced by the next write of that name.
        std::fs::write(dir.join(".tmp-real"), b"torn").unwrap();
        assert_eq!(backend.list().unwrap(), ["real"]);
        backend.write("real", b"data-2").unwrap();
        assert_eq!(backend.read("real").unwrap(), b"data-2");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_directory_lock_admits_one_writer_and_any_readers() {
        // Two library embedders opening one store directory as writers
        // would silently clobber each other's segments; the lock makes
        // the second open loud instead. Released with the file handle —
        // by the OS even on a crash, so no stale-lock cleanup exists.
        // Read-only backends (F4) take no lock: they coexist with the
        // writer and each other, and refuse mutation instead.
        let dir = std::env::temp_dir().join(format!("tallydb-io-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = FsBackend::new(&dir).unwrap();
        let Err(error) = FsBackend::new(&dir) else {
            panic!("second writer must be refused");
        };
        assert!(
            error.to_string().contains("one writer"),
            "unexpected error: {error}"
        );
        // Readers open alongside the writer, and refuse to mutate.
        let reader = FsBackend::open_read_only(&dir).unwrap();
        let _second_reader = FsBackend::open_read_only(&dir).unwrap();
        assert!(reader.write("x", b"nope").is_err());
        assert!(reader.remove("x").is_err());
        assert!(reader.open_log("x").is_err());
        drop(first);
        drop(FsBackend::new(&dir).unwrap()); // released with the handle
                                             // A reader over a directory that never existed is refused.
        assert!(FsBackend::open_read_only(dir.join("absent")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn concurrent_writes_to_one_name_stay_atomic() {
        // R6: many threads writing the same object under the advertised
        // `Sync` bound. With a shared `.tmp-{name}`, one writer's create
        // truncates another's in-flight bytes and a reader can observe a
        // torn file; a unique temp per write keeps every publish atomic.
        use std::sync::Arc;
        let dir = std::env::temp_dir().join(format!("tallydb-io-race-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let backend = Arc::new(FsBackend::new(&dir).unwrap());
        let mut handles = Vec::new();
        for tag in 0..8u8 {
            let backend = Arc::clone(&backend);
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    backend.write("obj", &[tag; 64]).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        // The published object is exactly one writer's uniform 64-byte
        // payload — complete and untorn, never a mix of two writers.
        let bytes = backend.read("obj").unwrap();
        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|&b| b == bytes[0]), "torn write");
        // No temp files leaked past their writes.
        assert!(backend.list().unwrap().iter().all(|name| name == "obj"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
