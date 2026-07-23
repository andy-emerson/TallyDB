//! Storage backends: where encoded segment bytes live.
//!
//! The trait is the WASM seam the design requires from day one: nothing
//! above it may assume a real filesystem, blocking I/O, or paths — a
//! backend is a flat namespace of named byte objects with atomic
//! publish. [`FsBackend`] (a directory of files) is the native
//! implementation; [`MemBackend`] backs tests and demonstrates the shape
//! an OPFS/WASM backend must fit. Ranged reads and mmap are recorded
//! follow-ups for when query-time pruning (M2.4) or a profiling number
//! asks for them — the trait grows additively then.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Mutex;

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
/// name, in unspecified order.
pub trait StorageBackend: Send + Sync {
    /// Publishes `bytes` under `name`, replacing any previous object.
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError>;

    /// Reads the object named `name`.
    fn read(&self, name: &str) -> Result<Vec<u8>, IoError>;

    /// Every published name.
    fn list(&self) -> Result<Vec<String>, IoError>;

    /// Removes the object named `name` (an error if absent).
    fn remove(&self, name: &str) -> Result<(), IoError>;
}

/// The native backend: one directory, one file per object. Writes go to
/// a dot-prefixed temporary file in the same directory, then rename —
/// atomic publish on POSIX filesystems; leftover temporaries from a
/// crash are invisible to `list` and overwritten by the next write.
pub struct FsBackend {
    dir: PathBuf,
}

impl FsBackend {
    /// A backend over `dir`, created if absent.
    pub fn new(dir: impl Into<PathBuf>) -> Result<FsBackend, IoError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|error| IoError::Backend(format!("creating {}: {error}", dir.display())))?;
        Ok(FsBackend { dir })
    }
}

impl StorageBackend for FsBackend {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
        let temp = self.dir.join(format!(".tmp-{name}"));
        let path = self.dir.join(name);
        std::fs::write(&temp, bytes)
            .map_err(|error| IoError::Backend(format!("writing {}: {error}", temp.display())))?;
        std::fs::rename(&temp, &path)
            .map_err(|error| IoError::Backend(format!("publishing {}: {error}", path.display())))
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
            if name.starts_with(".tmp-") {
                continue; // unpublished leftovers are invisible
            }
            if entry.path().is_file() {
                names.push(name);
            }
        }
        Ok(names)
    }

    fn remove(&self, name: &str) -> Result<(), IoError> {
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
    objects: Mutex<BTreeMap<String, Vec<u8>>>,
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
}
