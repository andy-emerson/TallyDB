/// Links the system LAPACK, taken as-is per this crate's contract — no
/// fork, no vendored build. On Linux this is `liblapack.so` (reference
/// LAPACK or OpenBLAS's provider via the distro alternatives system);
/// install with e.g. `apt-get install liblapack-dev`. Other platforms'
/// providers (Accelerate, MKL) are an M3 concern, not wired here yet.
fn main() {
    println!("cargo:rustc-link-lib=dylib=lapack");
}
