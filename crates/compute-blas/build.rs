fn main() {
    // System BLAS, linked as-is (the reference implementation shipped by
    // liblapack-dev/libblas-dev; OpenBLAS drops in via the alternatives
    // system). The TARGET=SANDYBRIDGE determinism note in the crate docs
    // applies when a from-source OpenBLAS is ever pinned.
    println!("cargo:rustc-link-lib=dylib=blas");
}
