fn main() {
    // Python extension modules on macOS must allow undefined Python symbols
    // to be resolved at runtime when loaded by the Python interpreter.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
