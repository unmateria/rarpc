use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Find the x64 cl.exe from Visual Studio 2022 Enterprise (or any VS install).
fn find_cl_exe() -> Option<PathBuf> {
    // Prefer the Hostx64/x64 toolchain (64-bit host, 64-bit target)
    let candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64",
    ];
    for dir in &candidates {
        let p = PathBuf::from(dir).join("cl.exe");
        if p.exists() {
            return Some(PathBuf::from(dir));
        }
    }

    // Fallback: glob for any MSVC version in VS 2022
    let bases = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    ];
    for base in &bases {
        let base_path = PathBuf::from(base);
        if let Ok(entries) = std::fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("bin").join("Hostx64").join("x64");
                if candidate.join("cl.exe").exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

fn compile_cuda_to_ptx(src: &str, kernels_dir: &PathBuf, out_dir: &PathBuf, cl_bin_dir: Option<&PathBuf>) {
    let src_path = PathBuf::from(src);
    let stem = src_path.file_stem().unwrap().to_str().unwrap();
    let out_path = out_dir.join(format!("{}.ptx", stem));

    println!("cargo:rerun-if-changed={}", src_path.display());

    let mut cmd = Command::new("nvcc");
    cmd.arg("--ptx")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--extra-device-vectorization")
        // Target virtual arch sm_89 (Ada Lovelace — RTX 4060 Ti, 4070+).
        // The PTX is still JIT'd by the driver but reflects full Ada ISA:
        // IADD3, LOP3, SHF.WRAP, improved register alloc. Runs on any Ada
        // or newer GPU; older GPUs (Ampere, Turing) won't load — adjust if
        // you need Turing support (compute_75 is the older floor).
        .arg("-arch=compute_89")
        // ptxas optimisation flags (these reach the back-end optimizer).
        .arg("-Xptxas")
        .arg("-O3")
        .arg("-Xptxas")
        .arg("--allow-expensive-optimizations=true")
        .arg("-Xptxas")
        .arg("-warn-spills")
        // -lineinfo: embed source/line info so Nsight Compute can
        // correlate SASS samples with .cu lines without hurting codegen.
        .arg("-lineinfo")
        .arg(format!("-I{}", kernels_dir.display()))
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path);

    // On Windows, nvcc needs cl.exe in PATH
    if let Some(cl_dir) = cl_bin_dir {
        let current_path = env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", cl_dir.display(), current_path);
        cmd.env("PATH", new_path);
    }

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "Failed to launch nvcc (is CUDA Toolkit in PATH?): {}",
            e
        )
    });

    if !status.success() {
        panic!("nvcc failed for {} (exit code {:?})", src, status.code());
    }

    println!("cargo:warning=Compiled {} → {}.ptx", stem, stem);
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let kernels_dir = manifest_dir.join("kernels");

    let cl_dir = find_cl_exe();
    if cl_dir.is_none() {
        println!(
            "cargo:warning=cl.exe not found; nvcc may fail. \
             Run cargo from a Visual Studio Developer Command Prompt."
        );
    }
    let cl_dir_ref = cl_dir.as_ref();

    compile_cuda_to_ptx("kernels/rar3_kdf.cu",     &kernels_dir, &out_dir, cl_dir_ref);
    compile_cuda_to_ptx("kernels/rar5_kdf.cu",     &kernels_dir, &out_dir, cl_dir_ref);
    compile_cuda_to_ptx("kernels/rar15_filter.cu", &kernels_dir, &out_dir, cl_dir_ref);

    // Re-run when any header changes
    for header in &["common.cuh", "sha1_device.cuh", "sha1_hc.cuh", "sha1_hc_switch.cuh", "sha1_hc_carry.inc", "sha256_device.cuh", "aes_device.cuh"] {
        println!(
            "cargo:rerun-if-changed={}",
            kernels_dir.join(header).display()
        );
    }
}
