//! RAR 1.5 GPU probabilistic filter (Arq B M1).
//!
//! Decrypts the first K bytes of the packed stream per candidate and runs a
//! reduced Unpack15 state machine; surviving candidates are passed to the CPU
//! for strict verify. See `kernels/rar15_filter.cu` + `rar15_filter_cpu`.

use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::gpu::batch::pack_passwords;
use crate::gpu::context::GpuContext;
use crate::rar::rar15::{Rar15FilterParams, Rar15Info};

const MAX_K_BYTES: usize = 512;

pub struct Rar15Gpu {
    stream: Arc<CudaStream>,
    func:   CudaFunction,
    d_packed: CudaSlice<u8>, // constant-like, re-uploaded per batch via global-memory fallback
    params: Rar15FilterParams,
    unp_size: i64,
    k_bytes:  i32,
}

impl Rar15Gpu {
    /// Build a GPU filter bound to one archive. Truncates and uploads the
    /// first `params.k_bytes` of the packed stream.
    pub fn new_for_archive(
        ctx: &GpuContext,
        info: &Rar15Info,
        params: Rar15FilterParams,
    ) -> Result<Self> {
        let ptx = Ptx::from_src(crate::RAR15_PTX);
        let module = ctx.ctx.load_module(ptx)?;
        let func = module.load_function("rar15_filter")?;

        let k = params.k_bytes.min(MAX_K_BYTES).min(info.packed_data.len());
        let mut buf = vec![0u8; MAX_K_BYTES];
        buf[..k].copy_from_slice(&info.packed_data[..k]);
        let d_packed = ctx.stream.clone_htod(&buf)?;

        Ok(Self {
            stream: ctx.stream.clone(),
            func,
            d_packed,
            params,
            unp_size: info.unp_size as i64,
            k_bytes:  k as i32,
        })
    }

    /// Run the filter over a batch. Returns the indices of survivors.
    pub fn filter_batch(&self, passwords: &[Vec<u8>]) -> Result<Vec<usize>> {
        if passwords.is_empty() { return Ok(Vec::new()); }

        let n = passwords.len();
        let (host_flat, host_lengths) = pack_passwords(passwords);
        let d_pw  = self.stream.clone_htod(&host_flat)?;
        let d_len = self.stream.clone_htod(&host_lengths)?;

        let n_words = (n + 31) / 32;
        let host_bitmap = vec![0u32; n_words];
        let mut d_bitmap = self.stream.clone_htod(&host_bitmap)?;

        const BLOCK: u32 = 32;
        let grid = ((n as u32) + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid_dim:  (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32        = n as i32;
        let k_bytes      = self.k_bytes;
        let n_iters      = self.params.n_iters as i32;
        let dest_max     = self.params.dest_max;
        let unp_size     = self.unp_size;

        let mut args = self.stream.launch_builder(&self.func);
        args.arg(&d_pw);
        args.arg(&d_len);
        args.arg(&n_i32);
        args.arg(&self.d_packed);
        args.arg(&k_bytes);
        args.arg(&n_iters);
        args.arg(&dest_max);
        args.arg(&unp_size);
        args.arg(&mut d_bitmap);
        unsafe { args.launch(cfg) }?;
        self.stream.synchronize()?;

        let bitmap: Vec<u32> = self.stream.clone_dtoh(&d_bitmap)?;
        let mut survivors = Vec::new();
        for (wi, w) in bitmap.iter().enumerate() {
            let mut bits = *w;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                let idx = wi * 32 + b;
                if idx < n { survivors.push(idx); }
                bits &= bits - 1;
            }
        }
        Ok(survivors)
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
}

