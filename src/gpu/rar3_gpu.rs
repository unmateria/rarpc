use anyhow::{bail, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::gpu::batch::pack_passwords;
use crate::gpu::context::GpuContext;

const NO_MATCH: i32 = -1;
const RAR3_ITERS: u32 = 0x40000;      // 262144
const RAR3_LOOP_CNT: u32 = 0x4000;    // 16384 iters per launch
const RAR3_NUM_LOOPS: u32 = RAR3_ITERS / RAR3_LOOP_CNT; // 16
const TMPS_SIZE: usize = 396;         // sizeof(rar3_tmps)

pub struct Rar3Gpu {
    stream: Arc<CudaStream>,
    func_init: CudaFunction,
    func_loop: CudaFunction,
    func_loop_ilp2: CudaFunction,
    func_loop_lb: CudaFunction,
    func_comp: CudaFunction,
    func_crack_mono: CudaFunction,
}

impl Rar3Gpu {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let ptx = Ptx::from_src(crate::RAR3_PTX);
        let module = ctx.ctx.load_module(ptx)?;
        let func_init = module.load_function("rar3_init")?;
        let func_loop = module.load_function("rar3_loop")?;
        let func_loop_ilp2 = module.load_function("rar3_loop_ilp2")?;
        let func_loop_lb = module.load_function("rar3_loop_lb")?;
        let func_comp = module.load_function("rar3_comp")?;
        let func_crack_mono = module.load_function("rar3_crack")?;
        Ok(Self {
            stream: ctx.stream.clone(),
            func_init,
            func_loop,
            func_loop_ilp2,
            func_loop_lb,
            func_comp,
            func_crack_mono,
        })
    }

    pub fn crack_batch(
        &self,
        passwords_utf16: &[Vec<u8>],
        salt: &[u8; 8],
        enc_block: &[u8; 16],
        check_mode: i32,
        head_type: i32,
        file_crc: u32,
        pack_size: i32,
    ) -> Result<Option<usize>> {
        if use_mono() {
            return self.crack_batch_mono(
                passwords_utf16, salt, enc_block, check_mode, head_type, file_crc, pack_size,
            );
        }
        // Kernel selection: largeblock is default (+~10% vs standard).
        // RARPC_RAR3_ILP=2 → ILP2 kernel; RARPC_RAR3_CLASSIC=1 → old switch-based loop.
        let kernel = if use_ilp2() {
            LoopKernel::Ilp2
        } else if use_classic() {
            LoopKernel::Standard
        } else {
            LoopKernel::Lb
        };
        self.crack_batch_split(
            passwords_utf16, salt, enc_block, check_mode, head_type, file_crc, pack_size,
            kernel,
        )
    }

    fn crack_batch_split(
        &self,
        passwords_utf16: &[Vec<u8>],
        salt: &[u8; 8],
        enc_block: &[u8; 16],
        check_mode: i32,
        head_type: i32,
        file_crc: u32,
        pack_size: i32,
        kernel: LoopKernel,
    ) -> Result<Option<usize>> {
        if passwords_utf16.is_empty() { return Ok(None); }

        let n = passwords_utf16.len() as i32;
        let (flat, lengths) = pack_passwords(passwords_utf16);
        let stream = &self.stream;

        let d_passwords: CudaSlice<u8>  = stream.clone_htod(&flat)?;
        let d_lengths:   CudaSlice<i32> = stream.clone_htod(&lengths)?;
        let d_salt:      CudaSlice<u8>  = stream.clone_htod(salt)?;
        let d_enc:       CudaSlice<u8>  = stream.clone_htod(enc_block)?;
        let mut d_result: CudaSlice<i32> = stream.clone_htod(&[NO_MATCH])?;

        // Allocate tmps buffer (396 bytes per thread)
        let tmps_bytes = (n as usize) * TMPS_SIZE;
        let zeros = vec![0u8; tmps_bytes];
        let mut d_tmps: CudaSlice<u8> = stream.clone_htod(&zeros)?;

        // ── rar3_init ──
        {
            const BLOCK: u32 = 256;
            let grid = ((n as u32) + BLOCK - 1) / BLOCK;
            let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            let mut args = stream.launch_builder(&self.func_init);
            args.arg(&d_passwords);
            args.arg(&d_lengths);
            args.arg(&n);
            args.arg(&d_salt);
            args.arg(&mut d_tmps);
            unsafe { args.launch(cfg) }?;
        }

        // ── rar3_loop × 16 ──
        match kernel {
            LoopKernel::Ilp2 => {
                const BLOCK: u32 = 64;
                let threads_needed = ((n as u32) + 1) / 2;
                let grid = (threads_needed + BLOCK - 1) / BLOCK;
                let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
                for launch in 0..RAR3_NUM_LOOPS {
                    let loop_pos: u32 = launch * RAR3_LOOP_CNT;
                    let loop_cnt: u32 = RAR3_LOOP_CNT;
                    let mut args = stream.launch_builder(&self.func_loop_ilp2);
                    args.arg(&mut d_tmps);
                    args.arg(&n);
                    args.arg(&loop_pos);
                    args.arg(&loop_cnt);
                    unsafe { args.launch(cfg) }?;
                }
            }
            LoopKernel::Lb => {
                const BLOCK: u32 = 256;
                let grid = ((n as u32) + BLOCK - 1) / BLOCK;
                let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
                for launch in 0..RAR3_NUM_LOOPS {
                    let loop_pos: u32 = launch * RAR3_LOOP_CNT;
                    let loop_cnt: u32 = RAR3_LOOP_CNT;
                    let mut args = stream.launch_builder(&self.func_loop_lb);
                    args.arg(&mut d_tmps);
                    args.arg(&n);
                    args.arg(&loop_pos);
                    args.arg(&loop_cnt);
                    unsafe { args.launch(cfg) }?;
                }
            }
            LoopKernel::Standard => {
                const BLOCK: u32 = 128;
                let grid = ((n as u32) + BLOCK - 1) / BLOCK;
                let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
                for launch in 0..RAR3_NUM_LOOPS {
                    let loop_pos: u32 = launch * RAR3_LOOP_CNT;
                    let loop_cnt: u32 = RAR3_LOOP_CNT;
                    let mut args = stream.launch_builder(&self.func_loop);
                    args.arg(&mut d_tmps);
                    args.arg(&n);
                    args.arg(&loop_pos);
                    args.arg(&loop_cnt);
                    unsafe { args.launch(cfg) }?;
                }
            }
        }

        // ── rar3_comp ──
        {
            const BLOCK: u32 = 256;
            let grid = ((n as u32) + BLOCK - 1) / BLOCK;
            let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            let mut args = stream.launch_builder(&self.func_comp);
            args.arg(&d_tmps);
            args.arg(&n);
            args.arg(&d_enc);
            args.arg(&check_mode);
            args.arg(&head_type);
            args.arg(&file_crc);
            args.arg(&pack_size);
            args.arg(&mut d_result);
            unsafe { args.launch(cfg) }?;
        }

        stream.synchronize()?;

        let result: Vec<i32> = stream.clone_dtoh(&d_result)?;
        let idx = result[0];

        if idx == NO_MATCH {
            Ok(None)
        } else if (idx as usize) < passwords_utf16.len() {
            Ok(Some(idx as usize))
        } else {
            bail!("GPU returned out-of-range index {}", idx)
        }
    }

    fn crack_batch_mono(
        &self,
        passwords_utf16: &[Vec<u8>],
        salt: &[u8; 8],
        enc_block: &[u8; 16],
        check_mode: i32,
        head_type: i32,
        file_crc: u32,
        pack_size: i32,
    ) -> Result<Option<usize>> {
        if passwords_utf16.is_empty() { return Ok(None); }

        let n = passwords_utf16.len() as i32;
        let (flat, lengths) = pack_passwords(passwords_utf16);
        let stream = &self.stream;

        let d_passwords: CudaSlice<u8>  = stream.clone_htod(&flat)?;
        let d_lengths:   CudaSlice<i32> = stream.clone_htod(&lengths)?;
        let d_salt:      CudaSlice<u8>  = stream.clone_htod(salt)?;
        let d_enc:       CudaSlice<u8>  = stream.clone_htod(enc_block)?;
        let mut d_result: CudaSlice<i32> = stream.clone_htod(&[NO_MATCH])?;

        const BLOCK: u32 = 128;
        let grid = ((n as u32) + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };

        let mut args = stream.launch_builder(&self.func_crack_mono);
        args.arg(&d_passwords);
        args.arg(&d_lengths);
        args.arg(&n);
        args.arg(&d_salt);
        args.arg(&d_enc);
        args.arg(&check_mode);
        args.arg(&head_type);
        args.arg(&file_crc);
        args.arg(&pack_size);
        args.arg(&mut d_result);
        unsafe { args.launch(cfg) }?;

        stream.synchronize()?;

        let result: Vec<i32> = stream.clone_dtoh(&d_result)?;
        let idx = result[0];

        if idx == NO_MATCH {
            Ok(None)
        } else if (idx as usize) < passwords_utf16.len() {
            Ok(Some(idx as usize))
        } else {
            bail!("GPU returned out-of-range index {}", idx)
        }
    }
}

#[derive(Clone, Copy)]
enum LoopKernel { Standard, Ilp2, Lb }

/// RARPC_RAR3_MONO=1 → monolithic kernel (parity reference).
fn use_mono() -> bool {
    matches!(std::env::var("RARPC_RAR3_MONO").as_deref(), Ok("1") | Ok("true"))
}

/// RARPC_RAR3_ILP=2 → ILP=2 loop (each thread handles 2 passwords).
fn use_ilp2() -> bool {
    matches!(std::env::var("RARPC_RAR3_ILP").as_deref(), Ok("2"))
}

/// RARPC_RAR3_CLASSIC=1 → use old switch_buf_carry loop (for comparison).
fn use_classic() -> bool {
    matches!(std::env::var("RARPC_RAR3_CLASSIC").as_deref(), Ok("1") | Ok("true"))
}
