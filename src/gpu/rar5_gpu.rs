use anyhow::{bail, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::gpu::batch::pack_passwords;
use crate::gpu::context::GpuContext;
use crate::rar::rar5::Rar5Info;

const NO_MATCH: i32 = -1;

/// RAR5 GPU cracker. Owns an Arc<CudaStream> so Engine can store it without
/// a lifetime tied to GpuContext — PTX is compiled once on construction.
pub struct Rar5Gpu {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

/// Uploaded + launched batch waiting on GPU. Moves ownership of the host
/// buffers into the struct: async memcpy_htod for regular Vec is recorded
/// without sync-on-drop, so the buffers must outlive the stream sync that
/// `fetch_result` performs.
pub struct Rar5InFlight {
    _host_flat:     Vec<u8>,
    _host_lengths:  Vec<i32>,
    _host_salt:     [u8; 16],
    _host_psw:      [u8; 12],
    _host_result:   [i32; 1],

    d_passwords:    CudaSlice<u8>,
    d_lengths:      CudaSlice<i32>,
    d_salt:         CudaSlice<u8>,
    d_psw_check:    CudaSlice<u8>,
    d_result:       CudaSlice<i32>,

    n: i32,
    iter_count: i32,

    stream: Arc<CudaStream>,
    pws_owned: Vec<Vec<u8>>,
}

impl Rar5InFlight {
    /// How many candidates this batch carries.
    pub fn len(&self) -> usize { self.n as usize }
    /// Peek at the first password (for progress-bar display without sync).
    pub fn first_sample(&self) -> String {
        self.pws_owned
            .first()
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .unwrap_or_default()
    }
}

impl Rar5Gpu {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let ptx = Ptx::from_src(crate::RAR5_PTX);
        let module = ctx.ctx.load_module(ptx)?;
        let func = module.load_function("rar5_crack")?;
        Ok(Self { stream: ctx.stream.clone(), func })
    }

    /// The stream this instance was built on (used by the non-pipelined path).
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }

    // ── Non-pipelined path (legacy) ───────────────────────────────────────

    /// Synchronous: upload + launch + sync + fetch index. Kept for RAR5 callers
    /// that don't want to juggle an InFlight (and for every RAR3 code path).
    pub fn crack_batch(
        &self,
        passwords: &[Vec<u8>],
        info: &Rar5Info,
    ) -> Result<Option<usize>> {
        if passwords.is_empty() {
            return Ok(None);
        }
        // Clone into owned Vec for InFlight; then unwrap after fetch to
        // preserve the existing `Option<usize>` contract.
        let pws_owned = passwords.to_vec();
        let mut inflight = match self.upload_async(pws_owned, info, self.stream.clone())? {
            Some(i) => i,
            None    => return Ok(None),
        };
        self.launch_async(&mut inflight)?;
        inflight.stream.synchronize()?;
        let result: Vec<i32> = inflight.stream.clone_dtoh(&inflight.d_result)?;
        let idx = result[0];
        if idx == NO_MATCH {
            Ok(None)
        } else if (idx as usize) < passwords.len() {
            Ok(Some(idx as usize))
        } else {
            bail!("GPU returned out-of-range index {}", idx)
        }
    }

    // ── Pipelined path ────────────────────────────────────────────────────

    /// Issue the H→D copies for one batch on `stream`. Returns `None` if the
    /// batch is empty or the archive lacks a fast-check (PswCheckData).
    pub fn upload_async(
        &self,
        passwords: Vec<Vec<u8>>,
        info: &Rar5Info,
        stream: Arc<CudaStream>,
    ) -> Result<Option<Rar5InFlight>> {
        if passwords.is_empty() { return Ok(None); }
        let pcd = match &info.psw_check_data {
            Some(p) => p,
            None    => return Ok(None),
        };

        let mut host_psw = [0u8; 12];
        host_psw[..8].copy_from_slice(&pcd.init_v);
        host_psw[8..].copy_from_slice(&pcd.check);

        let host_salt = info.salt;
        let host_result = [NO_MATCH];
        let n = passwords.len() as i32;
        let (host_flat, host_lengths) = pack_passwords(&passwords);

        let d_passwords = stream.clone_htod(&host_flat)?;
        let d_lengths   = stream.clone_htod(&host_lengths)?;
        let d_salt      = stream.clone_htod(host_salt.as_slice())?;
        let d_psw_check = stream.clone_htod(host_psw.as_slice())?;
        let d_result    = stream.clone_htod(host_result.as_slice())?;

        Ok(Some(Rar5InFlight {
            _host_flat:    host_flat,
            _host_lengths: host_lengths,
            _host_salt:    host_salt,
            _host_psw:     host_psw,
            _host_result:  host_result,
            d_passwords, d_lengths, d_salt, d_psw_check, d_result,
            n,
            iter_count: info.iter_count as i32,
            stream,
            pws_owned: passwords,
        }))
    }

    /// Queue the kernel launch on the InFlight's stream. Non-blocking.
    pub fn launch_async(&self, inflight: &mut Rar5InFlight) -> Result<()> {
        const BLOCK: u32 = 256;
        let grid = ((inflight.n as u32) + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let n = inflight.n;
        let iter = inflight.iter_count;

        let mut args = inflight.stream.launch_builder(&self.func);
        args.arg(&inflight.d_passwords);
        args.arg(&inflight.d_lengths);
        args.arg(&n);
        args.arg(&inflight.d_salt);
        args.arg(&iter);
        args.arg(&inflight.d_psw_check);
        args.arg(&mut inflight.d_result);
        unsafe { args.launch(cfg) }?;
        Ok(())
    }

    /// Synchronise the InFlight's stream and fetch the matching password, if any.
    /// Consumes the InFlight so its host buffers are dropped after sync completes.
    pub fn fetch_result(&self, inflight: Rar5InFlight) -> Result<Option<String>> {
        inflight.stream.synchronize()?;
        let result: Vec<i32> = inflight.stream.clone_dtoh(&inflight.d_result)?;
        let idx = result[0];
        if idx == NO_MATCH {
            Ok(None)
        } else if (idx as usize) < inflight.pws_owned.len() {
            Ok(Some(
                String::from_utf8_lossy(&inflight.pws_owned[idx as usize]).into_owned(),
            ))
        } else {
            bail!("GPU returned out-of-range index {}", idx)
        }
    }
}
