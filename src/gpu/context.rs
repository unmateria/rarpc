use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

/// Information about a single CUDA device
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub vram_mb: usize,
}

/// Wrapper around a cudarc CudaContext + its default stream
pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub info: GpuInfo,
}

impl GpuContext {
    pub fn new(index: usize) -> Result<Self> {
        let ctx = CudaContext::new(index)
            .with_context(|| format!("Cannot open CUDA device {}", index))?;

        let name = ctx.name().unwrap_or_else(|_| format!("GPU-{}", index));
        let vram_mb = ctx.total_mem().unwrap_or(0) / 1024 / 1024;

        let stream = ctx.default_stream();

        let info = GpuInfo { index, name, vram_mb };

        Ok(Self { ctx, stream, info })
    }

    /// Allocate an additional non-blocking stream on this context. Used by the
    /// pipelined engine for CPU↔GPU overlap. The default stream (`self.stream`)
    /// stays the primary; callers never share the returned stream back.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>> {
        Ok(self.ctx.new_stream()?)
    }

    /// Enumerate all available CUDA devices
    pub fn enumerate() -> Result<Vec<GpuInfo>> {
        let count = CudaContext::device_count()
            .with_context(|| "Failed to query CUDA device count")? as usize;

        let mut result = Vec::new();
        for i in 0..count {
            if let Ok(ctx) = Self::new(i) {
                result.push(ctx.info);
            }
        }
        Ok(result)
    }
}
