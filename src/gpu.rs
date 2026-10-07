//! WebGPU/native GPU solver backend for pow-buster message families.
//!
//! The same Rust implementation targets native `wgpu` backends and browser WebGPU on `wasm32`.

use core::{fmt, num::NonZeroU8};
use std::{borrow::Cow, string::String, vec::Vec};

use bytemuck::{Pod, Zeroable};
use sha2::{Digest, Sha256};

const WG_SIZE: u32 = 64;
const STEPS: u32 = 128;
const HASHES_PER_WG: u64 = WG_SIZE as u64 * STEPS as u64;
const CANCEL_INTERVAL: u32 = 16;
const COUNTER_SPACE: u64 = 1_u64 << 54;

/// Error returned by the GPU solver.
#[derive(Debug, Clone)]
pub struct GpuError(String);

impl GpuError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

/// Successful GPU proof-of-work solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSolution {
    /// Solver nonce.
    pub nonce: u64,
    /// Solver digest/output words.
    pub hash: [u32; 8],
    /// Candidate hashes submitted through the winning batch.
    ///
    /// Workgroups stop cooperatively after a hit, so this is an upper bound on hashes actually executed.
    pub dispatched_hashes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    words: [u32; 16],
}

/// Reusable GPU device/context shared by all GPU solver instances.
///
/// Native builds use the best backend selected by `wgpu`; wasm builds use browser WebGPU.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    in_buf: wgpu::Buffer,
    generic_pipeline: wgpu::ComputePipeline,
    multi_pipeline: wgpu::ComputePipeline,
    generic_in_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    generic_bind_group: wgpu::BindGroup,
    multi_bind_group: wgpu::BindGroup,
    max_wgs: u32,
    adapter: String,
}

impl GpuContext {
    /// Create a GPU solver using the high-performance adapter selected by `wgpu`.
    pub async fn create() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::new(format!("request GPU adapter: {e}")))?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?}, {})", info.name, info.backend, info.driver);
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GpuError::new(format!("request GPU device: {e}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pow-buster-anubis"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pow-buster-anubis"),
            layout: None,
            module: &shader,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let generic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pow-buster-sha256-single-block"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(generic_single_block_shader_source())),
        });
        let generic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pow-buster-sha256-single-block"),
            layout: None,
            module: &generic_shader,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let multi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pow-buster-sha256-multi-layout"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(multi_layout_shader_source())),
        });
        let multi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pow-buster-sha256-multi-layout"),
            layout: None,
            module: &multi_shader,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pow-buster-anubis-input"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let generic_in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pow-buster-sha256-single-block-input"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pow-buster-anubis-output"),
            size: 48,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pow-buster-anubis-staging"),
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pow-buster-anubis"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let generic_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pow-buster-sha256-single-block"),
            layout: &generic_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: generic_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let multi_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pow-buster-sha256-multi-layout"),
            layout: &multi_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: generic_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            in_buf,
            generic_pipeline,
            multi_pipeline,
            generic_in_buf,
            out_buf,
            staging,
            bind_group,
            generic_bind_group,
            multi_bind_group,
            max_wgs: limits.max_compute_workgroups_per_dimension,
            adapter: adapter_name,
        })
    }

    /// Submit a small representative batch to force lazy browser/driver shader setup.
    ///
    /// Browser WebGPU implementations may defer substantial pipeline work until the first submit.
    /// Extensions can call this once during startup so the first real challenge does not pay that cost.
    pub async fn warm_up(&mut self) -> Result<(), GpuError> {
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&crate::sha256::IV);
        params.words[10] = 19 * 8;
        params.words[11] = u32::MAX;
        params.words[12] = u32::MAX;
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        self.queue
            .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pow-buster-anubis-warmup"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pow-buster-anubis-warmup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
        self.queue.submit([encoder.finish()]);
        let _ = self.read_result().await?;
        Ok(())
    }

    /// Human-readable adapter/backend description.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Solve an Anubis SHA-256 challenge.
    pub async fn solve(
        &mut self,
        prefix: &[u8],
        difficulty: NonZeroU8,
    ) -> Result<GpuSolution, GpuError> {
        self.solve_with_limit(prefix, difficulty, COUNTER_SPACE)
            .await?
            .ok_or_else(|| GpuError::new("GPU search space exhausted"))
    }

    /// Solve an Anubis SHA-256 challenge while dispatching at most `max_hashes` candidates.
    pub async fn solve_with_limit(
        &mut self,
        prefix: &[u8],
        difficulty: NonZeroU8,
        max_hashes: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        if prefix.len() % 64 != 0 {
            return Err(GpuError::new(format!(
                "GPU Anubis solver requires a block-aligned prefix, got {} bytes",
                prefix.len()
            )));
        }
        if difficulty.get() > 16 {
            return Err(GpuError::new("Anubis GPU difficulty must be <= 16"));
        }
        let bitlen = prefix
            .len()
            .checked_add(19)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| GpuError::new("challenge is too large for the GPU kernel"))?;

        let mid = midstate(prefix);
        let mask = crate::compute_mask_anubis(difficulty);
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&mid);
        params.words[10] = bitlen;
        params.words[11] = (mask >> 32) as u32;
        params.words[12] = mask as u32;
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);

        let expected = 16_u64.saturating_pow(u32::from(difficulty.get()));
        let desired = expected.saturating_mul(8).max(HASHES_PER_WG);
        let limit = max_hashes.min(COUNTER_SPACE);
        let mut base = 0_u64;

        while base < limit {
            let remaining = limit - base;
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            params.words[8] = base as u32;
            params.words[9] = (base >> 32) as u32;
            self.queue
                .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pow-buster-anubis-solve"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pow-buster-anubis-solve"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            self.queue.submit([encoder.finish()]);

            let values = self.read_result().await?;
            let dispatched = base.saturating_add(u64::from(wgs) * HASHES_PER_WG);
            if values[0] != 0 {
                let counter = (u64::from(values[2]) << 32) | u64::from(values[1]);
                let nonce = counter_to_decimal_nonce(counter);
                let hash: [u32; 8] = values[3..11]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid GPU result layout"))?;
                if !verify(prefix, nonce, hash, mask) {
                    return Err(GpuError::new("GPU returned an invalid Anubis proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: dispatched,
                }));
            }
            base = dispatched;
        }

        Ok(None)
    }

    /// Solve a parsed Anubis challenge descriptor.
    #[cfg(feature = "adapter")]
    pub async fn solve_descriptor(
        &mut self,
        descriptor: &crate::adapter::anubis::ChallengeDescriptor,
    ) -> Result<GpuSolution, GpuError> {
        if descriptor.rules().algorithm() == "preact" {
            return Err(GpuError::new(
                "preact challenges do not require GPU proof-of-work",
            ));
        }
        if !descriptor.supported() {
            return Err(GpuError::new("unsupported Anubis algorithm"));
        }
        let difficulty = NonZeroU8::new(descriptor.rules().difficulty())
            .ok_or_else(|| GpuError::new("Anubis difficulty must be non-zero"))?;
        self.solve(descriptor.challenge().as_ref().as_bytes(), difficulty)
            .await
    }

    async fn solve_single_block_inner<const TYPE: u8>(
        &mut self,
        message: &crate::message::SingleBlockMessage,
        target: u64,
        mask: u64,
        limit: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        if !matches!(
            TYPE,
            crate::solver::SOLVE_TYPE_LT
                | crate::solver::SOLVE_TYPE_GT
                | crate::solver::SOLVE_TYPE_MASK
        ) {
            return Err(GpuError::new("unsupported solve type"));
        }

        if TYPE == crate::solver::SOLVE_TYPE_MASK
            && target == 0
            && single_block_fast_compatible(message)
        {
            return self
                .solve_fast_single_block_message(message, mask, limit)
                .await;
        }

        self.solve_generic_single_block::<TYPE>(message, target, mask, limit)
            .await
    }

    async fn solve_fast_single_block_message(
        &mut self,
        message: &crate::message::SingleBlockMessage,
        mask: u64,
        max_hashes: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&message.prefix_state);
        params.words[10] = message.message[15];
        params.words[11] = (mask >> 32) as u32;
        params.words[12] = if cfg!(feature = "compare-64bit") {
            mask as u32
        } else {
            0
        };
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);

        let desired = expected_work::<{ crate::solver::SOLVE_TYPE_MASK }>(0, mask)
            .saturating_mul(8)
            .max(HASHES_PER_WG);
        let limit = max_hashes.min(COUNTER_SPACE);
        let mut base = 0_u64;

        while base < limit {
            let remaining = limit - base;
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            params.words[8] = base as u32;
            params.words[9] = (base >> 32) as u32;
            self.queue
                .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pow-buster-sha256-fast-single-block"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pow-buster-sha256-fast-single-block"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            self.queue.submit([encoder.finish()]);

            let values = self.read_result().await?;
            let dispatched = base.saturating_add(u64::from(wgs) * HASHES_PER_WG);
            if values[0] != 0 {
                let counter = (u64::from(values[2]) << 32) | u64::from(values[1]);
                let nonce = counter_to_decimal_nonce(counter);
                let hash: [u32; 8] = values[3..11]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid GPU result layout"))?;
                if !verify_fast_single_block(message, nonce, hash, mask) {
                    return Err(GpuError::new("GPU returned an invalid fast-path proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: dispatched,
                }));
            }
            base = dispatched;
        }

        Ok(None)
    }

    async fn solve_generic_single_block<const TYPE: u8>(
        &mut self,
        message: &crate::message::SingleBlockMessage,
        target: u64,
        mask: u64,
        max_hashes: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let search_space = single_block_search_space(message);
        let limit = max_hashes.min(search_space);
        if limit == 0 {
            return Ok(None);
        }

        let start = if message.nonce_addend == 0 {
            100_000_000_u64
        } else {
            0
        };
        let mut params = [0_u32; 40];
        params[..8].copy_from_slice(&message.prefix_state);
        params[8..24].copy_from_slice(&message.message.0);
        params[25] = message.digit_index as u32;
        params[26] = message.nonce_addend as u32;
        params[27] = (message.nonce_addend >> 32) as u32;
        params[28] = (target >> 32) as u32;
        params[29] = target as u32;
        params[30] = (mask >> 32) as u32;
        params[31] = mask as u32;
        params[32] = u32::from(TYPE);
        params[33] = u32::from(message.no_trailing_zeros)
            | (u32::from(cfg!(feature = "compare-64bit")) << 1);
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);

        let desired = expected_work::<TYPE>(target, mask)
            .saturating_mul(8)
            .max(HASHES_PER_WG);
        let mut offset = 0_u64;

        while offset < limit {
            let remaining = limit - offset;
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            let candidate_start = start + offset;
            let candidate_end = start + offset + batch;
            params[24] = candidate_start as u32;
            params[34] = candidate_end as u32;
            self.queue
                .write_buffer(&self.generic_in_buf, 0, bytemuck::cast_slice(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pow-buster-sha256-generic-single-block"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pow-buster-sha256-generic-single-block"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.generic_pipeline);
                pass.set_bind_group(0, &self.generic_bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            self.queue.submit([encoder.finish()]);

            let values = self.read_result().await?;
            let dispatched = offset.saturating_add(batch);
            if values[0] != 0 {
                let nonce = (u64::from(values[2]) << 32) | u64::from(values[1]);
                let hash: [u32; 8] = values[3..11]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid GPU result layout"))?;
                if !verify_single_block::<TYPE>(message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid generic proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: dispatched,
                }));
            }
            offset = dispatched;
        }

        Ok(None)
    }

    async fn dispatch_multi(
        &mut self,
        params: &[u32; 64],
        workgroups: u32,
    ) -> Result<Option<(u64, [u32; 8])>, GpuError> {
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        self.queue
            .write_buffer(&self.generic_in_buf, 0, bytemuck::cast_slice(params));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pow-buster-sha256-multi-layout"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pow-buster-sha256-multi-layout"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multi_pipeline);
            pass.set_bind_group(0, &self.multi_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
        self.queue.submit([encoder.finish()]);
        let values = self.read_result().await?;
        if values[0] == 0 {
            return Ok(None);
        }
        let nonce = (u64::from(values[2]) << 32) | u64::from(values[1]);
        let hash = values[3..11]
            .try_into()
            .map_err(|_| GpuError::new("invalid GPU result layout"))?;
        Ok(Some((nonce, hash)))
    }

    async fn read_result(&self) -> Result<Vec<u32>, GpuError> {
        let slice = self.staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::new(format!("poll GPU: {e}")))?;

        rx.await
            .map_err(|_| GpuError::new("GPU map callback dropped"))?
            .map_err(|e| GpuError::new(format!("map GPU result: {e}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| GpuError::new(format!("read mapped GPU result: {e}")))?;
        let values = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        Ok(values)
    }
}

/// Backwards-compatible name for the original Anubis-only GPU context.
pub type AnubisGpuSolver = GpuContext;

/// Async GPU counterpart of the CPU single-block SHA-256 solver family.
///
/// Anubis, mCaptcha and Cap.js all route through this protocol-agnostic message type.
#[derive(Debug, Clone)]
pub struct SingleBlockSolver {
    message: crate::message::SingleBlockMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::SingleBlockMessage> for SingleBlockSolver {
    fn from(message: crate::message::SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl SingleBlockSolver {
    /// Set the maximum number of candidates dispatched by this solver.
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    ///
    /// GPU workgroups cooperatively stop after a hit, so this is an upper bound on candidates
    /// actually executed by the device.
    pub fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve this SHA-256 single-block message on the GPU.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let result = gpu
            .solve_single_block_inner::<TYPE>(&self.message, target, mask, self.limit)
            .await?;
        self.attempted_nonces = result.as_ref().map_or_else(
            || self.limit.min(single_block_search_space(&self.message)),
            |solution| solution.dispatched_hashes.min(self.limit),
        );
        Ok(result)
    }

    /// Solve and return only the nonce.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU counterpart of the CPU double-block decimal SHA-256 solver.
pub struct DoubleBlockSolver {
    message: crate::message::DoubleBlockMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::DoubleBlockMessage> for DoubleBlockSolver {
    fn from(message: crate::message::DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl DoubleBlockSolver {
    /// Set the maximum number of candidates dispatched.
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    pub fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve this double-block decimal message on the GPU.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let search_space = if self.message.nonce_addend == 0 {
            900_000_000_u64
        } else {
            1_000_000_000
        };
        let limit = self.limit.min(search_space);
        let start = if self.message.nonce_addend == 0 {
            100_000_000_u64
        } else {
            0
        };
        let desired = expected_work::<TYPE>(target, mask)
            .saturating_mul(8)
            .max(HASHES_PER_WG);
        let mut offset = 0_u64;
        while offset < limit {
            let batch = desired
                .min(limit - offset)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            let mut p = [0_u32; 64];
            p[0] = 1;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = (start + offset) as u32;
            p[5] = batch as u32;
            p[6] = (target >> 32) as u32;
            p[7] = target as u32;
            p[8] = (mask >> 32) as u32;
            p[9] = mask as u32;
            p[16..24].copy_from_slice(&self.message.prefix_state.0);
            p[24..40].copy_from_slice(&self.message.message.0);
            p[40] = self.message.nonce_addend as u32;
            p[41] = (self.message.nonce_addend >> 32) as u32;
            let bitlen = self.message.message_length.saturating_mul(8);
            p[42] = (bitlen >> 32) as u32;
            p[43] = bitlen as u32;
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = offset + batch;
                if !verify_double_block::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid double-block proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            offset += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }

    /// Solve and return only the nonce.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// GPU router for decimal SHA-256 messages, mirroring [`crate::message::DecimalMessage`].
pub enum DecimalSolver {
    /// Single-block message.
    SingleBlock(SingleBlockSolver),
    /// Double-block message.
    DoubleBlock(DoubleBlockSolver),
}

impl From<crate::message::DecimalMessage> for DecimalSolver {
    fn from(message: crate::message::DecimalMessage) -> Self {
        match message {
            crate::message::DecimalMessage::SingleBlock(m) => Self::SingleBlock(m.into()),
            crate::message::DecimalMessage::DoubleBlock(m) => Self::DoubleBlock(m.into()),
        }
    }
}

impl DecimalSolver {
    /// Set the maximum number of candidates dispatched.
    pub fn set_limit(&mut self, limit: u64) {
        match self {
            Self::SingleBlock(s) => s.set_limit(limit),
            Self::DoubleBlock(s) => s.set_limit(limit),
        }
    }
    /// Number of candidates dispatched so far.
    pub fn get_attempted_nonces(&self) -> u64 {
        match self {
            Self::SingleBlock(s) => s.get_attempted_nonces(),
            Self::DoubleBlock(s) => s.get_attempted_nonces(),
        }
    }
    /// Solve a decimal SHA-256 message.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        match self {
            Self::SingleBlock(s) => s.solve::<TYPE>(gpu, target, mask).await,
            Self::DoubleBlock(s) => s.solve::<TYPE>(gpu, target, mask).await,
        }
    }
}

/// Async GPU GoAway SHA-256 solver.
pub struct GoAwaySolver {
    message: crate::message::GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::GoAwayMessage> for GoAwaySolver {
    fn from(message: crate::message::GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl GoAwaySolver {
    /// Set the maximum number of candidates dispatched.
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    pub fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    /// Solve a GoAway message on the GPU.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let limit = self.limit.min(u64::from(u32::MAX) + 1);
        let desired = expected_work::<TYPE>(target, mask)
            .saturating_mul(8)
            .max(HASHES_PER_WG);
        let mut offset = 0_u64;
        while offset < limit {
            let batch = desired
                .min(limit - offset)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            let mut p = [0_u32; 64];
            p[0] = 2;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = offset as u32;
            p[5] = batch as u32;
            p[6] = (target >> 32) as u32;
            p[7] = target as u32;
            p[8] = (mask >> 32) as u32;
            p[9] = mask as u32;
            p[16..24].copy_from_slice(&self.message.challenge);
            p[24] = self.message.high_word;
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = offset + batch;
                if !verify_goaway::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid GoAway proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            offset += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }
    /// Solve and return only the nonce.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU solver for binary SHA-256 nonce messages.
pub struct BinarySolver {
    message: crate::message::BinaryMessage,
    attempted_nonces: u64,
    limit: u64,
}
impl From<crate::message::BinaryMessage> for BinarySolver {
    fn from(message: crate::message::BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}
impl BinarySolver {
    /// Set the maximum number of candidates dispatched.
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    pub fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    /// Solve a binary-nonce SHA-256 message on the GPU.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let bits = u32::from(self.message.nonce_byte_count.get()) * 8;
        let space = if bits == 64 { u64::MAX } else { 1_u64 << bits };
        let limit = self.limit.min(space);
        let (templates, used) = binary_templates(&self.message);
        let desired = expected_work::<TYPE>(target, mask)
            .saturating_mul(8)
            .max(HASHES_PER_WG);
        let mut base = 0_u64;
        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            let mut p = [0_u32; 64];
            p[0] = 3;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = base as u32;
            p[4] = (base >> 32) as u32;
            p[5] = batch as u32;
            p[6] = (target >> 32) as u32;
            p[7] = target as u32;
            p[8] = (mask >> 32) as u32;
            p[9] = mask as u32;
            p[10] = used;
            p[11] = u32::from(self.message.nonce_byte_count.get());
            p[12] = self.message.salt_residual_len as u32;
            p[16..24].copy_from_slice(&self.message.prefix_state.0);
            p[24..40].copy_from_slice(&templates[0]);
            p[40..56].copy_from_slice(&templates[1]);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = base + batch;
                if !verify_binary::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid binary proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }
    /// Solve and return only the nonce.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &mut GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

fn single_block_search_space(message: &crate::message::SingleBlockMessage) -> u64 {
    if message.nonce_addend == 0 {
        900_000_000
    } else {
        1_000_000_000
    }
}

fn single_block_fast_compatible(message: &crate::message::SingleBlockMessage) -> bool {
    if message.digit_index != 10
        || message.nonce_addend != 1_000_000_000_000_000_000
        || message.no_trailing_zeros
    {
        return false;
    }
    let words = &message.message.0;
    words[0] == 0x3130_3030
        && words[1] == 0x3030_3030
        && (words[2] & 0xffff_0000) == 0x3030_0000
        && words[4] == 0x0000_0080
}

fn expected_work<const TYPE: u8>(target: u64, mask: u64) -> u64 {
    let compare_mask = if cfg!(feature = "compare-64bit") {
        mask
    } else {
        mask & 0xffff_ffff_0000_0000
    };
    if TYPE == crate::solver::SOLVE_TYPE_MASK {
        return 1_u64
            .checked_shl(compare_mask.count_ones())
            .unwrap_or(u64::MAX);
    }
    let bits = if cfg!(feature = "compare-64bit") {
        64
    } else {
        32
    };
    let t = if bits == 64 {
        target as u128
    } else {
        (target >> 32) as u128
    };
    let space = 1_u128 << bits;
    let success = if TYPE == crate::solver::SOLVE_TYPE_LT {
        t.max(1)
    } else {
        space.saturating_sub(t).saturating_sub(1).max(1)
    };
    (space / success).min(u64::MAX as u128) as u64
}

fn midstate(prefix: &[u8]) -> [u32; 8] {
    let mut state = crate::sha256::IV;
    for block in prefix.chunks_exact(64) {
        let words = core::array::from_fn(|i| {
            u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap())
        });
        crate::sha256::digest_block(&mut state, &words);
    }
    state
}

fn counter_to_decimal_nonce(counter: u64) -> u64 {
    let mut nonce = 1_u64;
    for digit in (0..18).rev() {
        nonce = nonce * 10 + ((counter >> (digit * 3)) & 7);
    }
    nonce
}

fn verify(prefix: &[u8], nonce: u64, hash: [u32; 8], mask: u64) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(nonce.to_string().as_bytes());
    let digest = hasher.finalize();
    let expected =
        core::array::from_fn(|i| u32::from_be_bytes(digest[i * 4..i * 4 + 4].try_into().unwrap()));
    expected == hash && ((((hash[0] as u64) << 32) | hash[1] as u64) & mask) == 0
}

fn digest_single_block_message(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
) -> Option<[u32; 8]> {
    let suffix = nonce.checked_sub(message.nonce_addend)?;
    if suffix >= 1_000_000_000 {
        return None;
    }
    let mut words = message.message.0;
    let mut x = suffix as u32;
    for j in 0..9 {
        let digit = x % 10;
        x /= 10;
        let pos = message.digit_index + 8 - j;
        let word = pos / 4;
        let shift = (3 - (pos % 4)) * 8;
        words[word] = (words[word] & !(0xff_u32 << shift)) | ((u32::from(b'0') + digit) << shift);
    }
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &words);
    Some(state)
}

fn result_matches<const TYPE: u8>(hash: [u32; 8], target: u64, mask: u64) -> bool {
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    if cfg!(feature = "compare-64bit") {
        if TYPE == crate::solver::SOLVE_TYPE_LT {
            value < target
        } else if TYPE == crate::solver::SOLVE_TYPE_GT {
            value > target
        } else {
            (value & mask) == (target & mask)
        }
    } else {
        let value = hash[0];
        let target = (target >> 32) as u32;
        let mask = (mask >> 32) as u32;
        if TYPE == crate::solver::SOLVE_TYPE_LT {
            value < target
        } else if TYPE == crate::solver::SOLVE_TYPE_GT {
            value > target
        } else {
            (value & mask) == (target & mask)
        }
    }
}

fn verify_single_block<const TYPE: u8>(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    digest_single_block_message(message, nonce)
        .is_some_and(|expected| expected == hash && result_matches::<TYPE>(hash, target, mask))
}

fn verify_fast_single_block(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    mask: u64,
) -> bool {
    let nonce = nonce.to_string();
    if nonce.len() != 19 {
        return false;
    }
    let mut block = [0_u8; 64];
    block[..19].copy_from_slice(nonce.as_bytes());
    block[19] = 0x80;
    block[60..64].copy_from_slice(&message.message[15].to_be_bytes());
    let words =
        core::array::from_fn(|i| u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap()));
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &words);
    state == hash && result_matches::<{ crate::solver::SOLVE_TYPE_MASK }>(hash, 0, mask)
}

fn multi_layout_shader_source() -> String {
    let constants = crate::sha256::K32
        .iter()
        .map(|k| format!("0x{k:08x}u"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"
const K:array<u32,64>=array<u32,64>({constants});
const IV:array<u32,8>=array<u32,8>(0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u);
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn less64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah<bh||(ah==bh&&al<bl);}}
fn greater64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah>bh||(ah==bh&&al>bl);}}
fn target_ok(h0:u32,h1:u32)->bool{{
 let full=(P[2]&1u)!=0u;let ty=P[1];
 if(ty==1u){{return select(h0<P[6],less64(h0,h1,P[6],P[7]),full);}}
 if(ty==2u){{return select(h0>P[6],greater64(h0,h1,P[6],P[7]),full);}}
 let hi=(h0&P[8])==(P[6]&P[8]);return hi&&select(true,(h1&P[9])==(P[7]&P[9]),full);
}}
fn compress(s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w:array<u32,64>;for(var i=0u;i<16u;i=i+1u){{w[i]=block[i];}}
 for(var i=16u;i<64u;i=i+1u){{let s0=rotr(w[i-15u],7u)^rotr(w[i-15u],18u)^(w[i-15u]>>3u);let s1=rotr(w[i-2u],17u)^rotr(w[i-2u],19u)^(w[i-2u]>>10u);w[i]=w[i-16u]+s0+w[i-7u]+s1;}}
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 for(var i=0u;i<64u;i=i+1u){{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+K[i]+w[i];let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}
 return array<u32,8>(s[0]+a,s[1]+b,s[2]+c,s[3]+d,s[4]+e,s[5]+f,s[6]+g,s[7]+h);
}}
fn publish(nlo:u32,nhi:u32,h:array<u32,8>){{if(target_ok(h[0],h[1])&&atomicExchange(&R.flag,1u)==0u){{R.nonce_lo=nlo;R.nonce_hi=nhi;for(var i=0u;i<8u;i=i+1u){{R.hash[i]=h[i];}}}}}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=gid.x*{STEPS}u;
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let idx=first+step;if(idx>=P[5]){{return;}}
  let clo=P[3]+idx;let carry=select(0u,1u,clo<P[3]);let chi=P[4]+carry;let mode=P[0];
  if(mode==1u){{
   var block=array<u32,16>(P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31],P[32],P[33],P[34],P[35],P[36],P[37],P[38],P[39]);
   var x=clo;for(var j=0u;j<9u;j=j+1u){{let digit=x%10u;x=x/10u;let pos=62u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;block[wi]=(block[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);}}
   let st=array<u32,8>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);let mid=compress(st,block);
   var tail=array<u32,16>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,P[42],P[43]);let h=compress(mid,tail);
   let nlo=P[40]+clo;let nc=select(0u,1u,nlo<P[40]);publish(nlo,P[41]+nc,h);
  }} else if(mode==2u){{
   let block=array<u32,16>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23],P[24],clo,0x80000000u,0u,0u,0u,0u,320u);
   publish(clo,P[24],compress(IV,block));
  }} else if(mode==3u){{
   var blocks=array<u32,32>(P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31],P[32],P[33],P[34],P[35],P[36],P[37],P[38],P[39],P[40],P[41],P[42],P[43],P[44],P[45],P[46],P[47],P[48],P[49],P[50],P[51],P[52],P[53],P[54],P[55]);
   for(var j=0u;j<P[11];j=j+1u){{let byte=select((clo>>(j*8u))&0xffu,(chi>>((j-4u)*8u))&0xffu,j>=4u);let pos=P[12]+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;blocks[wi]=(blocks[wi]&~(0xffu<<sh))|(byte<<sh);}}
   let b0=array<u32,16>(blocks[0],blocks[1],blocks[2],blocks[3],blocks[4],blocks[5],blocks[6],blocks[7],blocks[8],blocks[9],blocks[10],blocks[11],blocks[12],blocks[13],blocks[14],blocks[15]);
   let st=array<u32,8>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);var h=compress(st,b0);
   if(P[10]==2u){{let b1=array<u32,16>(blocks[16],blocks[17],blocks[18],blocks[19],blocks[20],blocks[21],blocks[22],blocks[23],blocks[24],blocks[25],blocks[26],blocks[27],blocks[28],blocks[29],blocks[30],blocks[31]);h=compress(h,b1);}}
   publish(clo,chi,h);
  }}
 }}
}}
"#,
        CANCEL_INTERVAL - 1
    )
}

fn verify_double_block<const TYPE: u8>(
    message: &crate::message::DoubleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    let Some(mut suffix) = nonce.checked_sub(message.nonce_addend) else {
        return false;
    };
    if suffix >= 1_000_000_000 {
        return false;
    }
    let mut block = message.message.0;
    for j in (0..9).rev() {
        let digit = (suffix % 10) as u32;
        suffix /= 10;
        let pos = crate::message::DoubleBlockMessage::DIGIT_IDX as usize + j;
        let wi = pos / 4;
        let sh = (3 - (pos % 4)) * 8;
        block[wi] = (block[wi] & !(0xffu32 << sh)) | ((u32::from(b'0') + digit) << sh);
    }
    let mut state = message.prefix_state.0;
    crate::sha256::digest_block(&mut state, &block);
    let mut tail = [0u32; 16];
    let bits = message.message_length * 8;
    tail[14] = (bits >> 32) as u32;
    tail[15] = bits as u32;
    crate::sha256::digest_block(&mut state, &tail);
    state == hash && result_matches::<TYPE>(hash, target, mask)
}
fn verify_goaway<const TYPE: u8>(
    message: &crate::message::GoAwayMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    if (nonce >> 32) as u32 != message.high_word {
        return false;
    }
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&message.challenge);
    block[8] = message.high_word;
    block[9] = nonce as u32;
    block[10] = 0x8000_0000;
    block[15] = 320;
    let mut state = crate::sha256::IV;
    crate::sha256::digest_block(&mut state, &block);
    state == hash && result_matches::<TYPE>(hash, target, mask)
}
fn binary_templates(message: &crate::message::BinaryMessage) -> ([[u32; 16]; 2], u32) {
    let mut bytes = [[0u8; 64]; 2];
    bytes[0][..message.salt_residual_len]
        .copy_from_slice(&message.salt_residual[..message.salt_residual_len]);
    let mut ptr = message.salt_residual_len;
    let mut cur = 0usize;
    for _ in 0..message.nonce_byte_count.get() {
        ptr += 1;
        if ptr == 64 {
            cur = 1;
            ptr = 0;
        }
    }
    bytes[cur][ptr] = 0x80;
    ptr += 1;
    if ptr + 8 > 64 {
        cur = 1;
    }
    bytes[cur][56..64].copy_from_slice(&((message.message_length as u64) * 8).to_be_bytes());
    (
        core::array::from_fn(|b| {
            core::array::from_fn(|i| {
                u32::from_be_bytes(bytes[b][i * 4..i * 4 + 4].try_into().unwrap())
            })
        }),
        (cur + 1) as u32,
    )
}
fn verify_binary<const TYPE: u8>(
    message: &crate::message::BinaryMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    let (blocks, used) = binary_templates(message);
    let mut flat = [0u32; 32];
    flat[..16].copy_from_slice(&blocks[0]);
    flat[16..].copy_from_slice(&blocks[1]);
    let bytes = nonce.to_le_bytes();
    for j in 0..usize::from(message.nonce_byte_count.get()) {
        let pos = message.salt_residual_len + j;
        let wi = pos / 4;
        let sh = (3 - (pos % 4)) * 8;
        flat[wi] = (flat[wi] & !(0xffu32 << sh)) | (u32::from(bytes[j]) << sh);
    }
    let mut state = message.prefix_state.0;
    crate::sha256::digest_block(&mut state, flat[..16].try_into().unwrap());
    if used == 2 {
        crate::sha256::digest_block(&mut state, flat[16..32].try_into().unwrap());
    }
    state == hash && result_matches::<TYPE>(hash, target, mask)
}

fn generic_single_block_shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            rounds.push_str(&format!("w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",(t-2)&15,(t-2)&15,(t-2)&15,(t-7)&15,(t-15)&15,(t-15)&15,(t-15)&15));
        }
        rounds.push_str(&format!("{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"));
    }
    format!(
        r#"
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn less64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah<bh||(ah==bh&&al<bl);}}
fn greater64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah>bh||(ah==bh&&al>bl);}}
fn target_ok(h0:u32,h1:u32)->bool{{
 let full=(P[33]&2u)!=0u; let ty=P[32];
 if(ty==1u){{return select(h0<P[28],less64(h0,h1,P[28],P[29]),full);}}
 if(ty==2u){{return select(h0>P[28],greater64(h0,h1,P[28],P[29]),full);}}
 let hi=(h0&P[30])==(P[28]&P[30]);
 return hi&&select(true,(h1&P[31])==(P[29]&P[31]),full);
}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=P[24]+gid.x*{STEPS}u;
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let candidate=first+step;if(candidate>=P[34]){{return;}}
  if((P[33]&1u)!=0u&&candidate%10u==0u){{continue;}}
  var words=array<u32,16>(P[8],P[9],P[10],P[11],P[12],P[13],P[14],P[15],P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);
  var x=candidate;
  for(var j=0u;j<9u;j= j+1u){{
   let digit=x%10u;x=x/10u;let pos=P[25]+8u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;
   words[wi]=(words[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);
  }}
  var w0=words[0];var w1=words[1];var w2=words[2];var w3=words[3];var w4=words[4];var w5=words[5];var w6=words[6];var w7=words[7];
  var w8=words[8];var w9=words[9];var w10=words[10];var w11=words[11];var w12=words[12];var w13=words[13];var w14=words[14];var w15=words[15];
  var a=P[0];var b=P[1];var c=P[2];var d=P[3];var e=P[4];var f=P[5];var g=P[6];var h=P[7];
  {rounds}
  let r0=P[0]+a;let r1=P[1]+b;
  if(target_ok(r0,r1)){{if(atomicExchange(&R.flag,1u)==0u){{
   let nlo=P[26]+candidate;let carry=select(0u,1u,nlo<P[26]);R.nonce_lo=nlo;R.nonce_hi=P[27]+carry;
   R.hash[0]=r0;R.hash[1]=r1;R.hash[2]=P[2]+c;R.hash[3]=P[3]+d;R.hash[4]=P[4]+e;R.hash[5]=P[5]+f;R.hash[6]=P[6]+g;R.hash[7]=P[7]+h;
  }}}}
 }}
}}
"#,
        CANCEL_INTERVAL - 1
    )
}

fn shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            rounds.push_str(&format!("w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",(t-2)&15,(t-2)&15,(t-2)&15,(t-7)&15,(t-15)&15,(t-15)&15,(t-15)&15));
        }
        rounds.push_str(&format!("{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"));
    }
    format!(
        r#"
struct InParams {{ s0:u32,s1:u32,s2:u32,s3:u32,s4:u32,s5:u32,s6:u32,s7:u32,base_lo:u32,base_hi:u32,bitlen:u32,mask0:u32,mask1:u32 }};
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<uniform> P:InParams;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let gid_lo=gid.x<<7u;let gid_hi=gid.x>>25u;let base_lo=P.base_lo+gid_lo;var base_hi=P.base_hi+gid_hi;if(base_lo<P.base_lo){{base_hi=base_hi+1u;}}
 let d2=(base_lo>>6u)&7u;let d3=(base_lo>>9u)&7u;let d4=(base_lo>>12u)&7u;let d5=(base_lo>>15u)&7u;let d6=(base_lo>>18u)&7u;let d7=(base_lo>>21u)&7u;let d8=(base_lo>>24u)&7u;let d9=(base_lo>>27u)&7u;let d10=((base_lo>>30u)|(base_hi<<2u))&7u;let d11=(base_hi>>1u)&7u;let d12=(base_hi>>4u)&7u;let d13=(base_hi>>7u)&7u;let d14=(base_hi>>10u)&7u;let d15=(base_hi>>13u)&7u;let d16=(base_hi>>16u)&7u;let d17=(base_hi>>19u)&7u;
 let nw0=0x31303030u|(d17<<16u)|(d16<<8u)|d15;let nw1=0x30303030u|(d14<<24u)|(d13<<16u)|(d12<<8u)|d11;let nw2=0x30303030u|(d10<<24u)|(d9<<16u)|(d8<<8u)|d7;let nw3=0x30303030u|(d6<<24u)|(d5<<16u)|(d4<<8u)|d3;var nw4=0x30303080u|(d2<<24u);
 const GM:array<u32,7>=array<u32,7>(0x00000100u,0x00000200u,0x00000400u,0x00010000u,0x00020000u,0x00040000u,0x01000000u);
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u && atomicLoad(&R.flag)!=0u){{return;}}
  var w0=nw0;var w1=nw1;var w2=nw2;var w3=nw3;var w4=nw4;
  var w5=0u;var w6=0u;var w7=0u;var w8=0u;var w9=0u;var w10=0u;var w11=0u;var w12=0u;var w13=0u;var w14=0u;var w15=P.bitlen;
  var a=P.s0;var b=P.s1;var c=P.s2;var d=P.s3;var e=P.s4;var f=P.s5;var g=P.s6;var h=P.s7;
  {rounds}
  let r0=P.s0+a;let r1=P.s1+b;
  if((r0&P.mask0)==0u&&(r1&P.mask1)==0u){{if(atomicExchange(&R.flag,1u)==0u){{let gray=step^(step>>1u);let clo=base_lo+gray;let carry=select(0u,1u,clo<base_lo);R.nonce_lo=clo;R.nonce_hi=base_hi+carry;R.hash[0]=r0;R.hash[1]=r1;R.hash[2]=P.s2+c;R.hash[3]=P.s3+d;R.hash[4]=P.s4+e;R.hash[5]=P.s5+f;R.hash[6]=P.s6+g;R.hash[7]=P.s7+h;}}}}
  let ns=step+1u;if(ns<{STEPS}u){{nw4=nw4^GM[countTrailingZeros(ns)];}}
 }}
}}
"#,
        CANCEL_INTERVAL - 1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_conversion_is_decimal_octal_digits() {
        assert_eq!(counter_to_decimal_nonce(0), 1_000_000_000_000_000_000);
        assert_eq!(
            counter_to_decimal_nonce(0o1234567),
            1_000_000_000_001_234_567
        );
        assert!(counter_to_decimal_nonce(COUNTER_SPACE - 1) < u64::MAX);
    }
}
