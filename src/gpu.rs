use anyhow::{Context, Result};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

const SHADER_SRC: &str = include_str!("kernels.metal");

pub struct Gpu {
    pub device: Device,
    pub queue: CommandQueue,
    q8_0_matvec: ComputePipelineState,
    vec_add: ComputePipelineState,
    vec_add_inplace: ComputePipelineState,
    silu_hadamard: ComputePipelineState,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().context("no Metal device")?;
        let queue = device.new_command_queue();

        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Metal compile: {e}"))?;

        Ok(Self {
            q8_0_matvec: pipeline(&device, &lib, "q8_0_matvec")?,
            vec_add: pipeline(&device, &lib, "vec_add")?,
            vec_add_inplace: pipeline(&device, &lib, "vec_add_inplace")?,
            silu_hadamard: pipeline(&device, &lib, "silu_hadamard")?,
            queue,
            device,
        })
    }

    // -- buffer helpers -------------------------------------------------------

    pub fn buf_from_bytes(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buf_from_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as _,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buf_zeros(&self, n: usize) -> Buffer {
        self.device
            .new_buffer(n as u64 * 4, MTLResourceOptions::StorageModeShared)
    }

    pub fn read_f32(&self, buf: &Buffer, n: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
    }

    pub fn write_f32(&self, buf: &Buffer, data: &[f32]) {
        let dst = buf.contents() as *mut f32;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
    }

    // -- kernels --------------------------------------------------------------

    /// Q8_0 matrix × vector.
    /// `w_buf`: the mmap Metal buffer (zero-copy), `w_offset`: byte offset into it for this tensor.
    pub fn q8_0_matvec(&self, w_buf: &Buffer, w_offset: u64, x: &Buffer, n: usize, k: usize) -> Buffer {
        let out = self.buf_zeros(n);
        let rows = n as u32;
        let cols = k as u32;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q8_0_matvec);
        enc.set_buffer(0, Some(w_buf), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(&out), 0);
        // Use set_bytes for scalar params: no Metal buffer allocation/deallocation overhead.
        enc.set_bytes(3, 4, &rows as *const u32 as _);
        enc.set_bytes(4, 4, &cols as *const u32 as _);
        enc.set_bytes(5, 8, &w_offset as *const u64 as _);
        dispatch_1d(enc, n * 32, 256);  // 32 threads (one simdgroup) per output row
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        out
    }

    /// out[i] = a[i] + b[i]
    pub fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        let out = self.buf_zeros(n);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.vec_add);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(&out), 0);
        dispatch_1d(enc, n, 256);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        out
    }

    /// a[i] += b[i]  (in-place)
    pub fn add_inplace(&self, a: &Buffer, b: &Buffer, n: usize) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.vec_add_inplace);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        dispatch_1d(enc, n, 256);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// out[i] = silu(gate[i]) * up[i]
    pub fn silu_hadamard(&self, gate: &Buffer, up: &Buffer, n: usize) -> Buffer {
        let out = self.buf_zeros(n);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.silu_hadamard);
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(&out), 0);
        dispatch_1d(enc, n, 256);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        out
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}

fn pipeline(device: &Device, lib: &Library, name: &str) -> Result<ComputePipelineState> {
    let func = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("get_function({name}): {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("pipeline({name}): {e}"))
}

fn dispatch_1d(enc: &metal::ComputeCommandEncoderRef, n: usize, tg_size: usize) {
    let tg = MTLSize { width: tg_size as u64, height: 1, depth: 1 };
    let ng = MTLSize {
        width: n.div_ceil(tg_size) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(ng, tg);
}
