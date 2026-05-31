use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const Q8_0_BLOCK: usize = 34; // 2-byte f16 scale + 32 × i8

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub file_offset: u64,
    pub byte_size: u64,
    pub kind: u32,
    pub shape: Vec<u64>,
}

impl TensorMeta {
    pub fn rows(&self) -> usize {
        if self.shape.len() >= 2 { self.shape[1] as usize } else { 1 }
    }
    pub fn cols(&self) -> usize {
        self.shape[0] as usize
    }
}

pub struct TensorStore {
    mmap: Arc<Mmap>,
    /// Zero-copy Metal buffer wrapping the entire mmap.
    pub mmap_buf: Buffer,
    pub index: HashMap<String, TensorMeta>,
}

impl TensorStore {
    pub fn open(path: &str, device: &Device) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {path}"))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.context("mmap")?);

        // Zero-copy Metal buffer wrapping the entire mmap.
        // Metal requires both the pointer and length to be page-aligned (4096 bytes on macOS).
        // The mmap base pointer is always page-aligned; round the length up to the page boundary.
        // The extra bytes at the end of the last page are OS-zero-filled and never accessed by kernels.
        const PAGE: usize = 4096;
        let rounded_len = mmap.len().div_ceil(PAGE) * PAGE;
        let mmap_buf = unsafe {
            device.new_buffer_with_bytes_no_copy(
                mmap.as_ptr() as *mut _,
                rounded_len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };

        let data_start = find_data_start(path)?;

        let mut container = gguf_rs::get_gguf_container_array_size(path, 0)?;
        let model = container.decode()?;

        let mut index = HashMap::new();
        for t in model.tensors() {
            index.insert(
                t.name.clone(),
                TensorMeta {
                    file_offset: data_start + t.offset,
                    byte_size: t.size,
                    kind: t.kind,
                    shape: t.shape.clone(),
                },
            );
        }

        Ok(Self { mmap, mmap_buf, index })
    }

    /// Raw bytes for a tensor (CPU-side, from mmap).
    pub fn get(&self, name: &str) -> Result<&[u8]> {
        let meta = self.index.get(name)
            .with_context(|| format!("tensor '{name}' not in GGUF"))?;
        let start = meta.file_offset as usize;
        let end = start + meta.byte_size as usize;
        if end > self.mmap.len() {
            bail!("tensor '{name}' out of file bounds");
        }
        Ok(&self.mmap[start..end])
    }

    pub fn meta(&self, name: &str) -> Result<&TensorMeta> {
        self.index.get(name).with_context(|| format!("tensor '{name}' not in GGUF"))
    }

    // -- dequantization helpers (CPU) -----------------------------------------

    pub fn dequant_q8_0_row(row_bytes: &[u8]) -> Vec<f32> {
        let blocks = row_bytes.len() / Q8_0_BLOCK;
        let mut out = Vec::with_capacity(blocks * 32);
        for b in 0..blocks {
            let off = b * Q8_0_BLOCK;
            let scale_bits = u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            for k in 0..32 {
                out.push(scale * (row_bytes[off + 2 + k] as i8) as f32);
            }
        }
        out
    }

    pub fn dequant_f16_row(row_bytes: &[u8]) -> Vec<f32> {
        row_bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    }
}

fn find_data_start(path: &str) -> Result<u64> {
    let mut file = File::open(path)?;
    let magic = file.read_i32::<LittleEndian>()?;
    let bo = match magic {
        0x46554747 => gguf_rs::ByteOrder::LE,
        0x47475546 => gguf_rs::ByteOrder::BE,
        _ => bail!("not a GGUF file"),
    };

    let pos = Arc::new(AtomicU64::new(4));
    let reader = CountingReader { inner: file, pos: pos.clone() };
    let mut container = gguf_rs::GGUFContainer::new(bo, Box::new(reader), 0);
    let model = container.decode()?;

    let after_header = pos.load(Ordering::SeqCst);
    let align = model.metadata().get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32);

    Ok(after_header.div_ceil(align) * align)
}

struct CountingReader {
    inner: File,
    pos: Arc<AtomicU64>,
}

impl Read for CountingReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos.fetch_add(n as u64, Ordering::SeqCst);
        Ok(n)
    }
}
