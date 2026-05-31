use std::collections::HashMap;

use anyhow::Result;
use metal::Buffer;

use crate::gpu::Gpu;
use crate::tensor::{TensorStore, GGML_F16, GGML_Q8_0, Q8_0_BLOCK};

pub struct LlamaModel {
    pub arch: Arch,
    store: TensorStore,
    gpu: Gpu,
    /// Lazily-uploaded weight buffers: upload once, reuse every forward pass.
    weight_cache: HashMap<String, Buffer>,
}

#[derive(Clone, Debug)]
pub struct Arch {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub vocab_size: usize,
    pub rope_base: f32,
}

struct KvCache {
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
}

impl KvCache {
    fn new(n_layers: usize) -> Self {
        Self { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers] }
    }
    fn push(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>) {
        self.k[layer].push(k);
        self.v[layer].push(v);
    }
}

impl LlamaModel {
    pub fn load(path: &str) -> Result<Self> {
        let gpu = Gpu::new()?;
        // Pass gpu.device so TensorStore can (optionally) create the mmap buffer.
        let store = TensorStore::open(path, &gpu.device)?;

        let mut container = gguf_rs::get_gguf_container_array_size(path, 0)?;
        let model = container.decode()?;
        let meta = model.metadata();

        let get_u = |key: &str, default: usize| {
            meta.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default)
        };
        let get_f = |key: &str, default: f32| {
            meta.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
        };

        let hidden     = get_u("llama.embedding_length", 4096);
        let n_layers   = get_u("llama.block_count", 32);
        let n_heads    = get_u("llama.attention.head_count", 32);
        let n_kv_heads = get_u("llama.attention.head_count_kv", n_heads);
        let ffn_hidden = get_u("llama.feed_forward_length", hidden * 4);
        let vocab_size = get_u("llama.vocab_size", 32000);
        let rope_base  = get_f("llama.rope.freq_base", 10000.0);

        let head_dim = store.index.get("blk.0.attn_q.weight")
            .and_then(|m| m.shape.get(1).copied())
            .map(|d| d as usize / n_heads)
            .filter(|&d| d > 0)
            .unwrap_or(hidden / n_heads);

        Ok(Self {
            arch: Arch { hidden, n_layers, n_heads, n_kv_heads, head_dim, ffn_hidden, vocab_size, rope_base },
            store,
            gpu,
            weight_cache: HashMap::new(),
        })
    }

    pub fn generate(&mut self, tokens: &[u32], max_new: usize, vocab: &[String]) -> Result<()> {
        let mut kv = KvCache::new(self.arch.n_layers);
        let mut last = 0u32;
        let mut generated: Vec<u32> = Vec::new();

        // Prefill
        let t0 = std::time::Instant::now();
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = self.forward(tok, pos, &mut kv)?;
            last = argmax(&logits);
            if pos + 1 == tokens.len() {
                print_token(last, vocab);
                generated.push(last);
            }
        }
        let prefill_ms = t0.elapsed().as_millis();
        let n_prompt = tokens.len();
        eprintln!(
            "\nprefill: {n_prompt} tokens in {prefill_ms}ms  ({:.1} t/s)",
            n_prompt as f64 / (prefill_ms as f64 / 1000.0)
        );

        // Decode
        let t1 = std::time::Instant::now();
        let mut n_decoded = 0usize;
        let mut pos = tokens.len();
        for _ in 0..max_new {
            let mut logits = self.forward(last, pos, &mut kv)?;
            apply_repetition_penalty(&mut logits, &generated, 1.3);
            last = argmax(&logits);
            if last == 2 { break; }   // </s> EOS
            print_token(last, vocab);
            generated.push(last);
            pos += 1;
            n_decoded += 1;
        }
        let decode_ms = t1.elapsed().as_millis();
        println!();
        if n_decoded > 0 {
            eprintln!(
                "decode:  {n_decoded} tokens in {decode_ms}ms  ({:.1} t/s)",
                n_decoded as f64 / (decode_ms as f64 / 1000.0)
            );
        }
        Ok(())
    }

    fn forward(&mut self, token: u32, pos: usize, kv: &mut KvCache) -> Result<Vec<f32>> {
        let arch = self.arch.clone();
        let mut x = self.embed(token)?;
        let t_fwd = std::time::Instant::now();
        for layer in 0..arch.n_layers {
            let t_layer = std::time::Instant::now();
            x = self.block(x, layer, pos, kv)?;
            if layer < 3 || layer == arch.n_layers - 1 {
                eprintln!("  layer {layer:2}: {}ms", t_layer.elapsed().as_millis());
            }
        }
        eprintln!("  all layers: {}ms", t_fwd.elapsed().as_millis());
        let norm_w = self.f32_weights("output_norm.weight")?;
        x = rms_norm(&x, &norm_w, 1e-5);
        self.lm_head(&x)
    }

    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let bytes = self.store.get("token_embd.weight")?;
        let meta  = self.store.meta("token_embd.weight")?;
        let vocab_rows = meta.shape.get(1).copied().unwrap_or(meta.shape[0]) as usize;
        let row = token as usize;
        anyhow::ensure!(row < vocab_rows, "token {token} >= vocab {vocab_rows}");
        match meta.kind {
            GGML_Q8_0 => {
                let rb = meta.cols() / 32 * Q8_0_BLOCK;
                Ok(TensorStore::dequant_q8_0_row(&bytes[row * rb..][..rb]))
            }
            GGML_F16 => {
                let rb = meta.cols() * 2;
                Ok(TensorStore::dequant_f16_row(&bytes[row * rb..][..rb]))
            }
            k => anyhow::bail!("unsupported embedding dtype {k}"),
        }
    }

    fn block(&mut self, x: Vec<f32>, layer: usize, pos: usize, kv: &mut KvCache) -> Result<Vec<f32>> {
        let arch = self.arch.clone();

        // --- attention ---
        let attn_norm_w = self.f32_weights(&format!("blk.{layer}.attn_norm.weight"))?;
        let xn = rms_norm(&x, &attn_norm_w, 1e-5);
        let xn_buf = self.gpu.buf_from_f32(&xn);

        let q_dim  = self.tensor_rows(&format!("blk.{layer}.attn_q.weight"))?;
        let kv_dim = self.tensor_rows(&format!("blk.{layer}.attn_k.weight"))?;
        let head_dim = q_dim / arch.n_heads;

        let q_buf  = self.matvec(&format!("blk.{layer}.attn_q.weight"),      &xn_buf, q_dim,  arch.hidden)?;
        let k_buf  = self.matvec(&format!("blk.{layer}.attn_k.weight"),      &xn_buf, kv_dim, arch.hidden)?;
        let v_buf  = self.matvec(&format!("blk.{layer}.attn_v.weight"),      &xn_buf, kv_dim, arch.hidden)?;

        let mut q = self.gpu.read_f32(&q_buf, q_dim).to_vec();
        let mut k = self.gpu.read_f32(&k_buf, kv_dim).to_vec();
        let     v = self.gpu.read_f32(&v_buf, kv_dim).to_vec();

        rope(&mut q, arch.n_heads,    head_dim, pos, arch.rope_base);
        rope(&mut k, arch.n_kv_heads, head_dim, pos, arch.rope_base);
        kv.push(layer, k, v);

        let attn_out = attention(&q, &kv.k[layer], &kv.v[layer], arch.n_heads, arch.n_kv_heads, head_dim);
        let attn_buf = self.gpu.buf_from_f32(&attn_out);
        let x_buf    = self.gpu.buf_from_f32(&x);
        let o_proj   = self.matvec(&format!("blk.{layer}.attn_output.weight"), &attn_buf, arch.hidden, q_dim)?;
        let res1     = self.gpu.add(&x_buf, &o_proj, arch.hidden);

        // --- ffn ---
        let res1_vec    = self.gpu.read_f32(&res1, arch.hidden).to_vec();
        let ffn_norm_w  = self.f32_weights(&format!("blk.{layer}.ffn_norm.weight"))?;
        let xn2         = rms_norm(&res1_vec, &ffn_norm_w, 1e-5);
        let xn2_buf     = self.gpu.buf_from_f32(&xn2);

        let gate = self.matvec(&format!("blk.{layer}.ffn_gate.weight"), &xn2_buf, arch.ffn_hidden, arch.hidden)?;
        let up   = self.matvec(&format!("blk.{layer}.ffn_up.weight"),   &xn2_buf, arch.ffn_hidden, arch.hidden)?;
        let mid  = self.gpu.silu_hadamard(&gate, &up, arch.ffn_hidden);
        let down = self.matvec(&format!("blk.{layer}.ffn_down.weight"), &mid, arch.hidden, arch.ffn_hidden)?;

        let out = self.gpu.add(&res1, &down, arch.hidden);
        Ok(self.gpu.read_f32(&out, arch.hidden).to_vec())
    }

    fn lm_head(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let x_buf = self.gpu.buf_from_f32(x);
        let name = if self.store.index.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let vocab = self.arch.vocab_size;
        let hidden = self.arch.hidden;
        let out = self.matvec(name, &x_buf, vocab, hidden)?;
        Ok(self.gpu.read_f32(&out, vocab).to_vec())
    }

    // -- helpers --

    fn tensor_rows(&self, name: &str) -> Result<usize> {
        let m = self.store.meta(name)?;
        Ok(m.shape.get(1).copied().unwrap_or(m.shape[0]) as usize)
    }

    /// Q8_0 matvec with lazy weight caching.
    /// The first call for each tensor copies the mmap slice into a Metal buffer;
    /// every subsequent call reuses that buffer — zero copies at steady state.
    fn matvec(&mut self, name: &str, x: &Buffer, n: usize, k: usize) -> Result<Buffer> {
        let upload_ms = if !self.weight_cache.contains_key(name) {
            let t = std::time::Instant::now();
            let bytes = self.store.get(name)?;
            let buf = self.gpu.buf_from_bytes(bytes);
            self.weight_cache.insert(name.to_string(), buf);
            Some(t.elapsed().as_millis())
        } else { None };

        let t = std::time::Instant::now();
        let w = &self.weight_cache[name];
        let out = self.gpu.q8_0_matvec(w, 0, x, n, k);
        let dispatch_ms = t.elapsed().as_millis();

        if self.weight_cache.len() <= 10 || upload_ms.is_some() {
            let upload_str = upload_ms.map(|ms| format!(", upload={ms}ms")).unwrap_or_default();
            eprintln!("    matvec {name}: dispatch={dispatch_ms}ms{upload_str}");
        }
        Ok(out)
    }

    fn f32_weights(&self, name: &str) -> Result<Vec<f32>> {
        let b = self.store.get(name)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

// ---------------------------------------------------------------------------
// CPU math
// ---------------------------------------------------------------------------

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w.iter()).map(|(xi, wi)| xi * inv * wi).collect()
}

fn rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, base: f32) {
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..head_dim / 2 {
            let theta = pos as f32 / base.powf(2.0 * i as f32 / head_dim as f32);
            let (s, c) = theta.sin_cos();
            let (x0, x1) = (x[off + 2*i], x[off + 2*i + 1]);
            x[off + 2*i]     = x0 * c - x1 * s;
            x[off + 2*i + 1] = x0 * s + x1 * c;
        }
    }
}

fn attention(
    q: &[f32], k_cache: &[Vec<f32>], v_cache: &[Vec<f32>],
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> Vec<f32> {
    let seq  = k_cache.len();
    let gqa  = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h   = h / gqa;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        let mut scores: Vec<f32> = (0..seq).map(|t| {
            let k_head = &k_cache[t][kv_h * head_dim..(kv_h + 1) * head_dim];
            scale * dot(q_head, k_head)
        }).collect();

        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter_mut().map(|s| { *s = (*s - max).exp(); *s }).sum();
        scores.iter_mut().for_each(|s| *s /= sum);

        for t in 0..seq {
            let v_head = &v_cache[t][kv_h * head_dim..(kv_h + 1) * head_dim];
            for i in 0..head_dim {
                out[h * head_dim + i] += scores[t] * v_head[i];
            }
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Divide logits of recently-generated tokens by `penalty` (or multiply if logit < 0).
/// Suppresses repetition without temperature sampling.
fn apply_repetition_penalty(logits: &mut [f32], seen: &[u32], penalty: f32) {
    for &id in seen {
        if let Some(l) = logits.get_mut(id as usize) {
            if *l > 0.0 { *l /= penalty; } else { *l *= penalty; }
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn print_token(id: u32, vocab: &[String]) {
    if let Some(tok) = vocab.get(id as usize) {
        print!("{}", tok.replace('\u{0120}', " ").replace("<0x0A>", "\n"));
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}
