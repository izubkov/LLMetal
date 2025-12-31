# Inference Path Report: Apple Metal Q8 Devstral

This document traces the complete inference path in mistral_rs for Apple Metal devices with Q8 quantized GGUF models.

## Overview

The inference pipeline follows this flow:
1. **Model Loading**: Load GGUF Q8 quantized model files
2. **Device Initialization**: Set up Apple Metal device
3. **Tokenizer Initialization**: Load and configure tokenizer
4. **Forward Pass**: Execute model forward pass through transformer layers
5. **Sampling**: Sample next token from logits
6. **Decoding**: Decode tokens to text output

---

## 1. Model Loading (GGUF Q8)

### Entry Point
**File**: `mistral_rs/mistralrs/src/text_model.rs`

```rust
TextModelBuilder::new("model_id")
    .with_isq(IsqType::Q8_0)  // Q8 quantization
    .build()
    .await?
```

### GGUF Loader
**File**: `mistral_rs/mistralrs-core/src/pipeline/gguf.rs`

The `GGUFLoader` loads quantized model files:

```rust
impl Loader for GGUFLoader {
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,  // Metal device
        ...
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>
```

**Key Steps**:
- Reads GGUF file headers and metadata
- Loads Q8 quantized tensors using `candle_core::quantized::ggml_file`
- Maps tensors to Metal device
- Creates `GGUFPipeline` with quantized model weights

### Q8 Quantization Support
**File**: `mistral_rs/mistralrs-quant/src/lib.rs`

```rust
pub enum IsqType {
    Q8_0,  // Q8 quantization type
    Q8_1,
    ...
}
```

**File**: `mistral_rs/mistralrs-quant/src/gguf/mod.rs`

The `GgufMatMul` struct handles Q8 quantized matrix operations:

```rust
pub struct GgufMatMul {
    pub(crate) w: QMatMul,  // Quantized matrix multiplication
    pub(crate) b: Option<Tensor>,
}

impl QuantMethod for GgufMatMul {
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward(a)?;  // Q8 matmul on Metal
        ...
    }
}
```

---

## 2. Device Initialization (Apple Metal)

### Metal Device Setup
**File**: `mistral_rs/mistralrs-core/src/lib.rs`

Metal device is initialized when creating the pipeline:

```rust
#[cfg(feature = "metal")]
objc::rc::autoreleasepool(move || {
    // Metal device is automatically selected
    let device = Device::Metal(...);
    ...
})
```

**File**: `mistral_rs/mistralrs/src/model.rs`

```rust
pub fn best_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        Ok(Device::Cpu)
    } else {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0)  // Get Metal device
        }
        ...
    }
}
```

### Metal Kernels for Quantization
**File**: `mistral_rs/mistralrs-quant/src/metal_kernels/`

Metal shader kernels handle Q8 operations:
- `quantized.metal`: Q8 dequantization and matmul kernels
- Optimized for Apple Silicon's Neural Engine and GPU

---

## 3. Tokenizer Initialization

### Tokenizer Loading
**File**: `mistral_rs/mistralrs-core/src/pipeline/gguf.rs`

```rust
let tokenizer = get_tokenizer(
    &paths.get_tokenizer_filename()?,
    Some(tokenizer_json),
    &token_source,
    revision,
    silent,
)?;
```

**File**: `mistral_rs/mistralrs-core/src/utils/tokenizer.rs`

Tokenizer is loaded from `tokenizer.json`:
- Uses `tokenizers` crate
- Handles BPE/Unigram tokenization
- Maps tokens to IDs and vice versa

---

## 4. Forward Pass

### Pipeline Forward
**File**: `mistral_rs/mistralrs-core/src/pipeline/normal.rs`

```rust
impl Pipeline for NormalPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult> {
        // Extract input IDs, positions, etc.
        let logits = self.model.forward(
            &input_ids,
            &seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
        )?;
        ...
    }
}
```

### Model Forward Pass
**File**: `mistral_rs/mistralrs-core/src/models/quantized_llama.rs`

The quantized LLaMA model forward pass:

```rust
impl ModelWeights {
    pub fn forward(
        &self,
        xs: &Tensor,
        start_offsets: &[usize],
        context_lens: Option<&[usize]>,
        position_ids: Option<&Tensor>,
        paged_attn_meta: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        // 1. Token embeddings
        let xs = self.embeddings.forward(xs)?;
        
        // 2. Apply RoPE positional embeddings
        // 3. Pass through transformer layers
        for layer in &self.layers {
            xs = layer.forward(&xs, mask, start_offsets, &mut kv_cache, ...)?;
        }
        
        // 4. Final layer norm
        let xs = self.norm.forward(&xs)?;
        
        // 5. Output projection (logits)
        MatMul.qmethod_matmul(&xs, &*self.output)  // Q8 matmul
    }
}
```

### Layer Forward Pass
**File**: `mistral_rs/mistralrs-core/src/models/quantized_llama.rs`

Each transformer layer:

```rust
impl LayerWeights {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        ...
    ) -> Result<Tensor> {
        // 1. Attention norm (RMSNorm)
        let x_norm = self.attention_norm.forward(x)?;
        
        // 2. Self-attention (Q8 matmuls)
        let attn_out = self.forward_attn(&x_norm, mask, start_offsets, kv_cache, ...)?;
        let x = (x + attn_out)?;
        
        // 3. FFN norm (RMSNorm)
        let x_norm = self.ffn_norm.forward(&x)?;
        
        // 4. MLP/MoE (Q8 matmuls)
        let mlp_out = self.mlp_or_moe.forward(&x_norm)?;
        let x = (x + mlp_out)?;
        
        Ok(x)
    }
}
```

### Q8 Matrix Multiplication
**File**: `mistral_rs/mistralrs-quant/src/gguf/mod.rs`

```rust
impl QuantMethod for GgufMatMul {
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // Q8 matmul: (FP16 activations @ INT8 weights) -> FP32 -> FP16
        let x = self.w.forward(a)?;  // Uses Metal kernels
        if let Some(ref b) = self.b {
            x.broadcast_add(b)
        } else {
            Ok(x)
        }
    }
}
```

**Optimization**: Q8 weights are stored as INT8, activations as FP16. Metal kernels perform:
- INT8 × FP16 → INT32 accumulation
- Scale to FP32
- Convert to FP16 output

---

## 5. Sampling

### Sampler
**File**: `mistral_rs/mistralrs-core/src/sampler.rs`

```rust
impl Sampler {
    pub fn sample(
        &self,
        logits: &Tensor,
        rng: &mut Isaac64Rng,
        context: &[u32],
    ) -> Result<u32> {
        // 1. Apply logits processors
        let mut logits = self.apply_logits_processors(logits, context)?;
        
        // 2. Apply temperature, top-k, top-p
        if let Some(temp) = self.temperature {
            logits = (logits / temp)?;
        }
        
        // 3. Sample token ID
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let token_id = self.sample_from_probs(&probs, rng)?;
        
        Ok(token_id)
    }
}
```

---

## 6. Text Generation Loop

### Engine Processing
**File**: `mistral_rs/mistralrs-core/src/engine.rs`

The engine orchestrates the generation loop:

```rust
// Prefill: Process prompt tokens
for token in prompt_tokens {
    let logits = pipeline.forward_inputs(...)?;
}

// Decode: Generate tokens one by one
loop {
    let logits = pipeline.forward_inputs(...)?;
    let token_id = sampler.sample(&logits, &mut rng, &context)?;
    context.push(token_id);
    
    // Decode and print
    let text = tokenizer.decode(&[token_id], false)?;
    print!("{}", text);
    
    if token_id == eos_token_id {
        break;
    }
}
```

### Response Output
**File**: `mistral_rs/mistralrs/src/text_model.rs`

```rust
let response = model.send_chat_request(messages).await?;
println!("{}", response.choices[0].message.content.as_ref().unwrap());
```

---

## Key Optimizations for Apple Metal

### 1. Q8 Quantization
- **Weights**: INT8 format (50% memory reduction vs FP16)
- **Activations**: FP16 (native Metal support)
- **Scales**: FP32 (precision for scaling)
- **Kernels**: Metal shaders for INT8×FP16 matmul

### 2. Metal-Specific Optimizations
- **Unified Memory**: No CPU-GPU transfers
- **Command Buffers**: Batched operations
- **SIMD Groups**: Efficient parallel execution
- **Neural Engine**: Leverages Apple Silicon acceleration

### 3. Memory Management
- **KV Cache**: Efficient attention caching
- **Paged Attention**: Optional for long sequences
- **Prefix Caching**: Reuse common prefixes

### 4. Quantization Guard
**File**: `mistral_rs/mistralrs-quant/src/lib.rs`

```rust
impl QuantizeOntoGuard {
    pub fn acquire(&self, device: &Device) -> QuantizeOntoDropGuard<'_> {
        #[cfg(feature = "metal")]
        if let Device::Metal(dev) = device {
            // Wait for outstanding work to avoid encoder conflicts
            dev.wait_until_completed()?;
        }
        ...
    }
}
```

---

## File References Summary

| Component | File Path |
|-----------|-----------|
| Model Builder | `mistral_rs/mistralrs/src/text_model.rs` |
| GGUF Loader | `mistral_rs/mistralrs-core/src/pipeline/gguf.rs` |
| Q8 Quantization | `mistral_rs/mistralrs-quant/src/gguf/mod.rs` |
| Quantized Model | `mistral_rs/mistralrs-core/src/models/quantized_llama.rs` |
| Forward Pass | `mistral_rs/mistralrs-core/src/pipeline/normal.rs` |
| Sampling | `mistral_rs/mistralrs-core/src/sampler.rs` |
| Metal Kernels | `mistral_rs/mistralrs-quant/src/metal_kernels/` |
| Device Setup | `mistral_rs/mistralrs-core/src/lib.rs` |
| Tokenizer | `mistral_rs/mistralrs-core/src/utils/tokenizer.rs` |

---

## Complete Inference Flow Diagram

```
User Input (text)
    ↓
Tokenizer.encode() → token_ids: Vec<u32>
    ↓
Model Loading:
  - Load GGUF Q8 weights → Metal device
  - Initialize tokenizer
  - Set up Metal command queue
    ↓
Prefill Phase:
  For each prompt token:
    - Embedding lookup (FP16)
    - Forward through layers:
      * Attention (Q8 matmul)
      * MLP (Q8 matmul)
      * Layer norm (FP16)
    - Update KV cache
    ↓
Decode Phase (loop):
  - Forward pass (single token)
  - Extract logits (vocab_size)
  - Sample token_id
  - Decode token_id → text
  - Print text
  - If EOS: break
    ↓
Final Response (text)
```

---

## Notes

- All Q8 operations use Metal-optimized kernels
- FP16 activations leverage Apple Silicon's half-precision units
- INT8 weights reduce memory bandwidth
- Unified memory architecture eliminates transfer overhead
- Command buffer batching improves GPU utilization

