# MistralRS GGUF Inference Path for Apple Metal

This document traces the complete inference path for running a GGUF quantized model (Q8) on Apple Metal using mistral.rs.

---

## 1. Device Initialization

### Metal Device Creation
**File:** `mistralrs/src/model.rs:12-24`

```rust
pub fn best_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}
```

The `best_device()` function automatically selects the Metal device when compiled with the `metal` feature flag.

---

## 2. Model Builder Initialization

### GgufModelBuilder
**File:** `mistralrs/src/gguf.rs:17-79`

```rust
pub struct GgufModelBuilder {
    pub(crate) model_id: String,
    pub(crate) files: Vec<String>,
    pub(crate) tok_model_id: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) prefix_cache_n: Option<usize>,
    // ... more fields
}
```

Key defaults:
- Token source: HuggingFace cache (`~/.cache/huggingface/token`)
- Max sequences: 32
- Prefix cache: 16 sequences
- Device mapping: Auto with text model defaults

---

## 3. GGUF Model Loading

### GGUFLoaderBuilder
**File:** `mistralrs-core/src/pipeline/gguf.rs:125-216`

```rust
impl GGUFLoaderBuilder {
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self
```

### Content Loading from GGUF Files
**File:** `mistralrs-core/src/gguf/content.rs:49-110`

```rust
pub fn from_readers(readers: &'a mut [&'a mut R]) -> Result<Self> {
    let mut contents = Vec::new();
    for reader in readers.iter_mut() {
        contents.push(gguf_file::Content::read(reader)?);
    }
    // Detect architecture (llama, phi3, qwen, etc.)
    let arch = ct.metadata["general.architecture"].to_string()
        .and_then(GGUFArchitecture::from_value)?;
    // ...
}
```

### Architecture Detection
**File:** `mistralrs-core/src/gguf/mod.rs:14-32`

Supported architectures for GGUF:
- `Llama` - Maps to QLlama
- `Phi2` - Maps to QPhi
- `Phi3` - Maps to QPhi3
- `Starcoder2` - Maps to QStarcoder2
- `Qwen2` - Maps to QQwen
- `Qwen3` - Maps to QQwen3
- `Qwen3MoE` - Maps to QQwen3MoE

---

## 4. Tokenizer Initialization

### GGUF Tokenizer Conversion
**File:** `mistralrs-core/src/gguf/gguf_tokenizer.rs:69-144`

```rust
pub fn convert_gguf_to_hf_tokenizer<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Result<GgufTokenizerConversion> {
    // Extract tokenizer metadata
    let props = PropsGGUF::try_from(metadata)?;
    
    // Build tokenizer based on model type
    let (mut tokenizer, kind) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        "gpt2" => bpe_tokenizer(&props)?,
        _ => bail!("Tokenizer model not supported"),
    };
    // ...
}
```

---

## 5. Quantized Model Weights Loading

### Model Configuration from GGUF
**File:** `mistralrs-core/src/models/quantized_llama.rs:398-644`

```rust
impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        // Extract model properties
        let PropsGGUF { head_count, block_count, embedding_length, ... } = ...;
        
        // Load embeddings
        let tok_embeddings = ct.tensor("token_embd.weight", device)?.dequantize(device)?;
        
        // Load layers with quantized weights (GgufMatMul)
        for layer_idx in 0..block_count {
            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            // ... load all layer weights
            layers.push(LayerWeights { ... });
        }
    }
}
```

### Quantized Matrix Multiplication
**File:** `mistralrs-core/src/models/quantized_llama.rs:29-42`

```rust
struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,  // GgufMatMul
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;
        let y = &(candle_nn::ops::silu(&w1)? * w3)?;
        MatMul.qmethod_matmul(y, &*self.feed_forward_w2)
    }
}
```

---

## 6. Pipeline Creation

### GGUFPipeline Structure
**File:** `mistralrs-core/src/pipeline/gguf.rs:76-85`

```rust
pub struct GGUFPipeline {
    model: Model,  // enum of QLlama, QPhi, etc.
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}
```

---

## 7. Engine Initialization

### MistralRsBuilder
**File:** `mistralrs-core/src/lib.rs:271-366`

```rust
pub struct MistralRsBuilder {
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    method: SchedulerConfig,
    search_embedding_model: Option<SearchEmbeddingModel>,
    // ...
}
```

### Engine Thread for Metal (with autoreleasepool)
**File:** `mistralrs-core/src/lib.rs:413-437`

```rust
let engine_handler = thread::spawn(move || {
    #[cfg(feature = "metal")]
    objc::rc::autoreleasepool(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            let engine = Engine::new(
                tx_for_engine, rx, pipeline, method,
                no_kv_cache, no_prefix_cache, prefix_cache_n,
                // ...
            ).expect("Engine creation failed.");
            Arc::new(engine).run().await;
        })
    });
    // ...
});
```

---

## 8. Request Processing

### Chat Request Creation
**File:** `mistralrs/src/model.rs:106-148`

```rust
pub async fn send_chat_request<R: RequestLike>(
    &self,
    mut request: R,
) -> anyhow::Result<ChatCompletionResponse> {
    let (tx, mut rx) = channel(1);
    
    let request = Request::Normal(Box::new(NormalRequest {
        messages: request.take_messages(),
        sampling_params: request.take_sampling_params(),
        response: tx,
        is_streaming: false,
        // ...
    }));
    
    self.runner.get_sender(None)?.send(request).await?;
    // Wait for response...
}
```

---

## 9. Forward Pass Execution

### Model Forward
**File:** `mistralrs-core/src/models/quantized_llama.rs:647-700`

```rust
impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            // Apply attention norm
            let x = layer.attention_norm.forward(&x)?;
            // Self-attention with RoPE
            let attn = layer.forward_attn(&x, mask, start_offsets, cache, metadata)?;
            let x = (attn + residual)?;
            // MLP
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            layer_in = (x + residual)?;
        }
        
        // Final norm and output projection
        let layer_in = self.norm.forward(&layer_in)?;
        MatMul.qmethod_matmul(&layer_in, &*self.output)
    }
}
```

### Attention Implementation
**File:** `mistralrs-core/src/models/quantized_llama.rs:140-211`

```rust
fn forward_attn(&self, x: &Tensor, ...) -> Result<Tensor> {
    // Compute Q, K, V projections
    let q = MatMul.qmethod_matmul(x, &*self.attention_wq)?.to_dtype(self.dtype)?;
    let k = MatMul.qmethod_matmul(x, &*self.attention_wk)?.to_dtype(self.dtype)?;
    let v = MatMul.qmethod_matmul(x, &*self.attention_wv)?.to_dtype(self.dtype)?;
    
    // Apply RoPE
    let (q, k) = self.rotary.forward(&q, &k, start_offsets)?;
    
    // Run attention (with or without PagedAttention)
    let y = match &self.paged_attn {
        Some(paged_attn) => paged_attn.forward(&q, &k, &v, ...),
        None => Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?,
    };
    
    // Output projection
    MatMul.qmethod_matmul(&y, &*self.attention_wo)
}
```

---

## 10. Sampling

### Sampling Process
**File:** `mistralrs-core/src/pipeline/sampling.rs:28-68`

```rust
pub(crate) async fn finish_or_add_toks_to_seq(
    this: &dyn Pipeline,
    prefix_cacher: &mut PrefixCacheManagerV2,
    seq: &mut Sequence,
    logprobs: Logprobs,
    eos_tok: Option<&[u32]>,
    use_prefix_cacher: bool,
) -> Result<()> {
    let mut is_done = seq.is_done(logprobs.token, eos_tok, this.get_metadata().max_seq_len);
    seq.add_token(logprobs.clone(), ...);
    // ...
}
```

### SamplingParams
**File:** `mistralrs-core/src/sampler.rs:31-68`

```rust
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub frequency_penalty: Option<f32>,
    pub max_len: Option<usize>,
    // ...
}

impl SamplingParams {
    pub fn deterministic() -> Self {
        Self {
            temperature: None,
            top_k: Some(1),  // Greedy sampling
            // ...
        }
    }
}
```

---

## 11. Response Generation

### Streaming Response
**File:** `mistralrs/src/model.rs:51-102`

```rust
pub struct Stream<'a> {
    _server: &'a Model,
    rx: Receiver<Response>,
}

impl Stream<'_> {
    pub async fn next(&mut self) -> Option<Response> {
        self.rx.recv().await
    }
}
```

---

## Metal-Specific Optimizations

### Memory Management
**File:** `mistralrs-core/src/utils/memory_usage.rs:30-37`

```rust
#[cfg(feature = "metal")]
Device::Metal(dev) => {
    let max = dev.device().recommended_max_working_set_size();
    let current = dev.device().current_allocated_size();
    let avail = max.saturating_sub(current);
    Ok(avail)
}
```

### PagedAttention for Metal
**File:** `mistralrs-core/src/paged_attention/layers/mod.rs:1-4`

PagedAttention is supported on Metal (and CUDA) when the feature is enabled:
```rust
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub mod paged_attention;
```

### Autoreleasepool for Metal
**File:** `mistralrs-core/src/lib.rs:415-437`

The engine thread runs inside an `objc::rc::autoreleasepool` on Metal to properly manage Objective-C memory.

---

## Build Configuration

### Cargo Features for Metal
**File:** `mistralrs/Cargo.toml:37`

```toml
[features]
metal = ["mistralrs-core/metal"]
```

Build command:
```bash
cargo build --release --features metal
```

---

## Summary Flow

1. **Device** → `Device::new_metal(0)` creates Metal device
2. **Builder** → `GgufModelBuilder::new(model_id, files)` configures loading
3. **Content** → GGUF file readers parse metadata and tensors
4. **Architecture** → `GGUFArchitecture::from_value()` detects model type
5. **Tokenizer** → Built from GGUF metadata or external file
6. **Weights** → `ModelWeights::from_gguf()` loads quantized tensors to device
7. **Pipeline** → `GGUFPipeline` wraps model, tokenizer, and metadata
8. **Engine** → `MistralRsBuilder::build()` creates engine with scheduler
9. **Request** → `model.send_chat_request(messages)` submits request
10. **Forward** → Model executes transformer layers with quantized matmul
11. **Sample** → Logits are sampled to produce tokens
12. **Response** → Tokens decoded and returned as `ChatCompletionResponse`

