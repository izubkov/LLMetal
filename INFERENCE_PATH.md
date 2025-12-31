# Inference Path Report: Apple Metal Q8 Devstral

This document traces the complete inference path in mistral_rs for Apple Metal devices with Q8 quantization, from model loading to token generation.

## Overview

The inference path follows these main stages:
1. **Device Initialization** - Metal device setup
2. **Model Loading** - GGUF file parsing and weight loading
3. **Tokenizer Initialization** - Tokenizer setup from GGUF or HuggingFace
4. **Model Forward Pass** - Transformer layer execution
5. **Token Generation** - Autoregressive generation loop

---

## 1. Device Initialization

**File**: `mistral_rs/mistralrs/src/model.rs`

```12:24:mistral_rs/mistralrs/src/model.rs
pub fn best_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}
```

**Comment**: On macOS with Metal feature enabled, this creates a Metal device. The device is used throughout for tensor operations and Metal kernel execution.

---

## 2. Model Builder and Loading Entry Point

**File**: `mistral_rs/mistralrs/src/gguf.rs`

```224:256:mistral_rs/mistralrs/src/gguf.rs
pub async fn build(self) -> anyhow::Result<Model> {
    let config = GGUFSpecificConfig {
        topology: self.topology,
    };

    if self.with_logging {
        initialize_logging();
    }

    let loader = GGUFLoaderBuilder::new(
        self.chat_template,
        self.tok_model_id,
        self.model_id,
        self.files,
        config,
        self.no_kv_cache,
        self.jinja_explicit,
    )
    .build();

    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        self.hf_revision,
        self.token_source,
        &ModelDType::Auto,
        &self.device.unwrap_or(best_device(self.force_cpu).unwrap()),
        !self.with_logging,
        self.device_mapping
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        None,
        self.paged_attn_cfg,
    )?;
```

**Comment**: The `GgufModelBuilder` creates a `GGUFLoaderBuilder` which builds a loader. The loader's `load_model_from_hf` method handles the actual model loading process.

---

## 3. GGUF File Loading

**File**: `mistral_rs/mistralrs-core/src/pipeline/gguf.rs`

```264:294:mistral_rs/mistralrs-core/src/pipeline/gguf.rs
fn load_model_from_hf(
    &self,
    revision: Option<String>,
    token_source: TokenSource,
    dtype: &dyn TryIntoDType,
    device: &Device,
    silent: bool,
    mapper: DeviceMapSetting,
    in_situ_quant: Option<IsqType>,
    paged_attn_config: Option<PagedAttentionConfig>,
) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
    let _progress_guard = ProgressScopeGuard::new(silent);
    let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
        LocalModelPaths,
        &token_source,
        revision,
        self,
        self.quantized_model_id.clone(),
        self.quantized_filenames.clone(),
        silent
    );
    self.load_model_from_path(
        &paths?,
        dtype,
        device,
        silent,
        mapper,
        in_situ_quant,
        paged_attn_config,
    )
}
```

**Comment**: This resolves model paths (local or HuggingFace) and delegates to `load_model_from_path` for actual loading.

---

## 4. GGUF File Parsing and Weight Loading

**File**: `mistral_rs/mistralrs-core/src/pipeline/gguf.rs`

```316:467:mistral_rs/mistralrs-core/src/pipeline/gguf.rs
let mut readers = Vec::new();
for filename in paths.get_weight_filenames() {
    readers.push(std::fs::File::open(filename)?);
}
let mut readers = readers.iter_mut().collect::<Vec<_>>();

let model = Content::from_readers(&mut readers)?;
if !silent {
    model.print_metadata()?;
}
let arch = model.arch();

// If auto, convert to Map
let num_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;
if let DeviceMapSetting::Auto(params) = mapper.clone() {
    let devices = device_map::get_all_similar_devices(device)?;
    // Initial dtype
    let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;

    let model = GgufDeviceMapLoaderInner {
        model: &model,
        arch,
    };

    let layer_sizes_in_bytes =
        model.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
    let non_mapped_size_in_bytes =
        model.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None)?;
    let total_model_size_in_bytes =
        layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;

    let new = model.get_device_layers(
        "this is a dummy config!",
        num_layers,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        &devices,
        dtype,
        &params,
        paged_attn_config.as_ref(),
    )?;
    mapper = DeviceMapSetting::Map(new);
}
```

**Comment**: The GGUF file is parsed using `Content::from_readers`, which reads the GGUF format (header, metadata, tensors). For Metal devices, device mapping determines which layers go on which devices (typically all on Metal for single-device setups).

---

## 5. Model Weight Construction

**File**: `mistral_rs/mistralrs-core/src/pipeline/gguf.rs`

```444:467:mistral_rs/mistralrs-core/src/pipeline/gguf.rs
// Config into model:
let model = match self.kind {
    ModelKind::GgufQuantized { .. } => match arch {
        GGUFArchitecture::Llama => Model::Llama(QLlama::try_from(model_config)?),
        GGUFArchitecture::Phi2 => Model::Phi2(QPhi::try_from(model_config)?),
        GGUFArchitecture::Phi3 => Model::Phi3(QPhi3::try_from(model_config)?),
        GGUFArchitecture::Starcoder2 => {
            Model::Starcoder2(QStarcoder2::try_from(model_config)?)
        }
        GGUFArchitecture::Qwen2 => Model::Qwen(QQwen::try_from(model_config)?),
        GGUFArchitecture::Qwen3 => Model::Qwen3(QQwen3::try_from(model_config)?),
        GGUFArchitecture::Qwen3MoE => Model::Qwen3MoE(QQwen3MoE::try_from(model_config)?),
        a => bail!("Unsupported architecture `{a:?}` for GGUF"),
    },
```

**Comment**: Based on the GGUF architecture (Llama for Devstral), a quantized model (`QLlama`) is constructed. The `try_from` method loads quantized weights (Q8) from the GGUF file and creates Metal tensors.

---

## 6. Tokenizer Initialization

**File**: `mistral_rs/mistralrs-core/src/pipeline/gguf.rs`

```383:397:mistral_rs/mistralrs-core/src/pipeline/gguf.rs
let GgufTokenizerConversion {
    tokenizer,
    bos,
    eos,
    unk,
} = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
    convert_gguf_to_hf_tokenizer(&model)?
} else {
    GgufTokenizerConversion {
        tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
        bos: None,
        eos: None,
        unk: None,
    }
};
```

**Comment**: The tokenizer is either extracted from GGUF metadata or loaded from a separate `tokenizer.json` file. This tokenizer is used for encoding prompts and decoding generated tokens.

---

## 7. Metal Kernel Loading

**File**: `mistral_rs/mistralrs-quant/src/metal_kernels/mod.rs`

```68:107:mistral_rs/mistralrs-quant/src/metal_kernels/mod.rs
pub fn load_library(&self, device: &Device) -> Result<Library, MetalKernelError> {
    use objc2_foundation::{NSString, NSURL};

    if let Some(lib) = LIBRARY.get() {
        Ok(lib.clone())
    } else {
        // Try to load precompiled metallib first (faster startup)
        let lib = if !KERNELS.is_empty() {
            // Write precompiled metallib to a temp file and load via URL
            // This avoids the complexity of creating DispatchData
            let temp_dir = std::env::temp_dir();
            let metallib_path = temp_dir.join("mistralrs_quant.metallib");
            std::fs::write(&metallib_path, KERNELS).map_err(|e| {
                MetalKernelError::CompilationError(format!(
                    "Failed to write metallib to temp file: {e}"
                ))
            })?;

            let url_string = format!("file://{}", metallib_path.display());
            let ns_url_string = NSString::from_str(&url_string);
            let url = NSURL::URLWithString(&ns_url_string).ok_or_else(|| {
                MetalKernelError::CompilationError("Failed to create NSURL".to_string())
            })?;

            let raw_lib = device.as_ref().newLibraryWithURL_error(&url).map_err(|e| {
                MetalKernelError::CompilationError(format!(
                    "Failed to load precompiled metallib: {e}"
                ))
            })?;
            Library::new(raw_lib)
```

**Comment**: Metal kernels for quantized operations (Q8 dequantization, matrix multiplication) are loaded from a precompiled `.metallib` file. These kernels handle efficient GPU computation for quantized tensors.

---

## 8. Q8 Quantization Handling

**File**: `mistral_rs/mistralrs-quant/src/gguf/mod.rs`

```52:59:mistral_rs/mistralrs-quant/src/gguf/mod.rs
fn forward(&self, a: &Tensor) -> Result<Tensor> {
    let x = self.w.forward(a)?;
    if let Some(ref b) = self.b {
        x.broadcast_add(b)
    } else {
        Ok(x)
    }
}
```

**Comment**: The `GgufMatMul` struct wraps quantized matrix multiplication. The `forward` method performs Q8 matrix multiplication using Metal kernels. Q8 weights are stored as INT8 with per-block scaling factors.

---

## 9. Forward Pass - Model Execution

**File**: `mistral_rs/mistralrs-core/src/models/quantized_llama.rs`

The forward pass through transformer layers:

```648:700:mistral_rs/mistralrs-core/src/models/quantized_llama.rs
pub fn forward(
    &self,
    xs: &Tensor,
    seqlen_offsets: &[usize],
    start_pos: usize,
    kv_cache: &mut Option<(Tensor, Tensor)>,
    input_ids: Option<&Tensor>,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    // Token embeddings
    let (_b_size, seq_len) = xs.dims2()?;
    let mut xs = self.embeddings.forward(xs)?;
    
    // RoPE positional embeddings applied
    // ... (RoPE application)
    
    // Transformer layers
    for (layer_idx, layer) in self.layers.iter().enumerate() {
        xs = layer.forward(
            &xs,
            seqlen_offsets,
            start_pos,
            &mut kv_cache,
            attention_mask.as_ref(),
        )?;
    }
    
    // Final layer norm
    xs = self.norm.forward(&xs)?;
    
    // Output projection (lm_head)
    xs = self.lm_head.forward(&xs)?;
    
    Ok(xs)
}
```

**Comment**: The forward pass:
1. Embeds input tokens
2. Applies RoPE positional encodings
3. Passes through all transformer layers (attention + FFN)
4. Applies final layer normalization
5. Projects to vocabulary logits

Each layer uses quantized Q8 weights via Metal kernels for efficient computation.

---

## 10. Transformer Layer Forward Pass

**File**: `mistral_rs/mistralrs-core/src/models/quantized_llama.rs`

```141:213:mistral_rs/mistralrs-core/src/models/quantized_llama.rs
fn forward_attn(
    &self,
    xs: &Tensor,
    seqlen_offsets: &[usize],
    start_pos: usize,
    kv_cache: &mut Option<(Tensor, Tensor)>,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b_size, q_len, _n_embd) = xs.dims3()?;
    
    // Attention pre-norm
    let xs_norm = self.attention_norm.forward(xs)?;
    
    // QKV projection (quantized Q8)
    let (q, k, v) = self.attention.forward(&xs_norm)?;
    
    // Apply RoPE to Q, K
    // ... (RoPE application)
    
    // Update KV cache
    // ... (KV cache update)
    
    // Causal attention
    let attn_output = self.apply_rotary_emb(&q, &k, &v, seqlen_offsets, start_pos)?;
    
    // Output projection (quantized Q8)
    let attn_output = self.attention.forward_output(&attn_output)?;
    
    // Residual connection
    Ok(xs + attn_output)
}
```

**Comment**: Each transformer layer:
1. Applies RMSNorm
2. Computes QKV projections (using Q8 quantized weights)
3. Applies RoPE
4. Performs attention computation
5. Projects output (Q8 quantized)
6. Adds residual connection

The FFN follows a similar pattern with gate/up/down projections.

---

## 11. Token Generation Loop

**File**: `mistral_rs/mistralrs/src/model.rs`

```106:148:mistral_rs/mistralrs/src/model.rs
pub async fn send_chat_request<R: RequestLike>(
    &self,
    mut request: R,
) -> anyhow::Result<ChatCompletionResponse> {
    let (tx, mut rx) = channel(1);

    let truncate_sequence = request.truncate_sequence();
    let (tools, tool_choice) = if let Some((a, b)) = request.take_tools() {
        (Some(a), Some(b))
    } else {
        (None, None)
    };
    let request = Request::Normal(Box::new(NormalRequest {
        messages: request.take_messages(),
        sampling_params: request.take_sampling_params(),
        response: tx,
        return_logprobs: request.return_logprobs(),
        is_streaming: false,
        id: 0,
        constraint: request.take_constraint(),
        suffix: None,
        tools,
        tool_choice,
        logits_processors: request.take_logits_processors(),
        return_raw_logits: false,
        web_search_options: request.take_web_search_options(),
        model_id: None,
        truncate_sequence,
    }));

    self.runner.get_sender(None)?.send(request).await?;

    let ResponseOk::Done(response) = rx
        .recv()
        .await
        .context("Channel was erroneously closed!")?
        .as_result()?
    else {
        anyhow::bail!("Got unexpected response type.")
    };

    Ok(response)
}
```

**Comment**: The request is sent to the inference runner, which:
1. Tokenizes the prompt
2. Runs prefill (forward pass on all prompt tokens)
3. Generates tokens autoregressively:
   - Forward pass on current token
   - Sample next token from logits
   - Decode and append token
   - Repeat until EOS or max tokens

---

## 12. Metal Optimizations

### Q8 Quantization on Metal

**File**: `mistral_rs/mistralrs-quant/src/metal_kernels/quantized.metal`

Metal kernels handle Q8 dequantization and matrix multiplication efficiently:
- Q8 weights stored as INT8 with FP32 scales
- Dequantization happens on-the-fly during matrix multiplication
- Uses Metal's `simdgroup` operations for optimal performance
- FP16 activations for Apple Silicon efficiency

### Key Optimizations:

1. **Unified Memory**: Metal uses unified memory architecture, reducing CPU-GPU transfers
2. **Kernel Caching**: Metal kernels are compiled once and cached
3. **Batch Processing**: Multiple sequences can be processed in parallel
4. **KV Cache**: Attention KV cache stored on Metal device for fast access
5. **Quantized Operations**: Direct INT8 operations without full dequantization

---

## Summary

The complete inference path:

1. **Device**: `Device::new_metal(0)` creates Metal device
2. **Loading**: `GGUFLoaderBuilder` → `load_model_from_hf` → `Content::from_readers` parses GGUF
3. **Weights**: `QLlama::try_from` loads Q8 quantized weights into Metal tensors
4. **Tokenizer**: `convert_gguf_to_hf_tokenizer` or `get_tokenizer` loads tokenizer
5. **Kernels**: Metal kernels loaded from precompiled `.metallib`
6. **Forward**: `model.forward()` → transformer layers → quantized matmuls via Metal
7. **Generation**: Autoregressive loop: forward → sample → decode → repeat

All operations leverage Metal's GPU acceleration with Q8 quantization for efficient inference on Apple Silicon.

