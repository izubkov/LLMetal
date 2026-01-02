# Detailed Transformer Inference Path for Apple Metal (Q8 Quantization)

This document traces the complete, **step-by-step inference path** for running a GGUF quantized model on Apple Metal using mistral.rs. Each transformer operation is documented with code references.

---

## Overview: The 12 Steps of Inference

```
FOR EACH TOKEN:
  1. Token Embedding: token_id → hidden_states [batch, seq, hidden_size]

FOR EACH LAYER (repeated n_layers times):
  2. Pre-Attention RMS Norm
  3. Q/K/V Projections (quantized matmul)
  4. RoPE (Rotary Position Embedding)
  5. KV Cache Update
  6. SDPA (Scaled Dot-Product Attention)
  7. Output Projection
  8. Residual Connection (Post-Attention)
  9. Pre-FFN RMS Norm + Feed-Forward (SwiGLU)
  8. Residual Connection (Post-FFN)

AFTER ALL LAYERS:
  10. Final RMS Norm
  11. LM Head (Vocabulary Projection)
  12. Sampling (Temperature, Top-k, Top-p)
```

---

## Step 1: Token Embedding Lookup

Converts token IDs to dense vectors using an embedding table.

**File:** `mistralrs-core/src/models/quantized_llama.rs:655`

```rust
let mut layer_in = self.tok_embeddings.forward(x)?;
```

- **Input:** `token_ids` `[batch_size, seq_len]` - integers
- **Output:** `hidden_states` `[batch_size, seq_len, hidden_size]` - float tensors
- **Note:** The embedding table is typically dequantized from GGUF

---

## Step 2: RMS Normalization (Pre-Attention)

Root Mean Square Layer Normalization applied before attention.

**File:** `mistralrs-core/src/layers.rs:305-307`

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
}
```

**Formula:**
```
RMSNorm(x) = x * rsqrt(mean(x²) + eps) * weight
```

This is more efficient than LayerNorm as it doesn't center (subtract the mean).

---

## Step 3: Q, K, V Projections (Quantized MatMul)

Projects hidden states into Query, Key, Value spaces using quantized weights.

**File:** `mistralrs-core/src/models/quantized_llama.rs:151-159`

```rust
let q = MatMul.qmethod_matmul(x, &*self.attention_wq)?
    .to_dtype(self.dtype)?;
let k = MatMul.qmethod_matmul(x, &*self.attention_wk)?
    .to_dtype(self.dtype)?;
let v = MatMul.qmethod_matmul(x, &*self.attention_wv)?
    .to_dtype(self.dtype)?;
```

**For Q8_0 Quantization:**
- Weights stored as 8-bit integers with block-wise scaling
- Block size: 32 elements share one fp16 scale factor
- Dequantization: `w_float = w_int8 * scale`
- Memory reduction: ~4x vs fp32, ~2x vs fp16

**Grouped Query Attention (GQA):**
- Q has `n_heads * head_dim` dimensions
- K, V have `n_kv_heads * head_dim` dimensions (fewer heads, shared across groups)

**Reshape to attention format:**
```rust
let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?
         .transpose(1, 2)?;  // [batch, n_heads, seq, head_dim]
```

---

## Step 4: Rotary Position Embedding (RoPE)

Applies rotational position encoding to Q and K tensors.

**File:** `mistralrs-core/src/models/quantized_llama.rs:179`

```rust
let (q, k) = self.rotary.forward(&q, &k, start_offsets)?;
```

**RoPE Formula:**
```
For each position pos and dimension pair (i, i+1):
  theta_i = base^(-2i/dim)
  cos_theta = cos(pos * theta_i)
  sin_theta = sin(pos * theta_i)
  [q_new_i, q_new_{i+1}] = [q_i * cos - q_{i+1} * sin, q_i * sin + q_{i+1} * cos]
```

This encodes absolute position while allowing relative position attention.

---

## Step 5: KV Cache Update

Appends current K, V to the cache for autoregressive generation.

**File:** `mistralrs-core/src/models/quantized_llama.rs:196-200`

```rust
let (k, v) = kv_cache.append(&k, &v)?;
```

**During generation:**
- First token: `K_cache = K, V_cache = V`
- Subsequent tokens: `K_cache = concat([K_cache, K_new], dim=seq)`

This enables reusing previous computations.

---

## Step 6: Scaled Dot-Product Attention (SDPA)

The core attention mechanism.

**File:** `mistralrs-core/src/attention/mod.rs:106-155`

```rust
pub fn run_attention(
    &self, q: &Tensor, k: &Tensor, v: &Tensor,
    mask: Option<&Tensor>, flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    // ...
    if [q, k, v].into_iter().all(|x| x.device().is_metal())
        && all_head_dims_match && valid_head_dims.contains(&head_dim) && can_use_mask
    {
        return candle_nn::ops::sdpa(q, k, v, mask.as_ref(), /* ... */);
    }
    // ...
}
```

**Formula:**
```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(head_dim)) @ V
```

**Metal Optimizations:**
- Uses fused Metal SDPA kernel for valid head dimensions: `[32, 64, 72, 80, 96, 128]`
- Vector SDPA for single-token generation
- Full SDPA for prefill (prompt processing)

---

## Step 7: Output Projection

Projects attention output back to hidden_size dimensions.

**File:** `mistralrs-core/src/models/quantized_llama.rs:203-209`

```rust
let y = if mask.is_some() {
    y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
} else {
    y.reshape((b_sz, seq_len, ()))?
};
let y = MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)?;
```

---

## Step 8: Residual Connection (Post-Attention)

Adds the attention output to the original input (skip connection).

**File:** `mistralrs-core/src/models/quantized_llama.rs:691`

```rust
let x = (attn + residual)?;
```

This helps with gradient flow and preserves information.

---

## Step 9: Feed-Forward Network (SwiGLU)

The MLP block with SwiGLU activation.

**File:** `mistralrs-core/src/models/quantized_llama.rs:36-41`

```rust
impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;  // gate
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;  // up
        let y = &(candle_nn::ops::silu(&w1)? * w3)?;                   // SwiGLU
        MatMul.qmethod_matmul(y, &*self.feed_forward_w2)              // down
    }
}
```

**SwiGLU Formula:**
```
gate = W1(x)                      # gate projection
up = W3(x)                        # up projection
hidden = SiLU(gate) * up          # SwiGLU activation
output = W2(hidden)               # down projection

SiLU(x) = x * sigmoid(x)          # Swish activation
```

---

## Step 10: Final RMS Normalization

Applied after all layers, before the LM head.

**File:** `mistralrs-core/src/models/quantized_llama.rs:701`

```rust
let x = self.norm.forward(&layer_in)?;
```

---

## Step 11: LM Head (Vocabulary Projection)

Projects final hidden states to vocabulary logits.

**File:** `mistralrs-core/src/models/quantized_llama.rs:702-705`

```rust
extract_logits(
    &MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)?,
    context_lens,
)
```

- **Input:** `hidden_states` `[batch, seq, hidden_size]`
- **Output:** `logits` `[batch, seq, vocab_size]`

Often the LM head weights are tied to the embedding weights (transposed).

---

## Step 12: Sampling

Converts logits to probabilities and samples the next token.

**File:** `mistralrs-core/src/pipeline/sampling.rs`

**Sampling Options:**
- **Greedy:** `argmax(logits)`
- **Temperature:** `logits / temperature → softmax → sample`
- **Top-k:** Keep only top-k logits → softmax → sample
- **Top-p (nucleus):** Keep cumulative prob ≤ p → sample

---

## Complete Forward Pass (Pseudocode)

```python
def forward(token_ids, kv_caches):
    # Step 1: Token embedding
    hidden = embedding_table[token_ids]  # [batch, seq, hidden_size]
    
    for layer_idx in range(n_layers):
        residual = hidden
        
        # Step 2: Pre-attention normalization
        hidden = rms_norm(hidden, attn_norm_weight)
        
        # Step 3: Q, K, V projections (quantized matmul)
        Q = dequant_matmul(hidden, Wq)  # [batch, seq, n_heads * head_dim]
        K = dequant_matmul(hidden, Wk)  # [batch, seq, n_kv_heads * head_dim]
        V = dequant_matmul(hidden, Wv)
        
        # Reshape to [batch, heads, seq, head_dim]
        Q = Q.reshape(batch, seq, n_heads, head_dim).transpose(1, 2)
        K = K.reshape(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
        V = V.reshape(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
        
        # Step 4: Apply rotary position embedding
        Q, K = apply_rope(Q, K, positions, cos_cache, sin_cache)
        
        # Step 5: Update KV cache
        K, V = update_kv_cache(K, V, kv_caches[layer_idx])
        
        # Step 6: Scaled dot-product attention
        K = repeat_kv(K, n_kv_groups)  # GQA expansion
        V = repeat_kv(V, n_kv_groups)
        attn = softmax(Q @ K.T / sqrt(head_dim) + causal_mask) @ V
        
        # Step 7: Output projection
        attn = attn.transpose(1, 2).reshape(batch, seq, hidden_size)
        hidden = dequant_matmul(attn, Wo)
        
        # Step 8: Residual connection
        hidden = residual + hidden
        residual = hidden
        
        # Pre-FFN normalization
        hidden = rms_norm(hidden, ffn_norm_weight)
        
        # Step 9: Feed-forward network (SwiGLU)
        gate = dequant_matmul(hidden, W1)
        up = dequant_matmul(hidden, W3)
        hidden = silu(gate) * up
        hidden = dequant_matmul(hidden, W2)
        
        # Residual connection
        hidden = residual + hidden
    
    # Step 10: Final normalization
    hidden = rms_norm(hidden, final_norm_weight)
    
    # Step 11: Project to vocabulary
    logits = dequant_matmul(hidden, lm_head_weight)
    
    # Step 12: Sample next token
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
    
    return next_token
```

---

## Metal-Specific Optimizations

### 1. Fused SDPA Kernel
**File:** `mistralrs-core/src/attention/mod.rs:187-200`

```rust
if [q, k, v].into_iter().all(|x| x.device().is_metal())
    && all_head_dims_match
    && valid_head_dims.contains(&head_dim)
    && can_use_mask
{
    return candle_nn::ops::sdpa(q, k, v, mask.as_ref(), ...);
}
```

Valid head dimensions for Metal SDPA: `[32, 64, 72, 80, 96, 128]` (vector), `[32, 64, 72, 80, 96, 128]` (full).

### 2. Device Selection
**File:** `mistralrs/src/model.rs:12-24`

```rust
pub fn best_device(force_cpu: bool) -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}
```

### 3. Quantized Matrix Multiplication
**File:** `mistralrs-quant/src/gguf/mod.rs`

- Uses `GgufMatMul` for quantized weight operations
- Block-wise dequantization optimized for Metal

---

## Key Files Reference

| Component | File Path |
|-----------|-----------|
| Model Forward Pass | `mistralrs-core/src/models/quantized_llama.rs` |
| RMS Normalization | `mistralrs-core/src/layers.rs:291-307` |
| Attention (SDPA) | `mistralrs-core/src/attention/mod.rs` |
| Quantized MatMul | `mistralrs-quant/src/gguf/mod.rs` |
| RoPE Embedding | `mistralrs-core/src/layers.rs` (RotaryEmbedding) |
| Sampling | `mistralrs-core/src/pipeline/sampling.rs` |
| Metal Device | `mistralrs/src/model.rs:12-24` |
| GGUF Loading | `mistralrs-core/src/pipeline/gguf.rs` |
| Tokenizer Conversion | `mistralrs-core/src/gguf/gguf_tokenizer.rs` |

---

## Example: Devstral/Mistral Configuration

```rust
ModelConfig {
    n_layers: 32,
    vocab_size: 32000,
    hidden_size: 4096,
    n_heads: 32,
    n_kv_heads: 8,        // GQA with 4 groups
    head_dim: 128,
    intermediate_size: 14336,
    rms_norm_eps: 1e-5,
    rope_theta: 10000.0,
    max_seq_len: 32768,
    rope_dim: 128,
}
```

---

## Running the Detailed Inference Demo

```bash
# Compile and run the detailed inference demonstration
cargo run --release --bin inference_2

# The inference_2.rs file contains all 12 steps with detailed documentation
```
