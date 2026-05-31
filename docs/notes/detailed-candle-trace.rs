//! Detailed GGUF Q8 Inference Steps for Apple Metal
//!
//! This file demonstrates the DETAILED inference path showing all transformer
//! operations: token embedding, RMS normalization, attention (Q/K/V projections,
//! RoPE, SDPA), feed-forward network (SwiGLU), and sampling.
//!
//! Based on the actual mistralrs implementation in:
//! - mistralrs-core/src/models/quantized_llama.rs
//! - mistralrs-core/src/layers.rs
//! - mistralrs-core/src/attention/mod.rs
//!
//! Historical note: this was an early detailed trace sketch, not the active binary.

use anyhow::Result;
// Candle tensor library (used by mistralrs under the hood)
use candle_core::{
    quantized::QTensor,
    DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D,
};
use candle_nn::{Embedding, ops::rms_norm, ops::silu, ops::softmax_last_dim};

// =============================================================================
// MODEL CONFIGURATION
// =============================================================================

/// Model configuration extracted from GGUF metadata
/// These parameters define the transformer architecture
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of transformer layers (blocks)
    pub n_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension (embedding size)
    pub hidden_size: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (for GQA - Grouped Query Attention)
    pub n_kv_heads: usize,
    /// Dimension per head = hidden_size / n_heads
    pub head_dim: usize,
    /// Intermediate size for FFN (typically 4 * hidden_size or custom)
    pub intermediate_size: usize,
    /// RMS normalization epsilon
    pub rms_norm_eps: f32,
    /// RoPE base frequency
    pub rope_theta: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE dimension (usually == head_dim)
    pub rope_dim: usize,
}

// =============================================================================
// DETAILED INFERENCE STEPS
// =============================================================================

/// STEP 1: Token Embedding Lookup
/// 
/// Converts token IDs to dense vectors using an embedding table.
/// 
/// Input: token_ids [batch_size, seq_len] - integers
/// Output: hidden_states [batch_size, seq_len, hidden_size] - float tensors
/// 
/// This is the ONLY dequantized weight in typical GGUF models.
pub fn step1_token_embedding(
    token_ids: &Tensor,
    embedding_table: &Embedding,
) -> CandleResult<Tensor> {
    println!("  STEP 1: Token Embedding");
    println!("    Input shape: {:?}", token_ids.shape());
    
    // Simple lookup: for each token_id, fetch the corresponding row from embedding_table
    let hidden_states = embedding_table.forward(token_ids)?;
    
    println!("    Output shape: {:?}", hidden_states.shape());
    println!("    Operation: embedding_table[token_id] for each token");
    
    Ok(hidden_states)
}

/// STEP 2: RMS Normalization (Pre-Attention)
/// 
/// Root Mean Square Layer Normalization - used before attention and FFN.
/// 
/// Formula: RMSNorm(x) = x * rsqrt(mean(x²) + eps) * weight
/// 
/// This is more efficient than LayerNorm as it doesn't subtract the mean.
pub fn step2_rms_norm(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    step_name: &str,
) -> CandleResult<Tensor> {
    println!("  STEP 2: RMS Normalization ({})", step_name);
    println!("    Input shape: {:?}", x.shape());
    
    // rms_norm from candle_nn::ops handles the full computation
    // Internally it does:
    // 1. x_squared = x * x
    // 2. mean_x2 = mean(x_squared, dim=-1, keepdim=True)  
    // 3. rsqrt_val = rsqrt(mean_x2 + eps)
    // 4. normalized = x * rsqrt_val
    // 5. output = normalized * weight
    let normalized = rms_norm(&x.contiguous()?, weight, eps)?;
    
    println!("    Formula: x * rsqrt(mean(x²) + {}) * weight", eps);
    println!("    Output shape: {:?}", normalized.shape());
    
    Ok(normalized)
}

/// STEP 3: Q, K, V Projections (Quantized MatMul)
/// 
/// Projects hidden states into Query, Key, Value spaces using quantized weights.
/// 
/// For GGUF Q8_0 quantization:
/// - Weights stored as 8-bit integers with block-wise scaling
/// - Dequantization: w_float = w_int8 * scale
/// - MatMul: output = input @ W.T (transposed weight matrix)
/// 
/// Grouped Query Attention (GQA):
/// - Q has n_heads * head_dim = hidden_size dimensions
/// - K,V have n_kv_heads * head_dim dimensions (fewer heads, shared across groups)
pub fn step3_qkv_projection(
    hidden_states: &Tensor,
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<(Tensor, Tensor, Tensor)> {
    println!("  STEP 3: Q/K/V Projections (Quantized MatMul)");
    println!("    Input shape: {:?}", hidden_states.shape());
    
    // Dequantize and perform matrix multiplication
    // Q8_0 format: blocks of 32 values with shared scale factor
    let q = quantized_matmul(hidden_states, wq, device)?
        .to_dtype(dtype)?;
    let k = quantized_matmul(hidden_states, wk, device)?
        .to_dtype(dtype)?;
    let v = quantized_matmul(hidden_states, wv, device)?
        .to_dtype(dtype)?;
    
    println!("    Q shape (before reshape): {:?}", q.shape());
    println!("    K shape (before reshape): {:?}", k.shape());
    println!("    V shape (before reshape): {:?}", v.shape());
    
    // Reshape to [batch, seq_len, n_heads, head_dim] then transpose to [batch, n_heads, seq_len, head_dim]
    let (b_sz, seq_len, _) = hidden_states.dims3()?;
    
    let q = q.reshape((b_sz, seq_len, n_heads, head_dim))?
             .transpose(1, 2)?;
    let k = k.reshape((b_sz, seq_len, n_kv_heads, head_dim))?
             .transpose(1, 2)?;
    let v = v.reshape((b_sz, seq_len, n_kv_heads, head_dim))?
             .transpose(1, 2)?;
    
    println!("    Q reshaped: [batch={}, heads={}, seq={}, head_dim={}]", b_sz, n_heads, seq_len, head_dim);
    println!("    K reshaped: [batch={}, kv_heads={}, seq={}, head_dim={}]", b_sz, n_kv_heads, seq_len, head_dim);
    
    Ok((q, k, v))
}

/// STEP 4: Rotary Position Embedding (RoPE)
/// 
/// Applies rotational position encoding to Q and K tensors.
/// 
/// RoPE Formula:
/// - For each position pos and dimension pair (i, i+1):
/// - theta_i = base^(-2i/dim)
/// - cos_theta = cos(pos * theta_i)
/// - sin_theta = sin(pos * theta_i)
/// - [q_new_i, q_new_{i+1}] = [q_i * cos - q_{i+1} * sin, q_i * sin + q_{i+1} * cos]
/// 
/// This encodes absolute position while allowing relative position attention.
pub fn step4_rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    position_ids: &[usize],
    cos_cache: &Tensor,
    sin_cache: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    println!("  STEP 4: Rotary Position Embedding (RoPE)");
    println!("    Position IDs: {:?}", position_ids);
    
    // Select cos/sin values for current positions
    // cos_cache shape: [max_seq_len, head_dim]
    // We need to index by position_ids
    
    let q_rotated = apply_rotary_emb(q, cos_cache, sin_cache, position_ids)?;
    let k_rotated = apply_rotary_emb(k, cos_cache, sin_cache, position_ids)?;
    
    println!("    Formula: [x_rot, x_pass] where x_rot = x * cos + rotate_half(x) * sin");
    println!("    Q/K rotated shape: {:?}", q_rotated.shape());
    
    Ok((q_rotated, k_rotated))
}

/// STEP 5: KV Cache Update
/// 
/// Appends current K,V to the cache for autoregressive generation.
/// 
/// During generation:
/// - First token: K_cache = K, V_cache = V
/// - Subsequent tokens: K_cache = concat([K_cache, K_new], dim=seq)
/// 
/// This allows reusing previous computations.
pub fn step5_kv_cache_update(
    k: Tensor,
    v: Tensor,
    k_cache: &mut Option<Tensor>,
    v_cache: &mut Option<Tensor>,
) -> CandleResult<(Tensor, Tensor)> {
    println!("  STEP 5: KV Cache Update");
    
    let (k_full, v_full) = match (k_cache.take(), v_cache.take()) {
        (Some(k_prev), Some(v_prev)) => {
            println!("    Appending to existing cache");
            println!("    Previous cache seq_len: {}", k_prev.dim(2)?);
            let k_new = Tensor::cat(&[&k_prev, &k], 2)?;
            let v_new = Tensor::cat(&[&v_prev, &v], 2)?;
            println!("    New cache seq_len: {}", k_new.dim(2)?);
            (k_new, v_new)
        }
        _ => {
            println!("    Initializing new cache");
            (k, v)
        }
    };
    
    // Store updated cache
    *k_cache = Some(k_full.clone());
    *v_cache = Some(v_full.clone());
    
    Ok((k_full, v_full))
}

/// STEP 6: Scaled Dot-Product Attention (SDPA)
/// 
/// The core attention mechanism.
/// 
/// Formula: Attention(Q, K, V) = softmax(Q @ K.T / sqrt(head_dim)) @ V
/// 
/// With causal mask: only attend to previous positions (for autoregressive LM).
/// 
/// On Metal, this uses optimized kernels via candle_nn::ops::sdpa
pub fn step6_scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal_mask: Option<&Tensor>,
    n_kv_groups: usize,
    softmax_scale: f32,
) -> CandleResult<Tensor> {
    println!("  STEP 6: Scaled Dot-Product Attention");
    println!("    Q shape: {:?}", q.shape());
    println!("    K shape: {:?}", k.shape());
    println!("    V shape: {:?}", v.shape());
    println!("    Softmax scale: {} (= 1/sqrt({}))", softmax_scale, (1.0/softmax_scale).powi(2) as usize);
    
    // If using GQA, repeat K,V heads to match Q
    let (k, v) = if n_kv_groups > 1 {
        println!("    GQA: Repeating K,V {} times", n_kv_groups);
        let k = repeat_kv(k, n_kv_groups)?;
        let v = repeat_kv(v, n_kv_groups)?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };
    
    // On Metal, use optimized SDPA kernel
    // This computes: softmax(Q @ K.T * scale + mask) @ V
    // Args: q, k, v, mask, do_causal, scale, softcapping
    let attn_output = if q.device().is_metal() {
        println!("    Using Metal optimized SDPA kernel");
        let do_causal = causal_mask.is_some();
        let softcapping = 1.0_f32; // No softcapping (1.0 = disabled)
        candle_nn::ops::sdpa(q, &k, &v, causal_mask, do_causal, softmax_scale, softcapping)?
    } else {
        // Fallback: naive implementation
        println!("    Using naive SDPA implementation");
        naive_sdpa(q, &k, &v, causal_mask, softmax_scale)?
    };
    
    println!("    Attention output shape: {:?}", attn_output.shape());
    
    Ok(attn_output)
}

/// STEP 7: Output Projection
/// 
/// Projects attention output back to hidden_size dimensions.
/// 
/// attention_output [batch, n_heads, seq, head_dim] 
///   -> reshape [batch, seq, hidden_size]
///   -> matmul with Wo [batch, seq, hidden_size]
pub fn step7_output_projection(
    attn_output: &Tensor,
    wo: &QTensor,
    device: &Device,
) -> CandleResult<Tensor> {
    println!("  STEP 7: Output Projection");
    
    let (b_sz, n_heads, seq_len, head_dim) = attn_output.dims4()?;
    
    // Reshape: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
    let attn_output = attn_output
        .transpose(1, 2)?
        .reshape((b_sz, seq_len, n_heads * head_dim))?;
    
    println!("    Reshaped to: {:?}", attn_output.shape());
    
    // Project back to hidden_size with quantized matmul
    let output = quantized_matmul(&attn_output, wo, device)?;
    
    println!("    After Wo projection: {:?}", output.shape());
    
    Ok(output)
}

/// STEP 8: Residual Connection (Post-Attention)
/// 
/// Adds the attention output to the original input (skip connection).
/// 
/// output = input + attention(norm(input))
/// 
/// This helps with gradient flow during training and preserves information.
pub fn step8_residual_add(
    residual: &Tensor,
    attention_output: &Tensor,
) -> CandleResult<Tensor> {
    println!("  STEP 8: Residual Connection (Post-Attention)");
    let output = (residual + attention_output)?;
    println!("    output = residual + attention_output");
    Ok(output)
}

/// STEP 9: Feed-Forward Network (SwiGLU)
/// 
/// The MLP block with SwiGLU activation.
/// 
/// SwiGLU Formula:
///   gate = W1(x)   (gate projection)
///   up = W3(x)     (up projection)  
///   hidden = SiLU(gate) * up    (SwiGLU activation)
///   output = W2(hidden)         (down projection)
/// 
/// SiLU(x) = x * sigmoid(x) (Swish activation)
pub fn step9_feed_forward_network(
    x: &Tensor,
    w1_gate: &QTensor,
    w2_down: &QTensor,
    w3_up: &QTensor,
    device: &Device,
) -> CandleResult<Tensor> {
    println!("  STEP 9: Feed-Forward Network (SwiGLU)");
    println!("    Input shape: {:?}", x.shape());
    
    // Gate projection: hidden_size -> intermediate_size
    let gate = quantized_matmul(x, w1_gate, device)?;
    println!("    W1 (gate) output: {:?}", gate.shape());
    
    // Up projection: hidden_size -> intermediate_size
    let up = quantized_matmul(x, w3_up, device)?;
    println!("    W3 (up) output: {:?}", up.shape());
    
    // SwiGLU: SiLU(gate) * up
    // SiLU(x) = x * sigmoid(x)
    let hidden = (silu(&gate)? * up)?;
    println!("    After SwiGLU (silu(gate) * up): {:?}", hidden.shape());
    
    // Down projection: intermediate_size -> hidden_size
    let output = quantized_matmul(&hidden, w2_down, device)?;
    println!("    W2 (down) output: {:?}", output.shape());
    
    Ok(output)
}

/// STEP 10: Final Layer Normalization
/// 
/// RMS normalization before the output projection (LM head).
pub fn step10_final_norm(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> CandleResult<Tensor> {
    println!("  STEP 10: Final RMS Normalization");
    let normalized = rms_norm(&x.contiguous()?, weight, eps)?;
    println!("    Output shape: {:?}", normalized.shape());
    Ok(normalized)
}

/// STEP 11: LM Head (Vocabulary Projection)
/// 
/// Projects final hidden states to vocabulary logits.
/// 
/// hidden [batch, seq, hidden_size] -> logits [batch, seq, vocab_size]
/// 
/// Often tied to the embedding weights (output = input_embedding.T).
pub fn step11_lm_head(
    hidden_states: &Tensor,
    lm_head: &QTensor,
    device: &Device,
) -> CandleResult<Tensor> {
    println!("  STEP 11: LM Head (Vocabulary Projection)");
    println!("    Input shape: {:?}", hidden_states.shape());
    
    let logits = quantized_matmul(hidden_states, lm_head, device)?;
    
    println!("    Output logits shape: {:?}", logits.shape());
    Ok(logits)
}

/// STEP 12: Sampling - Select Next Token
/// 
/// Converts logits to probabilities and samples the next token.
/// 
/// Options:
/// - Greedy: argmax(logits)
/// - Temperature: logits / temperature -> softmax -> sample
/// - Top-k: keep only top-k logits -> softmax -> sample
/// - Top-p (nucleus): keep cumulative prob <= p -> sample
pub fn step12_sample_next_token(
    logits: &Tensor,
    temperature: Option<f64>,
    _top_k: Option<usize>,  // TODO: implement top-k sampling
    _top_p: Option<f64>,    // TODO: implement top-p (nucleus) sampling
) -> CandleResult<u32> {
    println!("  STEP 12: Sampling");
    
    // Get logits for the last token position only
    let (_, seq_len, _) = logits.dims3()?;
    let last_logits = logits.i((.., seq_len - 1, ..))?;
    println!("    Last position logits shape: {:?}", last_logits.shape());
    
    // Apply temperature scaling
    let scaled_logits = match temperature {
        Some(t) if t != 1.0 => {
            println!("    Applying temperature: {}", t);
            (last_logits.to_dtype(DType::F32)? / t)?
        }
        _ => last_logits.to_dtype(DType::F32)?
    };
    
    // Convert to probabilities
    let probs = softmax_last_dim(&scaled_logits)?;
    
    // For simplicity, use greedy (argmax) sampling here
    // In production, implement top-k/top-p properly
    let token_id = probs.argmax(D::Minus1)?
        .to_vec1::<u32>()?[0];
    
    println!("    Selected token ID: {}", token_id);
    
    Ok(token_id)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Quantized matrix multiplication (dequantize + matmul)
fn quantized_matmul(
    input: &Tensor,
    weight: &QTensor,
    device: &Device,
) -> CandleResult<Tensor> {
    // For Q8_0: dequantize the weight tensor first
    // In production, use fused kernels that operate on quantized data directly
    let weight_f = weight.dequantize(device)?;
    input.matmul(&weight_f.t()?)
}

/// Repeat KV heads for Grouped Query Attention
fn repeat_kv(x: &Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b_sz, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x_expanded = x
        .unsqueeze(2)?
        .expand((b_sz, n_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))?;
    Ok(x_expanded)
}

/// Apply rotary position embedding
fn apply_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &[usize],
) -> CandleResult<Tensor> {
    // Simplified RoPE application
    // In production, this is more complex with proper position indexing
    let seq_len = x.dim(2)?;
    let head_dim = x.dim(3)?;
    
    // Get cos/sin for the positions we need
    let cos = cos.narrow(0, positions[0], seq_len)?
        .reshape((1, 1, seq_len, head_dim))?;
    let sin = sin.narrow(0, positions[0], seq_len)?
        .reshape((1, 1, seq_len, head_dim))?;
    
    // Split x into two halves and apply rotation
    let x1 = x.narrow(3, 0, head_dim / 2)?;
    let x2 = x.narrow(3, head_dim / 2, head_dim / 2)?;
    
    let cos1 = cos.narrow(3, 0, head_dim / 2)?;
    let cos2 = cos.narrow(3, head_dim / 2, head_dim / 2)?;
    let sin1 = sin.narrow(3, 0, head_dim / 2)?;
    let sin2 = sin.narrow(3, head_dim / 2, head_dim / 2)?;
    
    // Rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos1)? - x2.broadcast_mul(&sin1)?)?;
    let r2 = (x1.broadcast_mul(&sin2)? + x2.broadcast_mul(&cos2)?)?;
    
    Tensor::cat(&[r1, r2], 3)
}

/// Naive SDPA implementation (fallback when Metal kernels not available)
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> CandleResult<Tensor> {
    // QK^T
    let attn_weights = q.matmul(&k.t()?)?;
    
    // Scale
    let attn_weights = (attn_weights * scale as f64)?;
    
    // Apply mask (causal)
    let attn_weights = match mask {
        Some(m) => attn_weights.broadcast_add(m)?,
        None => attn_weights,
    };
    
    // Softmax
    let attn_weights = softmax_last_dim(&attn_weights)?;
    
    // Attention @ V
    attn_weights.matmul(v)
}

/// Create causal attention mask
fn create_causal_mask(seq_len: usize, dtype: DType, device: &Device) -> CandleResult<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if j <= i { 0.0 } else { f32::NEG_INFINITY }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

// =============================================================================
// COMPLETE FORWARD PASS
// =============================================================================

/// Execute one complete forward pass through a single transformer layer
pub fn transformer_layer_forward(
    hidden_states: &Tensor,
    layer_idx: usize,
    // Layer weights (all quantized)
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    wo: &QTensor,
    w1_gate: &QTensor,
    w2_down: &QTensor,
    w3_up: &QTensor,
    attn_norm_weight: &Tensor,
    ffn_norm_weight: &Tensor,
    // Config
    config: &ModelConfig,
    // Positional embeddings
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    positions: &[usize],
    // KV cache
    k_cache: &mut Option<Tensor>,
    v_cache: &mut Option<Tensor>,
    // Other
    device: &Device,
) -> CandleResult<Tensor> {
    println!("\n=== TRANSFORMER LAYER {} ===", layer_idx);
    
    // Save for residual connection
    let residual = hidden_states.clone();
    
    // Pre-attention RMS norm
    let normed = step2_rms_norm(hidden_states, attn_norm_weight, config.rms_norm_eps, "pre-attention")?;
    
    // Q, K, V projections
    let (q, k, v) = step3_qkv_projection(
        &normed, wq, wk, wv,
        config.n_heads, config.n_kv_heads, config.head_dim,
        DType::F16, device
    )?;
    
    // Apply RoPE
    let (q, k) = step4_rotary_embedding(&q, &k, positions, cos_cache, sin_cache)?;
    
    // Update KV cache
    let (k, v) = step5_kv_cache_update(k, v, k_cache, v_cache)?;
    
    // Create causal mask
    let seq_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let mask = if seq_len > 1 {
        Some(create_causal_mask(kv_len, q.dtype(), device)?)
    } else {
        None
    };
    
    // Scaled dot-product attention
    let n_kv_groups = config.n_heads / config.n_kv_heads;
    let softmax_scale = 1.0 / (config.head_dim as f32).sqrt();
    let attn_output = step6_scaled_dot_product_attention(
        &q, &k, &v, mask.as_ref(), n_kv_groups, softmax_scale
    )?;
    
    // Output projection
    let attn_output = step7_output_projection(&attn_output, wo, device)?;
    
    // Residual connection
    let hidden_states = step8_residual_add(&residual, &attn_output)?;
    
    // Save for FFN residual
    let residual = hidden_states.clone();
    
    // Pre-FFN RMS norm
    let normed = step2_rms_norm(&hidden_states, ffn_norm_weight, config.rms_norm_eps, "pre-FFN")?;
    
    // Feed-forward network (SwiGLU)
    let ffn_output = step9_feed_forward_network(&normed, w1_gate, w2_down, w3_up, device)?;
    
    // Residual connection
    let hidden_states = step8_residual_add(&residual, &ffn_output)?;
    
    println!("=== END LAYER {} ===\n", layer_idx);
    
    Ok(hidden_states)
}

// =============================================================================
// MAIN FUNCTION - DEMONSTRATION
// =============================================================================

fn main() -> Result<()> {
    println!("=========================================================");
    println!("  DETAILED TRANSFORMER INFERENCE STEPS FOR APPLE METAL  ");
    println!("=========================================================\n");
    
    println!("This demonstrates the step-by-step inference process:");
    println!();
    println!("FOR EACH TOKEN:");
    println!("  1. Token Embedding: token_id -> hidden_states [batch, seq, hidden_size]");
    println!();
    println!("FOR EACH LAYER (repeated n_layers times):");
    println!("  2. Pre-Attention RMS Norm: normalize hidden_states");
    println!("  3. Q/K/V Projections: quantized matmul with dequantization");
    println!("  4. RoPE: apply rotary position embeddings to Q, K");
    println!("  5. KV Cache: append new K,V to cache");
    println!("  6. SDPA: softmax(Q @ K.T / sqrt(d)) @ V (Metal optimized)");
    println!("  7. Output Projection: attention_out @ Wo");
    println!("  8. Residual: hidden += attention_output");
    println!("  9. Pre-FFN RMS Norm: normalize");
    println!("  9. Feed-Forward (SwiGLU): silu(x @ W1) * (x @ W3) @ W2");
    println!("  8. Residual: hidden += ffn_output");
    println!();
    println!("AFTER ALL LAYERS:");
    println!("  10. Final RMS Norm");
    println!("  11. LM Head: hidden @ output_weight -> logits [vocab_size]");
    println!("  12. Sampling: temperature, top-k, top-p -> next token");
    println!();
    
    println!("QUANTIZATION (Q8_0):");
    println!("  - Weights stored as int8 with block-wise scales");
    println!("  - Block size: 32 elements share one fp16 scale");
    println!("  - Dequantize: w_float = w_int8 * scale");
    println!("  - Memory reduction: ~4x vs fp32, ~2x vs fp16");
    println!();
    
    println!("METAL OPTIMIZATIONS:");
    println!("  - SDPA uses fused Metal kernel (vector or full attention)");
    println!("  - PagedAttention for efficient KV cache memory");
    println!("  - Automatic device mapping across Metal devices");
    println!("  - Autoreleasepool for proper ObjC memory management");
    println!();
    
    println!("This file is kept as a historical note, not as an active binary.");
    println!();
    println!("Example model config (Devstral/Mistral):");
    let example_config = ModelConfig {
        n_layers: 32,
        vocab_size: 32000,
        hidden_size: 4096,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 14336,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 32768,
        rope_dim: 128,
    };
    println!("{:#?}", example_config);
    
    Ok(())
}

// =============================================================================
// PSEUDO-CODE SUMMARY
// =============================================================================
//
// def forward(token_ids, kv_caches):
//     # Step 1: Token embedding
//     hidden = embedding_table[token_ids]  # [batch, seq, hidden_size]
//     
//     for layer_idx in range(n_layers):
//         residual = hidden
//         
//         # Step 2: Pre-attention normalization
//         hidden = rms_norm(hidden, attn_norm_weight)
//         
//         # Step 3: Q, K, V projections (quantized matmul)
//         Q = dequant_matmul(hidden, Wq)  # [batch, seq, n_heads * head_dim]
//         K = dequant_matmul(hidden, Wk)  # [batch, seq, n_kv_heads * head_dim]
//         V = dequant_matmul(hidden, Wv)  # [batch, seq, n_kv_heads * head_dim]
//         
//         # Reshape to [batch, heads, seq, head_dim]
//         Q = Q.reshape(batch, seq, n_heads, head_dim).transpose(1, 2)
//         K = K.reshape(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
//         V = V.reshape(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
//         
//         # Step 4: Apply rotary position embedding
//         Q, K = apply_rope(Q, K, positions, cos_cache, sin_cache)
//         
//         # Step 5: Update KV cache
//         K, V = update_kv_cache(K, V, kv_caches[layer_idx])
//         
//         # Step 6: Scaled dot-product attention
//         # If GQA: repeat K,V to match Q heads
//         K = repeat_kv(K, n_kv_groups)
//         V = repeat_kv(V, n_kv_groups)
//         # Compute attention
//         attn = softmax(Q @ K.T / sqrt(head_dim) + causal_mask) @ V
//         
//         # Step 7: Output projection
//         attn = attn.transpose(1, 2).reshape(batch, seq, hidden_size)
//         hidden = dequant_matmul(attn, Wo)
//         
//         # Step 8: Residual connection
//         hidden = residual + hidden
//         residual = hidden
//         
//         # Pre-FFN normalization
//         hidden = rms_norm(hidden, ffn_norm_weight)
//         
//         # Step 9: Feed-forward network (SwiGLU)
//         gate = dequant_matmul(hidden, W1)  # gate projection
//         up = dequant_matmul(hidden, W3)    # up projection
//         hidden = silu(gate) * up           # SwiGLU activation
//         hidden = dequant_matmul(hidden, W2)  # down projection
//         
//         # Residual connection
//         hidden = residual + hidden
//     
//     # Step 10: Final normalization
//     hidden = rms_norm(hidden, final_norm_weight)
//     
//     # Step 11: Project to vocabulary
//     logits = dequant_matmul(hidden, lm_head_weight)  # [batch, seq, vocab_size]
//     
//     # Step 12: Sample next token
//     next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
//     
//     return next_token
