// =============================================================================
// PSEUDO-CODE SUMMARY (CUDA GPU)
// =============================================================================
//
// def forward(token_id, kv_caches, pos):
//     # -------------------------------------------------------------------------
//     # CUDA Graph: Record once, replay with updated parameters
//     # -------------------------------------------------------------------------
//     graph.wrap(lambda: build_forward_graph(token_id, pos))
//     graph.launch(stream)
//
// def build_forward_graph(token_id, pos):
//     # Step 1: Token embedding
//     # KERNEL: copy_embedding<<<grid, block>>>
//     hidden = half_to_float(embedding_table[token_id])  # float[dim]
//     
//     # KV cache ring buffer (StreamingLLM)
//     kv_sink = KV_SINKS if pos >= max_seq_len else 0
//     kv_pos = kv_sink + (pos - kv_sink) % (max_seq_len - kv_sink)
//     kv_len = min(pos + 1, max_seq_len)
//     
//     for layer_idx in range(n_layers):
//         residual = hidden
//         
//         # Step 2: Pre-attention normalization
//         # KERNEL: rmsnorm<<<1, 1024>>> (single block, warp reductions)
//         hidden = rms_norm(hidden, attn_norm_weight)
//         
//         # Step 3: Q, K, V projections
//         # KERNEL: fused_qkv_matmul_clip<<<q_dim+2*kv_dim, 32>>>
//         # FUSED: All three matmuls + clipping in one kernel
//         Q = matmul(hidden, Wq)  # float[n_heads * head_dim]
//         K = matmul(hidden, Wk)  # float[n_kv_heads * head_dim]
//         V = matmul(hidden, Wv)  # float[n_kv_heads * head_dim]
//         Q, K, V = clip(Q, K, V, qkv_clip)
//         
//         # Step 4 & 5: RoPE + KV cache update
//         # KERNEL: fused_rope_and_cache_update<<<blocks, 1024>>>
//         # FUSED: RoPE rotation + fp32→fp16 conversion + cache write
//         Q = apply_rope(Q, pos)
//         K = apply_rope(K, pos)
//         kb[kv_pos] = float_to_half(K)  # half[max_seq_len, n_kv_heads, head_dim]
//         vb[kv_pos] = float_to_half(V)  # half[max_seq_len, n_kv_heads, head_dim]
//         
//         # Step 5.1: Rotate sink tokens (when pos >= max_seq_len)
//         # KERNEL: rotate_sink_tokens<<<blocks, 1024>>>
//         if kv_sink > 0:
//             for r in range(kv_sink):
//                 kb[r] = apply_rope(kb[r], 1)
//         
//         # Step 6: Scaled dot-product attention (3 kernels)
//         # KERNEL: attn_dot<<<(kv_len/32, n_heads/group), (32, group)>>>
//         att = (Q @ kb[:kv_len].T) / sqrt(head_dim)  # float[n_heads, kv_len]
//         
//         # KERNEL: attn_softmax<<<n_heads, 32>>> (block_all_reduce_max/sum)
//         att = softmax(att)
//         
//         # KERNEL: att_mix<<<n_heads, (32, seq_stride)>>> (prefetch 16 values)
//         attn_out = att @ vb[:kv_len]  # float[n_heads, head_dim]
//         
//         # Step 7: Output projection + residual
//         # KERNEL: fused_matmul_add_residuals<<<dim/32, 1024>>>
//         # FUSED: matmul + residual add in one kernel
//         hidden = residual + matmul(attn_out, Wo)
//         
//         residual = hidden
//         
//         # Step 8: Pre-FFN normalization
//         # KERNEL: rmsnorm<<<1, 1024>>>
//         hidden = rms_norm(hidden, ffn_norm_weight)
//         
//         # Step 9: Feed-forward network (SwiGLU)
//         # KERNEL: fused_ffn_w1_w3_glu_act<<<hidden_dim, 32>>>
//         # FUSED: Both matmuls + activation + multiply in one kernel
//         gate = matmul(hidden, W1)
//         up = matmul(hidden, W3)
//         hb = silu(gate) * up  # float[hidden_dim]
//         
//         # KERNEL: fused_matmul_add_residuals<<<dim/32, 1024>>>
//         # FUSED: down projection + residual add
//         hidden = residual + matmul(hb, W2)
//     
//     # Step 10: Final normalization
//     # KERNEL: rmsnorm<<<1, 1024>>>
//     hidden = rms_norm(hidden, final_norm_weight)
//     
//     # Step 11: Project to vocabulary
//     # KERNEL: matmul_wide<<<vocab_size/32, 1024>>> (block transpose for coalesced writes)
//     logits = matmul(hidden, lm_head_weight)  # float[vocab_size] (pinned host)
//     
//     # Step 12: Sample next token (CPU)
//     cudaStreamSynchronize(stream)
//     next_token = sample(logits, temperature, top_k, top_p)
//     
//     return next_token
//
// =============================================================================
// KEY CUDA OPTIMIZATIONS vs METAL/CPU
// =============================================================================
//
// 1. KERNEL FUSION:
//    - fused_qkv_matmul_clip: 3 matmuls + clip → 1 kernel
//    - fused_rope_and_cache_update: RoPE + cache write → 1 kernel  
//    - fused_ffn_w1_w3_glu_act: 2 matmuls + activation + multiply → 1 kernel
//    - fused_matmul_add_residuals: matmul + residual → 1 kernel
//
// 2. CUDA GRAPHS:
//    - Record kernel sequence once, replay with parameter updates
//    - Dynamic params: pos, kv_pos, kv_len, token_id
//    - Reduces CPU dispatch overhead
//
// 3. WARP-LEVEL PRIMITIVES:
//    - __shfl_down_sync / __shfl_xor_sync for reductions
//    - Block transpose for coalesced 128-byte writes
//    - Shared memory for cross-warp communication
//
// 4. MEMORY PRECISION:
//    - Weights: fp16 (dequant on-the-fly)
//    - KV cache: fp16 (half memory footprint)
//    - Activations: fp32 (numerical stability)
//    - Logits: fp32 pinned host (zero-copy DMA)
//
// 5. STREAMING ATTENTION (StreamingLLM):
//    - Ring buffer KV cache with sink tokens
//    - Rotate sink token RoPE each step

