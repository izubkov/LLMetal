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
