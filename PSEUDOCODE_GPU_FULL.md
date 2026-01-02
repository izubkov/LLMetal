# YALM GPU Inference Pseudocode (CUDA)

## High-Level Overview

The GPU inference uses CUDA Graphs for efficient kernel dispatch. The graph is built once during the first forward pass, then updated and replayed for subsequent passes.

## Data Types & Buffers

```
// GPU Device Buffers (allocated via cudaMalloc)
x:        float[dim]                    // current activations
xb:       float[dim]                    // activation buffer (post-norm)
xb2:      float[dim]                    // secondary activation buffer
hb:       float[hidden_dim]             // FFN hidden state
q:        float[n_heads * head_dim]     // query vectors
k:        float[n_kv_heads * head_dim]  // key vectors
v:        float[n_kv_heads * head_dim]  // value vectors
att:      float[n_heads * max_seq_len]  // attention scores
kb:       half[max_seq_len, n_kv_heads, head_dim]  // key cache (fp16)
vb:       half[max_seq_len, n_kv_heads, head_dim]  // value cache (fp16)

// Host-pinned buffer for results
logits:   float[vocab_size]             // output logits (pinned host memory)
```

## CUDA Graph Execution Model

```python
def forward_cuda(token, pos, mode):
    # Wrap kernel dispatches in CUDA Graph
    graph.wrap(lambda: forward_cuda_build_graph(token, pos, mode))
    
    # Launch the pre-recorded graph
    cudaGraphLaunch(graph.instance, stream)
    
    if mode == OUTPUT_LOGITS:
        cudaStreamSynchronize(stream)  # wait for logits
```

## Forward Pass with GPU Kernels

```python
def forward_cuda_build_graph(token, pos, mode):
    # ==========================================================================
    # Step 1: Token Embedding Lookup
    # ==========================================================================
    # Kernel: copy_embedding_half<<<grid, block, stream>>>
    #   grid  = (dim + max_threads_per_block - 1) / max_threads_per_block
    #   block = max_threads_per_block
    # 
    # Each thread copies one element: x[i] = __half2float(embedding[token * dim + i])
    
    copy_embedding<<<grid, block, stream>>>(
        token_embedding_table,  # half* or float*
        dim,
        token,
        x                       # output: float[dim]
    )
    
    # ==========================================================================
    # KV Cache Ring Buffer Management (StreamingLLM)
    # ==========================================================================
    kv_sink = KV_SINKS if pos >= max_seq_len else 0
    kv_pos = kv_sink + (pos - kv_sink) % (max_seq_len - kv_sink)
    kv_len = max_seq_len if pos >= max_seq_len else pos + 1
    
    # ==========================================================================
    # Step 2-9: Process Each Transformer Layer
    # ==========================================================================
    for layer_idx in range(n_layers):
        block_cuda(layer_idx, pos, kv_sink, kv_pos, kv_len)
    
    if mode == HYDRATE_KV_CACHE:
        return  # skip logits computation during prompt processing
    
    # ==========================================================================
    # Step 10: Final RMS Normalization
    # ==========================================================================
    # Kernel: rmsnorm<<<1, max_threads_per_block, stream>>>
    #   Single block, parallel reduction for sum of squares
    #   Thread cooperation via warp shuffles and shared memory
    
    rmsnorm<<<1, max_threads_per_block, stream>>>(
        x,                    # input
        rms_final_weight,     # float[dim]
        dim,
        norm_eps,
        x                     # output (in-place)
    )
    
    # ==========================================================================
    # Step 11: LM Head - Project to Vocabulary
    # ==========================================================================
    # Kernel: matmul_wide<<<vocab_size/32, warp_size*32, stream>>>
    #   Each warp computes one row of output
    #   Block-level transpose for coalesced memory writes
    
    matmul_wide<<<vocab_size/32, warp_size*32, stream>>>(
        wcls,                 # half* or float* [vocab_size, dim]
        x,                    # float[dim]
        dim,
        vocab_size,
        logits                # float[vocab_size] (pinned host memory)
    )
```

## Transformer Block GPU Implementation

```python
def block_cuda(layer_idx, pos, kv_sink, kv_pos, kv_len):
    # ==========================================================================
    # Step 2: Pre-Attention RMS Normalization
    # ==========================================================================
    # Kernel: rmsnorm<<<1, max_threads_per_block, stream>>>
    #   Parallel reduction: sum(x[i]^2) using warp_all_reduce_sum
    #   Then scale: out[i] = x[i] * rsqrt(rms + eps) * weight[i]
    
    rmsnorm<<<1, max_threads_per_block, stream>>>(
        x,                    # input: float[dim]
        rms_att_weight,       # float[dim]
        dim,
        norm_eps,
        xb                    # output: float[dim]
    )
    
    # ==========================================================================
    # Step 3: Fused Q, K, V Projections with Clipping
    # ==========================================================================
    # Kernel: fused_qkv_matmul_clip<<<total_rows, warp_size, stream>>>
    #   total_rows = q_dim + 2 * kv_dim
    #   Each warp computes one output row (Q, K, or V)
    #   Warp-level reduction for dot product
    #   Applies clipping: clamp(result, -qkv_clip, qkv_clip)
    
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    total_rows = q_dim + 2 * kv_dim
    
    fused_qkv_matmul_clip<<<total_rows, warp_size, stream>>>(
        wq,                   # half* [q_dim, dim]
        wk,                   # half* [kv_dim, dim]
        wv,                   # half* [kv_dim, dim]
        xb,                   # float[dim]
        dim,
        q_dim,
        kv_dim,
        qkv_clip,
        q,                    # output: float[q_dim]
        k,                    # output: float[kv_dim]
        v                     # output: float[kv_dim]
    )
    
    # ==========================================================================
    # Step 4 & 5: Fused RoPE + KV Cache Update
    # ==========================================================================
    # Kernel: fused_rope_and_cache_update<<<num_blocks, max_threads_per_block>>>
    #   Each thread handles 2 consecutive elements (complex rotation pair)
    #   Q: Apply RoPE in-place
    #   K: Apply RoPE and store to kb (fp16)
    #   V: Direct copy to vb (fp16)
    #
    # RoPE formula per pair (i, i+1):
    #   freq = 1.0 / theta^(i % head_dim / rotary_dim)
    #   cos_val = cos(pos * freq)
    #   sin_val = sin(pos * freq)
    #   out[i]   = in[i] * cos_val - in[i+1] * sin_val
    #   out[i+1] = in[i] * sin_val + in[i+1] * cos_val
    
    max_dim = max(n_heads * head_dim, n_kv_heads * head_dim)
    threads_needed = (max_dim + 1) / 2
    num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block
    
    # Uses CUDA Graph node API for dynamic parameter updates
    graph.add_or_update_kernel_node("fused_rope_and_cache_update",
        fused_rope_and_cache_update<<<num_blocks, max_threads_per_block, stream>>>(
            q,                # float[n_heads * head_dim]
            k,                # float[n_kv_heads * head_dim]
            v,                # float[n_kv_heads * head_dim]
            head_dim,
            n_heads,
            n_kv_heads,
            pos,              # dynamic: current position
            kv_pos,           # dynamic: position in ring buffer
            rope_theta,
            rotary_dim,
            q,                # output: Q updated in-place
            kb,               # output: half[max_seq_len, n_kv_heads, head_dim]
            vb                # output: half[max_seq_len, n_kv_heads, head_dim]
        )
    )
    
    # ==========================================================================
    # Step 5.1: Rotate Sink Tokens (StreamingLLM)
    # ==========================================================================
    # Only when pos >= max_seq_len (past context window)
    # Sink tokens must maintain correct relative positions
    
    if kv_sink > 0:
        threads_needed = (kv_dim + 1) / 2
        num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block
        
        graph.add_or_update_kernel_node("rotate_sink_tokens",
            rotate_sink_tokens<<<num_blocks, max_threads_per_block, stream>>>(
                kb,           # half* key cache
                kv_sink,      # number of sink tokens
                kv_dim,       # n_kv_heads * head_dim
                head_dim,
                rope_theta,
                rotary_dim
            )
        )
    
    # ==========================================================================
    # Step 6a: Attention Dot Products
    # ==========================================================================
    # Kernel: attn_dot<<<blocks, threads>>>
    #   threads.x = warp_size (32)
    #   threads.y = n_heads / n_kv_heads (GQA group size)
    #   blocks.x = (kv_len + warp_size - 1) / warp_size
    #   blocks.y = (n_heads + threads.y - 1) / threads.y
    #
    # Each thread computes: score = (Q[h] · K[t]) / sqrt(head_dim)
    
    threads = dim3(warp_size, n_heads / n_kv_heads)
    blocks = dim3(
        (kv_len + warp_size - 1) / warp_size,
        (n_heads + threads.y - 1) / threads.y
    )
    
    graph.add_or_update_kernel_node("attn_dot",
        attn_dot<<<blocks, threads, stream>>>(
            kb,               # half[max_seq_len, n_kv_heads, head_dim]
            q,                # float[n_heads, head_dim]
            head_dim,
            kv_len,           # dynamic: current sequence length
            max_seq_len,
            n_heads,
            n_kv_heads,
            att               # output: float[n_heads, max_seq_len]
        )
    )
    
    # ==========================================================================
    # Step 6b: Attention Softmax
    # ==========================================================================
    # Kernel: attn_softmax<<<n_heads, warp_size, stream>>>
    #   One block per attention head
    #   Two-pass algorithm with block-level reductions:
    #     Pass 1: Find max via block_all_reduce_max
    #     Pass 2: Compute exp(x - max), sum, normalize
    
    graph.add_or_update_kernel_node("attn_softmax",
        attn_softmax<<<n_heads, warp_size, stream>>>(
            att,              # input: float[n_heads, max_seq_len]
            kv_len,           # dynamic: current sequence length
            max_seq_len,
            n_heads,
            att               # output: in-place softmax
        )
    )
    
    # ==========================================================================
    # Step 6c: Attention Value Mixing
    # ==========================================================================
    # Kernel: att_mix<<<n_heads, threads>>>
    #   threads.x = warp_size (32)
    #   threads.y = min(kv_len, max_threads_per_block / warp_size)
    #   blocks.x = n_heads
    #
    # Computes: out[h, i] = sum_t(att[h, t] * V[t, kv_head, i])
    # Uses shared memory for warp-level accumulation
    # Prefetches 16 values at a time for memory latency hiding
    
    threads = dim3(warp_size, min(kv_len, max_threads_per_block / warp_size))
    blocks = dim3(n_heads)
    
    graph.add_or_update_kernel_node("att_mix",
        att_mix<<<blocks, threads, stream>>>(
            vb,               # half[max_seq_len, n_kv_heads, head_dim]
            att,              # float[n_heads, max_seq_len] (softmax weights)
            head_dim,
            n_heads,
            n_kv_heads,
            kv_len,           # dynamic
            max_seq_len,
            xb2               # output: float[n_heads, head_dim]
        )
    )
    
    # ==========================================================================
    # Step 7: Output Projection + Residual
    # ==========================================================================
    # Kernel: fused_matmul_add_residuals<<<dim/32, warp_size*32, stream>>>
    #   Each warp computes one output element
    #   Fuses: x += Wo @ attention_output
    #   Block transpose for coalesced writes
    
    fused_matmul_add_residuals<<<dim/32, warp_size*32, stream>>>(
        wo,                   # half* [dim, n_heads * head_dim]
        xb2,                  # float[n_heads * head_dim]
        q_dim,                # n_heads * head_dim
        dim,
        x                     # output: x += matmul result (residual add)
    )
    
    # ==========================================================================
    # Step 8: Pre-FFN RMS Normalization
    # ==========================================================================
    rmsnorm<<<1, max_threads_per_block, stream>>>(
        x,
        rms_ffn_weight,
        dim,
        norm_eps,
        xb
    )
    
    # ==========================================================================
    # Step 9a: Fused FFN Gate + Up + Activation (SwiGLU/GeGLU)
    # ==========================================================================
    # Kernel: fused_ffn_w1_w3_glu_act<<<hidden_dim, warp_size, stream>>>
    #   Each warp computes one hidden dimension element
    #   Fuses three operations:
    #     gate = W1 @ x
    #     up   = W3 @ x
    #     hb   = activation(gate) * up
    #
    # SiLU activation: x / (1 + exp(-x))
    # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    fused_ffn_w1_w3_glu_act<<<hidden_dim, warp_size, stream>>>(
        w1,                   # half* [hidden_dim, dim]
        w3,                   # half* [hidden_dim, dim]
        xb,                   # float[dim]
        dim,
        hidden_dim,
        hb                    # output: float[hidden_dim]
    )
    
    # ==========================================================================
    # Step 9b: Down Projection + Residual
    # ==========================================================================
    # Kernel: fused_matmul_add_residuals<<<dim/32, warp_size*32, stream>>>
    #   Fuses: x += W2 @ hb
    
    fused_matmul_add_residuals<<<dim/32, warp_size*32, stream>>>(
        w2,                   # half* [dim, hidden_dim]
        hb,                   # float[hidden_dim]
        hidden_dim,
        dim,
        x                     # output: x += matmul result (residual add)
    )
```

## CUDA Kernel Implementation Details

### Warp-Level Primitives

```cuda
// Warp reduction for dot products
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// Block-level reduction via shared memory
__device__ float block_all_reduce_sum(float val) {
    __shared__ float shared[32];  // max 32 warps per block
    int wid = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    
    val = warp_all_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
        val = warp_all_reduce_sum(val);
        if (lane == 0) shared[wid] = val;
    }
    __syncthreads();
    return shared[0];
}
```

### Matrix-Vector Multiplication

```cuda
// Each warp computes one row
__device__ float matmul_row(const half* row, const float* x, int offset, int dim) {
    float sum = 0.0;
    for (int j = offset; j < dim; j += warpSize) {
        sum += __half2float(row[j]) * x[j];
    }
    return warp_reduce_sum(sum);
}
```

### Block Transpose for Coalesced Writes

```cuda
// Transpose warp results for 128-byte coalesced stores
__device__ float blocktranspose(float v, float def) {
    __shared__ float sm[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    
    if (lane == 0) sm[warp] = v;
    __syncthreads();
    
    return lane < blockDim.x / warpSize ? sm[lane] : def;
}
```

## Memory Layout Summary

| Buffer | Shape | Precision | Location | Notes |
|--------|-------|-----------|----------|-------|
| Weights (Wq, Wk, Wv, Wo, W1, W2, W3) | varies | fp16 | GPU | Dequantized on-the-fly |
| Activations (x, xb, xb2, hb, q, k, v) | varies | fp32 | GPU | Working precision |
| KV Cache (kb, vb) | [max_seq_len, n_kv_heads, head_dim] | fp16 | GPU | Reduced memory footprint |
| Attention Scores (att) | [n_heads, max_seq_len] | fp32 | GPU | Full precision for softmax |
| Logits | [vocab_size] | fp32 | Pinned Host | Zero-copy DMA |

## CUDA Graph Optimization

```python
# First forward pass: Record graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
forward_cuda_build_graph(token, pos, mode)
cudaStreamEndCapture(stream, &graph)
cudaGraphInstantiate(&instance, graph)

# Subsequent passes: Update dynamic parameters and replay
for node_name, params in dynamic_nodes:
    cudaGraphExecKernelNodeSetParams(instance, nodes[node_name], &params)
cudaGraphLaunch(instance, stream)
```

Dynamic parameters updated per forward pass:
- `pos` (token position)
- `kv_pos` (ring buffer position)
- `kv_len` (current sequence length)
- `token` (input token ID)

