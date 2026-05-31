#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Q8_0 matrix-vector multiply — simdgroup reduction (coalesced)
//   W  : [rows, cols] in Q8_0 layout (34 bytes per block-of-32)
//         block = [f16 scale (2 bytes)] [32 × int8 weights]
//   x  : [cols] float32 input
//   out: [rows] float32 output
//
//   Launch with rows*32 total threads, 256 threads/threadgroup.
//   Each simdgroup of 32 threads computes one output row:
//     lane t handles Q8_0 blocks t, t+32, t+64, ... of that row.
//   Consecutive threads → consecutive 34-byte blocks → coalesced reads.
//   simd_sum() reduces partial sums within the simdgroup.
// ---------------------------------------------------------------------------
kernel void q8_0_matvec(
    device const uint8_t* W [[buffer(0)]],
    device const float*   x [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    constant ulong& W_off   [[buffer(5)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tid / 32;
    if (row >= rows) return;

    const uint blocks_per_row = cols / 32;
    const uint block_stride   = 34;

    float acc = 0.0f;
    ulong base = W_off + (ulong)row * (ulong)blocks_per_row * block_stride;

    // Stride by 32 so consecutive lanes hit consecutive blocks (coalesced).
    for (uint b = lane; b < blocks_per_row; b += 32) {
        ulong bo = base + (ulong)b * block_stride;
        uint16_t scale_bits = (uint16_t)W[bo] | ((uint16_t)W[bo + 1] << 8);
        float scale = (float)as_type<half>(scale_bits);
        uint xi = b * 32;
        for (uint k = 0; k < 32; k++) {
            acc += scale * (float)(int8_t)W[bo + 2 + k] * x[xi + k];
        }
    }

    // Reduce partial sums across the 32 lanes; first lane writes result.
    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
}

// ---------------------------------------------------------------------------
// Element-wise add (residual stream)
// ---------------------------------------------------------------------------
kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*     out [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// In-place residual add: a[i] += b[i]
// ---------------------------------------------------------------------------
kernel void vec_add_inplace(
    device float*       a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    a[i] += b[i];
}

// ---------------------------------------------------------------------------
// SiLU(gate) ⊙ up  →  FFN hidden
// ---------------------------------------------------------------------------
kernel void silu_hadamard(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}
