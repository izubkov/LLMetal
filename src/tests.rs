/// Unit tests and GPU micro-benchmarks.
/// Run GPU bench: cargo test bench_gpu -- --ignored --nocapture

#[cfg(test)]
mod tests {
    use crate::tensor::TensorStore;
    use crate::tokenizer::PromptTokenizer;

    // -------------------------------------------------------------------------
    // Dequantization
    // -------------------------------------------------------------------------

    fn make_q8_0_block(scale_f16_bits: u16, quants: [i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale_f16_bits.to_le_bytes());
        for q in quants {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn dequant_q8_0_all_ones() {
        // scale = 1.0 (f16 bits = 0x3C00), quants = [1; 32]
        let block = make_q8_0_block(0x3C00, [1i8; 32]);
        let out = TensorStore::dequant_q8_0_row(&block);
        assert_eq!(out.len(), 32);
        for v in &out {
            let diff = (v - 1.0f32).abs();
            assert!(diff < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn dequant_q8_0_scale_two() {
        // scale = 2.0 (f16 bits = 0x4000), quants alternating [-1, 1]
        let mut quants = [0i8; 32];
        for (i, q) in quants.iter_mut().enumerate() {
            *q = if i % 2 == 0 { -1 } else { 1 };
        }
        let block = make_q8_0_block(0x4000, quants);
        let out = TensorStore::dequant_q8_0_row(&block);
        assert_eq!(out.len(), 32);
        for (i, v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { -2.0f32 } else { 2.0f32 };
            assert!((v - expected).abs() < 1e-3, "idx {i}: expected {expected}, got {v}");
        }
    }

    #[test]
    fn dequant_q8_0_two_blocks() {
        // Two blocks: first all +1 with scale 1.0, second all +2 with scale 0.5
        let mut bytes = make_q8_0_block(0x3C00, [1i8; 32]);  // scale=1.0, q=1 → 1.0
        bytes.extend(make_q8_0_block(0x3800, [2i8; 32]));    // scale=0.5, q=2 → 1.0
        let out = TensorStore::dequant_q8_0_row(&bytes);
        assert_eq!(out.len(), 64);
        for v in &out {
            assert!((v - 1.0f32).abs() < 1e-2, "expected ~1.0, got {v}");
        }
    }

    // -------------------------------------------------------------------------
    // Tokenizer
    // -------------------------------------------------------------------------

    fn tiny_vocab() -> Vec<String> {
        // Simulate a tiny GPT-2/Tekken vocab.
        // Ġ = U+0120 (space prefix).
        vec![
            "<unk>".to_string(),   // 0
            "<s>".to_string(),     // 1 BOS
            "</s>".to_string(),    // 2 EOS
            "h".to_string(),       // 3
            "e".to_string(),       // 4
            "l".to_string(),       // 5
            "o".to_string(),       // 6
            "\u{0120}".to_string(),// 7 — lone space token
            "\u{0120}h".to_string(),        // 8
            "\u{0120}he".to_string(),       // 9
            "\u{0120}hel".to_string(),      // 10
            "\u{0120}hell".to_string(),     // 11
            "\u{0120}hello".to_string(),    // 12
            "hello".to_string(),   // 13
        ]
    }

    #[test]
    fn tokenizer_bos_prepended() {
        let tok = PromptTokenizer::new(tiny_vocab());
        let ids = tok.tokenize_bos("hello");
        assert_eq!(ids[0], 1, "BOS must be first token");
    }

    #[test]
    fn tokenizer_greedy_longest_match() {
        let tok = PromptTokenizer::new(tiny_vocab());
        // " hello" normalises to "Ġhello"; longest match from pos=0 should be id=12 "Ġhello"
        let ids = tok.tokenize("hello");
        assert_eq!(ids, vec![12], "expected Ġhello (id=12)");
    }

    #[test]
    fn tokenizer_no_unknown_for_known_chars() {
        let tok = PromptTokenizer::new(tiny_vocab());
        let ids = tok.tokenize("hello");
        assert!(!ids.contains(&u32::MAX), "no unknowns expected for vocab-covered input");
    }

    // -------------------------------------------------------------------------
    // CPU math (rms_norm, RoPE correctness smoke test)
    // -------------------------------------------------------------------------

    fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        x.iter().zip(w.iter()).map(|(xi, wi)| xi * inv * wi).collect()
    }

    #[test]
    fn rms_norm_unit_weights_normalises() {
        let x = vec![3.0f32, 4.0];  // RMS = sqrt((9+16)/2) = sqrt(12.5)
        let w = vec![1.0f32, 1.0];
        let out = rms_norm(&x, &w, 1e-8);
        // After norm, RMS of out should be ~1.0
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-5, "RMS should be ~1.0 after normalisation, got {rms}");
    }

    #[test]
    fn rms_norm_scales_with_weight() {
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let w = vec![2.0f32, 2.0, 2.0, 2.0];
        let out = rms_norm(&x, &w, 1e-8);
        // x is constant so norm(x) = x/1.0 = [1,1,1,1]; multiplied by w=2 → [2,2,2,2]
        for v in &out {
            assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
        }
    }

    // -------------------------------------------------------------------------
    // GPU micro-benchmark — ignored by default, run with:
    //   cargo test bench_gpu -- --ignored --nocapture
    // -------------------------------------------------------------------------
    #[test]
    #[ignore]
    fn bench_gpu_matvec() {
        use crate::gpu::Gpu;
        use std::time::Instant;

        let gpu = Gpu::new().expect("Metal device");

        // Simulate Devstral ffn_gate: 13824 output rows × 5120 input cols, Q8_0
        let rows: usize = 13824;
        let cols: usize = 5120;
        let q8_0_bytes = rows * (cols / 32) * 34; // 34 bytes per Q8_0 block

        // Synthetic Q8_0 weight buffer (scale=1.0 = 0x3C00 in f16, quants=1)
        let mut weight_data = vec![0u8; q8_0_bytes];
        for chunk in weight_data.chunks_exact_mut(34) {
            chunk[0] = 0x00; chunk[1] = 0x3C; // f16 1.0
            for b in &mut chunk[2..] { *b = 1u8; } // quant = 1 → value 1.0
        }
        let w_buf = gpu.buf_from_bytes(&weight_data);

        // Input vector
        let x_data = vec![1.0f32; cols];
        let x_buf  = gpu.buf_from_f32(&x_data);

        // Warmup
        for _ in 0..3 {
            let _ = gpu.q8_0_matvec(&w_buf, 0, &x_buf, rows, cols);
        }

        // Timed runs
        let n_runs = 20;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = gpu.q8_0_matvec(&w_buf, 0, &x_buf, rows, cols);
        }
        let elapsed = t0.elapsed();
        let ms_per = elapsed.as_secs_f64() * 1000.0 / n_runs as f64;
        let gb_per_s = (q8_0_bytes as f64 / 1e9) / (elapsed.as_secs_f64() / n_runs as f64);

        eprintln!("\n=== GPU bench: ffn_gate ({rows}×{cols} Q8_0, {:.1}MB) ===", q8_0_bytes as f64 / 1e6);
        eprintln!("  {ms_per:.1}ms per matmul  |  {gb_per_s:.1} GB/s effective memory bandwidth");
        eprintln!("  M1 peak: 68 GB/s (base) / 200 GB/s (Pro) / 400 GB/s (Max)");

        // Also bench a smaller attention matmul
        let rows_q: usize = 4096;
        let q8_attn = rows_q * (cols / 32) * 34;
        let mut attn_data = vec![0u8; q8_attn];
        for chunk in attn_data.chunks_exact_mut(34) {
            chunk[0] = 0x00; chunk[1] = 0x3C;
            for b in &mut chunk[2..] { *b = 1u8; }
        }
        let w_attn = gpu.buf_from_bytes(&attn_data);
        let t1 = Instant::now();
        for _ in 0..n_runs {
            let _ = gpu.q8_0_matvec(&w_attn, 0, &x_buf, rows_q, cols);
        }
        let ms_attn = t1.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;
        let gb_attn = (q8_attn as f64 / 1e9) / (t1.elapsed().as_secs_f64() / n_runs as f64);
        eprintln!("  attn_q ({rows_q}×{cols}, {:.1}MB): {ms_attn:.1}ms | {gb_attn:.1} GB/s", q8_attn as f64/1e6);
    }
}
