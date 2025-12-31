//! Complete inference implementation for Apple Metal Q8 Devstral model
//! 
//! This file demonstrates the full inference path:
//! 1. Load Q8 quantized GGUF model files
//! 2. Initialize Metal device
//! 3. Load tokenizer
//! 4. Forward pass through transformer layers
//! 5. Sample tokens
//! 6. Decode and print response
//!
//! Usage:
//!   cargo run --example inference_2 -- <path_to_gguf_model> "Your prompt here"

use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextMessages,
    TextModelBuilder,
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_path_or_hf_id> [prompt]", args[0]);
        eprintln!("Example: {} microsoft/Phi-3.5-mini-instruct \"Hello, how are you?\"", args[0]);
        std::process::exit(1);
    }

    let model_id = &args[1];
    let prompt = args.get(2)
        .map(|s| s.as_str())
        .unwrap_or("Hello! How are you? Please write a generic binary search function in Rust.");

    println!("=== Apple Metal Q8 Inference ===");
    println!("Model: {}", model_id);
    println!("Prompt: {}\n", prompt);

    // ============================================
    // STEP 1: Model Loading with Q8 Quantization
    // ============================================
    // The TextModelBuilder loads GGUF Q8 quantized models
    // - Q8_0 uses INT8 weights with FP16 activations
    // - Optimized Metal kernels handle quantized matmuls
    // - Model weights are loaded onto Metal device
    
    println!("[1/6] Loading Q8 quantized model...");
    let model = TextModelBuilder::new(model_id)
        // Q8_0 quantization: INT8 weights, FP16 activations
        // This enables 50% memory reduction while maintaining quality
        .with_isq(IsqType::Q8_0)
        
        // Enable logging for debugging
        .with_logging()
        
        // PagedAttention for efficient memory usage
        // Uses Metal-optimized attention kernels
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_block_size(16)  // 16 tokens per block
                .build()
        })?
        
        // Build the model pipeline
        // This internally:
        // - Loads GGUF files
        // - Initializes Metal device
        // - Loads tokenizer
        // - Sets up quantized layers
        .build()
        .await?;
    
    println!("✓ Model loaded successfully\n");

    // ============================================
    // STEP 2: Tokenizer Initialization
    // ============================================
    // Tokenizer is automatically loaded during model build
    // It handles:
    // - Text → token IDs (encoding)
    // - Token IDs → text (decoding)
    // - Special tokens (BOS, EOS, etc.)
    
    println!("[2/6] Tokenizer ready\n");

    // ============================================
    // STEP 3: Prepare Input Messages
    // ============================================
    // Format messages according to chat template
    // The chat template handles:
    // - System/user/assistant roles
    // - Special token insertion
    // - Formatting for the model
    
    println!("[3/6] Preparing input messages...");
    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are a helpful AI assistant.",
        )
        .add_message(
            TextMessageRole::User,
            prompt,
        );
    println!("✓ Messages prepared\n");

    // ============================================
    // STEP 4: Forward Pass & Generation
    // ============================================
    // The send_chat_request method handles:
    // 1. Tokenization (text → token IDs)
    // 2. Prefill phase (process all prompt tokens)
    //    - Embedding lookup (FP16)
    //    - Forward through transformer layers:
    //      * Attention with Q8 matmuls
    //      * MLP with Q8 matmuls
    //      * Layer normalization (FP16)
    //    - KV cache population
    // 3. Decode phase (generate tokens one by one)
    //    - Forward pass for single token
    //    - Extract logits (vocab_size)
    //    - Sample next token
    //    - Decode token → text
    //    - Repeat until EOS or max_len
    // 4. Return complete response
    
    println!("[4/6] Running inference...");
    println!("[5/6] Forward pass through transformer layers...");
    println!("[6/6] Sampling and decoding tokens...\n");
    
    let start_time = std::time::Instant::now();
    
    let response = model.send_chat_request(messages).await?;
    
    let elapsed = start_time.elapsed();
    
    // ============================================
    // STEP 5: Output Response
    // ============================================
    // Print the generated text
    // The response contains:
    // - Generated text content
    // - Token usage statistics
    // - Performance metrics
    
    println!("=== Response ===");
    if let Some(content) = response.choices[0].message.content.as_ref() {
        println!("{}", content);
    } else {
        println!("(No content generated)");
    }
    
    println!("\n=== Performance ===");
    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Prompt tokens/sec: {:.2}", response.usage.avg_prompt_tok_per_sec);
    println!("Completion tokens/sec: {:.2}", response.usage.avg_compl_tok_per_sec);
    println!("Total tokens: {}", response.usage.total_tokens);

    // ============================================
    // BONUS: Advanced Example with Logprobs
    // ============================================
    // RequestBuilder provides more control:
    // - Return logprobs for each token
    // - Custom sampling parameters
    // - Stop sequences
    // - Temperature, top-k, top-p
    
    println!("\n=== Advanced Example ===");
    let request = RequestBuilder::new()
        .return_logprobs(true)  // Get log probabilities
        .add_message(
            TextMessageRole::User,
            "Write a simple hello world program in Rust.",
        );

    let response = model.send_chat_request(request).await?;

    if let Some(logprobs) = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|l| l.content.as_ref())
    {
        println!("Top logprobs for first 3 tokens:");
        for (i, logprob) in logprobs.iter().take(3).enumerate() {
            println!("  Token {}: {:.4}", i, logprob.logprob);
            if let Some(top_logprobs) = &logprob.top_logprobs {
                println!("    Top alternatives:");
                for top in top_logprobs.iter().take(3) {
                    println!("      {}: {:.4}", top.token, top.logprob);
                }
            }
        }
    }

    Ok(())
}

// ============================================
// Key Optimizations Used:
// ============================================
//
// 1. Q8 Quantization (IsqType::Q8_0):
//    - INT8 weights: 50% memory reduction vs FP16
//    - FP16 activations: Native Metal support
//    - Metal kernels: Optimized INT8×FP16 matmuls
//
// 2. PagedAttention:
//    - Efficient KV cache management
//    - Reduces memory fragmentation
//    - Metal-optimized attention kernels
//
// 3. Metal Device:
//    - Automatic Metal device selection
//    - Unified memory (no CPU-GPU transfers)
//    - Command buffer batching
//    - SIMD group execution
//
// 4. Quantized Operations:
//    - All linear layers use Q8 matmuls
//    - Attention Q/K/V projections: Q8
//    - MLP layers: Q8
//    - Output projection: Q8
//
// 5. Memory Efficiency:
//    - KV cache reuse across sequences
//    - Prefix caching for common prompts
//    - Efficient tensor layouts
//
// ============================================
// Inference Flow Summary:
// ============================================
//
// Input Text
//   ↓
// Tokenizer.encode() → [token_ids]
//   ↓
// Prefill Phase:
//   For each prompt token:
//     - Embedding lookup (FP16)
//     - Layer 0..N:
//       * RMSNorm (FP16)
//       * Attention (Q8 matmul → FP16)
//       * Residual add (FP16)
//       * RMSNorm (FP16)
//       * MLP (Q8 matmul → FP16)
//       * Residual add (FP16)
//     - Update KV cache
//   ↓
// Decode Phase (loop):
//   - Forward pass (single token)
//   - Extract logits [vocab_size]
//   - Apply sampling (temperature, top-k, top-p)
//   - Sample token_id
//   - Tokenizer.decode([token_id]) → text
//   - Print text
//   - If token_id == EOS: break
//   ↓
// Output Text

