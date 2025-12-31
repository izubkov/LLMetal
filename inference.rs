//! Single-file inference for Apple Metal Q8 Devstral model
//! 
//! This file demonstrates the complete inference path from model loading
//! to token generation using mistral_rs with Metal backend and Q8 quantization.
//!
//! Usage:
//!   cargo run --bin inference --features metal -- <model_id_or_path> <gguf_file> "Your prompt here"
//!
//! Example:
//!   cargo run --bin inference --features metal -- bartowski/devstral-7b-v0.1 devstral-7b-v0.1-Q8_0.gguf "Hello, how are you?"

use anyhow::Result;
use mistralrs::{
    GgufModelBuilder, TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <model_id_or_path> <gguf_file> <prompt>", args[0]);
        eprintln!("Example: {} bartowski/devstral-7b-v0.1 devstral-7b-v0.1-Q8_0.gguf \"Hello, how are you?\"", args[0]);
        eprintln!("\nNote: The model_id_or_path is the HuggingFace model ID or local directory");
        eprintln!("      The gguf_file is the specific GGUF file to load");
        std::process::exit(1);
    }

    let model_id = &args[1];
    let gguf_file = &args[2];
    let prompt = &args[3];

    println!("=== Apple Metal Q8 Devstral Inference ===");
    println!("Model ID/Path: {}", model_id);
    println!("GGUF File: {}", gguf_file);
    println!("Prompt: {}\n", prompt);

    // Step 1: Build GGUF model loader
    // This automatically detects and uses Metal device if available (via best_device())
    // The device selection happens in mistralrs/src/model.rs::best_device()
    println!("[1/4] Initializing Metal device...");
    let model = GgufModelBuilder::new(
        model_id,                      // Model ID or path (HuggingFace or local)
        vec![gguf_file.to_string()],   // GGUF file paths
    )
    .with_logging()                    // Enable logging for debugging
    .build()
    .await?;

    println!("[2/4] Model loaded successfully!");
    println!("      - Q8 quantized weights loaded into Metal device");
    println!("      - Tokenizer initialized");
    println!("      - Metal kernels ready\n");

    // Step 2: Prepare messages
    // For Devstral/LLaMA models, we can use simple user messages
    println!("[3/4] Tokenizing prompt...");
    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::User,
            prompt,
        );

    // Step 3: Run inference
    // This triggers:
    //   - Token encoding
    //   - Prefill phase (forward pass on prompt tokens)
    //   - Autoregressive generation (forward pass + sampling loop)
    println!("[4/4] Running inference (forward pass + generation)...\n");
    let response = model.send_chat_request(messages).await?;

    // Step 4: Print response
    println!("=== Response ===");
    if let Some(content) = response.choices[0].message.content.as_ref() {
        println!("{}", content);
    } else {
        println!("No content in response");
    }

    // Print performance metrics
    println!("\n=== Performance Metrics ===");
    println!("Prompt tokens/sec: {:.2}", response.usage.avg_prompt_tok_per_sec);
    println!("Completion tokens/sec: {:.2}", response.usage.avg_compl_tok_per_sec);
    println!("Prompt tokens: {}", response.usage.prompt_tokens);
    println!("Completion tokens: {}", response.usage.completion_tokens);
    println!("Total tokens: {}", response.usage.total_tokens);

    Ok(())
}

