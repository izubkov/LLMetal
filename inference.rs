//! Single-file inference for Apple Metal Q8 Devstral model
//! 
//! This file demonstrates the complete inference path from model loading
//! to token generation using mistral_rs with Metal backend and Q8 quantization.
//!
//! Usage:
//!   cargo run --bin inference --features metal -- <gguf_file_path> "Your prompt here"
//!
//! Example:
//!   cargo run --bin inference --features metal -- ../LLMs/devstralQ8_0.gguf "Hello, how are you?"
//!
//! Note: For local files, provide the full path (absolute or relative) to the GGUF file.
//!       The tokenizer will be automatically extracted from the GGUF file itself.

use anyhow::Result;
use mistralrs::{
    GgufModelBuilder, TextMessageRole, TextMessages,
};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <gguf_file_path> <prompt>", args[0]);
        eprintln!("Example: {} ../LLMs/devstralQ8_0.gguf \"Hello, how are you?\"", args[0]);
        eprintln!("\nNote: For local files, provide the full path to the GGUF file.");
        eprintln!("      The tokenizer will be extracted from the GGUF file itself.");
        std::process::exit(1);
    }

    let gguf_file_path = &args[1];
    let prompt = &args[2];

    println!("=== Apple Metal Q8 Devstral Inference ===");
    println!("GGUF File: {}", gguf_file_path);
    println!("Prompt: {}\n", prompt);

    // Convert to absolute path and extract directory and filename
    let gguf_path = Path::new(gguf_file_path);
    let gguf_absolute = if gguf_path.is_absolute() {
        gguf_path.to_path_buf()
    } else {
        std::env::current_dir()?.join(gguf_path)
    };

    if !gguf_absolute.exists() {
        eprintln!("Error: GGUF file not found: {}", gguf_absolute.display());
        std::process::exit(1);
    }

    let model_dir = gguf_absolute.parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid GGUF file path"))?
        .to_path_buf();
    let gguf_filename = gguf_absolute.file_name()
        .ok_or_else(|| anyhow::anyhow!("Invalid GGUF file path"))?
        .to_string_lossy()
        .to_string();

    // For local files, we need to pass the directory path (with trailing slash)
    // and just the filename (not full path)
    let model_dir_str = model_dir.to_string_lossy().to_string();
    let model_dir_with_slash = if model_dir_str.ends_with('/') {
        model_dir_str
    } else {
        format!("{}/", model_dir_str)
    };

    println!("[1/4] Initializing Metal device...");
    println!("      Model directory: {}", model_dir.display());
    println!("      GGUF filename: {}", gguf_filename);

    // Step 1: Build GGUF model loader
    // For local files, pass the directory as model_id and just the filename
    // This matches the pattern from examples/gguf_locally/main.rs
    let model = GgufModelBuilder::new(
        &model_dir_with_slash,         // Local directory path (must end with /)
        vec![gguf_filename],           // Just the filename, not full path
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

