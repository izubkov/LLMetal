//! Standalone GGUF Q8 Inference for Apple Metal
//!
//! This file demonstrates the complete inference path for running a GGUF quantized
//! model (like Devstral Q8) on Apple Metal using mistral.rs.
//!
//! Build with: cargo build --release --features metal
//!
//! Dependencies (add to Cargo.toml):
//! ```toml
//! [dependencies]
//! mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", features = ["metal"] }
//! tokio = { version = "1", features = ["full", "rt-multi-thread"] }
//! anyhow = "1.0"
//!
//! [features]
//! metal = ["mistralrs/metal"]
//! ```

use anyhow::Result;
use mistralrs::{
    // Core types
    GgufModelBuilder,
    TextMessageRole,
    TextMessages,
    RequestBuilder,
    // Streaming response types
    Response,
    ChatCompletionChunkResponse,
    ChunkChoice,
    Delta,
    // Sampling
    SamplingParams,
    // PagedAttention (optional but recommended for Metal)
    PagedAttentionMetaBuilder,
    MemoryGpuConfig,
    // Device mapping
    DeviceMapSetting,
    AutoDeviceMapParams,
};
use std::io::{self, Write};

/// Configuration for the model and inference
struct InferenceConfig {
    /// HuggingFace model ID or local path containing the GGUF file
    model_id: String,
    /// GGUF filename(s) - can be split across multiple files
    gguf_files: Vec<String>,
    /// Optional: Model ID to source tokenizer from (if not embedded in GGUF)
    tokenizer_model_id: Option<String>,
    /// Optional: Path to chat template file
    chat_template: Option<String>,
    /// Maximum generation length
    max_tokens: usize,
    /// Temperature for sampling (None = greedy)
    temperature: Option<f64>,
    /// Top-p sampling
    top_p: Option<f64>,
    /// Top-k sampling
    top_k: Option<usize>,
    /// Context size for PagedAttention
    context_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            // Example: Devstral Q8 from bartowski
            model_id: "bartowski/devstral-small-2505-GGUF".to_string(),
            gguf_files: vec!["devstral-small-2505-Q8_0.gguf".to_string()],
            tokenizer_model_id: Some("mistralai/devstral-small-2505".to_string()),
            chat_template: None,
            max_tokens: 2048,
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            context_size: 8192,
        }
    }
}

/// Build the model with Metal optimizations
async fn build_model(config: &InferenceConfig) -> Result<mistralrs::Model> {
    // Initialize the GGUF model builder
    let mut builder = GgufModelBuilder::new(&config.model_id, config.gguf_files.clone());

    // Source tokenizer from a separate model if needed
    // This is useful when the GGUF doesn't have an embedded tokenizer
    if let Some(ref tok_model_id) = config.tokenizer_model_id {
        builder = builder.with_tok_model_id(tok_model_id);
    }

    // Set custom chat template if provided
    if let Some(ref template) = config.chat_template {
        builder = builder.with_chat_template(template);
    }

    // Enable logging for debugging
    builder = builder.with_logging();

    // Enable throughput logging to see tokens/sec
    builder = builder.with_throughput_logging();

    // Configure automatic device mapping for optimal memory usage
    // This distributes model layers across available Metal devices
    builder = builder.with_device_mapping(DeviceMapSetting::Auto(
        AutoDeviceMapParams::default_text(),
    ));

    // Enable PagedAttention for better memory efficiency on Metal
    // This is a key optimization for running larger context sizes
    builder = builder.with_paged_attn(|| {
        PagedAttentionMetaBuilder::default()
            // Set GPU memory allocation based on context size
            .with_gpu_memory(MemoryGpuConfig::ContextSize(config.context_size))
            // Block size for paged attention (16 is a good default)
            .with_block_size(16)
            .build()
    })?;

    // Set prefix cache for faster repeated prompts
    // This caches computed KV for common prefixes
    builder = builder.with_prefix_cache_n(Some(16));

    // Set maximum concurrent sequences
    builder = builder.with_max_num_seqs(4);

    // Build and return the model
    let model = builder.build().await?;

    println!("Model loaded successfully on Metal!");
    
    // Print model config
    if let Ok(config) = model.config() {
        println!("Device: {:?}", config.device);
        if let Some(max_seq) = config.max_seq_len {
            println!("Max sequence length: {}", max_seq);
        }
    }

    Ok(model)
}

/// Create sampling parameters
fn create_sampling_params(config: &InferenceConfig) -> SamplingParams {
    SamplingParams {
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: None,
        top_n_logprobs: 0,
        frequency_penalty: None,
        presence_penalty: None,
        repetition_penalty: Some(1.1), // Slight repetition penalty
        stop_toks: None,
        max_len: Some(config.max_tokens),
        logits_bias: None,
        n_choices: 1,
        dry_params: None,
    }
}

/// Run inference with simple text messages (non-streaming)
async fn run_simple_inference(
    model: &mistralrs::Model,
    system_prompt: &str,
    user_message: &str,
) -> Result<String> {
    // Build the messages
    let messages = TextMessages::new()
        .add_message(TextMessageRole::System, system_prompt)
        .add_message(TextMessageRole::User, user_message);

    // Send the request and wait for response
    let response = model.send_chat_request(messages).await?;

    // Extract the response content
    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .map(|s| s.to_string())
        .unwrap_or_default();

    // Print usage statistics
    println!("\n--- Usage Statistics ---");
    println!(
        "Prompt tokens/sec: {:.2}",
        response.usage.avg_prompt_tok_per_sec
    );
    println!(
        "Completion tokens/sec: {:.2}",
        response.usage.avg_compl_tok_per_sec
    );
    println!(
        "Total tokens: {} (prompt) + {} (completion)",
        response.usage.prompt_tokens, response.usage.completion_tokens
    );

    Ok(content)
}

/// Run inference with streaming output
async fn run_streaming_inference(
    model: &mistralrs::Model,
    config: &InferenceConfig,
    system_prompt: &str,
    user_message: &str,
) -> Result<String> {
    // Use RequestBuilder for more control over sampling
    let request = RequestBuilder::new()
        .add_message(TextMessageRole::System, system_prompt)
        .add_message(TextMessageRole::User, user_message)
        .set_sampling(create_sampling_params(config));

    // Get streaming response
    let mut stream = model.stream_chat_request(request).await?;

    let stdout = io::stdout();
    let mut lock = stdout.lock();
    let mut full_response = String::new();

    // Process chunks as they arrive
    while let Some(chunk) = stream.next().await {
        match chunk {
            Response::Chunk(ChatCompletionChunkResponse { choices, .. }) => {
                if let Some(ChunkChoice {
                    delta: Delta {
                        content: Some(content),
                        ..
                    },
                    ..
                }) = choices.first()
                {
                    // Print and accumulate the content
                    write!(lock, "{}", content)?;
                    lock.flush()?;
                    full_response.push_str(content);
                }
            }
            Response::Done(response) => {
                // Final response with usage stats
                println!("\n\n--- Usage Statistics ---");
                println!(
                    "Prompt tokens/sec: {:.2}",
                    response.usage.avg_prompt_tok_per_sec
                );
                println!(
                    "Completion tokens/sec: {:.2}",
                    response.usage.avg_compl_tok_per_sec
                );
            }
            Response::InternalError(e) | Response::ValidationError(e) => {
                eprintln!("\nError: {}", e);
                anyhow::bail!("Inference error: {}", e);
            }
            _ => {}
        }
    }

    println!(); // Final newline
    Ok(full_response)
}

/// Run multi-turn conversation
async fn run_conversation(
    model: &mistralrs::Model,
    config: &InferenceConfig,
) -> Result<()> {
    println!("\n=== Multi-turn Conversation ===\n");

    let system_prompt = "You are a helpful coding assistant specializing in Rust.";

    // Turn 1
    println!("User: What is the ownership system in Rust?");
    println!("Assistant: ");
    let _ = run_streaming_inference(
        model,
        config,
        system_prompt,
        "What is the ownership system in Rust? Explain briefly.",
    )
    .await?;

    println!("\n");

    // Turn 2 (note: this is a new context, not maintaining history)
    // For true multi-turn, you would accumulate messages
    println!("User: Show me an example of borrowing.");
    println!("Assistant: ");
    let _ = run_streaming_inference(
        model,
        config,
        system_prompt,
        "Show me a simple example of borrowing in Rust.",
    )
    .await?;

    Ok(())
}

/// Tokenize text for inspection
async fn tokenize_example(model: &mistralrs::Model) -> Result<()> {
    let text = "Hello, how are you?";
    
    // Tokenize raw text
    let tokens = model
        .tokenize(
            either::Either::Right(text.to_string()),
            None,  // No tools
            true,  // Add special tokens
            false, // Don't add generation prompt
            None,  // Enable thinking
        )
        .await?;

    println!("\n=== Tokenization Example ===");
    println!("Text: \"{}\"", text);
    println!("Tokens: {:?}", tokens);
    println!("Token count: {}", tokens.len());

    // Detokenize back
    let decoded = model.detokenize(tokens.clone(), false).await?;
    println!("Decoded: \"{}\"", decoded);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("===========================================");
    println!("  MistralRS GGUF Inference on Apple Metal  ");
    println!("===========================================\n");

    // Create configuration
    // Modify this for your specific model
    let config = InferenceConfig {
        // For a local GGUF file, use the directory path:
        // model_id: "/path/to/your/gguf/directory".to_string(),
        // gguf_files: vec!["model-Q8_0.gguf".to_string()],
        
        // For HuggingFace hosted GGUF:
        model_id: "bartowski/devstral-small-2505-GGUF".to_string(),
        gguf_files: vec!["devstral-small-2505-Q8_0.gguf".to_string()],
        tokenizer_model_id: Some("mistralai/devstral-small-2505".to_string()),
        chat_template: None,
        max_tokens: 1024,
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        context_size: 8192,
    };

    println!("Loading model: {}", config.model_id);
    println!("GGUF files: {:?}", config.gguf_files);
    println!();

    // Build the model
    let model = build_model(&config).await?;

    // Example 1: Simple non-streaming inference
    println!("\n=== Simple Inference (Non-streaming) ===\n");
    let response = run_simple_inference(
        &model,
        "You are a helpful assistant.",
        "Write a haiku about Rust programming.",
    )
    .await?;
    println!("Response:\n{}", response);

    // Example 2: Streaming inference
    println!("\n=== Streaming Inference ===\n");
    println!("User: Write a function to calculate fibonacci numbers in Rust.");
    println!("Assistant: ");
    let _ = run_streaming_inference(
        &model,
        &config,
        "You are a Rust expert. Provide concise, working code.",
        "Write a function to calculate fibonacci numbers in Rust. Include iterative and recursive versions.",
    )
    .await?;

    // Example 3: Tokenization
    tokenize_example(&model).await?;

    // Example 4: Multi-turn conversation
    run_conversation(&model, &config).await?;

    println!("\n=== Inference Complete ===");
    Ok(())
}

// ============================================================================
// Alternative: Local GGUF file loading
// ============================================================================

/// Load a model from a local GGUF file
#[allow(dead_code)]
async fn load_local_gguf(
    gguf_path: &str,
    chat_template_path: Option<&str>,
) -> Result<mistralrs::Model> {
    // For local files, the model_id is the directory containing the GGUF
    let model_dir = std::path::Path::new(gguf_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string());
    
    let gguf_filename = std::path::Path::new(gguf_path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .expect("Invalid GGUF path");

    let mut builder = GgufModelBuilder::new(&model_dir, vec![gguf_filename]);

    // Set chat template if provided
    if let Some(template) = chat_template_path {
        builder = builder.with_chat_template(template);
    }

    builder = builder
        .with_logging()
        .with_throughput_logging()
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_gpu_memory(MemoryGpuConfig::ContextSize(4096))
                .build()
        })?;

    let model = builder.build().await?;
    Ok(model)
}

// ============================================================================
// Usage Notes for Devstral Q8
// ============================================================================
//
// Devstral is based on the Mistral architecture, which maps to `llama` in GGUF.
// The Q8_0 quantization provides a good balance of quality and speed on Metal.
//
// Recommended settings for Devstral:
// - Context size: 8192 or higher (model supports up to 32k)
// - Temperature: 0.7 for creative tasks, 0.1-0.3 for coding
// - Top-p: 0.9
// - Repetition penalty: 1.1
//
// Memory requirements (approximate):
// - Q8_0: ~8GB for the base model
// - Additional memory for KV cache depends on context size
//
// Performance tips:
// 1. PagedAttention significantly improves throughput
// 2. Prefix caching helps with repeated prompts
// 3. Batch multiple requests when possible (max_num_seqs)
// 4. Use streaming for long responses to reduce perceived latency

