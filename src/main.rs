use gguf_rs::get_gguf_container;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }

    let mut container = get_gguf_container(&args[1])?;
    let model = container.decode()?;

    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());

    // Load model parameters
    let vocab_size = model
        .metadata()
        .get("llama.vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let dim = model
        .metadata()
        .get("llama.embedding_length")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let n_layers = model
        .metadata()
        .get("llama.block_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let n_heads = model
        .metadata()
        .get("llama.attention.head_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let n_heads_kv = model
        .metadata()
        .get("llama.attention.head_count_kv")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let head_dim = dim / n_heads;

    Ok(())
}