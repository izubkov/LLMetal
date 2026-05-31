
Some old attempt to vibe-code basic inference.

```Rust
use gguf_rs::get_gguf_container;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::f32::consts::SQRT_2;
use std::io::{self, Write};

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
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);

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

    let head_dim = dim / n_heads;

    // Load vocabulary (simplified - using token IDs directly)
    let vocab: Vec<String> = (0..vocab_size).map(|i| i.to_string()).collect();

    // Load weights
    let token_embedding = load_tensor(&model, "token_embd.weight")?;
    let output_weights = load_tensor(&model, "output.weight")?;

    // Load transformer weights for each layer
    let mut transformer_weights = Vec::new();
    for layer in 0..n_layers {
        transformer_weights.push(LayerWeights {
            attn_q: load_tensor(&model, &format!("blk.{}.attn_q.weight", layer))?,
            attn_k: load_tensor(&model, &format!("blk.{}.attn_k.weight", layer))?,
            attn_v: load_tensor(&model, &format!("blk.{}.attn_v.weight", layer))?,
            attn_output: load_tensor(&model, &format!("blk.{}.attn_output.weight", layer))?,
            ffn_gate: load_tensor(&model, &format!("blk.{}.ffn_gate.weight", layer))?,
            ffn_up: load_tensor(&model, &format!("blk.{}.ffn_up.weight", layer))?,
            ffn_down: load_tensor(&model, &format!("blk.{}.ffn_down.weight", layer))?,
            attn_norm: load_tensor(&model, &format!("blk.{}.attn_norm.weight", layer))?
                .row(0)
                .to_owned(),
            ffn_norm: load_tensor(&model, &format!("blk.{}.ffn_norm.weight", layer))?
                .row(0)
                .to_owned(),
        });
    }

    let output_norm = load_tensor(&model, "output_norm.weight")?.row(0).to_owned();

    // Interactive chat loop
    println!("\nEnter your prompt (type 'quit' to exit):");
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") {
            break;
        }

        // Simple tokenization (split by whitespace and map to IDs)
        let tokens: Vec<usize> = input
            .split_whitespace()
            .filter_map(|word| word.parse::<usize>().ok())
            .filter(|&id| id < vocab_size)
            .collect();

        if tokens.is_empty() {
            println!("Please enter valid token IDs (numbers)");
            continue;
        }

        println!("Generating response...");

        // Generate response
        let response = generate_response(
            &tokens,
            &token_embedding,
            &transformer_weights,
            &output_norm,
            &output_weights,
            dim,
            n_heads,
            vocab_size,
            20,
        )?;

        println!("Response: {}", response);
        println!();
    }

    Ok(())
}

struct LayerWeights {
    attn_q: Array2<f32>,
    attn_k: Array2<f32>,
    attn_v: Array2<f32>,
    attn_output: Array2<f32>,
    ffn_gate: Array2<f32>,
    ffn_up: Array2<f32>,
    ffn_down: Array2<f32>,
    attn_norm: Array1<f32>,
    ffn_norm: Array1<f32>,
}

fn load_tensor(
    model: &gguf_rs::GGUFModel,
    name: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let tensor = model
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or(format!("Tensor {} not found", name))?;

    let shape = tensor.shape.clone();

    let data: Vec<f32> = tensor.data().iter().map(|&x| x as f32).collect();

    if shape.len() == 1 {
        Ok(Array2::from_shape_vec((1, shape[0]), data)?)
    } else {
        Ok(Array2::from_shape_vec((shape[0], shape[1]), data)?)
    }
}

fn rms_norm(x: ArrayView1<f32>, weight: ArrayView1<f32>) -> Array1<f32> {
    let ss = x.mapv(|v| v * v).sum() / x.len() as f32 + 1e-5;
    let scale = 1.0 / ss.sqrt();
    &x * scale * &weight
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax(x: ArrayView1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Array1<f32> = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

fn generate_response(
    prompt_tokens: &[usize],
    token_embedding: &Array2<f32>,
    transformer_weights: &[LayerWeights],
    output_norm: &Array1<f32>,
    output_weights: &Array2<f32>,
    dim: usize,
    n_heads: usize,
    vocab_size: usize,
    max_length: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut generated_tokens = prompt_tokens.to_vec();
    let mut hidden_state = Array1::zeros(dim);

    // Process prompt tokens
    for &token in prompt_tokens {
        let embedding = token_embedding.row(token);
        hidden_state = embedding.to_owned();

        // Forward through all layers
        for weights in transformer_weights {
            let residual = hidden_state.clone();

            // Attention
            let normed = rms_norm(hidden_state.view(), weights.attn_norm.view());
            let q = normed.dot(&weights.attn_q);
            let k = normed.dot(&weights.attn_k);
            let v = normed.dot(&weights.attn_v);

            // Simple attention (no causal mask for simplicity)
            let attention = q.dot(&k.t()) / (dim as f32).sqrt();
            let attention_weights = softmax(attention.view());
            let attended = attention_weights.dot(&v);
            let attn_output = attended.dot(&weights.attn_output);
            hidden_state = &residual + &attn_output;

            // FFN
            let residual = hidden_state.clone();
            let normed = rms_norm(hidden_state.view(), weights.ffn_norm.view());
            let gate = normed.dot(&weights.ffn_gate).mapv(silu);
            let up = normed.dot(&weights.ffn_up);
            let ffn_output = (&gate * &up).dot(&weights.ffn_down);
            hidden_state = &residual + &ffn_output;
        }
    }

    // Generate new tokens
    for _ in 0..max_length {
        // Final normalization and output projection
        let normed = rms_norm(hidden_state.view(), output_norm.view());
        let logits = normed.dot(&output_weights.t());

        // Sample next token (greedy sampling)
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        generated_tokens.push(next_token);

        // Stop if we hit a special token or max length
        if next_token >= vocab_size - 10 {
            // Simple stopping condition
            break;
        }

        // Update hidden state with new token
        let embedding = token_embedding.row(next_token);
        hidden_state = embedding.to_owned();

        // Forward through transformer layers
        for weights in transformer_weights {
            let residual = hidden_state.clone();

            let normed = rms_norm(hidden_state.view(), weights.attn_norm.view());
            let q = normed.dot(&weights.attn_q);
            let k = normed.dot(&weights.attn_k);
            let v = normed.dot(&weights.attn_v);

            let attention = q.dot(&k.t()) / (dim as f32).sqrt();
            let attention_weights = softmax(attention.view());
            let attended = attention_weights.dot(&v);
            let attn_output = attended.dot(&weights.attn_output);
            hidden_state = &residual + &attn_output;

            let residual = hidden_state.clone();
            let normed = rms_norm(hidden_state.view(), weights.ffn_norm.view());
            let gate = normed.dot(&weights.ffn_gate).mapv(silu);
            let up = normed.dot(&weights.ffn_up);
            let ffn_output = (&gate * &up).dot(&weights.ffn_down);
            hidden_state = &residual + &ffn_output;
        }
    }

    // Convert token IDs back to "text" (just numbers for simplicity)
    Ok(generated_tokens[prompt_tokens.len()..]
        .iter()
        .map(|&t| t.to_string())
        .collect::<Vec<String>>()
        .join(" "))
}
```