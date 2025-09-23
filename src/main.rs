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

    // Initialize tokenizer
    let tokenizer = DevstralTokenizer::new(vocab_size);
    
    // Example usage: encode a prompt
    let prompt = "Hello, how are you today?";
    let encoded_tokens = tokenizer.encode(prompt);
    
    println!("Original prompt: {}", prompt);
    println!("Encoded tokens: {:?}", encoded_tokens);
    println!("Decoded text: {}", tokenizer.decode(&encoded_tokens));

    Ok(())
}
// Custom Tokenizer for Q8 Devstral
struct DevstralTokenizer {
    vocab_size: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    unk_token_id: u32,
}

impl DevstralTokenizer {
    fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            bos_token_id: 1,  // Beginning of sequence
            eos_token_id: 2,  // End of sequence
            pad_token_id: 0,  // Padding
            unk_token_id: 3,  // Unknown token
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Add BOS token
        tokens.push(self.bos_token_id);
        
        // Simple word-based tokenization with fallback to character-based
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for word in words {
            // Try to encode as word token (simplified hash-based approach)
            let word_hash = self.word_to_token_id(word);
            tokens.push(word_hash);
        }
        
        // Add EOS token
        tokens.push(self.eos_token_id);
        
        tokens
    }

    fn word_to_token_id(&self, word: &str) -> u32 {
        // Simple hash-based tokenization for demonstration
        // In a real implementation, you'd use the actual vocabulary
        let hash = word.chars().fold(0u32, |acc, c| {
            acc.wrapping_mul(31).wrapping_add(c as u32)
        });
        
        // Map to valid token range (avoiding special tokens)
        let token_id = 4 + (hash % (self.vocab_size as u32 - 4));
        token_id
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        
        for &token in tokens {
            match token {
                t if t == self.bos_token_id => continue, // Skip BOS
                t if t == self.eos_token_id => break,    // Stop at EOS
                t if t == self.pad_token_id => continue, // Skip padding
                t if t == self.unk_token_id => result.push_str("<UNK>"),
                _ => {
                    // Simple reverse mapping (in real implementation, use actual vocab)
                    result.push_str(&format!("word_{}", token));
                    result.push(' ');
                }
            }
        }
        
        result.trim().to_string()
    }
}
