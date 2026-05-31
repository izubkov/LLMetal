use anyhow::Result;
use gguf_rs::{get_gguf_container, get_gguf_container_array_size};

#[derive(Debug, Clone)]
pub struct GgufModelInfo {
    pub path: String,
    pub family: String,
    pub parameters: String,
    pub file_type: String,
    pub tensor_count: usize,
    pub architecture: ModelArchitecture,
    /// Token strings from `tokenizer.ggml.tokens`, empty if not present.
    pub vocab: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelArchitecture {
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub layer_count: Option<usize>,
    pub head_count: Option<usize>,
    pub kv_head_count: Option<usize>,
    pub head_dim: Option<usize>,
    pub ffn_hidden_size: Option<usize>,
}

impl GgufModelInfo {
    pub fn load(path: &str) -> Result<Self> {
        // Load metadata with truncated arrays (fast path for inspect).
        let mut container = get_gguf_container(path)?;
        let model = container.decode()?;
        let metadata = model.metadata();

        let vocab_size = metadata
            .get("llama.vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let hidden_size = metadata
            .get("llama.embedding_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let layer_count = metadata
            .get("llama.block_count")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let head_count = metadata
            .get("llama.attention.head_count")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let kv_head_count = metadata
            .get("llama.attention.head_count_kv")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let ffn_hidden_size = metadata
            .get("llama.feed_forward_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let family = model.model_family().to_string();
        let parameters = model.model_parameters().to_string();
        let file_type = model.file_type().to_string();
        let tensor_count = model.num_tensor() as usize;

        // Load the full token list in a second pass.
        let vocab = Self::load_vocab(path).unwrap_or_default();
        let vocab_size = vocab_size.or_else(|| {
            if vocab.is_empty() { None } else { Some(vocab.len()) }
        });

        Ok(Self {
            path: path.to_string(),
            family,
            parameters,
            file_type,
            tensor_count,
            architecture: ModelArchitecture {
                vocab_size,
                hidden_size,
                layer_count,
                head_count,
                kv_head_count,
                head_dim: hidden_size.zip(head_count).map(|(d, h)| d / h),
                ffn_hidden_size,
            },
            vocab,
        })
    }

    fn load_vocab(path: &str) -> Result<Vec<String>> {
        let mut container = get_gguf_container_array_size(path, u64::MAX)?;
        let model = container.decode()?;
        let tokens = model
            .metadata()
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(tokens)
    }

    pub fn print_summary(&self) {
        println!("Model: {}", self.path);
        println!("  family:      {}", self.family);
        println!("  parameters:  {}", self.parameters);
        println!("  file type:   {}", self.file_type);
        println!("  tensors:     {}", self.tensor_count);
        println!("  architecture:");
        println!("    layers:    {}", fmt_opt(self.architecture.layer_count));
        println!("    hidden:    {}", fmt_opt(self.architecture.hidden_size));
        println!("    heads:     {}", fmt_opt(self.architecture.head_count));
        println!(
            "    kv heads:  {}",
            fmt_opt(self.architecture.kv_head_count)
        );
        println!("    head dim:  {}", fmt_opt(self.architecture.head_dim));
        println!(
            "    ffn:       {}",
            fmt_opt(self.architecture.ffn_hidden_size)
        );
        println!("    vocab:     {}", fmt_opt(self.architecture.vocab_size));
    }
}

fn fmt_opt(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".to_string(), |v| v.to_string())
}
