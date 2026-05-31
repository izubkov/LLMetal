use anyhow::Result;
use gguf_rs::get_gguf_container;

#[derive(Debug, Clone)]
pub struct GgufModelInfo {
    pub path: String,
    pub family: String,
    pub parameters: String,
    pub file_type: String,
    pub tensor_count: usize,
    pub architecture: ModelArchitecture,
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
        let mut container = get_gguf_container(path)?;
        let model = container.decode()?;
        let metadata = model.metadata();

        let vocab_size = metadata
            .get("llama.vocab_size")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .or_else(|| {
                metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|value| value.as_array())
                    .map(|values| values.len())
            });
        let hidden_size = metadata
            .get("llama.embedding_length")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let layer_count = metadata
            .get("llama.block_count")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let head_count = metadata
            .get("llama.attention.head_count")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let kv_head_count = metadata
            .get("llama.attention.head_count_kv")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let ffn_hidden_size = metadata
            .get("llama.feed_forward_length")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);

        Ok(Self {
            path: path.to_string(),
            family: model.model_family().to_string(),
            parameters: model.model_parameters().to_string(),
            file_type: model.file_type().to_string(),
            tensor_count: model.num_tensor(),
            architecture: ModelArchitecture {
                vocab_size,
                hidden_size,
                layer_count,
                head_count,
                kv_head_count,
                head_dim: hidden_size.zip(head_count).map(|(dim, heads)| dim / heads),
                ffn_hidden_size,
            },
        })
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
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}
