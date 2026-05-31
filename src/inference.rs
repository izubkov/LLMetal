use crate::gguf_loader::GgufModelInfo;
use crate::gpu::Gpu;
use crate::tokenizer::PromptTokenizer;

pub struct TransparentRunner {
    model: GgufModelInfo,
    gpu: Gpu,
    tokenizer: PromptTokenizer,
}

impl TransparentRunner {
    pub fn new(model: GgufModelInfo, gpu: Gpu) -> Self {
        let tokenizer = PromptTokenizer::new(model.vocab.clone());
        Self { model, gpu, tokenizer }
    }

    pub fn describe_prompt_pass(&self, prompt: &str) {
        let name = self.gpu.device_name();
        println!("LLMetal transparent inference trace");
        println!("  model:  {}", self.model.path);
        println!("  device: {}", name);
        println!("  prompt: {prompt:?}");
        println!("  tokens: {}", self.tokenizer.explain(prompt));
        println!();

        self.describe_prefill();
        self.describe_decode_token();
    }

    fn describe_prefill(&self) {
        let arch = &self.model.architecture;
        let layers = arch.layer_count.unwrap_or(0);
        let heads = arch.head_count.unwrap_or(0);
        let kv_heads = arch.kv_head_count.unwrap_or(heads);
        let hidden = arch.hidden_size.unwrap_or(0);
        let head_dim = arch.head_dim.unwrap_or(0);
        let ffn = arch.ffn_hidden_size.unwrap_or(0);

        println!("Prefill pass");
        println!("  1. Token ids index the embedding table -> hidden [{hidden}]");
        println!("  2. For each of {layers} transformer blocks:");
        println!("     a. RMSNorm before attention");
        println!("     b. Q/K/V projections as explicit quantized matmuls");
        println!("        Q heads: {heads}, KV heads: {kv_heads}, head dim: {head_dim}");
        println!("     c. RoPE rotates Q and K in-place by token position");
        println!("     d. K/V are appended to the layer cache");
        println!("     e. Attention is softmax(QK^T / sqrt(head_dim))V");
        println!("     f. Output projection is added back through the residual path");
        println!("     g. RMSNorm + SwiGLU feed-forward, then another residual add");
        println!("        FFN hidden size: {ffn}");
        println!("  3. Final RMSNorm + LM head produce logits for the last prompt token");
        println!();
    }

    fn describe_decode_token(&self) {
        println!("Decode loop");
        println!("  1. Feed only the last sampled token");
        println!("  2. Reuse per-layer K/V cache instead of replaying the full prompt");
        println!("  3. Run the same visible transformer block path for one position");
        println!("  4. Sample from the final logits and append that token to the stream");
        println!();
        println!(
            "This binary is intentionally a visible trace scaffold, not a polished runtime yet."
        );
    }
}
