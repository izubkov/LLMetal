mod gguf_loader;
mod gpu;
mod inference;
mod model;
mod tensor;
mod tests;
mod tokenizer;

use anyhow::{Context, Result, bail};
use gguf_loader::GgufModelInfo;
use inference::TransparentRunner;
use model::LlamaModel;

fn main() -> Result<()> {
    let command = Command::from_env()?;

    match command {
        Command::Inspect { model_path } => {
            let model = GgufModelInfo::load(&model_path)
                .with_context(|| format!("failed to inspect GGUF file: {model_path}"))?;
            model.print_summary();
        }
        Command::Trace { model_path, prompt } => {
            let model = GgufModelInfo::load(&model_path)
                .with_context(|| format!("failed to inspect GGUF file: {model_path}"))?;
            let runner = TransparentRunner::new(model, gpu::Gpu::new()?);
            runner.describe_prompt_pass(&prompt);
        }
        Command::Run { model_path, prompt, max_new } => {
            eprintln!("Loading model tensors (mmap)...");
            let model = LlamaModel::load(&model_path)?;
            eprintln!(
                "Architecture: {} layers, {} hidden, {} heads, {} kv-heads",
                model.arch.n_layers, model.arch.hidden, model.arch.n_heads, model.arch.n_kv_heads
            );

            eprintln!("Loading vocabulary...");
            let gguf = GgufModelInfo::load(&model_path)?;
            let vocab = gguf.vocab;
            let tokenizer = tokenizer::PromptTokenizer::new(vocab.clone());

            eprintln!("Tokenizing prompt...");
            let token_ids = tokenizer.tokenize_bos(&prompt);
            eprintln!("  {} tokens", token_ids.len());

            eprintln!("\n--- generation ---");
            let mut model = model;
            model.generate(&token_ids, max_new, &vocab)?;
        }
    }

    Ok(())
}

enum Command {
    Inspect { model_path: String },
    Trace { model_path: String, prompt: String },
    Run { model_path: String, prompt: String, max_new: usize },
}

impl Command {
    fn from_env() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let Some(command) = args.next() else {
            print_usage();
            bail!("missing command");
        };

        match command.as_str() {
            "inspect" => {
                let Some(model_path) = args.next() else {
                    print_usage();
                    bail!("missing GGUF path");
                };
                Ok(Self::Inspect { model_path })
            }
            "trace" => {
                let Some(model_path) = args.next() else {
                    print_usage();
                    bail!("missing GGUF path");
                };
                let prompt = args.collect::<Vec<_>>().join(" ");
                let prompt = if prompt.is_empty() {
                    "The quick brown fox".to_string()
                } else {
                    prompt
                };
                Ok(Self::Trace { model_path, prompt })
            }
            "run" => {
                let Some(model_path) = args.next() else {
                    print_usage();
                    bail!("missing GGUF path");
                };
                let mut max_new = 64;
                let mut prompt_words = Vec::new();
                loop {
                    match args.next().as_deref() {
                        Some("--max") => {
                            max_new = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
                        }
                        Some(w) => prompt_words.push(w.to_string()),
                        None => break,
                    }
                }
                let prompt = if prompt_words.is_empty() {
                    "Hello".to_string()
                } else {
                    prompt_words.join(" ")
                };
                Ok(Self::Run { model_path, prompt, max_new })
            }
            _ => {
                print_usage();
                bail!("unknown command: {command}");
            }
        }
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  llmetal inspect <model.gguf>");
    eprintln!("  llmetal trace   <model.gguf> [prompt]");
    eprintln!("  llmetal run     <model.gguf> [--max N] [prompt text]");
}
