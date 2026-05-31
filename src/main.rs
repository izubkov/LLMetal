mod gguf_loader;
mod inference;
mod metal;
mod tokenizer;

use anyhow::{Context, Result, bail};
use gguf_loader::GgufModelInfo;
use inference::TransparentRunner;

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
            let runner = TransparentRunner::new(model, metal::preferred_device());
            runner.describe_prompt_pass(&prompt);
        }
    }

    Ok(())
}

enum Command {
    Inspect { model_path: String },
    Trace { model_path: String, prompt: String },
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
    eprintln!("  llmetal trace <model.gguf> [prompt text]");
}
