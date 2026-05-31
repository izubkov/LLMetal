/// Greedy longest-match tokenizer over a GGUF vocab.
///
/// GGUF uses SentencePiece-style tokens where spaces are represented as the
/// `▁` (U+2581) character. This normaliser replaces leading/internal spaces
/// with `▁` so the greedy scan finds real token boundaries.
pub struct PromptTokenizer {
    /// Sorted vocab slice, indexed by token id.
    vocab: Vec<String>,
}

impl PromptTokenizer {
    pub fn new(vocab: Vec<String>) -> Self {
        Self { vocab }
    }

    /// Tokenize with BOS token (id=1) prepended, matching llama.cpp conventions.
    pub fn tokenize_bos(&self, prompt: &str) -> Vec<u32> {
        let mut ids = vec![1u32]; // BOS
        ids.extend(self.tokenize(prompt));
        ids
    }

    pub fn tokenize(&self, prompt: &str) -> Vec<u32> {
        if self.vocab.is_empty() {
            return Vec::new();
        }

        // GPT-2 / Tekken: prepend a space then replace all spaces with Ġ (U+0120)
        // so "hello world" → " hello world" → "Ġhello Ġworld" → "ĠhelloĠworld"
        let normalised = format!(" {prompt}").replace(' ', "\u{0120}");
        let chars: Vec<char> = normalised.chars().collect();
        let mut ids = Vec::new();
        let mut pos = 0;

        while pos < chars.len() {
            let mut best_len = 0;
            let mut best_id = 0u32;

            // Greedy scan: longest token that matches from `pos`.
            for (id, tok) in self.vocab.iter().enumerate() {
                let tok_chars: Vec<char> = tok.chars().collect();
                let len = tok_chars.len();
                if len > best_len && pos + len <= chars.len() {
                    if chars[pos..pos + len] == tok_chars[..] {
                        best_len = len;
                        best_id = id as u32;
                    }
                }
            }

            if best_len == 0 {
                // Unknown character — emit as a single byte token (fallback).
                ids.push(u32::MAX);
                pos += 1;
            } else {
                ids.push(best_id);
                pos += best_len;
            }
        }

        ids
    }

    pub fn explain(&self, prompt: &str) -> String {
        if self.vocab.is_empty() {
            let bytes = prompt.len();
            let words = prompt.split_whitespace().count();
            return format!(
                "no vocab in GGUF; estimate ~{} tokens ({bytes} bytes, {words} words)",
                (bytes as f64 / 3.5).ceil() as usize,
            );
        }

        let ids = self.tokenize_bos(prompt);
        let unknown = ids.iter().filter(|&&id| id == u32::MAX).count();
        let token_strs: Vec<&str> = ids
            .iter()
            .map(|&id| {
                if id == u32::MAX {
                    "<unk>"
                } else {
                    self.vocab[id as usize].as_str()
                }
            })
            .collect();

        if unknown > 0 {
            format!(
                "{} tokens [{} unknown]: {}",
                ids.len(),
                unknown,
                token_strs.join("|")
            )
        } else {
            format!("{} tokens: {}", ids.len(), token_strs.join("|"))
        }
    }
}
