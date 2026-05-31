pub struct PromptTokens<'a> {
    prompt: &'a str,
}

impl<'a> PromptTokens<'a> {
    pub fn from_prompt(prompt: &'a str) -> Self {
        Self { prompt }
    }

    pub fn explain(&self) -> String {
        let bytes = self.prompt.len();
        let words = self.prompt.split_whitespace().count();
        format!(
            "not tokenized here; tokenizer wiring is deliberately left explicit ({bytes} bytes, {words} whitespace chunks)"
        )
    }
}
