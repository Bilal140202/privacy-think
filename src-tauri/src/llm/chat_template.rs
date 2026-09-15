// Chat Template Engine for PrivacyThink
// Formats raw prompts into model-specific conversational templates
// (ChatML for Qwen, Llama 3 header tokens, Phi-4 tokens, Gemma turns, TinyLlama)
// and provides comprehensive stop sequences to prevent model rambling.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen,
    Llama3,
    Phi4,
    Gemma,
    TinyLlama,
    Generic,
}

impl ModelFamily {
    pub fn from_path_or_name(name_or_path: &str) -> Self {
        let lower = name_or_path.to_lowercase();
        if lower.contains("qwen") {
            ModelFamily::Qwen
        } else if lower.contains("llama-3") || lower.contains("llama3") {
            ModelFamily::Llama3
        } else if lower.contains("phi-4") || lower.contains("phi4") {
            ModelFamily::Phi4
        } else if lower.contains("gemma") {
            ModelFamily::Gemma
        } else if lower.contains("tinyllama") {
            ModelFamily::TinyLlama
        } else {
            ModelFamily::Generic
        }
    }
}

/// Check if a prompt already contains chat template markers
pub fn is_already_templated(prompt: &str) -> bool {
    prompt.contains("<|im_start|>")
        || prompt.contains("<|begin_of_text|>")
        || prompt.contains("<start_of_turn>")
        || prompt.contains("<|system|>")
        || prompt.contains("<|user|>")
        || prompt.contains("[INST]")
}

/// Split a prompt into system and user parts if explicit separators exist
fn extract_system_and_user(prompt: &str) -> (Option<String>, String) {
    let trimmed = prompt.trim();

    // Check for explicit SYSTEM: ... USER: ... or SYSTEM: ... CONTEXT: ...
    if let Some(sys_start) = trimmed.strip_prefix("SYSTEM:") {
        if let Some(user_idx) = sys_start.find("\nUSER:") {
            let sys = sys_start[..user_idx].trim().to_string();
            let user = sys_start[user_idx + 6..].trim().to_string();
            return (Some(sys), user);
        }
    }

    // Check for "You are a document analysis assistant... CONTEXT: ... QUESTION: ..."
    if trimmed.starts_with("You are a document analysis assistant") {
        if let Some(ctx_idx) = trimmed.find("\n\nCONTEXT:") {
            let sys = trimmed[..ctx_idx].trim().to_string();
            let user = trimmed[ctx_idx + 2..].trim().to_string();
            return (Some(sys), user);
        }
    }

    // Default: no separate system prompt detected in the raw text
    (None, trimmed.to_string())
}

/// Format a prompt according to the target model family
pub fn format_prompt_for_model(model_path: &Path, prompt: &str) -> String {
    if is_already_templated(prompt) {
        return prompt.to_string();
    }

    let model_str = model_path.file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    let family = ModelFamily::from_path_or_name(model_str);

    let (custom_system, user_content) = extract_system_and_user(prompt);
    let default_system = "You are a helpful, accurate, and concise AI assistant. Always respond in English.";
    let system_text = custom_system.as_deref().unwrap_or(default_system);

    match family {
        ModelFamily::Qwen | ModelFamily::Generic => {
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system_text, user_content
            )
        }
        ModelFamily::Llama3 => {
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                system_text, user_content
            )
        }
        ModelFamily::Phi4 => {
            format!(
                "<|im_start|>system<|im_sep|>\n{}<|im_end|>\n<|im_start|>user<|im_sep|>\n{}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n",
                system_text, user_content
            )
        }
        ModelFamily::Gemma => {
            format!(
                "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
                system_text, user_content
            )
        }
        ModelFamily::TinyLlama => {
            format!(
                "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                system_text, user_content
            )
        }
    }
}

/// Return all active stop sequences for a given model
pub fn get_stop_sequences_for_model(model_path: &Path) -> Vec<String> {
    let model_str = model_path.file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    let family = ModelFamily::from_path_or_name(model_str);

    let mut stops = vec![
        "[end of text]".to_string(),
        "<|im_end|>".to_string(),
        "<|endoftext|>".to_string(),
        "<|eot_id|>".to_string(),
        "<|end_of_text|>".to_string(),
        "<end_of_turn>".to_string(),
        "</s>".to_string(),
        "<|im_start|>".to_string(),
        "\nUser:".to_string(),
        "\nClient:".to_string(),
        "\nHuman:".to_string(),
    ];

    match family {
        ModelFamily::Qwen => {
            stops.push("<|im_end|>".to_string());
        }
        ModelFamily::Llama3 => {
            stops.push("<|eot_id|>".to_string());
            stops.push("<|start_header_id|>".to_string());
        }
        ModelFamily::Phi4 => {
            stops.push("<|im_end|>".to_string());
            stops.push("<|im_sep|>".to_string());
        }
        ModelFamily::Gemma => {
            stops.push("<end_of_turn>".to_string());
            stops.push("<start_of_turn>".to_string());
        }
        ModelFamily::TinyLlama => {
            stops.push("</s>".to_string());
        }
        ModelFamily::Generic => {}
    }

    stops.dedup();
    stops
}
