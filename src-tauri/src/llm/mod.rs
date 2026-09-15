// LLM Module - Local Language Model integration
// Provides GGUF model loading, text generation, and model management

pub mod types;
pub mod model_loader;
pub mod inference;
pub mod downloader;
pub mod llamacpp_subprocess;
pub mod chat_template;

pub use types::*;
pub use model_loader::ModelLoader;
pub use inference::InferenceEngine;
pub use chat_template::{format_prompt_for_model, get_stop_sequences_for_model};

