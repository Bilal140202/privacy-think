// LLM Types - Data structures for LLM operations

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,
    
    /// Context window size (default: 2048)
    #[serde(default = "default_context_size")]
    pub context_size: u32,
    
    /// Temperature for sampling (default: 0.7)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    
    /// Maximum tokens to generate (default: 512)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    
    /// Number of CPU threads to use (default: 4)
    #[serde(default = "default_threads")]
    pub threads: u8,
    
    /// Use memory-mapped files for model loading
    #[serde(default = "default_use_mmap")]
    pub use_mmap: bool,
}

fn default_context_size() -> u32 { 2048 }
fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> u32 { 512 }
fn default_threads() -> u8 { 4 }
fn default_use_mmap() -> bool { true }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            context_size: default_context_size(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            threads: default_threads(),
            use_mmap: default_use_mmap(),
        }
    }
}

impl ModelConfig {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            ..Default::default()
        }
    }
    
    pub fn with_context_size(mut self, size: u32) -> Self {
        self.context_size = size;
        self
    }
    
    pub fn with_threads(mut self, threads: u8) -> Self {
        self.threads = threads;
        self
    }
}

/// Configuration for text generation/inference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    /// Temperature for randomness (0.0 = deterministic, 1.0 = random)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling threshold
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    
    /// Top-k sampling (number of top tokens to consider)
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    
    /// Penalty for repeating tokens
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    
    /// Stop sequences that terminate generation
    #[serde(default = "default_stop_sequences")]
    pub stop_sequences: Vec<String>,

    /// llama-cli context window size (-c flag).
    /// - None  → auto-detected from available RAM at inference time:
    ///           <6 GB → 2048, 6–12 GB → 4096, ≥12 GB → 8192
    /// - Some(n) → use exactly n (user override from /settings)
    #[serde(default)]
    pub context_size: Option<u32>,
}

fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> u32 { 40 }
fn default_repeat_penalty() -> f32 { 1.1 }
fn default_stop_sequences() -> Vec<String> { 
    vec!["###".to_string(), "\n\n\n".to_string()] 
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repeat_penalty: default_repeat_penalty(),
            max_tokens: default_max_tokens(),
            stop_sequences: default_stop_sequences(),
            context_size: None,
        }
    }
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Model name (e.g., "tinyllama", "phi2")
    pub name: String,
    
    /// Model file size in megabytes
    pub size_mb: f64,
    
    /// Whether the model is currently loaded
    pub loaded: bool,
    
    /// Path to the model file
    pub path: String,
    
    /// Model description
    pub description: String,
    
    /// Recommended minimum RAM in GB
    pub min_ram_gb: u32,
}

/// Statistics from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceStats {
    /// Tokens generated per second
    pub tokens_per_second: f32,
    
    /// Total tokens generated
    pub total_tokens: u32,
    
    /// Memory used by the model in MB
    pub memory_used_mb: f32,
    
    /// Context usage (0.0 to 1.0)
    pub context_usage: f32,
    
    /// Whether a model is currently loaded
    pub model_loaded: bool,
}

/// Available models for download
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailableModel {
    pub name: String,
    pub display_name: String,
    pub filename: String,
    pub url: String,
    pub size_mb: u64,
    pub description: String,
    pub min_ram_gb: u32,
    pub downloaded: bool,
}

impl AvailableModel {
    pub fn qwen_05b() -> Self {
        Self {
            name: "qwen2.5-0.5b".to_string(),
            display_name: "Qwen 2.5 0.5B".to_string(),
            filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
            url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
            size_mb: 397,
            description: "Ultra-fast, lightweight model. Great for quick queries on any system.".to_string(),
            min_ram_gb: 4,
            downloaded: false,
        }
    }
    
    pub fn qwen_15b() -> Self {
        Self {
            name: "qwen2.5-1.5b".to_string(),
            display_name: "Qwen 2.5 1.5B".to_string(),
            filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
            url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
            size_mb: 1050,
            description: "Balanced speed and quality. Good for 8GB RAM systems.".to_string(),
            min_ram_gb: 6,
            downloaded: false,
        }
    }
    
    pub fn llama_32_3b() -> Self {
        Self {
            name: "llama-3.2-3b".to_string(),
            display_name: "Llama 3.2 3B".to_string(),
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
            url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
            size_mb: 2020,
            description: "Meta's latest small model. Excellent reasoning for its size.".to_string(),
            min_ram_gb: 8,
            downloaded: false,
        }
    }
    
    pub fn qwen_7b() -> Self {
        Self {
            name: "qwen2.5-7b".to_string(),
            display_name: "Qwen 2.5 7B ⭐".to_string(),
            filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_string(),
            url: "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_string(),
            size_mb: 4680,
            description: "Power model! Best quality responses. Needs 12GB+ RAM.".to_string(),
            min_ram_gb: 12,
            downloaded: false,
        }
    }
    
    pub fn phi_4_mini() -> Self {
        Self {
            name: "phi-4-mini".to_string(),
            display_name: "Phi-4-mini 3.8B".to_string(),
            filename: "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf".to_string(),
            url: "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf".to_string(),
            size_mb: 2400,
            description: "Microsoft's excellent reasoning model. Runs on 8GB RAM.".to_string(),
            min_ram_gb: 6,
            downloaded: false,
        }
    }
    
    pub fn gemma_3_4b() -> Self {
        Self {
            name: "gemma-3-4b".to_string(),
            display_name: "Gemma 3 4B".to_string(),
            filename: "google_gemma-3-4b-it-Q4_K_M.gguf".to_string(),
            url: "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf".to_string(),
            size_mb: 2600,
            description: "Google's best small model for instruction following.".to_string(),
            min_ram_gb: 6,
            downloaded: false,
        }
    }
    
    pub fn all() -> Vec<Self> {
        vec![
            Self::qwen_05b(),
            Self::qwen_15b(),
            Self::llama_32_3b(),
            Self::phi_4_mini(),
            Self::gemma_3_4b(),
            Self::qwen_7b(),
        ]
    }
}

/// Download progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub model_name: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub speed_mbps: f32,
    pub eta_seconds: u32,
}

/// LLM-related errors
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Failed to load model: {0}")]
    LoadError(String),
    
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("Model not loaded")]
    ModelNotLoaded,
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Invalid GGUF file: {0}")]
    InvalidGguf(String),
    
    #[error("Download failed: {0}")]
    DownloadError(String),
    
    #[error("Generation timeout")]
    Timeout,
    
    #[error("Generation cancelled")]
    Cancelled,
}

impl From<LlmError> for String {
    fn from(err: LlmError) -> Self {
        err.to_string()
    }
}
