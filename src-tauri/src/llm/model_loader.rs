// Model Loader - Handles GGUF model path resolution and llama.cpp keep-alive process management.
// Candle is NOT used here. All inference is delegated to llama-run via llamacpp_subprocess.

use crate::llm::types::{LlmError, ModelInfo, AvailableModel};
use crate::llm::llamacpp_subprocess;
use crate::utils::paths;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

/// Global model state - singleton pattern for loaded model
pub(crate) static MODEL_STATE: Lazy<Arc<RwLock<ModelState>>> = Lazy::new(|| {
    Arc::new(RwLock::new(ModelState::default()))
});

/// Internal state for the loaded model
#[derive(Default)]
pub(crate) struct ModelState {
    /// Currently loaded model info
    pub loaded_model: Option<LoadedModel>,
}

/// Represents a model that has been validated and whose llama-run process is running
pub(crate) struct LoadedModel {
    pub info: ModelInfo,
}

/// Model loader with singleton pattern for managing GGUF models
pub struct ModelLoader;

impl ModelLoader {
    /// Get the models directory path
    pub fn get_models_dir() -> Result<PathBuf, LlmError> {
        paths::get_models_dir().map_err(|e| LlmError::LoadError(e.to_string()))
    }

    /// Ensure the models directory exists, returning its path
    pub fn ensure_models_dir() -> Result<PathBuf, LlmError> {
        let dir = Self::get_models_dir()?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| LlmError::LoadError(format!("Cannot create models dir: {}", e)))?;
        Ok(dir)
    }

    /// Check if a model file exists (either downloaded or bundled)
    pub fn model_exists(model_name: &str) -> bool {
        Self::get_model_path(model_name).is_ok()
    }

    /// Get the path to a model file (checks AppData, then bundled resources)
    pub fn get_model_path(model_name: &str) -> Result<PathBuf, LlmError> {
        // Map model names to their actual filenames
        let filename = match model_name.to_lowercase().as_str() {
            "qwen2.5-0.5b" => "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-1.5b" => "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "llama-3.2-3b" => "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "qwen2.5-7b"   => "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "tinyllama"    => "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "phi-4-mini"   => "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
            "gemma-3-4b"   => "google_gemma-3-4b-it-Q4_K_M.gguf",
            _ => return Err(LlmError::ModelNotFound(format!("Unknown model: {}", model_name))),
        };

        // 1. Check AppData (user downloaded)
        if let Ok(models_dir) = Self::get_models_dir() {
            let path = models_dir.join(filename);
            if path.exists() {
                return Ok(path);
            }
        }

        // 2. Check current dir /bin (dev)
        let bin_path = PathBuf::from("./bin").join(filename);
        if bin_path.exists() {
            return Ok(bin_path);
        }

        Err(LlmError::ModelNotFound(format!(
            "Model '{}' not found. Please download it from the Models page.",
            model_name
        )))
    }

    /// Validate that the file at `path` is a valid GGUF file by checking its magic bytes.
    pub fn validate_gguf(path: &PathBuf) -> Result<(), LlmError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| LlmError::LoadError(format!("Cannot open GGUF file: {}", e)))?;

        let mut magic = [0u8; 4];
        use std::io::Read;
        file.read_exact(&mut magic)
            .map_err(|e| LlmError::LoadError(format!("Cannot read GGUF magic: {}", e)))?;

        // GGUF magic: "GGUF" = [0x47, 0x47, 0x55, 0x46]
        if &magic != b"GGUF" {
            return Err(LlmError::LoadError(format!(
                "File is not a valid GGUF model (bad magic bytes: {:?})",
                magic
            )));
        }

        Ok(())
    }

    /// Load a model:
    ///  1. Resolve the GGUF path
    ///  2. Validate GGUF magic bytes
    ///  3. Spawn (or re-spawn) the llama-run keep-alive process
    ///  4. Store ModelInfo in MODEL_STATE — inference happens in llamacpp_subprocess
    pub fn load_model(model_name: &str) -> Result<ModelInfo, LlmError> {
        info!("Loading model: {}", model_name);

        let model_path = Self::get_model_path(model_name)?;

        // Validate the GGUF file magic bytes
        Self::validate_gguf(&model_path)?;

        // Determine CPU thread count
        let threads = (std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4) as u8)
            .min(8);

        // Spawn (or respawn) the llama-run keep-alive process for this model
        llamacpp_subprocess::spawn_keep_alive(&model_path, threads)
            .map_err(|e| LlmError::LoadError(format!("Failed to start llama-run: {}", e)))?;

        // Get file size
        let metadata = std::fs::metadata(&model_path)
            .map_err(|e| LlmError::LoadError(format!("Cannot read file metadata: {}", e)))?;
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);

        // Per-model description / RAM estimate
        let (description, min_ram) = match model_name.to_lowercase().as_str() {
            "qwen2.5-0.5b" => ("Ultra-fast 0.5B model for quick queries".to_string(), 4),
            "qwen2.5-1.5b" => ("Balanced 1.5B model with good quality".to_string(), 6),
            "llama-3.2-3b" => ("Meta's 3B model with excellent reasoning".to_string(), 8),
            "qwen2.5-7b"   => ("Power model! Best quality responses".to_string(), 12),
            "tinyllama"    => ("Fast, efficient 1.1B model for basic queries".to_string(), 8),
            "phi-4-mini"   => ("Microsoft's excellent reasoning model. Runs on 8GB RAM.".to_string(), 6),
            "gemma-3-4b"   => ("Google's best small model for instruction following.".to_string(), 6),
            _              => ("Unknown model".to_string(), 8),
        };

        let model_info = ModelInfo {
            name: model_name.to_string(),
            size_mb,
            loaded: true,
            path: model_path.to_string_lossy().to_string(),
            description,
            min_ram_gb: min_ram,
        };

        // Store in global state
        {
            let mut state = MODEL_STATE.write();
            state.loaded_model = Some(LoadedModel {
                info: model_info.clone(),
            });
        }

        info!("Model '{}' ready ({:.1} MB) — llama-run process running", model_name, size_mb);
        Ok(model_info)
    }

    /// Get information about the currently loaded model
    pub fn get_loaded_model() -> Option<ModelInfo> {
        let state = MODEL_STATE.read();
        state.loaded_model.as_ref().map(|m| m.info.clone())
    }

    /// Check if a model is currently loaded
    pub fn is_model_loaded() -> bool {
        let state = MODEL_STATE.read();
        state.loaded_model.is_some()
    }

    /// Unload the current model from memory and kill the llama-run process
    pub fn unload_model() -> Result<(), LlmError> {
        let mut state = MODEL_STATE.write();

        if state.loaded_model.is_none() {
            warn!("No model loaded to unload");
            return Ok(());
        }

        let model_name = state.loaded_model.as_ref()
            .map(|m| m.info.name.clone())
            .unwrap_or_default();

        state.loaded_model = None;
        drop(state);

        // Kill the llama-run process
        llamacpp_subprocess::kill_current_process();

        info!("Model '{}' unloaded", model_name);
        Ok(())
    }

    /// Get list of all available models with their download status
    pub fn get_available_models() -> Vec<AvailableModel> {
        let mut models = AvailableModel::all();
        for model in &mut models {
            model.downloaded = Self::model_exists(&model.name);
        }
        models
    }

    // ----- tokenizer (kept for potential future use, currently NOT called during load) -----

    /// Ensure tokenizer is downloaded from the correct base model repository.
    /// GGUF repos (bartowski, Qwen-GGUF) do NOT contain tokenizer.json — only the
    /// base model repos do. We hardcode the correct base repo URL per model here.
    #[allow(dead_code)]
    pub fn ensure_tokenizer(model_name: &str) -> Result<PathBuf, LlmError> {
        let models_dir = Self::get_models_dir()?;
        let model_path = Self::get_model_path(model_name)?;
        let filename = model_path.file_name().unwrap_or_default().to_str().unwrap_or_default();
        let tokenizer_filename = format!("{}.tokenizer.json", filename);
        let tokenizer_path = models_dir.join(&tokenizer_filename);

        if tokenizer_path.exists() {
            return Ok(tokenizer_path);
        }

        // Hardcoded tokenizer URLs pointing to the correct base repos (not GGUF repos).
        let tokenizer_url = match model_name {
            "qwen2.5-0.5b" => "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
            "qwen2.5-1.5b" => "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json",
            "qwen2.5-7b"   => "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer.json",
            "llama-3.2-3b" => "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/tokenizer.json",
            "tinyllama"    => "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json",
            "phi-4-mini"   => "https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/tokenizer.json",
            "gemma-3-4b"   => "https://huggingface.co/google/gemma-3-4b-it/resolve/main/tokenizer.json",
            _ => return Err(LlmError::ModelNotFound(format!("No tokenizer URL for model: {}", model_name))),
        };

        info!("Downloading tokenizer from {}...", tokenizer_url);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| LlmError::LoadError(format!("Failed to build HTTP client: {}", e)))?;

        let response = tokio::runtime::Handle::current()
            .block_on(async {
                client.get(tokenizer_url).send().await
            })
            .map_err(|e| LlmError::LoadError(format!("Failed to download tokenizer: {}", e)))?;

        if !response.status().is_success() {
            return Err(LlmError::LoadError(format!(
                "Failed to download tokenizer. Status: {}",
                response.status()
            )));
        }

        let bytes = tokio::runtime::Handle::current()
            .block_on(async { response.bytes().await })
            .map_err(|e| LlmError::LoadError(format!("Failed to read tokenizer response: {}", e)))?;

        std::fs::write(&tokenizer_path, &bytes)
            .map_err(|e| LlmError::LoadError(format!("Failed to save tokenizer: {}", e)))?;

        info!("Tokenizer downloaded to {:?}", tokenizer_path);
        Ok(tokenizer_path)
    }
}
