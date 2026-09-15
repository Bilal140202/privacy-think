// LLM Commands - Tauri IPC commands for LLM operations
// Exposes model loading, text generation, and streaming to the frontend

use crate::llm::{
    downloader::ModelDownloader,
    inference::InferenceEngine,
    model_loader::ModelLoader,
    types::{AvailableModel, InferenceConfig, InferenceStats, ModelInfo},
};
use tauri::{Emitter, Window};
use tracing::{error, info};
use tauri::async_runtime::JoinHandle;
use std::collections::HashMap;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

static ACTIVE_DOWNLOADS: Lazy<Mutex<HashMap<String, JoinHandle<()>>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

/// Load a model into memory
#[tauri::command]
pub async fn load_model(model_name: String) -> Result<ModelInfo, String> {
    info!("Command: load_model({})", model_name);
    
    // Check if model file exists
    if !ModelLoader::model_exists(&model_name) {
        return Err(format!(
            "Model '{}' is not downloaded. Please download it first from the Models page.",
            model_name
        ));
    }
    
    // Unload any existing model first
    if ModelLoader::is_model_loaded() {
        info!("Unloading previous model before loading new one");
        ModelLoader::unload_model().map_err(|e| e.to_string())?;
    }
    
    // Load the new model
    ModelLoader::load_model(&model_name).map_err(|e| e.to_string())
}

/// Get information about the currently loaded model
#[tauri::command]
pub fn get_loaded_model() -> Option<ModelInfo> {
    ModelLoader::get_loaded_model()
}

/// Unload the current model from memory
#[tauri::command]
pub async fn unload_model() -> Result<(), String> {
    info!("Command: unload_model");
    // Also cancel any running inference
    InferenceEngine::cancel();
    ModelLoader::unload_model().map_err(|e| e.to_string())
}

/// Cancel any ongoing inference process
#[tauri::command]
pub fn cancel_inference() {
    info!("Command: cancel_inference");
    InferenceEngine::cancel();
}

/// Get list of all available models with download status
#[tauri::command]
pub fn get_available_models() -> Vec<AvailableModel> {
    ModelLoader::get_available_models()
}

/// Generate text synchronously (non-streaming)
#[tauri::command]
pub async fn generate(prompt: String, config: Option<InferenceConfig>) -> Result<String, String> {
    info!("Command: generate (prompt length: {})", prompt.len());
    let config = config.unwrap_or_default();
    InferenceEngine::generate_text(&prompt, &config)
        .await
        .map_err(|e| e.to_string())
}

/// Generate text with streaming (emits events to frontend)
#[tauri::command]
pub async fn generate_stream(
    window: Window,
    prompt: String,
    config: Option<InferenceConfig>,
) -> Result<(), String> {
    info!("Command: generate_stream (prompt length: {})", prompt.len());

    let config = config.unwrap_or_default();
    let window_for_tokens = window.clone();

    // Directly await async inference — no spawn_blocking needed
    let result = InferenceEngine::generate_stream(&prompt, &config, move |token| {
        if let Err(e) = window_for_tokens.emit("llm-token", token) {
            error!("Failed to emit llm-token event: {}", e);
        }
    })
    .await;

    match result {
        Ok(full_text) => {
            if let Err(e) = window.emit("llm-done", &full_text) {
                error!("Failed to emit llm-done event: {}", e);
            }
            Ok(())
        }
        Err(e) => {
            let error_msg = e.to_string();
            if let Err(emit_err) = window.emit("llm-error", &error_msg) {
                error!("Failed to emit llm-error event: {}", emit_err);
            }
            Err(error_msg)
        }
    }
}

/// Get inference statistics
#[tauri::command]
pub fn get_inference_stats() -> InferenceStats {
    InferenceEngine::get_stats()
}

/// Download a model with progress events
#[tauri::command]
pub async fn download_model_async(
    window: Window,
    model_name: String,
) -> Result<String, String> {
    info!("Command: download_model_async({})", model_name);
    
    // Find the model
    let model = AvailableModel::all()
        .into_iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| format!("Unknown model: {}", model_name))?;
    
    let window_clone = window.clone();
    let model_name_clone = model_name.clone();
    let model_name_for_map = model_name.clone();
    
    // Spawn the download in the background
    let handle = tauri::async_runtime::spawn(async move {
        // Download with progress reporting
        let result = ModelDownloader::download_model(&model, move |progress| {
            if let Err(e) = window_clone.emit("download-progress", &progress) {
                error!("Failed to emit download-progress event: {}", e);
            }
        })
        .await;
        
        // Remove from active downloads when completed or failed
        {
            let mut active = ACTIVE_DOWNLOADS.lock();
            active.remove(&model_name_clone);
        }

        match result {
            Ok(_) => {
                // Emit completion event
                if let Err(e) = window.emit("download-complete", &model_name_clone) {
                    error!("Failed to emit download-complete event: {}", e);
                }
            }
            Err(e) => {
                // Emit error event with structured payload
                let error_msg = e.to_string();
                #[derive(serde::Serialize, Clone)]
                struct DownloadErrorPayload {
                    model_name: String,
                    error: String,
                }
                let payload = DownloadErrorPayload {
                    model_name: model_name_clone.clone(),
                    error: error_msg.clone(),
                };
                if let Err(emit_err) = window.emit("download-error", &payload) {
                    error!("Failed to emit download-error event: {}", emit_err);
                }
            }
        }
    });

    // Store the handle in active downloads map
    {
        let mut active = ACTIVE_DOWNLOADS.lock();
        if let Some(old_handle) = active.insert(model_name_for_map, handle) {
            old_handle.abort();
        }
    }
    
    Ok("download_started".to_string())
}

/// Cancel an ongoing download
#[tauri::command]
pub fn cancel_download(model_name: String) -> Result<(), String> {
    info!("Command: cancel_download({})", model_name);
    
    // Abort the active download task
    {
        let mut active = ACTIVE_DOWNLOADS.lock();
        if let Some(handle) = active.remove(&model_name) {
            handle.abort();
            info!("Aborted active download task for {}", model_name);
        }
    }
    
    ModelDownloader::cancel_download(&model_name).map_err(|e| e.to_string())
}

/// Delete a downloaded model
#[tauri::command]
pub fn delete_model(model_name: String) -> Result<(), String> {
    info!("Command: delete_model({})", model_name);
    
    // Make sure model is not loaded before deleting
    if let Some(loaded) = ModelLoader::get_loaded_model() {
        if loaded.name == model_name {
            return Err("Cannot delete a model that is currently loaded. Unload it first.".to_string());
        }
    }
    
    ModelDownloader::delete_model(&model_name).map_err(|e| e.to_string())
}

/// Check models directory and get status
#[tauri::command]
pub fn check_models() -> Result<Vec<AvailableModel>, String> {
    info!("Command: check_models");
    
    // Ensure models directory exists
    ModelLoader::ensure_models_dir().map_err(|e| e.to_string())?;
    
    Ok(ModelLoader::get_available_models())
}
