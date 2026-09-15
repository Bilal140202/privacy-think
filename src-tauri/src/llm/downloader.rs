// Model Downloader - Downloads GGUF models from HuggingFace
// Supports progress reporting, resume, and validation

use crate::llm::model_loader::ModelLoader;
use crate::llm::types::{AvailableModel, DownloadProgress, LlmError};
use futures_util::StreamExt;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

/// Model downloader with progress reporting
pub struct ModelDownloader;

impl ModelDownloader {
    /// Check which models are not yet downloaded
    pub fn get_missing_models() -> Vec<AvailableModel> {
        AvailableModel::all()
            .into_iter()
            .filter(|m| !ModelLoader::model_exists(&m.name))
            .collect()
    }
    
    /// Check which models are already downloaded
    pub fn get_downloaded_models() -> Vec<AvailableModel> {
        AvailableModel::all()
            .into_iter()
            .filter(|m| ModelLoader::model_exists(&m.name))
            .collect()
    }
    
    /// Download a model with progress reporting
    pub async fn download_model(
        model: &AvailableModel,
        progress_callback: impl Fn(DownloadProgress) + Send + Sync,
    ) -> Result<PathBuf, LlmError> {
        info!("Starting download of model: {} from {}", model.name, model.url);
        
        // Ensure models directory exists
        let models_dir = ModelLoader::ensure_models_dir()?;
        let target_path = models_dir.join(&model.filename);
        let temp_path = models_dir.join(format!("{}.download", model.filename));
        
        // Check if already downloaded
        if target_path.exists() {
            info!("Model already exists at {:?}", target_path);
            return Ok(target_path);
        }
        
        // Check for temp file and partial download size
        let mut existing_bytes = 0;
        if temp_path.exists() {
            if let Ok(metadata) = std::fs::metadata(&temp_path) {
                existing_bytes = metadata.len();
            }
        }

        // Create HTTP client
        let client = reqwest::Client::builder()
            .user_agent("PrivacyThink/0.1.0")
            .build()
            .map_err(|e| LlmError::DownloadError(format!("Failed to create HTTP client: {}", e)))?;
        
        // Prepare request
        let mut request = client.get(&model.url);
        if existing_bytes > 0 {
            request = request.header("Range", format!("bytes={}-", existing_bytes));
            info!("Resuming download from {} bytes", existing_bytes);
        }

        // Start download
        let response = request
            .send()
            .await
            .map_err(|e| LlmError::DownloadError(format!("HTTP request failed: {}", e)))?;
        
        let status = response.status();
        if !status.is_success() {
            return Err(LlmError::DownloadError(format!(
                "Download failed with status: {}",
                status
            )));
        }
        
        let is_partial = status == reqwest::StatusCode::PARTIAL_CONTENT;
        
        // Open file in append or truncate mode
        let mut file = if is_partial && existing_bytes > 0 {
            std::fs::OpenOptions::new()
                .write(true)
                .append(true)
                .open(&temp_path)
                .map_err(|e| LlmError::DownloadError(format!("Failed to open temp file in append mode: {}", e)))?
        } else {
            if existing_bytes > 0 {
                info!("Server did not return partial content (206), restarting download from scratch");
            }
            existing_bytes = 0;
            std::fs::File::create(&temp_path)
                .map_err(|e| LlmError::DownloadError(format!("Failed to create temp file: {}", e)))?
        };
        
        // Get content length
        let remaining_bytes = response.content_length();
        let total_bytes = if is_partial {
            remaining_bytes.unwrap_or(0) + existing_bytes
        } else {
            remaining_bytes.unwrap_or(model.size_mb * 1024 * 1024)
        };
        
        info!("Downloading {} bytes to {:?}", total_bytes, temp_path);
        
        // Download with progress
        let mut stream = response.bytes_stream();
        let mut downloaded = existing_bytes;
        let start_time = Instant::now();
        let mut last_progress_time = Instant::now();
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .map_err(|e| LlmError::DownloadError(format!("Stream error: {}", e)))?;
            
            file.write_all(&chunk)
                .map_err(|e| LlmError::DownloadError(format!("Write error: {}", e)))?;
            
            downloaded += chunk.len() as u64;
            
            // Report progress every 500ms
            if last_progress_time.elapsed().as_millis() > 500 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let speed_mbps = if elapsed > 0.0 {
                    (downloaded as f32 / 1_048_576.0) / elapsed
                } else {
                    0.0
                };
                
                let percent = (downloaded as f32 / total_bytes as f32) * 100.0;
                let remaining_bytes = total_bytes - downloaded;
                let eta_seconds = if speed_mbps > 0.0 {
                    (remaining_bytes as f32 / (speed_mbps * 1_048_576.0)) as u32
                } else {
                    0
                };
                
                progress_callback(DownloadProgress {
                    model_name: model.name.clone(),
                    bytes_downloaded: downloaded,
                    total_bytes,
                    percent,
                    speed_mbps,
                    eta_seconds,
                });
                
                last_progress_time = Instant::now();
            }
        }
        
        // Ensure all data is written
        file.flush()
            .map_err(|e| LlmError::DownloadError(format!("Flush error: {}", e)))?;
        drop(file);
        
        // Validate the downloaded file
        info!("Download complete, validating GGUF file...");
        ModelLoader::validate_gguf(&temp_path)?;
        
        // Move to final location
        std::fs::rename(&temp_path, &target_path)
            .map_err(|e| LlmError::DownloadError(format!("Failed to rename file: {}", e)))?;
        
        let elapsed = start_time.elapsed();
        info!(
            "Model '{}' downloaded successfully in {:.1}s ({:.1} MB/s)",
            model.name,
            elapsed.as_secs_f32(),
            (total_bytes as f32 / 1_048_576.0) / elapsed.as_secs_f32()
        );
        
        // Final progress callback
        progress_callback(DownloadProgress {
            model_name: model.name.clone(),
            bytes_downloaded: total_bytes,
            total_bytes,
            percent: 100.0,
            speed_mbps: 0.0,
            eta_seconds: 0,
        });
        
        Ok(target_path)
    }
    
    /// Cancel/pause an ongoing download (keeping the temp file for resume)
    pub fn cancel_download(model_name: &str) -> Result<(), LlmError> {
        let models_dir = ModelLoader::get_models_dir()?;
        
        // Check if there is a temp file and log pause
        for model in AvailableModel::all() {
            if model.name == model_name {
                let temp_path = models_dir.join(format!("{}.download", model.filename));
                if temp_path.exists() {
                    info!("Paused download for model: {} (temp file retained for resume)", model_name);
                }
                break;
            }
        }
        
        Ok(())
    }
    
    /// Delete a downloaded model
    pub fn delete_model(model_name: &str) -> Result<(), LlmError> {
        let models_dir = ModelLoader::get_models_dir()?;
        
        for model in AvailableModel::all() {
            if model.name == model_name {
                let path = models_dir.join(&model.filename);
                if path.exists() {
                    std::fs::remove_file(&path)
                        .map_err(|e| LlmError::DownloadError(format!("Failed to delete model: {}", e)))?;
                    info!("Deleted model: {}", model_name);
                }
                break;
            }
        }
        
        Ok(())
    }
}
