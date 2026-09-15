// Embedding Generator - Local embedding generation using fastembed
//
// Uses the all-MiniLM-L6-v2 model (384 dimensions, ~80MB)
// Model is loaded once and cached for the lifetime of the application.

use fastembed::{TextEmbedding, EmbeddingModel, InitOptions};
use once_cell::sync::OnceCell;
use std::path::PathBuf;
use tracing::{info, warn, error};

/// Global embedding model instance (singleton pattern)
static EMBEDDING_MODEL: OnceCell<TextEmbedding> = OnceCell::new();

/// Result type for embedding operations
pub type EmbeddingResult<T> = Result<T, String>;

/// Embedding generator for document chunks
pub struct EmbeddingGenerator;

impl EmbeddingGenerator {
    /// Get the cache directory for embedding models
    fn get_cache_dir() -> Result<PathBuf, String> {
        let app_data = std::env::var("APPDATA")
            .map_err(|_| "APPDATA environment variable not found")?;
        let cache_dir = PathBuf::from(app_data)
            .join("PrivacyThink")
            .join("models")
            .join("embeddings");
        
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        
        Ok(cache_dir)
    }

    /// Initialize the embedding model (call once at startup)
    /// 
    /// This downloads the model on first run (~80MB) and caches it
    /// in %APPDATA%/PrivacyThink/models/embeddings/
    pub fn init() -> EmbeddingResult<()> {
        // Check if already initialized
        if EMBEDDING_MODEL.get().is_some() {
            info!("Embedding model already initialized");
            return Ok(());
        }
        
        info!("Initializing embedding model (all-MiniLM-L6-v2)...");
        
        let cache_dir = Self::get_cache_dir()?;
        info!("Model cache directory: {:?}", cache_dir);
        
        // Initialize the model with custom options using builder pattern
        let init_options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false)
            .with_cache_dir(cache_dir);
        
        let model = TextEmbedding::try_new(init_options)
            .map_err(|e| {
                error!("Failed to load embedding model: {}", e);
                format!("Failed to load embedding model: {}", e)
            })?;
        
        // Store in global singleton
        if EMBEDDING_MODEL.set(model).is_err() {
            warn!("Embedding model was already initialized by another thread");
        }
        
        info!("Embedding model initialized successfully");
        Ok(())
    }

    /// Check if the embedding model is initialized
    pub fn is_initialized() -> bool {
        EMBEDDING_MODEL.get().is_some()
    }

    /// Ensure the model is initialized, initializing it if necessary
    fn ensure_initialized() -> EmbeddingResult<()> {
        if !Self::is_initialized() {
            Self::init()?;
        }
        Ok(())
    }

    /// Generate embedding for a single text
    /// 
    /// Returns a 384-dimensional vector
    pub fn embed_text(text: &str) -> EmbeddingResult<Vec<f32>> {
        Self::ensure_initialized()?;
        
        let model = EMBEDDING_MODEL.get()
            .ok_or("Embedding model not initialized")?;
        
        // Generate embedding for single text
        let embeddings = model.embed(vec![text], None)
            .map_err(|e| format!("Embedding generation failed: {}", e))?;
        
        if embeddings.is_empty() {
            return Err("No embeddings generated".to_string());
        }
        
        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Generate embeddings for multiple texts (faster with batching)
    /// 
    /// Processes texts in batches of 32 for optimal performance
    pub fn embed_batch(texts: Vec<String>) -> EmbeddingResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        Self::ensure_initialized()?;
        
        let model = EMBEDDING_MODEL.get()
            .ok_or("Embedding model not initialized")?;
        
        // Process in batches of 32 for optimal performance
        let batch_size = 32;
        let mut all_embeddings = Vec::with_capacity(texts.len());
        
        for chunk_texts in texts.chunks(batch_size) {
            let batch_refs: Vec<&str> = chunk_texts.iter().map(|s| s.as_str()).collect();
            
            let embeddings = model.embed(batch_refs, None)
                .map_err(|e| format!("Batch embedding failed: {}", e))?;
            
            all_embeddings.extend(embeddings);
        }
        
        Ok(all_embeddings)
    }

    /// Get embedding dimensions (384 for all-MiniLM-L6-v2)
    pub const fn get_dimensions() -> usize {
        384
    }

    /// Get model information
    pub fn get_model_info() -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            name: "all-MiniLM-L6-v2".to_string(),
            dimensions: Self::get_dimensions(),
            max_sequence_length: 256,
            approximate_size_mb: 80,
            initialized: Self::is_initialized(),
        }
    }
}

/// Information about the embedding model
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingModelInfo {
    /// Model name
    pub name: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Maximum sequence length in tokens
    pub max_sequence_length: usize,
    /// Approximate model size in MB
    pub approximate_size_mb: usize,
    /// Whether the model is currently loaded
    pub initialized: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimensions() {
        assert_eq!(EmbeddingGenerator::get_dimensions(), 384);
    }

    #[test]
    fn test_model_info() {
        let info = EmbeddingGenerator::get_model_info();
        assert_eq!(info.name, "all-MiniLM-L6-v2");
        assert_eq!(info.dimensions, 384);
    }
}
