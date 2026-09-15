// Embedding Commands - Tauri IPC handlers for embedding generation

use crate::embeddings::{EmbeddingGenerator, generator::EmbeddingModelInfo};
use crate::document::types::DocumentChunk;

/// Initialize the embedding model
/// 
/// Downloads and loads the all-MiniLM-L6-v2 model (~80MB on first run)
#[tauri::command]
pub async fn init_embedding_model() -> Result<String, String> {
    // Run in blocking task to avoid blocking the async runtime
    tokio::task::spawn_blocking(|| {
        EmbeddingGenerator::init()
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))??;
    
    Ok("Embedding model loaded successfully".to_string())
}

/// Check if the embedding model is initialized
#[tauri::command]
pub async fn is_embedding_model_initialized() -> Result<bool, String> {
    Ok(EmbeddingGenerator::is_initialized())
}

/// Get embedding model information
#[tauri::command]
pub async fn get_embedding_model_info() -> Result<EmbeddingModelInfo, String> {
    Ok(EmbeddingGenerator::get_model_info())
}

/// Generate embedding for a single text
#[tauri::command]
pub async fn generate_embedding(text: String) -> Result<Vec<f32>, String> {
    tokio::task::spawn_blocking(move || {
        EmbeddingGenerator::embed_text(&text)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Generate embeddings for multiple texts (faster with batching)
#[tauri::command]
pub async fn generate_embeddings_batch(texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
    tokio::task::spawn_blocking(move || {
        EmbeddingGenerator::embed_batch(texts)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Generate embeddings for document chunks
/// 
/// Returns a list of (chunk_id, embedding) pairs
#[tauri::command]
pub async fn embed_chunks(chunks: Vec<DocumentChunk>) -> Result<Vec<ChunkEmbedding>, String> {
    tokio::task::spawn_blocking(move || {
        // Extract text from chunks
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        
        // Generate embeddings
        let embeddings = EmbeddingGenerator::embed_batch(texts)?;
        
        // Pair chunk data with embeddings
        let result: Vec<ChunkEmbedding> = chunks.iter()
            .zip(embeddings.iter())
            .map(|(chunk, emb)| ChunkEmbedding {
                chunk_id: chunk.id.clone(),
                document_id: chunk.document_id.clone(),
                embedding: emb.clone(),
            })
            .collect();
        
        Ok(result)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Get the embedding dimensions
#[tauri::command]
pub async fn get_embedding_dimensions() -> Result<usize, String> {
    Ok(EmbeddingGenerator::get_dimensions())
}

/// Chunk embedding result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkEmbedding {
    pub chunk_id: String,
    pub document_id: String,
    pub embedding: Vec<f32>,
}
