// Document Commands - Tauri IPC handlers for document extraction
//
// These commands enable the frontend to:
// - Extract text from various document formats
// - Get list of supported file types
// - Chunk documents for RAG pipelines

use crate::document::{
    extract_document, get_supported_extensions, is_supported, Document, DocumentChunk,
    DocumentError,
};
use std::path::PathBuf;
use tracing::{info, warn};

/// Extract text from a document file
/// 
/// # Arguments
/// * `file_path` - Absolute path to the file
/// 
/// # Returns
/// * `Document` - The extracted document with pages and metadata
#[tauri::command]
pub async fn extract_document_text(file_path: String) -> Result<Document, String> {
    info!("Extracting document: {}", file_path);
    
    let path = PathBuf::from(&file_path);
    
    // Perform extraction in a blocking task to not block async runtime
    let result = tokio::task::spawn_blocking(move || extract_document(&path))
        .await
        .map_err(|e| format!("Task failed: {}", e))?;
    
    match result {
        Ok(doc) => {
            info!(
                "Successfully extracted: {} ({} pages, {} chars)",
                doc.filename,
                doc.total_pages,
                doc.total_chars()
            );
            Ok(doc)
        }
        Err(e) => {
            warn!("Extraction failed for {}: {}", file_path, e);
            Err(e.to_string())
        }
    }
}

/// Get list of all supported file extensions
#[tauri::command]
pub fn get_supported_formats() -> Vec<String> {
    get_supported_extensions()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Check if a file extension is supported
#[tauri::command]
pub fn is_format_supported(extension: String) -> bool {
    is_supported(&extension)
}

/// Response for extract_and_chunk_document containing both document and chunks
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractAndChunkResult {
    pub document: Document,
    pub chunks: Vec<DocumentChunk>,
}

/// Chunk a document's text for RAG processing
/// 
/// # Arguments
/// * `file_path` - Path to the document
/// * `chunk_size` - Target chunk size in characters (default: 800)
/// * `overlap` - Overlap between chunks in characters (default: 100)
/// 
/// # Returns
/// Both the extracted document and its chunks for indexing
#[tauri::command]
pub async fn extract_and_chunk_document(
    file_path: String,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
) -> Result<ExtractAndChunkResult, String> {
    use crate::document::chunker::{chunk_document, ChunkConfig};
    
    let path = PathBuf::from(&file_path);
    let config = ChunkConfig {
        chunk_size: chunk_size.unwrap_or(800),
        overlap: overlap.unwrap_or(100),
        preserve_code_blocks: true,
    };
    
    // Extract document
    let document = tokio::task::spawn_blocking(move || extract_document(&path))
        .await
        .map_err(|e| format!("Task failed: {}", e))?
        .map_err(|e: DocumentError| e.to_string())?;
    
    // Chunk the document
    let chunks = chunk_document(&document, &config);
    
    info!(
        "Document '{}' chunked into {} pieces",
        document.filename,
        chunks.len()
    );
    
    Ok(ExtractAndChunkResult { document, chunks })
}

/// Get document preview (first N characters)
#[tauri::command]
pub async fn get_document_preview(
    file_path: String,
    max_chars: Option<usize>,
) -> Result<String, String> {
    let path = PathBuf::from(&file_path);
    let limit = max_chars.unwrap_or(500);
    
    let document = tokio::task::spawn_blocking(move || extract_document(&path))
        .await
        .map_err(|e| format!("Task failed: {}", e))?
        .map_err(|e: DocumentError| e.to_string())?;
    
    let full_text = document.full_text();
    let preview: String = full_text.chars().take(limit).collect();
    
    Ok(if full_text.len() > limit {
        format!("{}...", preview)
    } else {
        preview
    })
}

/// Batch extract multiple documents
#[tauri::command]
pub async fn extract_multiple_documents(
    file_paths: Vec<String>,
) -> Vec<Result<Document, String>> {
    let mut results = Vec::new();
    
    for file_path in file_paths {
        let path = PathBuf::from(&file_path);
        let result = tokio::task::spawn_blocking(move || extract_document(&path))
            .await
            .map_err(|e| format!("Task failed: {}", e))
            .and_then(|r| r.map_err(|e| e.to_string()));
        
        results.push(result);
    }
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_supported_formats() {
        let formats = get_supported_formats();
        assert!(formats.contains(&"txt".to_string()));
        assert!(formats.contains(&"py".to_string()));
        assert!(formats.contains(&"pdf".to_string()));
    }
    
    #[test]
    fn test_is_format_supported() {
        assert!(is_format_supported("txt".to_string()));
        assert!(is_format_supported("py".to_string()));
        assert!(!is_format_supported("exe".to_string()));
    }
}
