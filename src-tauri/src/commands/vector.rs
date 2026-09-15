// Vector Store Commands - Tauri IPC handlers for vector database operations

use crate::vector::{VectorStore, SearchResult, DocumentInfo, ChunkRecord};
use crate::vector::types::{VectorStoreStats, SearchFilters};
use crate::document::types::{Document, DocumentChunk};
use crate::embeddings::EmbeddingGenerator;
use tauri::AppHandle;
use tracing::info;

/// Index a document with automatic embedding generation.
/// After indexing, fires the LKOS background pipeline (section detection + summary).
#[tauri::command]
pub async fn index_document(
    document: Document,
    chunks: Vec<DocumentChunk>,
    app_handle: AppHandle,
) -> Result<String, String> {
    let doc_id = document.id.clone();
    let doc_filename = document.filename.clone();

    // Run indexing in spawn_blocking (CPU-bound embedding + SQLite writes)
    tokio::task::spawn_blocking(move || {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = EmbeddingGenerator::embed_batch(texts)?;
        let store = VectorStore::global()?;
        let store = store.lock();
        store.add_document(&document, &chunks, &embeddings)?;
        Ok::<_, String>(())
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))??;

    // Fire-and-forget background LKOS pipeline
    let doc_id_clone = doc_id.clone();
    let app_clone = app_handle.clone();
    tauri::async_runtime::spawn(async move {
        crate::background::run_background_pipeline(doc_id_clone, app_clone).await;
    });

    info!("Indexed document {} and launched background pipeline", doc_filename);
    Ok(format!("Indexed {} with background pipeline started", doc_filename))
}

/// Search for similar chunks using a text query
/// 
/// Generates embedding for query and searches the vector store
#[tauri::command]
pub async fn search_similar_text(
    query: String,
    top_k: Option<usize>,
    doc_ids: Option<Vec<String>>,
) -> Result<Vec<SearchResult>, String> {
    let top_k = top_k.unwrap_or(5);
    
    tokio::task::spawn_blocking(move || {
        // Generate embedding for query
        let query_embedding = EmbeddingGenerator::embed_text(&query)?;
        
        // Search vector store
        let store = VectorStore::global()?;
        let store = store.lock();
        store.search_similar(&query_embedding, top_k, doc_ids)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Search for similar chunks using a pre-computed embedding
#[tauri::command]
pub async fn search_similar_embedding(
    query_embedding: Vec<f32>,
    top_k: Option<usize>,
    doc_ids: Option<Vec<String>>,
) -> Result<Vec<SearchResult>, String> {
    let top_k = top_k.unwrap_or(5);
    
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.search_similar(&query_embedding, top_k, doc_ids)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// List all indexed documents
#[tauri::command]
pub async fn list_indexed_documents() -> Result<Vec<DocumentInfo>, String> {
    tokio::task::spawn_blocking(|| {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.list_documents()
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Get a specific indexed document by ID
#[tauri::command]
pub async fn get_indexed_document(doc_id: String) -> Result<Option<DocumentInfo>, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.get_document(&doc_id)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Get chunks for a specific document
#[tauri::command]
pub async fn get_document_chunks(doc_id: String) -> Result<Vec<ChunkRecord>, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.get_chunks(&doc_id)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Delete an indexed document
#[tauri::command]
pub async fn delete_indexed_document(doc_id: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.delete_document(&doc_id)?;
        Ok(format!("Deleted document {}", doc_id))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Check if a document is indexed
#[tauri::command]
pub async fn is_document_indexed(doc_id: String) -> Result<bool, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.document_exists(&doc_id)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Get vector store statistics
#[tauri::command]
pub async fn get_vector_store_stats() -> Result<VectorStoreStats, String> {
    tokio::task::spawn_blocking(|| {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.get_stats()
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

// ==================== HYBRID SEARCH COMMANDS ====================

/// Hybrid search combining vector similarity and keyword (FTS5) search
/// 
/// This provides +30% accuracy improvement over vector-only search
#[tauri::command]
pub async fn search_with_hybrid(
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: Option<usize>,
    filters: Option<SearchFilters>,
) -> Result<Vec<SearchResult>, String> {
    let top_k = top_k.unwrap_or(5);
    let filters = filters.unwrap_or_default();
    
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.hybrid_search(&query_text, &query_embedding, top_k, &filters)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Assemble smart context from search results
/// 
/// Manages token budget and includes source metadata for better prompts
#[tauri::command]
pub async fn assemble_smart_context(
    chunks: Vec<SearchResult>,
    max_tokens: Option<usize>,
    include_metadata: Option<bool>,
) -> Result<String, String> {
    let max_tokens = max_tokens.unwrap_or(2000);
    let include_metadata = include_metadata.unwrap_or(true);
    
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        Ok(store.assemble_context(chunks, max_tokens, include_metadata))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Get the pre-built summary for a document (LKOS Phase 2)
/// Returns None if the summary hasn't been generated yet
#[tauri::command]
pub async fn get_document_summary(doc_id: String) -> Result<Option<String>, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.get_document_summary(&doc_id)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Update document readiness state (used internally and by background pipeline)
#[tauri::command]
pub async fn get_document_readiness(doc_id: String) -> Result<Option<String>, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        match store.get_document(&doc_id)? {
            Some(doc) => Ok(doc.readiness_state),
            None => Ok(None),
        }
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Result payload for Phase 4 parallel search command
#[derive(serde::Serialize)]
pub struct ParallelSearchResult {
    pub chunks: Vec<SearchResult>,
    pub prebuilt_summary: Option<String>,
}

/// LKOS Phase 4 — Parallel Search Command
///
/// FIX (2026-08-10): Replaced the previous 4 separate `spawn_blocking` calls that all
/// contended for the single `VectorStore` Mutex with a single `spawn_blocking` that acquires
/// the lock once and performs all operations sequentially inside it.
///
/// Root cause of old deadlock: tokio::join! spawned 3 blocking tasks simultaneously. All 3
/// tried to lock the same `parking_lot::Mutex<VectorStore>`. Under load this exhausted the
/// blocking thread pool with threads that were all blocked on each other. The previous code had
/// zero actual parallelism (the Mutex serialized them anyway) but maximum contention overhead.
///
/// New design: single lock acquisition → vector search → FTS search → summary fetch → RRF merge.
/// Lock held for the minimum time. No concurrent contention possible.
#[tauri::command]
pub async fn search_parallel(
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: Option<usize>,
    filters: Option<SearchFilters>,
    doc_id: Option<String>,
) -> Result<ParallelSearchResult, String> {
    let top_k = top_k.unwrap_or(10);
    let filters = filters.unwrap_or_default();

    // Single spawn_blocking: acquire the lock once, do all work, release.
    let (merged_chunks, prebuilt_summary) = tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let sl = store.lock();

        let vector_chunks = sl.search_similar(&query_embedding, top_k, filters.clone().document_ids)?;
        let fts_chunks = sl.fts_search_only(&query_text, top_k, &filters)?;
        let prebuilt_summary = if let Some(id) = doc_id {
            sl.get_document_summary(&id)?
        } else {
            None
        };
        let merged = sl.reciprocal_rank_fusion(vector_chunks, fts_chunks);

        Ok::<_, String>((merged, prebuilt_summary))
    })
    .await
    .map_err(|e| format!("search_parallel task failed: {}", e))??;

    Ok(ParallelSearchResult {
        chunks: merged_chunks,
        prebuilt_summary,
    })
}

/// LKOS Phase 5 — Search chunks by entity name across all documents
#[tauri::command]
pub async fn search_by_entity(entity: String, top_k: Option<usize>) -> Result<Vec<SearchResult>, String> {
    let top_k = top_k.unwrap_or(10);
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.search_by_entity(&entity, top_k)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// LKOS Phase 5 — Get top entities for a document (for UI display)
#[tauri::command]
pub async fn get_document_entities(doc_id: String, limit: Option<usize>) -> Result<Vec<(String, String)>, String> {
    let limit = limit.unwrap_or(5);
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store = store.lock();
        store.get_top_entities_for_document(&doc_id, limit)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

