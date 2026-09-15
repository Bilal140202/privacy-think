// RAG Commands - Tauri IPC handlers for RAG prompt functions

use crate::rag::{detect_document_type, get_prompt, get_prompt_template, PromptTemplate};
use crate::vector::{store::VectorStore, types::SearchResult};

/// Detect the document type based on filename and content sample
/// Returns: "general", "legal", "code", "medical", or "financial"
#[tauri::command]
pub fn detect_doc_type(filename: String, content_sample: String) -> String {
    let doc_type = detect_document_type(&filename, &content_sample);
    doc_type.as_str().to_string()
}

/// Get the optimized prompt for a given document type
/// Fills in the context and question placeholders
#[tauri::command]
pub fn get_optimized_prompt(
    doc_type: String,
    context: String,
    question: String,
) -> String {
    let template = PromptTemplate::parse_str(&doc_type);
    get_prompt(&template, &context, &question)
}

/// Get just the prompt template (without context/question filled in)
#[tauri::command]
pub fn get_prompt_template_raw(doc_type: String) -> String {
    let template = PromptTemplate::parse_str(&doc_type);
    get_prompt_template(&template).to_string()
}

/// Get first N chunks of a document (for summary requests)
/// If document_id is None, gets chunks from all documents
#[tauri::command]
pub async fn get_document_start(
    document_id: Option<String>,
    chunk_count: usize,
) -> Result<Vec<SearchResult>, String> {
    tokio::task::spawn_blocking(move || {
        let store = VectorStore::global()?;
        let store_lock = store.lock();
        
        if let Some(doc_id) = document_id {
            // Get first N chunks of specific document
            store_lock.get_document_chunks(&doc_id, chunk_count)
        } else {
            // Get first N chunks across all documents
            store_lock.get_all_document_starts(chunk_count)
        }
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}
