// Vector Store Types - Data structures for the vector database

use serde::{Deserialize, Serialize};

/// Search result from similarity query
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    /// Chunk ID
    pub chunk_id: String,
    /// Chunk text content
    pub text: String,
    /// Source page number (if applicable)
    pub page_number: Option<i32>,
    /// Parent document ID
    pub document_id: String,
    /// Document filename
    pub document_name: String,
    /// File type (e.g., "pdf", "docx", "code")
    pub file_type: String,
    /// Similarity score (0-1, higher is better)
    pub similarity: f32,
    /// Document type for domain-specific handling (e.g., "legal", "code", "general")
    #[serde(default)]
    pub doc_type: Option<String>,
    /// Section title if detected
    #[serde(default)]
    pub section_title: Option<String>,
    /// LKOS Retrieval Source: "vector" | "keyword" | "hybrid" | "entity"
    #[serde(default)]
    pub match_source: Option<String>,
    /// LKOS Positional Authority multiplier applied to this chunk
    #[serde(default)]
    pub authority_score: Option<f32>,
}

/// Search filters for narrowing hybrid search results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchFilters {
    /// Filter by document type (e.g., "legal", "code", "general")
    #[serde(default)]
    pub doc_type: Option<String>,
    /// Filter by specific document IDs
    #[serde(default)]
    pub document_ids: Option<Vec<String>>,
    /// Filter by section title
    #[serde(default)]
    pub section_title: Option<String>,
}

/// Document information from the vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentInfo {
    /// Document ID
    pub id: String,
    /// Original filename
    pub filename: String,
    /// Full path to the file
    pub path: String,
    /// File type (e.g., "pdf", "docx", "code")
    pub file_type: String,
    /// Total pages/sections
    pub total_pages: i32,
    /// File size in bytes
    pub size_bytes: i64,
    /// When the document was indexed
    pub created_at: String,
    /// Number of chunks indexed
    pub chunk_count: i32,
    /// Document type for domain-specific handling (e.g., "legal", "code", "general")
    #[serde(default)]
    pub doc_type: Option<String>,
    /// Pre-generated LLM summary of the document
    #[serde(default)]
    pub summary: Option<String>,
    /// Processing/readiness state: "indexing" | "ready" | "summarizing" | "complete"
    #[serde(default)]
    pub readiness_state: Option<String>,
    /// Number of sections identified in the document
    #[serde(default)]
    pub section_count: Option<i32>,
    /// Timestamp when the summary was generated
    #[serde(default)]
    pub summary_generated_at: Option<String>,
}

/// Chunk record stored in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkRecord {
    /// Chunk ID
    pub id: String,
    /// Parent document ID
    pub document_id: String,
    /// Chunk text content
    pub text: String,
    /// Source page number (if applicable)
    pub page_number: Option<i32>,
    /// Index within document
    pub chunk_index: i32,
    /// Token count (approximate)
    pub token_count: Option<i32>,
    /// Programming language (for code chunks)
    pub language: Option<String>,
}

/// Statistics about the vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorStoreStats {
    /// Total number of indexed documents
    pub document_count: i32,
    /// Total number of indexed chunks
    pub chunk_count: i32,
    /// Database file size in bytes
    pub database_size_bytes: u64,
}
