// Document Types - Data structures for document extraction

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Represents an extracted document
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    /// Unique identifier (UUID)
    pub id: String,
    /// Original filename
    pub filename: String,
    /// Full path to the file
    pub path: PathBuf,
    /// Type of file
    pub file_type: FileType,
    /// Extracted pages/sections
    pub pages: Vec<Page>,
    /// Total number of pages
    pub total_pages: usize,
    /// Document metadata
    pub metadata: DocumentMetadata,
    /// Extraction timestamp
    pub created_at: DateTime<Utc>,
}

/// Represents a page or section of extracted text
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Page {
    /// Page number (1-indexed)
    pub number: usize,
    /// Extracted text content
    pub text: String,
    /// Character count
    pub char_count: usize,
    /// Line count (for code files)
    pub line_count: Option<usize>,
    /// Programming language (for code files)
    pub language: Option<String>,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentMetadata {
    /// File size in bytes
    pub size_bytes: u64,
    /// File extension (lowercase)
    pub extension: String,
    /// Whether the file is code
    pub is_code: bool,
    /// Whether OCR was or would be required
    pub requires_ocr: bool,
    /// Extraction time in milliseconds
    pub extraction_time_ms: u64,
}

/// File type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum FileType {
    /// PDF document
    Pdf,
    /// Plain text file
    Text,
    /// Microsoft Word document
    Word,
    /// Source code with language name
    Code(String),
    /// Image (requires OCR)
    Image,
    /// Data files (JSON, XML, YAML, CSV)
    Data,
    /// Unknown or unsupported file type
    Unknown,
}

impl Document {
    /// Get the full text content of the document
    pub fn full_text(&self) -> String {
        self.pages
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
    
    /// Get total character count
    pub fn total_chars(&self) -> usize {
        self.pages.iter().map(|p| p.char_count).sum()
    }
    
    /// Get summary suitable for display
    pub fn summary(&self) -> String {
        let chars = self.total_chars();
        let file_type_str = match &self.file_type {
            FileType::Pdf => "PDF".to_string(),
            FileType::Text => "Text".to_string(),
            FileType::Word => "Word".to_string(),
            FileType::Code(lang) => format!("{} Code", lang),
            FileType::Image => "Image (OCR)".to_string(),
            FileType::Data => "Data".to_string(),
            FileType::Unknown => "Unknown".to_string(),
        };
        format!(
            "{} - {} pages, {} chars",
            file_type_str, self.total_pages, chars
        )
    }
}

/// Document chunk for RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentChunk {
    /// Unique chunk identifier
    pub id: String,
    /// Parent document ID
    pub document_id: String,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Chunk text content
    pub text: String,
    /// Character count
    pub char_count: usize,
    /// Source page number (if applicable)
    pub source_page: Option<usize>,
}

/// Error types for document extraction
#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("File too large: {0}")]
    FileTooLarge(String),
    
    #[error("Unsupported file type: {0}")]
    UnsupportedType(String),
    
    #[error("Extraction failed: {0}")]
    ExtractionError(String),
    
    #[error("IO error: {0}")]
    IoError(String),
}

impl From<std::io::Error> for DocumentError {
    fn from(err: std::io::Error) -> Self {
        DocumentError::IoError(err.to_string())
    }
}

impl From<DocumentError> for String {
    fn from(err: DocumentError) -> Self {
        err.to_string()
    }
}
