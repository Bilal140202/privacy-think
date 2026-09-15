// Document Module - Universal document extraction for PrivacyThink
//
// This module provides text extraction from various file formats
// for use in RAG (Retrieval-Augmented Generation) pipelines.
//
// Supported formats:
// - Text: TXT, MD, CSV, LOG, HTML, XML, JSON, YAML, TOML
// - Code: Python, JavaScript, TypeScript, Rust, C/C++, Java, Go, etc.
// - Documents: PDF, DOCX (Word)
// - Images: PNG, JPG, BMP, TIFF, GIF (OCR placeholder - requires Tesseract)

pub mod types;
pub mod extractors;
pub mod chunker;

// Re-export main types and functions
pub use types::{Document, DocumentChunk, DocumentError, DocumentMetadata, FileType, Page};
pub use extractors::{extract_document, get_supported_extensions, is_supported};
pub use chunker::{chunk_document, ChunkConfig};
