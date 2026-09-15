// Text Extractor - Handles plain text files
// Supports: TXT, MD, CSV, LOG, HTML, XML, JSON, YAML, TOML

use crate::document::types::*;
use chrono::Utc;
use std::fs;
use std::path::Path;
use uuid::Uuid;

pub struct TextExtractor;

impl TextExtractor {
    /// Check if this extractor supports the given extension
    pub fn supports(extension: &str) -> bool {
        matches!(
            extension.to_lowercase().as_str(),
            "txt" | "md" | "csv" | "log" | "html" | "htm" |
            "xml" | "json" | "yaml" | "yml" | "toml" | "ini" | "cfg"
        )
    }
    
    /// Extract text from a text-based file
    pub fn extract(path: &Path) -> Result<Document, DocumentError> {
        // Read entire file as UTF-8
        let content = fs::read_to_string(path)
            .map_err(|e| DocumentError::ExtractionError(format!("Failed to read file: {}", e)))?;
        
        let metadata_fs = fs::metadata(path)
            .map_err(|e| DocumentError::IoError(format!("Cannot read metadata: {}", e)))?;
        
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let file_type = match extension.as_str() {
            "json" | "xml" | "yaml" | "yml" | "csv" | "toml" => FileType::Data,
            _ => FileType::Text,
        };
        
        let line_count = content.lines().count();
        
        // For text files, treat entire content as one "page"
        let page = Page {
            number: 1,
            text: content.clone(),
            char_count: content.len(),
            line_count: Some(line_count),
            language: None,
        };
        
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        Ok(Document {
            id: Uuid::new_v4().to_string(),
            filename,
            path: path.to_path_buf(),
            file_type,
            pages: vec![page],
            total_pages: 1,
            metadata: DocumentMetadata {
                size_bytes: metadata_fs.len(),
                extension,
                is_code: false,
                requires_ocr: false,
                extraction_time_ms: 0,
            },
            created_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_supports() {
        assert!(TextExtractor::supports("txt"));
        assert!(TextExtractor::supports("md"));
        assert!(TextExtractor::supports("json"));
        assert!(!TextExtractor::supports("pdf"));
        assert!(!TextExtractor::supports("py"));
    }
}
