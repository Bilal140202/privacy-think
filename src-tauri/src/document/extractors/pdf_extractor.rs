// PDF Extractor - Extracts text from PDF documents
// Uses pdf-extract crate for text extraction

use crate::document::types::*;
use chrono::Utc;
use std::fs;
use std::path::Path;
use uuid::Uuid;

pub struct PdfExtractor;

impl PdfExtractor {
    /// Check if this extractor supports the given extension
    pub fn supports(extension: &str) -> bool {
        extension.to_lowercase() == "pdf"
    }
    
    /// Check if a page appears to be scanned (minimal text)
    fn is_likely_scanned(text: &str) -> bool {
        // If page has less than 50 non-whitespace characters, likely scanned
        text.chars().filter(|c| !c.is_whitespace()).count() < 50
    }
    
    /// Extract text from a PDF file
    #[cfg(feature = "pdf")]
    pub fn extract(path: &Path) -> Result<Document, DocumentError> {
        use lopdf::Document as LopdfDocument;
        
        let metadata_fs = fs::metadata(path)
            .map_err(|e| DocumentError::IoError(format!("Cannot read metadata: {}", e)))?;

        // Load PDF document
        let doc = LopdfDocument::load(path)
            .map_err(|e| DocumentError::ExtractionError(format!("Failed to load PDF: {}", e)))?;
        
        let mut pages = Vec::new();
        let page_ids = doc.get_pages();
        
        for (page_num, _page_id) in page_ids.iter() {
            let page_text = doc.extract_text(&[*page_num as u32])
                .unwrap_or_default();
            
            pages.push(Page {
                number: *page_num as usize,
                text: page_text.clone(),
                char_count: page_text.len(),
                line_count: Some(page_text.lines().count()),
                language: None,
            });
        }
        
        if pages.is_empty() {
            pages.push(Page {
                number: 1,
                text: String::new(),
                char_count: 0,
                line_count: Some(0),
                language: None,
            });
        }
        
        let total_pages = pages.len();
        let requires_ocr = pages.iter().any(|p| Self::is_likely_scanned(&p.text));
        
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        Ok(Document {
            id: Uuid::new_v4().to_string(),
            filename,
            path: path.to_path_buf(),
            file_type: FileType::Pdf,
            pages,
            total_pages,
            metadata: DocumentMetadata {
                size_bytes: metadata_fs.len(),
                extension: "pdf".to_string(),
                is_code: false,
                requires_ocr,
                extraction_time_ms: 0,
            },
            created_at: Utc::now(),
        })
    }
    
    /// Fallback extraction without pdf-extract feature
    #[cfg(not(feature = "pdf"))]
    pub fn extract(path: &Path) -> Result<Document, DocumentError> {
        let metadata_fs = fs::metadata(path)
            .map_err(|e| DocumentError::IoError(format!("Cannot read metadata: {}", e)))?;
        
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Return placeholder document - PDF extraction not available
        let placeholder_text = format!(
            "[PDF extraction not available]\n\n\
            File: {}\n\
            Size: {} bytes\n\n\
            To enable PDF extraction, add pdf-extract to your dependencies.",
            filename,
            metadata_fs.len()
        );
        
        let page = Page {
            number: 1,
            text: placeholder_text.clone(),
            char_count: placeholder_text.len(),
            line_count: Some(placeholder_text.lines().count()),
            language: None,
        };
        
        Ok(Document {
            id: Uuid::new_v4().to_string(),
            filename,
            path: path.to_path_buf(),
            file_type: FileType::Pdf,
            pages: vec![page],
            total_pages: 1,
            metadata: DocumentMetadata {
                size_bytes: metadata_fs.len(),
                extension: "pdf".to_string(),
                is_code: false,
                requires_ocr: true,  // Indicate PDF needs processing
                extraction_time_ms: 0,
            },
            created_at: Utc::now(),
        })
    }
}
