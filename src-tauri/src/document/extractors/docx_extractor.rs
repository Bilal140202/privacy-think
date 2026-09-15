// DOCX Extractor - Extracts text from Microsoft Word documents
//
// DOCX files are ZIP archives containing XML files.
// This extractor reads document.xml to get the text content.

use crate::document::types::*;
use chrono::Utc;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use uuid::Uuid;
use tracing::debug;

pub struct DocxExtractor;

impl DocxExtractor {
    /// Check if this extractor supports the given extension
    pub fn supports(extension: &str) -> bool {
        extension.to_lowercase() == "docx"
    }
    
    /// Extract text from a DOCX file
    pub fn extract(path: &Path) -> Result<Document, DocumentError> {
        let file = File::open(path)
            .map_err(|e| DocumentError::ExtractionError(format!("Failed to open DOCX: {}", e)))?;
        
        let metadata_fs = std::fs::metadata(path)
            .map_err(|e| DocumentError::IoError(format!("Cannot read metadata: {}", e)))?;
        
        let reader = BufReader::new(file);
        
        // Open DOCX as a ZIP archive
        let mut archive = zip::ZipArchive::new(reader)
            .map_err(|e| DocumentError::ExtractionError(format!("Failed to open DOCX archive: {}", e)))?;
        
        // Find and read document.xml
        let mut document_xml = String::new();
        
        // Try to find document.xml 
        let doc_path = "word/document.xml";
        match archive.by_name(doc_path) {
            Ok(mut file) => {
                file.read_to_string(&mut document_xml)
                    .map_err(|e| DocumentError::ExtractionError(format!("Failed to read document.xml: {}", e)))?;
            }
            Err(e) => {
                return Err(DocumentError::ExtractionError(format!(
                    "Could not find document.xml in DOCX: {}", e
                )));
            }
        }
        
        debug!("Read document.xml: {} bytes", document_xml.len());
        
        // Parse XML and extract text
        let text = Self::extract_text_from_xml(&document_xml)?;
        
        let line_count = text.lines().count();
        
        let page = Page {
            number: 1,
            text: text.clone(),
            char_count: text.len(),
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
            file_type: FileType::Word,
            pages: vec![page],
            total_pages: 1,
            metadata: DocumentMetadata {
                size_bytes: metadata_fs.len(),
                extension: "docx".to_string(),
                is_code: false,
                requires_ocr: false,
                extraction_time_ms: 0,
            },
            created_at: Utc::now(),
        })
    }
    
    /// Extract text content from Word document XML
    fn extract_text_from_xml(xml: &str) -> Result<String, DocumentError> {
        let mut text = String::new();
        let mut in_text_element = false;
        let mut in_paragraph = false;
        
        // Simple XML parsing without external dependency
        // Look for <w:t> elements which contain the actual text
        let mut chars = xml.chars().peekable();
        let mut current_tag = String::new();
        let mut reading_tag = false;
        
        while let Some(c) = chars.next() {
            match c {
                '<' => {
                    reading_tag = true;
                    current_tag.clear();
                }
                '>' => {
                    reading_tag = false;
                    
                    // Check what tag we just read
                    let tag = current_tag.trim();
                    
                    // Start of text element
                    if tag.starts_with("w:t") && !tag.starts_with("w:t/") {
                        in_text_element = true;
                    }
                    // End of text element
                    else if tag == "/w:t" {
                        in_text_element = false;
                    }
                    // Start of paragraph
                    else if tag.starts_with("w:p") && !tag.ends_with("/") && !tag.starts_with("w:pPr") {
                        in_paragraph = true;
                    }
                    // End of paragraph - add newline
                    else if tag == "/w:p" {
                        if in_paragraph {
                            text.push('\n');
                        }
                        in_paragraph = false;
                    }
                    // Line break
                    else if tag == "w:br" || tag.starts_with("w:br ") {
                        text.push('\n');
                    }
                    // Tab
                    else if tag == "w:tab" {
                        text.push('\t');
                    }
                    
                    current_tag.clear();
                }
                _ => {
                    if reading_tag {
                        current_tag.push(c);
                    } else if in_text_element {
                        // Collect text content
                        text.push(c);
                    }
                }
            }
        }
        
        // Clean up the text
        let cleaned = text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        
        debug!("Extracted {} characters from DOCX", cleaned.len());
        
        Ok(cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_supports() {
        assert!(DocxExtractor::supports("docx"));
        assert!(DocxExtractor::supports("DOCX"));
        assert!(!DocxExtractor::supports("doc"));
        assert!(!DocxExtractor::supports("pdf"));
    }
    
    #[test]
    fn test_extract_text_from_xml() {
        let xml = r#"<?xml version="1.0"?>
            <w:document>
                <w:body>
                    <w:p>
                        <w:r>
                            <w:t>Hello World</w:t>
                        </w:r>
                    </w:p>
                    <w:p>
                        <w:r>
                            <w:t>Second paragraph</w:t>
                        </w:r>
                    </w:p>
                </w:body>
            </w:document>"#;
        
        let text = DocxExtractor::extract_text_from_xml(xml).expect("failed to extract text from XML");
        assert!(text.contains("Hello World"));
        assert!(text.contains("Second paragraph"));
    }
}
