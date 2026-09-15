// Image Extractor - OCR for images (placeholder implementation)
//
// Full OCR requires Tesseract to be installed on the system.
// This implementation provides a fallback that extracts image metadata
// and flags the file as requiring OCR processing.
//
// To enable full OCR:
// 1. Install Tesseract OCR on the system
// 2. Add tesseract-rs crate to dependencies
// 3. Implement the full OCR extract method

use crate::document::types::*;
use chrono::Utc;
use std::fs;
use std::path::Path;
use uuid::Uuid;
use tracing::info;

pub struct ImageExtractor;

impl ImageExtractor {
    /// Check if this extractor supports the given extension
    pub fn supports(extension: &str) -> bool {
        matches!(
            extension.to_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "tif" | "gif" | "webp"
        )
    }
    
    /// Get image format name from extension
    fn get_format_name(extension: &str) -> &str {
        match extension.to_lowercase().as_str() {
            "png" => "PNG",
            "jpg" | "jpeg" => "JPEG",
            "bmp" => "BMP",
            "tiff" | "tif" => "TIFF",
            "gif" => "GIF",
            "webp" => "WebP",
            _ => "Image",
        }
    }
    
    /// Extract text from an image file (placeholder implementation)
    /// 
    /// This returns metadata about the image and flags it for OCR processing.
    /// Full OCR would require Tesseract integration.
    pub fn extract(path: &Path) -> Result<Document, DocumentError> {
        let metadata_fs = fs::metadata(path)
            .map_err(|e| DocumentError::IoError(format!("Cannot read metadata: {}", e)))?;
        
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let format_name = Self::get_format_name(&extension);
        
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Try to get basic image info (dimensions)
        let image_info = Self::get_image_info(path);
        
        // Create placeholder text with image information
        let placeholder_text = match image_info {
            Some((width, height)) => {
                format!(
                    "[Image: {}]\n\n\
                    Format: {}\n\
                    Dimensions: {}x{} pixels\n\
                    Size: {} bytes\n\n\
                    ⚠️ OCR Not Available\n\
                    To extract text from this image, Tesseract OCR must be installed.\n\n\
                    This image has been flagged for OCR processing. When OCR is available,\n\
                    the text content will be extracted and made searchable.",
                    filename, format_name, width, height, metadata_fs.len()
                )
            }
            None => {
                format!(
                    "[Image: {}]\n\n\
                    Format: {}\n\
                    Size: {} bytes\n\n\
                    ⚠️ OCR Not Available\n\
                    To extract text from this image, Tesseract OCR must be installed.\n\n\
                    This image has been flagged for OCR processing.",
                    filename, format_name, metadata_fs.len()
                )
            }
        };
        
        info!("Image '{}' flagged for OCR ({} bytes)", filename, metadata_fs.len());
        
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
            file_type: FileType::Image,
            pages: vec![page],
            total_pages: 1,
            metadata: DocumentMetadata {
                size_bytes: metadata_fs.len(),
                extension,
                is_code: false,
                requires_ocr: true,  // Flag for future OCR processing
                extraction_time_ms: 0,
            },
            created_at: Utc::now(),
        })
    }
    
    /// Try to get image dimensions by reading file headers
    fn get_image_info(path: &Path) -> Option<(u32, u32)> {
        // Read first bytes to detect format and dimensions
        let bytes = fs::read(path).ok()?;
        
        if bytes.len() < 24 {
            return None;
        }
        
        // PNG: Check for PNG signature and read IHDR chunk
        if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) && bytes.len() >= 24 {
            let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
            let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
            return Some((width, height));
        }
        
        // JPEG: Look for SOF0 marker (0xFF 0xC0)
        if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            let mut i = 2;
            while i < bytes.len() - 9 {
                if bytes[i] == 0xFF {
                    let marker = bytes[i + 1];
                    // SOF0 through SOF3 markers
                    if (0xC0..=0xC3).contains(&marker) {
                        let height = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
                        let width = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]) as u32;
                        return Some((width, height));
                    }
                    // Skip to next marker
                    if marker != 0x00 && marker != 0xFF && i + 3 < bytes.len() {
                        let length = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
                        i += 2 + length;
                        continue;
                    }
                }
                i += 1;
            }
        }
        
        // BMP: Read dimensions from header
        if bytes.starts_with(b"BM") && bytes.len() >= 26 {
            let width = u32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
            let height = u32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]).abs_diff(0);
            return Some((width, height));
        }
        
        // GIF: Read dimensions from header
        if bytes.starts_with(b"GIF") && bytes.len() >= 10 {
            let width = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
            let height = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
            return Some((width, height));
        }
        
        None
    }
    
    /// Check if Tesseract is available on the system (for future use)
    #[allow(dead_code)]
    pub fn is_ocr_available() -> bool {
        // Check if tesseract is in PATH
        std::process::Command::new("tesseract")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_supports() {
        assert!(ImageExtractor::supports("png"));
        assert!(ImageExtractor::supports("jpg"));
        assert!(ImageExtractor::supports("jpeg"));
        assert!(ImageExtractor::supports("bmp"));
        assert!(ImageExtractor::supports("tiff"));
        assert!(ImageExtractor::supports("gif"));
        assert!(ImageExtractor::supports("webp"));
        assert!(!ImageExtractor::supports("txt"));
        assert!(!ImageExtractor::supports("pdf"));
    }
    
    #[test]
    fn test_get_format_name() {
        assert_eq!(ImageExtractor::get_format_name("png"), "PNG");
        assert_eq!(ImageExtractor::get_format_name("jpg"), "JPEG");
        assert_eq!(ImageExtractor::get_format_name("jpeg"), "JPEG");
        assert_eq!(ImageExtractor::get_format_name("bmp"), "BMP");
    }
}
