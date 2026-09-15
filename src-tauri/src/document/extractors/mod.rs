// Document Extractors - Router and module exports

use crate::document::types::*;
use std::path::Path;
use std::time::Instant;
use tracing::info;

pub mod text_extractor;
pub mod code_extractor;
pub mod pdf_extractor;
pub mod docx_extractor;
pub mod image_extractor;

// Re-export extractors
pub use text_extractor::TextExtractor;
pub use code_extractor::CodeExtractor;
pub use pdf_extractor::PdfExtractor;
pub use docx_extractor::DocxExtractor;
pub use image_extractor::ImageExtractor;

/// Maximum file size for extraction (100 MB)
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Supported file extensions (50+ formats)
pub fn get_supported_extensions() -> Vec<&'static str> {
    vec![
        // Text
        "txt", "md", "csv", "log", "html", "htm", "xml", "json", "yaml", "yml", "toml", "ini", "cfg",
        // Code (30+ languages)
        "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "rs", "cpp", "cc", "cxx", "c", "h", "hpp",
        "java", "go", "php", "rb", "swift", "kt", "kts", "scala", "cs", "fs", "dart", "lua", "r",
        "sql", "sh", "bash", "ps1", "bat", "cmd", "zig", "nim", "ex", "exs", "erl", "hs", "ml",
        "vue", "svelte",
        // Documents
        "pdf", "docx",
        // Images (OCR placeholder)
        "png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp",
    ]
}

/// Check if a file extension is supported
pub fn is_supported(extension: &str) -> bool {
    let ext = extension.to_lowercase();
    TextExtractor::supports(&ext) 
        || CodeExtractor::supports(&ext) 
        || PdfExtractor::supports(&ext)
        || DocxExtractor::supports(&ext)
        || ImageExtractor::supports(&ext)
}

/// Main extraction router - detects file type and routes to appropriate extractor
pub fn extract_document(path: &Path) -> Result<Document, DocumentError> {
    let start = Instant::now();
    
    // Validate file exists
    if !path.exists() {
        return Err(DocumentError::FileNotFound(format!("{:?}", path)));
    }
    
    // Get file metadata
    let metadata = std::fs::metadata(path)
        .map_err(|e| DocumentError::IoError(format!("Cannot read file metadata: {}", e)))?;
    
    // Check file size
    if metadata.len() > MAX_FILE_SIZE {
        return Err(DocumentError::FileTooLarge(format!(
            "File is {} MB, max is {} MB",
            metadata.len() / (1024 * 1024),
            MAX_FILE_SIZE / (1024 * 1024)
        )));
    }
    
    // Get extension
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    info!("Extracting document: {:?} (extension: {})", path, extension);
    
    // Route to appropriate extractor based on file type
    let mut document = if CodeExtractor::supports(&extension) {
        CodeExtractor::extract(path)?
    } else if TextExtractor::supports(&extension) {
        TextExtractor::extract(path)?
    } else if PdfExtractor::supports(&extension) {
        PdfExtractor::extract(path)?
    } else if DocxExtractor::supports(&extension) {
        DocxExtractor::extract(path)?
    } else if ImageExtractor::supports(&extension) {
        ImageExtractor::extract(path)?
    } else {
        return Err(DocumentError::UnsupportedType(format!("Unsupported: .{}", extension)));
    };
    
    // Add extraction time
    document.metadata.extraction_time_ms = start.elapsed().as_millis() as u64;
    
    info!(
        "Extracted {} pages, {} chars in {}ms",
        document.total_pages,
        document.total_chars(),
        document.metadata.extraction_time_ms
    );
    
    Ok(document)
}

/// Extract multiple documents from a list of paths
pub fn extract_documents(paths: &[&Path]) -> Vec<Result<Document, DocumentError>> {
    paths.iter().map(|p| extract_document(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_supported() {
        assert!(is_supported("txt"));
        assert!(is_supported("py"));
        assert!(is_supported("pdf"));
        assert!(!is_supported("exe"));
        assert!(!is_supported("dll"));
    }
    
    #[test]
    fn test_get_supported_extensions() {
        let extensions = get_supported_extensions();
        assert!(extensions.contains(&"txt"));
        assert!(extensions.contains(&"py"));
        assert!(extensions.contains(&"pdf"));
    }
}
