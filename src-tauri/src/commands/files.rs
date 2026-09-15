// File commands - Secure file operations

use crate::models::{FileMetadata, ValidationResult};
use crate::utils::paths;
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;

/// Allowed file extensions for document processing
const ALLOWED_EXTENSIONS: &[&str] = &["pdf", "txt", "docx", "doc", "md", "rtf"];

/// Maximum file size in bytes (100 MB)
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Open a native file dialog and return the selected file path
#[tauri::command]
pub async fn open_file_dialog(_allowed_extensions: Vec<String>) -> Result<Option<String>, String> {
    info!("Opening file dialog");
    
    // The actual dialog is handled by the dialog plugin from the frontend
    // The actual dialog is handled by the dialog plugin from the frontend
    Ok(None)
}

/// Read metadata for a file
#[tauri::command]
pub async fn read_file_metadata(path: String) -> Result<FileMetadata, String> {
    info!("Reading file metadata: {}", path);
    
    let file_path = PathBuf::from(&path);
    
    // Validate path exists
    if !file_path.exists() {
        return Err(format!("File not found: {}", path));
    }
    
    // Get file metadata
    let metadata = std::fs::metadata(&file_path)
        .map_err(|e| format!("Failed to read file metadata: {}", e))?;
    
    if !metadata.is_file() {
        return Err("Path is not a file".to_string());
    }
    
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    
    let extension = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let size_bytes = metadata.len();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
    
    let modified = metadata.modified()
        .map(|t| {
            let datetime: chrono::DateTime<chrono::Utc> = t.into();
            datetime.format("%Y-%m-%d %H:%M:%S").to_string()
        })
        .unwrap_or_else(|_| "Unknown".to_string());
    
    Ok(FileMetadata {
        name: file_name,
        path: path.clone(),
        extension,
        size_bytes,
        size_mb,
        modified_date: modified,
        is_readable: true,
    })
}

/// Validate a file for processing
#[tauri::command]
pub async fn validate_file(path: String) -> Result<ValidationResult, String> {
    info!("Validating file: {}", path);
    
    let file_path = PathBuf::from(&path);
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    
    // Check file exists
    if !file_path.exists() {
        errors.push("File does not exist".to_string());
        return Ok(ValidationResult {
            is_valid: false,
            path: path.clone(),
            errors,
            warnings,
        });
    }
    
    // Check it's a file, not a directory
    let metadata = match std::fs::metadata(&file_path) {
        Ok(m) => m,
        Err(e) => {
            errors.push(format!("Cannot read file: {}", e));
            return Ok(ValidationResult {
                is_valid: false,
                path: path.clone(),
                errors,
                warnings,
            });
        }
    };
    
    if !metadata.is_file() {
        errors.push("Path is a directory, not a file".to_string());
        return Ok(ValidationResult {
            is_valid: false,
            path: path.clone(),
            errors,
            warnings,
        });
    }
    
    // Check extension
    let extension = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    if !ALLOWED_EXTENSIONS.contains(&extension.as_str()) {
        errors.push(format!(
            "File type '.{}' is not supported. Allowed: {}",
            extension,
            ALLOWED_EXTENSIONS.join(", ")
        ));
    }
    
    // Check file size
    let size = metadata.len();
    if size > MAX_FILE_SIZE {
        errors.push(format!(
            "File is too large: {:.1}MB (maximum: {}MB)",
            size as f64 / (1024.0 * 1024.0),
            MAX_FILE_SIZE / (1024 * 1024)
        ));
    }
    
    if size == 0 {
        warnings.push("File is empty".to_string());
    }
    
    // Check for path traversal attacks
    if path.contains("..") {
        errors.push("Invalid path: directory traversal not allowed".to_string());
    }
    
    // Try to read the file to verify it's not corrupted
    match std::fs::File::open(&file_path) {
        Ok(_) => {}
        Err(e) => {
            errors.push(format!("Cannot open file: {}", e));
        }
    }
    
    Ok(ValidationResult {
        is_valid: errors.is_empty(),
        path: path.clone(),
        errors,
        warnings,
    })
}

/// Copy a file to secure storage with a UUID filename
#[tauri::command]
pub async fn copy_to_secure_storage(source_path: String) -> Result<String, String> {
    info!("Copying file to secure storage: {}", source_path);
    
    let source = PathBuf::from(&source_path);
    
    // Validate source file
    if !source.exists() {
        return Err("Source file does not exist".to_string());
    }
    
    // Get the documents directory
    let docs_dir = paths::get_documents_dir()
        .map_err(|e| format!("Failed to get documents directory: {}", e))?;
    
    // Ensure directory exists
    std::fs::create_dir_all(&docs_dir)
        .map_err(|e| format!("Failed to create documents directory: {}", e))?;
    
    // Generate new filename with UUID
    let extension = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");
    
    let new_filename = format!("{}.{}", Uuid::new_v4(), extension);
    let dest_path = docs_dir.join(&new_filename);
    
    // Copy the file
    std::fs::copy(&source, &dest_path)
        .map_err(|e| format!("Failed to copy file: {}", e))?;
    
    info!("File copied to: {:?}", dest_path);
    
    Ok(dest_path.to_string_lossy().to_string())
}
