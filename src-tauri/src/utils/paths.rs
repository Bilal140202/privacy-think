// Path utilities - App data directory handling

use std::path::PathBuf;
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum PathError {
    #[error("Could not find app data directory")]
    NoAppDataDir,
    
    #[error("Failed to create directory: {0}")]
    CreateDirError(#[from] std::io::Error),
}

/// Application name used for data directory
const APP_NAME: &str = "PrivacyThink";

/// Get the base app data directory path
/// On Windows: %APPDATA%/PrivacyThink
pub fn get_app_data_dir() -> Result<PathBuf, PathError> {
    let base = dirs::data_dir().ok_or(PathError::NoAppDataDir)?;
    Ok(base.join(APP_NAME))
}

/// Ensure the app data directory exists, creating it if necessary
pub fn ensure_app_data_dir() -> Result<PathBuf, PathError> {
    let path = get_app_data_dir()?;
    
    if !path.exists() {
        info!("Creating app data directory: {:?}", path);
        std::fs::create_dir_all(&path)?;
    }
    
    // Also create subdirectories
    let subdirs = ["documents", "logs", "cache", "config", "models"];
    for subdir in subdirs {
        let subpath = path.join(subdir);
        if !subpath.exists() {
            std::fs::create_dir_all(&subpath)?;
        }
    }
    
    Ok(path)
}

/// Get the documents directory path
pub fn get_documents_dir() -> Result<PathBuf, PathError> {
    let base = get_app_data_dir()?;
    Ok(base.join("documents"))
}

/// Get the models directory path
pub fn get_models_dir() -> Result<PathBuf, PathError> {
    let base = get_app_data_dir()?;
    Ok(base.join("models"))
}

/// Get the logs directory path
pub fn get_logs_dir() -> Result<PathBuf, PathError> {
    let base = get_app_data_dir()?;
    Ok(base.join("logs"))
}

/// Get the cache directory path
pub fn get_cache_dir() -> Result<PathBuf, PathError> {
    let base = get_app_data_dir()?;
    Ok(base.join("cache"))
}

/// Get the config directory path
pub fn get_config_dir() -> Result<PathBuf, PathError> {
    let base = get_app_data_dir()?;
    Ok(base.join("config"))
}

/// Validate a path to prevent directory traversal attacks
pub fn validate_path(path: &str) -> bool {
    // Reject paths with parent directory references
    if path.contains("..") {
        return false;
    }
    
    // Reject paths that try to escape the app directory
    let normalized = PathBuf::from(path);
    if let Ok(canonical) = normalized.canonicalize() {
        if let Ok(app_dir) = get_app_data_dir() {
            return canonical.starts_with(&app_dir);
        }
    }
    
    true // Allow paths outside app dir for user-selected files
}

/// Sanitize a filename by removing potentially dangerous characters
pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '-' || *c == '_' || *c == ' ')
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("test.pdf"), "test.pdf");
        assert_eq!(sanitize_filename("test<script>.pdf"), "testscript.pdf");
        assert_eq!(sanitize_filename("  test  "), "test");
    }
}
