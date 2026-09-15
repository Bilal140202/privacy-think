// Document models - Data structures for documents

use serde::{Deserialize, Serialize};

/// Application information returned by get_app_info command
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppInfo {
    pub version: String,
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub data_directory: String,
    pub available_ram_gb: f64,
}

/// System requirements check result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemCheck {
    pub meets_requirements: bool,
    pub ram_gb: f64,
    pub cpu_cores: u32,
    pub free_disk_gb: f64,
    pub os_version: String,
    pub is_windows_10_plus: bool,
    pub warnings: Vec<String>,
}

/// File metadata returned by read_file_metadata command
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileMetadata {
    pub name: String,
    pub path: String,
    pub extension: String,
    pub size_bytes: u64,
    pub size_mb: f64,
    pub modified_date: String,
    pub is_readable: bool,
}

/// File validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationResult {
    pub is_valid: bool,
    pub path: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Document status in the library
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DocumentStatus {
    Pending,
    Processing,
    Ready,
    Error,
}

/// Document entry in the library
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    pub id: String,
    pub name: String,
    pub original_path: String,
    pub secure_path: String,
    pub extension: String,
    pub size_mb: f64,
    pub status: DocumentStatus,
    pub added_date: String,
    pub last_accessed: Option<String>,
}

/// Detailed storage breakdown of app data and system disk
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageBreakdown {
    pub models_size_bytes: u64,
    pub documents_size_bytes: u64,
    pub vector_db_size_bytes: u64,
    pub cache_size_bytes: u64,
    pub total_app_size_bytes: u64,
    pub free_disk_gb: f64,
    pub total_disk_gb: f64,
}

