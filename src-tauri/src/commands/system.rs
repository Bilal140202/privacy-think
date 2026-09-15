// System commands - System information and health checks

use crate::models::{AppInfo, SystemCheck};
use crate::utils::paths;
use sysinfo::System;
use tracing::{debug, info};

/// Simple health check command
#[tauri::command]
pub fn ping() -> String {
    debug!("Ping received");
    "pong".to_string()
}

/// Get application information
#[tauri::command]
pub fn get_app_info() -> Result<AppInfo, String> {
    info!("Getting app info");
    
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let data_dir = paths::get_app_data_dir()
        .map_err(|e| format!("Failed to get data directory: {}", e))?;
    
    let total_memory_gb = sys.total_memory() as f64 / 1_073_741_824.0; // Convert to GB
    
    Ok(AppInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        os: std::env::consts::OS.to_string(),
        os_version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
        arch: std::env::consts::ARCH.to_string(),
        data_directory: data_dir.to_string_lossy().to_string(),
        available_ram_gb: total_memory_gb,
    })
}

/// Check if system meets minimum requirements
#[tauri::command]
pub async fn check_requirements() -> Result<SystemCheck, String> {
    tokio::task::spawn_blocking(|| {
        info!("Checking system requirements");
        
        let mut sys = System::new_all();
        sys.refresh_all();
        
        // Get RAM in GB
        let ram_gb = sys.total_memory() as f64 / 1_073_741_824.0;
        
        // Get CPU cores
        let cpu_cores = sys.cpus().len() as u32;
        
        // Get free disk space (for the app data drive)
        let free_disk_gb = get_free_disk_space_gb().unwrap_or(0.0);
        
        // Check Windows version
        let os_version = System::os_version().unwrap_or_else(|| "Unknown".to_string());
        let is_windows_10_plus = check_windows_version(&os_version);
        
        // Minimum requirements (7.5GB allows for hardware-reserved memory on 8GB systems)
        let min_ram_gb = 7.5;
        let min_cpu_cores = 4;
        let min_free_disk_gb = 5.0;
        
        let meets_requirements = ram_gb >= min_ram_gb 
            && cpu_cores >= min_cpu_cores 
            && free_disk_gb >= min_free_disk_gb
            && is_windows_10_plus;
        
        let mut warnings = Vec::new();
        
        if ram_gb < min_ram_gb {
            warnings.push(format!("Insufficient RAM: {:.1}GB (minimum: 8GB)", ram_gb));
        }
        if cpu_cores < min_cpu_cores {
            warnings.push(format!("Insufficient CPU cores: {} (minimum: {})", cpu_cores, min_cpu_cores));
        }
        if free_disk_gb < min_free_disk_gb {
            warnings.push(format!("Insufficient disk space: {:.1}GB (minimum: {}GB)", free_disk_gb, min_free_disk_gb));
        }
        if !is_windows_10_plus {
            warnings.push(format!("Windows 10 or later required (detected: {})", os_version));
        }
        
        Ok(SystemCheck {
            meets_requirements,
            ram_gb,
            cpu_cores,
            free_disk_gb,
            os_version,
            is_windows_10_plus,
            warnings,
        })
    }).await.map_err(|e| format!("Task failed: {}", e))?
}

/// Get the application data path, creating it if it doesn't exist
#[tauri::command]
pub async fn get_app_data_path() -> Result<String, String> {
    tokio::task::spawn_blocking(|| {
        info!("Getting app data path");
        
        let path = paths::ensure_app_data_dir()
            .map_err(|e| format!("Failed to create app data directory: {}", e))?;
        
        Ok(path.to_string_lossy().to_string())
    }).await.map_err(|e| format!("Task failed: {}", e))?
}

/// Get detailed storage breakdown of app data and system disk
#[tauri::command]
pub async fn get_storage_breakdown() -> Result<crate::models::StorageBreakdown, String> {
    tokio::task::spawn_blocking(|| {
        let app_data_dir = paths::get_app_data_dir()
            .map_err(|e| format!("Failed to get data directory: {}", e))?;

        let models_dir = app_data_dir.join("models");
        let documents_dir = app_data_dir.join("documents");
        let cache_dir = app_data_dir.join("cache");
        let logs_dir = app_data_dir.join("logs");

        let models_size_bytes = get_dir_size_bytes(&models_dir);
        let documents_size_bytes = get_dir_size_bytes(&documents_dir);
        let cache_size_bytes = get_dir_size_bytes(&cache_dir) + get_dir_size_bytes(&logs_dir);

        // Vector DB files: vector.db, vector.db-wal, vector.db-shm
        let mut vector_db_size_bytes = 0u64;
        for db_file in &["vector.db", "vector.db-wal", "vector.db-shm"] {
            let p = app_data_dir.join(db_file);
            if let Ok(meta) = std::fs::metadata(p) {
                vector_db_size_bytes += meta.len();
            }
        }

        let total_app_size_bytes = models_size_bytes + documents_size_bytes + vector_db_size_bytes + cache_size_bytes;

        // Drive statistics
        let (free_disk_gb, total_disk_gb) = get_disk_capacity_gb();

        Ok(crate::models::StorageBreakdown {
            models_size_bytes,
            documents_size_bytes,
            vector_db_size_bytes,
            cache_size_bytes,
            total_app_size_bytes,
            free_disk_gb,
            total_disk_gb,
        })
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Clear application cache and log files
#[tauri::command]
pub async fn clear_app_cache() -> Result<u64, String> {
    tokio::task::spawn_blocking(|| {
        let app_data_dir = paths::get_app_data_dir()
            .map_err(|e| format!("Failed to get data directory: {}", e))?;

        let cache_dir = app_data_dir.join("cache");
        let logs_dir = app_data_dir.join("logs");

        let mut bytes_freed = 0u64;

        for dir in &[cache_dir, logs_dir] {
            if dir.exists() {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Ok(meta) = std::fs::metadata(&path) {
                            let len = meta.len();
                            if path.is_file() {
                                if std::fs::remove_file(&path).is_ok() {
                                    bytes_freed += len;
                                }
                            } else if path.is_dir() {
                                if std::fs::remove_dir_all(&path).is_ok() {
                                    bytes_freed += len;
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Cleared {} bytes from app cache and logs", bytes_freed);
        Ok(bytes_freed)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Helper function to compute total recursive size of a directory
fn get_dir_size_bytes(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total_size = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Ok(meta) = std::fs::metadata(&entry_path) {
                    total_size += meta.len();
                }
            } else if entry_path.is_dir() {
                total_size += get_dir_size_bytes(&entry_path);
            }
        }
    }
    total_size
}

/// Get free and total disk space in GB for the system drive
fn get_disk_capacity_gb() -> (f64, f64) {
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    for disk in disks.list() {
        let mount = disk.mount_point().to_string_lossy();
        if mount.starts_with("C:") || mount == "/" {
            let free = disk.available_space() as f64 / 1_073_741_824.0;
            let total = disk.total_space() as f64 / 1_073_741_824.0;
            return (free, total);
        }
    }
    if let Some(disk) = disks.list().first() {
        let free = disk.available_space() as f64 / 1_073_741_824.0;
        let total = disk.total_space() as f64 / 1_073_741_824.0;
        return (free, total);
    }
    (0.0, 0.0)
}

/// Get free disk space in GB for the system drive
fn get_free_disk_space_gb() -> Option<f64> {
    let (free, _) = get_disk_capacity_gb();
    if free > 0.0 {
        Some(free)
    } else {
        None
    }
}

/// Check if Windows version is 10 or later
/// Handles various version string formats:
/// - "10.0.19045" (standard format)
/// - "11 (26200)" (Windows 11 insider format)
/// - "10" or "11" (simple format)
fn check_windows_version(version: &str) -> bool {
    // Try to extract the major version number
    let version_trimmed = version.trim();
    
    // First, try parsing formats like "11 (26200)" or "10 (19045)"
    if let Some(first_part) = version_trimmed.split_whitespace().next() {
        if let Ok(major_num) = first_part.parse::<u32>() {
            return major_num >= 10;
        }
    }
    
    // Try parsing formats like "10.0.19045"
    if let Some(major) = version_trimmed.split('.').next() {
        if let Ok(major_num) = major.parse::<u32>() {
            return major_num >= 10;
        }
    }
    
    // If version contains "10" or "11" anywhere, assume it's valid
    if version.contains("10") || version.contains("11") {
        return true;
    }
    
    false
}

