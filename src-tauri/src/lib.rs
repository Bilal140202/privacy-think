// PrivacyThink - Privacy-first, local-only document analysis
// Main library module

pub mod commands;
pub mod models;
pub mod utils;
pub mod llm;
pub mod document;
pub mod embeddings;
pub mod vector;
pub mod rag;
pub mod background;

use tauri::Manager;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize logging for the application (writes to both stdout and %APPDATA%/PrivacyThink/logs/app.log)
fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "privacythink=debug,tauri=info".into());

    let app_log_dir = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("PrivacyThink")
        .join("logs");

    let _ = std::fs::create_dir_all(&app_log_dir);
    let log_file_path = app_log_dir.join("app.log");

    if let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_file_path)
    {
        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Arc::new(file));

        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .with(file_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }
}

/// Load a Windows ICO file and extract the best resolution for taskbar display
/// Windows taskbar prefers 32x32 icons, but we'll take the best available
fn load_ico_icon(ico_bytes: &[u8]) -> Option<tauri::image::Image<'static>> {
    use std::io::Cursor;
    let cursor = Cursor::new(ico_bytes);
    let icon_dir = ico::IconDir::read(cursor).ok()?;
    
    // Find the best icon entry (prefer 32x32 for taskbar, otherwise largest)
    let best_entry = icon_dir.entries().iter()
        .max_by_key(|e| {
            let size = e.width();
            // Prefer 32x32 for taskbar, give it highest priority
            if size == 32 { u32::MAX } else { size }
        })?;
    
    let image = best_entry.decode().ok()?;
    let rgba = image.rgba_data();
    let width = image.width();
    let height = image.height();
    
    Some(tauri::image::Image::new_owned(rgba.to_vec(), width, height))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_logging();
    info!("Starting PrivacyThink v{}", env!("CARGO_PKG_VERSION"));

    // Ensure app data directory exists on startup
    if let Err(e) = utils::paths::ensure_app_data_dir() {
        eprintln!("Failed to create app data directory: {}", e);
    }
    
    // Ensure models directory exists on startup
    if let Err(e) = llm::ModelLoader::ensure_models_dir() {
        eprintln!("Failed to create models directory: {}", e);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // Focus the main window when a second instance is launched
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
                let _ = window.show();
            }
        }))
        .setup(|app| {
            // Setup system tray
            use tauri::menu::{MenuBuilder, MenuItemBuilder};
            use tauri::tray::TrayIconBuilder;

            let show_hide = MenuItemBuilder::with_id("show_hide", "Show/Hide").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            
            let menu = MenuBuilder::new(app)
                .item(&show_hide)
                .separator()
                .item(&quit)
                .build()?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .on_menu_event(|app, event| {
                    match event.id().as_ref() {
                        "show_hide" => {
                            if let Some(window) = app.get_webview_window("main") {
                                if window.is_visible().unwrap_or(false) {
                                    let _ = window.hide();
                                } else {
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
                            }
                        }
                        "quit" => {
                            app.exit(0);
                        }
                        _ => {}
                    }
                })
                .build(app)?;

            // Initialize embedding model in background (downloads ~80MB on first run)
            std::thread::spawn(|| {
                info!("Starting background embedding model initialization...");
                match embeddings::EmbeddingGenerator::init() {
                    Ok(_) => info!("Embedding model initialized successfully"),
                    Err(e) => tracing::warn!("Failed to init embeddings on startup: {} (will retry on first use)", e),
                }
            });

            // Ensure the main window has the correct icon (especially important for frameless windows)
            if let Some(window) = app.get_webview_window("main") {
                // Load icon from embedded .ico bytes (Windows prefers ICO format with multiple sizes)
                let icon_bytes = include_bytes!("../icons/icon.ico");
                match load_ico_icon(icon_bytes) {
                    Some(icon) => {
                        if let Err(e) = window.set_icon(icon) {
                            tracing::warn!("Failed to set window icon: {}", e);
                        } else {
                            info!("Window icon set successfully from ICO file");
                        }
                    }
                    None => {
                        tracing::warn!("Failed to parse ICO file, trying PNG fallback");
                        // Fallback to PNG if ICO parsing fails
                        let png_bytes = include_bytes!("../icons/icon.png");
                        if let Ok(img) = image::load_from_memory(png_bytes) {
                            let rgba = img.to_rgba8();
                            let (width, height) = rgba.dimensions();
                            let icon = tauri::image::Image::new_owned(rgba.into_raw(), width, height);
                            let _ = window.set_icon(icon);
                        }
                    }
                }

                // Explicitly show, unminimize, and focus the window on startup
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }

            info!("PrivacyThink initialized successfully");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // System commands
            commands::system::get_app_info,
            commands::system::check_requirements,
            commands::system::get_app_data_path,
            commands::system::get_storage_breakdown,
            commands::system::clear_app_cache,
            commands::system::ping,
            // File commands
            commands::files::open_file_dialog,
            commands::files::read_file_metadata,
            commands::files::validate_file,
            commands::files::copy_to_secure_storage,
            // LLM commands
            commands::llm::load_model,
            commands::llm::get_loaded_model,
            commands::llm::unload_model,
            commands::llm::get_available_models,
            commands::llm::generate,
            commands::llm::generate_stream,
            commands::llm::cancel_inference,
            commands::llm::get_inference_stats,
            commands::llm::download_model_async,
            commands::llm::cancel_download,
            commands::llm::delete_model,
            commands::llm::check_models,
            // Document extraction commands
            commands::documents::extract_document_text,
            commands::documents::get_supported_formats,
            commands::documents::is_format_supported,
            commands::documents::extract_and_chunk_document,
            commands::documents::get_document_preview,
            commands::documents::extract_multiple_documents,
            // Embedding commands
            commands::embeddings::init_embedding_model,
            commands::embeddings::is_embedding_model_initialized,
            commands::embeddings::get_embedding_model_info,
            commands::embeddings::generate_embedding,
            commands::embeddings::generate_embeddings_batch,
            commands::embeddings::embed_chunks,
            commands::embeddings::get_embedding_dimensions,
            // Vector store commands
            commands::vector::index_document,
            commands::vector::search_similar_text,
            commands::vector::search_similar_embedding,
            commands::vector::list_indexed_documents,
            commands::vector::get_indexed_document,
            commands::vector::get_document_chunks,
            commands::vector::delete_indexed_document,
            commands::vector::is_document_indexed,
            commands::vector::get_vector_store_stats,
            commands::vector::search_with_hybrid,
            commands::vector::assemble_smart_context,
            // RAG prompt commands
            commands::rag::detect_doc_type,
            commands::rag::get_optimized_prompt,
            commands::rag::get_prompt_template_raw,
            commands::rag::get_document_start,
            // LKOS Phase 2, 4 & 5 commands
            commands::vector::get_document_summary,
            commands::vector::get_document_readiness,
            commands::vector::search_parallel,
            commands::vector::search_by_entity,
            commands::vector::get_document_entities,
            // Feedback commands
            commands::feedback::send_feedback,
        ])
        .run(tauri::generate_context!())
        .expect("error while running PrivacyThink application");
}

