// llama.cpp CLI subprocess wrapper
// Runs llama-cli.exe as a one-shot subprocess for inference
// ARCHITECTURE: Prompts are written to a temp file and passed via -f flag,
// NOT as CLI arguments. This avoids Windows 8191-char CLI limit and
// STATUS_STACK_BUFFER_OVERRUN (0xC0000409) crashes with large document prompts.

use std::path::Path;
use std::process::Stdio;
use tokio::io::{BufReader, AsyncBufReadExt, AsyncReadExt};
use tokio::process::{Command, Child};
use tracing::{info, warn};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use sysinfo::System;

use crate::llm::types::InferenceConfig;


/// Strip ANSI escape codes from a string (e.g., [0m, [1;32m, etc.)
fn strip_ansi_codes(s: &str) -> String {
    static ANSI_REGEX: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07").unwrap()
    });
    ANSI_REGEX.replace_all(s, "").to_string()
}

/// Global tracking for the current active inference process
static CURRENT_PROCESS: Lazy<Mutex<Option<Child>>> = Lazy::new(|| Mutex::new(None));

/// Kill any currently running inference process
pub fn kill_current_process() {
    let mut lock = CURRENT_PROCESS.lock();
    if let Some(mut child) = lock.take() {
        info!("Killing current inference process...");
        let _ = child.start_kill();
        tauri::async_runtime::spawn(async move {
            let _ = child.wait().await;
        });
    }
}

/// Get the PID of the currently running inference process
pub fn get_current_pid() -> Option<u32> {
    let lock = CURRENT_PROCESS.lock();
    lock.as_ref().and_then(|child| child.id())
}

#[cfg(target_os = "windows")]
const LLAMA_RUN_NAME: &str = "llama-cli.exe";

#[cfg(not(target_os = "windows"))]
const LLAMA_RUN_NAME: &str = "llama-cli";

pub fn get_llama_cli_path() -> Result<std::path::PathBuf, String> {
    // 1. ./bin/ relative to cwd (dev, same folder as working dir)
    let dev_path = std::path::PathBuf::from("./bin").join(LLAMA_RUN_NAME);
    if dev_path.exists() {
        return Ok(dev_path);
    }
    
    if let Ok(cwd) = std::env::current_dir() {
        // 2. {cwd}/bin/
        let bin_path = cwd.join("bin").join(LLAMA_RUN_NAME);
        if bin_path.exists() {
            return Ok(bin_path);
        }
        // 3. {cwd}/src-tauri/bin/ — standard Tauri sidecar location during dev
        let tauri_bin_path = cwd.join("src-tauri").join("bin").join(LLAMA_RUN_NAME);
        if tauri_bin_path.exists() {
            return Ok(tauri_bin_path);
        }
    }
    
    if let Ok(exe_dir) = std::env::current_exe() {
        if let Some(parent) = exe_dir.parent() {
            // 4. Next to the exe
            let bin_path = parent.join("bin").join(LLAMA_RUN_NAME);
            if bin_path.exists() { return Ok(bin_path); }
            // 5. In resources/bin/ (bundled Tauri release)
            let res_path = parent.join("resources").join("bin").join(LLAMA_RUN_NAME);
            if res_path.exists() { return Ok(res_path); }
            // 6. src-tauri/bin/ relative to exe parent chain (monorepo layout)
            let tauri_rel = parent.join("..").join("src-tauri").join("bin").join(LLAMA_RUN_NAME);
            if tauri_rel.exists() { return Ok(tauri_rel); }
        }
    }
    
    Err(format!(
        "llama-cli not found. Please ensure {} is placed in the bin/ directory next to the executable, or in src-tauri/bin/ during development.",
        LLAMA_RUN_NAME
    ))
}

pub fn is_llama_cli_available() -> bool {
    get_llama_cli_path().is_ok()
}

pub use crate::utils::text::safe_slice;


/// Regex to detect llama.cpp's own progress/loading lines on stdout.
/// These are noise we must skip — but ONLY before the first real output token.
static LLAMA_PROGRESS_LINE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(^\s*load|^\s*llama_|gguf_|build:|llm_load|print_info|n_ctx|n_batch|\[\s*\d+\s*/|^\s*\.+\s*$|^\s*>\s*$)"
    ).unwrap()
});

/// spawn_keep_alive is a no-op in one-shot mode but checks binary availability
pub fn spawn_keep_alive(model_path: &Path, _threads: u8) -> Result<(), String> {
    if !model_path.exists() {
        return Err(format!("Model path does not exist: {:?}", model_path));
    }
    let _ = get_llama_cli_path()?;
    info!("Inference binary validated for model: {:?}", model_path);
    Ok(())
}

/// Compute a safe llama-cli context window size based on total system RAM.
///
/// RAM buckets (conservative — leaves headroom for OS, model weights, embeddings):
///   < 6 GB  → 2048   (minimal; avoids OOM on constrained systems)
///   6–12 GB → 4096   (standard; fits most Q4 models comfortably)
///   ≥ 12 GB → 8192   (extended; allows longer RAG context windows)
///
/// The user can override this via the /settings Performance panel, which passes
/// `context_size: Some(n)` in the InferenceConfig.
fn compute_safe_context_window() -> u32 {
    let mut sys = System::new();
    sys.refresh_memory();
    let ram_gb = sys.total_memory() as f64 / 1_073_741_824.0;
    if ram_gb < 6.0 {
        info!("Context window: 2048 (RAM {:.1}GB < 6GB)", ram_gb);
        2048
    } else if ram_gb < 12.0 {
        info!("Context window: 4096 (RAM {:.1}GB 6-12GB)", ram_gb);
        4096
    } else {
        info!("Context window: 8192 (RAM {:.1}GB >= 12GB)", ram_gb);
        8192
    }
}


/// Write the prompt to a temporary file and return the path.
/// This is critical: passing large prompts as CLI arguments (-p) hits Windows'
/// 8191-char command-line limit and causes STATUS_STACK_BUFFER_OVERRUN (0xC0000409).
/// Using -f (file) has no such limit and handles any UTF-8 content safely.
fn write_prompt_to_tempfile(prompt: &str) -> Result<std::path::PathBuf, String> {
    let temp_dir = std::env::temp_dir().join("privacythink");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to create temp dir: {}", e))?;
    
    let prompt_file = temp_dir.join("llm_prompt.txt");
    
    // Normalize line endings for consistency
    let cleaned = prompt.replace("\r\n", "\n").replace('\r', "\n");
    
    std::fs::write(&prompt_file, cleaned.as_bytes())
        .map_err(|e| format!("Failed to write prompt to temp file: {}", e))?;
    
    info!("Wrote prompt to temp file: {:?} ({} bytes)", prompt_file, cleaned.len());
    Ok(prompt_file)
}

pub async fn run_inference<F>(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    token_callback: F,
) -> Result<String, String>
where
    F: Fn(&str) + Send + Sync,
{
    if !model_path.exists() {
        return Err(format!("Model file not found at: {:?}", model_path));
    }

    let cli_path = get_llama_cli_path()?;

    info!("Spawning llama-cli process...");
    info!("Model: {:?}", model_path);
    info!("Prompt raw length: {} chars", prompt.len());

    // ── Format prompt with model-aware chat template ──────────────────────────
    let templated_prompt = crate::llm::chat_template::format_prompt_for_model(model_path, prompt);
    info!("Templated prompt length: {} chars", templated_prompt.len());

    // ── Write prompt to temp file (avoids Windows CLI arg length crash) ──────
    let prompt_file = write_prompt_to_tempfile(&templated_prompt)?;

    // Prepare full stop sequences (model specific + config)
    let mut active_stop_sequences = crate::llm::chat_template::get_stop_sequences_for_model(model_path);
    for seq in &config.stop_sequences {
        if !active_stop_sequences.contains(seq) {
            active_stop_sequences.push(seq.clone());
        }
    }

    // Kill any existing process first
    kill_current_process();

    // ── Build command ──────────────────────────────────────────────────────────
    // Determine context window: use user override if provided, else auto-detect from RAM.
    let ctx_size = config.context_size.unwrap_or_else(compute_safe_context_window);
    info!("Using context window: {} tokens", ctx_size);

    // Hardware acceleration and thread tuning:
    // Determine CPU threads: leave 1-2 cores for OS / UI responsiveness
    let available_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let threads = if available_cores > 2 { available_cores - 1 } else { available_cores };
    info!("Inference thread allocation: {} threads (of {} detected cores)", threads, available_cores);

    // llama-cli -m <model> -f <prompt_file> -n <max_tokens> -c <ctx> --temp <temp> -ngl 99 -fa -t <threads> -no-cnv --no-display-prompt --simple-io
    let mut cmd = Command::new(&cli_path);
    cmd.arg("-m").arg(model_path);
    cmd.arg("-f").arg(&prompt_file);           // File-based prompt (no CLI length limit)
    cmd.arg("-n").arg(config.max_tokens.to_string());
    cmd.arg("-c").arg(ctx_size.to_string());  // RAM-aware context window (not hardcoded)
    cmd.arg("--temp").arg(config.temperature.to_string());
    cmd.arg("-t").arg(threads.to_string());    // Maximize multi-core CPU performance
    cmd.arg("-ngl").arg("99");                 // Offload all layers to GPU if present (auto falls back to CPU if no GPU)
    cmd.arg("-fa");                            // Flash Attention (drastically faster prompt evaluation)
    cmd.arg("--simple-io");                    // Direct unbuffered I/O for clean token streaming
    cmd.arg("-no-cnv");
    cmd.arg("--no-display-prompt");            // Don't echo the prompt back in stdout

    cmd.stdin(Stdio::null()); // no stdin needed
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn llama-cli: {}", e))?;

    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;

    // Monitor stderr in the background and log everything for diagnostics
    let stderr_handle = tokio::spawn(async move {
        let mut stderr_reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = stderr_reader.next_line().await {
            // Log all stderr — includes model loading progress which is useful
            warn!("llama-cli stderr: {}", line);
        }
    });

    // Track the process globally so it can be cancelled
    {
        let mut lock = CURRENT_PROCESS.lock();
        *lock = Some(child);
    }

    let mut full_output = String::new();
    let mut response_started = false;
    let load_timeout = std::time::Duration::from_secs(120);
    let mut is_timeout = false;

    // Read directly from stdout stream in chunks for real-time fluid streaming
    let mut stdout_stream = stdout;
    let mut buffer = [0u8; 1024];
    let mut startup_buf = String::new();

    loop {
        let per_chunk_timeout = if response_started {
            std::time::Duration::from_secs(30)
        } else {
            load_timeout
        };

        let read_result = tokio::time::timeout(
            per_chunk_timeout,
            stdout_stream.read(&mut buffer),
        ).await;

        match read_result {
            Ok(Ok(0)) => {
                // EOF — subprocess finished
                info!("llama-cli stdout EOF — generation complete");
                break;
            }
            Ok(Ok(n)) => {
                let chunk_str = String::from_utf8_lossy(&buffer[..n]);
                let clean = strip_ansi_codes(&chunk_str);

                if !response_started {
                    startup_buf.push_str(&clean);
                    // Filter out any startup banner/progress lines that might appear on stdout
                    if startup_buf.contains('\n') {
                        let mut lines = startup_buf.split('\n').collect::<Vec<_>>();
                        let remainder = lines.pop().unwrap_or("").to_string();

                        for line in lines {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            if LLAMA_PROGRESS_LINE.is_match(trimmed) {
                                info!("[startup] {}", trimmed);
                                continue;
                            }
                            // Real generation token reached
                            response_started = true;
                            info!("[response started] first token: {:?}", safe_slice(trimmed, 80));

                            let mut line_to_emit = format!("{}\n", line);
                            let mut line_stopped = false;
                            for seq in &active_stop_sequences {
                                if let Some(idx) = line_to_emit.find(seq.as_str()) {
                                    line_to_emit.truncate(idx);
                                    line_stopped = true;
                                }
                            }

                            if !line_to_emit.is_empty() {
                                token_callback(&line_to_emit);
                                full_output.push_str(&line_to_emit);
                            }

                            if line_stopped {
                                info!("Stop sequence detected in startup line — ending generation");
                                break;
                            }
                        }
                        startup_buf = remainder;
                    }
                } else {
                    // Check if chunk contains any stop sequence
                    let mut chunk_to_emit = clean.clone();
                    let mut chunk_stopped = false;

                    for seq in &active_stop_sequences {
                        if let Some(idx) = chunk_to_emit.find(seq.as_str()) {
                            chunk_to_emit.truncate(idx);
                            chunk_stopped = true;
                        }
                    }

                    if !chunk_to_emit.is_empty() {
                        token_callback(&chunk_to_emit);
                        full_output.push_str(&chunk_to_emit);
                    }

                    if chunk_stopped {
                        info!("Stop sequence detected in chunk — ending generation");
                        break;
                    }
                }

                // Check full_output against all active stop sequences
                let mut full_stopped = false;
                for seq in &active_stop_sequences {
                    if let Some(idx) = full_output.find(seq.as_str()) {
                        full_output.truncate(idx);
                        full_stopped = true;
                        break;
                    }
                }

                if full_stopped {
                    info!("Stop sequence detected in full output — ending generation");
                    break;
                }

                // Enforce max_tokens limit
                let word_count = full_output.split_whitespace().count();
                let approx_tokens = (word_count as f32 * 1.33) as u32;
                if approx_tokens >= config.max_tokens {
                    info!("Reached max tokens limit ({}), stopping.", config.max_tokens);
                    break;
                }
            }
            Ok(Err(e)) => {
                return Err(format!("Failed to read stdout: {}", e));
            }
            Err(_) => {
                if !response_started {
                    warn!("Timed out waiting for model to start generating ({}s)", load_timeout.as_secs());
                } else {
                    warn!("Per-token timeout hit after 30s — ending generation with partial output");
                }
                is_timeout = true;
                break;
            }
        }
    }

    // If there is any remaining un-emitted text in startup_buf
    if !startup_buf.is_empty() && !LLAMA_PROGRESS_LINE.is_match(startup_buf.trim()) {
        let mut rem = startup_buf;
        for seq in &active_stop_sequences {
            if let Some(idx) = rem.find(seq.as_str()) {
                rem.truncate(idx);
            }
        }
        if !rem.is_empty() {
            token_callback(&rem);
            full_output.push_str(&rem);
        }
    }

    // Wait for process to exit
    let child_opt = {
        let mut lock = CURRENT_PROCESS.lock();
        lock.take()
    };

    if let Some(mut child) = child_opt {
        // Kill subprocess if it was stopped early by stop sequence or timeout
        let _ = child.start_kill();
        let status = child.wait().await
            .map_err(|e| format!("Failed to wait for process: {}", e))?;
        
        let _ = stderr_handle.await;

        if !status.success() && !is_timeout && full_output.trim().is_empty() {
            warn!("llama-cli exit error status: {:?}", status);
            return Err(format!(
                "Inference failed (exit code {:?}). This may indicate insufficient memory for the model. Try a smaller model or close other applications.",
                status.code()
            ));
        }
    }
    
    // Truncate any trailing stop sequences from the final returned string
    let mut final_output = full_output;
    for seq in &active_stop_sequences {
        if let Some(idx) = final_output.find(seq.as_str()) {
            final_output.truncate(idx);
        }
    }

    // Clean up the temp prompt file
    if let Err(e) = std::fs::remove_file(&prompt_file) {
        warn!("Failed to clean up temp prompt file: {}", e);
    }

    info!("Generation complete, output length: {} chars", final_output.len());
    Ok(final_output.trim().to_string())
}