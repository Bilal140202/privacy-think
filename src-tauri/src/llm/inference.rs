// Inference Engine - Text generation via llama-cli subprocess
// All inference is delegated to llamacpp_subprocess (llama-cli.exe).
// The candle-based in-process path has been removed — it only supported LLaMA
// architecture models and failed for Qwen / Phi / Gemma GGUF files.
//
// ASYNC NOTE: generate_stream and generate_text are now async to avoid the
// spawn_blocking + block_on anti-pattern that caused thread-pool deadlocks.

use crate::llm::types::{InferenceConfig, InferenceStats, LlmError};
use crate::llm::model_loader::MODEL_STATE;
use crate::llm::llamacpp_subprocess;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;
use once_cell::sync::Lazy;

static SYS_STATS: Lazy<Mutex<InferenceStats>> = Lazy::new(|| {
    Mutex::new(InferenceStats {
        tokens_per_second: 0.0,
        total_tokens: 0,
        memory_used_mb: 0.0,
        context_usage: 0.0,
        model_loaded: false,
    })
});

pub struct InferenceEngine;

impl InferenceEngine {
    /// Generate text from a prompt (async, non-streaming).
    /// Internally calls generate_stream and accumulates the result.
    pub async fn generate_text(
        prompt: &str,
        config: &InferenceConfig,
    ) -> Result<String, LlmError> {
        let full_output = Arc::new(Mutex::new(String::new()));
        let output_clone = full_output.clone();
        
        Self::generate_stream(prompt, config, move |token| {
            output_clone.lock().push_str(token);
        })
        .await?;
        
        let result = full_output.lock().clone();
        Ok(result)
    }

    /// Generate text with a streaming callback (async).
    /// Directly awaits run_inference — no spawn_blocking or block_on needed.
    pub async fn generate_stream<F>(
        prompt: &str,
        config: &InferenceConfig,
        callback: F,
    ) -> Result<String, LlmError>
    where
        F: Fn(&str) + Send + Sync,
    {
        let start_time = Instant::now();
        info!("Starting streaming generation via llama-cli subprocess");

        // Ensure a model is loaded (its path is needed for the subprocess call)
        let model_path = {
            let state = MODEL_STATE.read();
            let loaded = state.loaded_model.as_ref().ok_or(LlmError::ModelNotLoaded)?;
            std::path::PathBuf::from(&loaded.info.path)
        };

        // Directly await the async subprocess — no thread pool indirection
        let result = llamacpp_subprocess::run_inference(
            &model_path,
            prompt,
            config,
            callback,
        )
        .await
        .map_err(LlmError::InferenceError)?;

        let elapsed = start_time.elapsed();
        let word_count = result.split_whitespace().count();
        let approx_tokens = (word_count as f32 * 1.33) as u32;
        let tps = if elapsed.as_secs_f32() > 0.0 {
            approx_tokens as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };

        {
            let mut stats = SYS_STATS.lock();
            stats.total_tokens += approx_tokens;
            stats.tokens_per_second = tps;
            stats.context_usage = 0.0; // managed internally by llama-cli
        }

        info!("Generation complete in {:.2}s via llama-cli", elapsed.as_secs_f32());
        Ok(result)
    }

    /// Cancel any currently running generation by killing the llama-cli process.
    pub fn cancel() {
        info!("Cancelling inference — killing llama-cli process");
        llamacpp_subprocess::kill_current_process();
    }

    pub fn get_stats() -> InferenceStats {
        let model_loaded = MODEL_STATE.read().loaded_model.is_some();
        let mut stats = SYS_STATS.lock().clone();
        stats.model_loaded = model_loaded;
        if model_loaded {
            stats.memory_used_mb = MODEL_STATE.read().loaded_model.as_ref()
                .map(|m| m.info.size_mb as f32)
                .unwrap_or(0.0);
        } else {
            stats.memory_used_mb = 0.0;
            stats.total_tokens = 0;
            stats.tokens_per_second = 0.0;
            stats.context_usage = 0.0;
        }
        stats
    }

    pub fn clear_cache() {
        info!("Clearing cache — restarting llama-cli keep-alive process");
        llamacpp_subprocess::kill_current_process();
    }
}
