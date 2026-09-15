// LKOS Background Knowledge Pipeline
// Phases 2 + 3 — Runs after document indexing completes.
//
// Pipeline stages (all async, fire-and-forget):
//   1. Section detection (regex, zero-cost)
//   2. Knowledge extraction — entities, keywords, authority score (Phase 3)
//   3. Entity indexing into entity_index table (Phase 5)
//   4. Mark document as "ready"
//   5. LLM summary generation → mark "complete" (Phase 2, optional)

use tauri::{AppHandle, Emitter};
use tracing::{info, warn, error};
use crate::vector::store::VectorStore;
use crate::rag::KnowledgeObject;
use crate::llm::inference::InferenceEngine;
use crate::llm::model_loader::ModelLoader;
use crate::llm::types::InferenceConfig;

/// Entry point: spawned as a fire-and-forget Tokio task from `index_document`.
pub async fn run_background_pipeline(doc_id: String, app_handle: AppHandle) {
    info!("LKOS background pipeline starting for doc: {}", doc_id);

    // ── Stage 1: Section detection ─────────────────────────────────────
    let section_count = detect_and_store_sections(&doc_id).await.unwrap_or(0);
    info!("Detected {} sections in doc: {}", section_count, doc_id);

    // ── Stage 2+3: Knowledge extraction + Entity indexing (Phase 3 + 5) ─
    if let Err(e) = extract_and_store_knowledge(&doc_id).await {
        warn!("Knowledge extraction failed for doc {}: {}", doc_id, e);
        // Non-fatal — pipeline continues
    } else {
        info!("Knowledge extraction complete for doc: {}", doc_id);
    }

    // ── Stage 4: Mark document as "ready" ──────────────────────────────
    {
        let store = match VectorStore::global() {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to access vector store: {}", e);
                return;
            }
        };
        let _ = store.lock().update_document_readiness(&doc_id, "ready");
    }
    let _ = app_handle.emit("document-ready", doc_id.clone());
    let _ = app_handle.emit("document-status", (doc_id.clone(), "ready"));
    info!("Doc marked ready: {}", doc_id);

    // ── Stage 5: LLM summary (skipped if no model loaded) ──────────────
    if !ModelLoader::is_model_loaded() {
        info!("No LLM model loaded — skipping summary generation for doc: {}", doc_id);
        return;
    }

    // Transition to "summarizing"
    {
        let store = match VectorStore::global() {
            Ok(s) => s,
            Err(e) => { error!("Store access failed: {}", e); return; }
        };
        let _ = store.lock().update_document_readiness(&doc_id, "summarizing");
    }
    let _ = app_handle.emit("document-status", (doc_id.clone(), "summarizing"));

    // Run LLM summary
    match generate_document_summary(&doc_id).await {
        Ok(summary) => {
            let store = match VectorStore::global() {
                Ok(s) => s,
                Err(e) => { error!("Store access failed: {}", e); return; }
            };
            let sl = store.lock();
            if let Err(e) = sl.update_document_summary(&doc_id, &summary) {
                error!("Failed to save summary for doc {}: {}", doc_id, e);
            } else {
                let _ = sl.update_document_readiness(&doc_id, "complete");
                info!("Summary saved for doc: {}", doc_id);
                let _ = app_handle.emit("document-summary-ready", doc_id.clone());
                let _ = app_handle.emit("document-status", (doc_id.clone(), "complete"));
            }
        }
        Err(e) => {
            warn!("Summary generation failed for doc {}: {}", doc_id, e);
            let store = VectorStore::global().ok();
            if let Some(s) = store {
                let _ = s.lock().update_document_readiness(&doc_id, "ready");
            }
            let _ = app_handle.emit("document-status", (doc_id.clone(), "ready"));
        }
    }
}

/// Stage 1 — Section detection via heading heuristics.
/// Annotates section_title on each chunk in-place.
async fn detect_and_store_sections(doc_id: &str) -> Result<i32, String> {
    let store = VectorStore::global()?;
    let chunks = {
        let sl = store.lock();
        sl.get_chunks(doc_id)?
    };

    let heading_keywords: &[&str] = &[
        "section", "chapter", "introduction", "conclusion", "summary",
        "background", "appendix", "abstract", "overview", "methodology",
        "results", "discussion", "references",
    ];

    let mut section_count: i32 = 0;
    let mut current_section: Option<String> = None;

    for chunk in &chunks {
        let mut found_heading: Option<String> = None;
        for line in chunk.text.lines().take(5) {
            let trimmed = line.trim();
            if trimmed.len() < 4 || trimmed.len() > 120 {
                continue;
            }
            let lower = trimmed.to_lowercase();
            let is_all_caps = trimmed.chars().all(|c| !c.is_alphabetic() || c.is_uppercase())
                && trimmed.chars().any(|c| c.is_alphabetic());
            let ends_with_colon = trimmed.ends_with(':');
            let starts_with_keyword = heading_keywords.iter().any(|k| lower.starts_with(k));

            if is_all_caps || ends_with_colon || starts_with_keyword {
                found_heading = Some(trimmed.to_string());
                section_count += 1;
                break;
            }
        }

        if found_heading.is_some() {
            current_section = found_heading;
        }

        if let Some(sec) = &current_section {
            let sl = store.lock();
            let _ = sl.update_chunk_section(&chunk.id, sec);
        }
    }

    {
        let sl = store.lock();
        let _ = sl.update_document_section_count(doc_id, section_count);
    }

    Ok(section_count)
}

/// Stage 2+3 — KnowledgeObject extraction and entity indexing.
/// For each chunk: extracts entities/keywords/authority_score, stores as JSON, indexes entities.
async fn extract_and_store_knowledge(doc_id: &str) -> Result<(), String> {
    let store = VectorStore::global()?;
    let chunks = {
        let sl = store.lock();
        sl.get_chunks_for_knowledge(doc_id)?
    };

    let total = chunks.len() as i32;
    if total == 0 {
        return Ok(());
    }

    info!("Extracting knowledge from {} chunks in doc: {}", total, doc_id);

    for (idx, chunk) in chunks.iter().enumerate() {
        let ko = KnowledgeObject::extract(&chunk.id, &chunk.text, idx as i32, total);

        let json = ko.to_json();
        let entities = ko.entities.clone();
        let entity_types = ko.entity_types.clone();

        {
            let sl = store.lock();
            // Store KnowledgeObject JSON
            if let Err(e) = sl.store_chunk_knowledge(&chunk.id, &json) {
                warn!("Failed to store knowledge for chunk {}: {}", chunk.id, e);
            }
            // Index entities for cross-document search (Phase 5)
            if !entities.is_empty() {
                if let Err(e) = sl.index_chunk_entities(&chunk.id, doc_id, &entities, &entity_types) {
                    warn!("Failed to index entities for chunk {}: {}", chunk.id, e);
                }
            }
        }
    }

    Ok(())
}

/// Stage 5 — Generate an LLM summary from the document's first 5 chunks.
async fn generate_document_summary(doc_id: &str) -> Result<String, String> {
    let store = VectorStore::global()?;

    let chunks = {
        let sl = store.lock();
        sl.get_document_chunks(doc_id, 5)?
    };

    if chunks.is_empty() {
        return Err("No chunks available for summary".to_string());
    }

    let mut doc_text = String::with_capacity(4096);
    for chunk in &chunks {
        doc_text.push_str(&chunk.text);
        doc_text.push_str("\n\n");
    }

    // UTF-8 safe truncation: walk char boundary instead of raw byte index
    let truncated = if doc_text.len() > 3000 {
        let boundary = doc_text.char_indices()
            .map(|(i, _)| i)
            .filter(|&i| i <= 3000)
            .next_back()
            .unwrap_or(0);
        &doc_text[..boundary]
    } else {
        &doc_text
    };

    let prompt = format!(
        "SYSTEM: You are a document analysis assistant. Write a dense 3-4 sentence summary of the following document. Be factual and specific. Do not add commentary. Always respond in English.\nUSER:\nDOCUMENT:\n{}\n\nTASK: Provide a clear, concise 3-4 sentence summary of the document above.",
        truncated
    );

    let config = InferenceConfig {
        temperature: 0.2,
        max_tokens: 256,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        stop_sequences: vec!["###".to_string(), "\n\n\n".to_string()],
        context_size: None,  // auto-detect from RAM
    };


    info!("Generating LLM summary (prompt: {} chars)", prompt.len());
    let summary = InferenceEngine::generate_text(&prompt, &config)
        .await
        .map_err(|e| format!("Inference error: {:?}", e))?;

    if summary.trim().is_empty() {
        return Err("LLM returned empty summary".to_string());
    }

    Ok(summary.trim().to_string())
}
