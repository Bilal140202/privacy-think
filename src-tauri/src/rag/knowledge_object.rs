// LKOS Phase 3 — KnowledgeObject Abstraction
//
// Enriches every chunk at index-time with structured metadata:
// - Named entities (people, orgs, dates, amounts)
// - Top keywords (TF-based)
// - Authority score (position-based weight for RRF ranking)
//
// This metadata is stored as JSON in chunks.knowledge_json and applied
// at query time to boost high-authority, entity-rich chunks in RRF fusion.

use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use regex::Regex;

// Regexes compiled once at program startup — NOT per chunk call.
// Previously these lived inside extract_entities() which ran per chunk (potentially
// hundreds of times per document), recompiling the same patterns each time.
static YEAR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(1[89]\d{2}|20[0-9]{2})\b").unwrap()
});
static MONEY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\$[\d,]+(?:\.\d+)?[KMBkmb]?").unwrap()
});
static PCT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d+(?:\.\d+)?%").unwrap()
});
static NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20})){1,3}\b").unwrap()
});

/// Enriched knowledge metadata for a single document chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeObject {
    pub chunk_id: String,
    /// Named entities extracted (persons, orgs, dates, amounts)
    pub entities: Vec<String>,
    /// Top content keywords (stopword-filtered)
    pub keywords: Vec<String>,
    /// Authority score multiplier for RRF ranking (1.0 = normal)
    pub authority_score: f32,
    /// Entity types parallel to entities vec
    pub entity_types: Vec<String>,
}

/// English stopwords to filter out from keyword extraction
static STOPWORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "from", "have", "are",
    "was", "were", "been", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "but", "its", "into", "than", "then", "when",
    "where", "which", "who", "whom", "what", "how", "why", "all", "any",
    "each", "more", "also", "such", "their", "they", "them", "these", "those",
    "been", "has", "had", "does", "did", "just", "about", "over", "some",
    "only", "very", "well", "both", "other", "while", "even", "through",
    "after", "before", "between", "during", "under", "within", "without",
    "there", "here", "being", "been", "however", "whether", "although",
];

impl KnowledgeObject {
    /// Extract a KnowledgeObject from chunk text and positional metadata.
    ///
    /// # Arguments
    /// * `chunk_id` - The chunk's unique ID
    /// * `text` - The chunk's raw text
    /// * `chunk_index` - 0-based position in the document
    /// * `total_chunks` - Total number of chunks in the document
    pub fn extract(
        chunk_id: impl Into<String>,
        text: &str,
        chunk_index: i32,
        total_chunks: i32,
    ) -> Self {
        let chunk_id = chunk_id.into();
        let (entities, entity_types) = extract_entities(text);
        let keywords = extract_keywords(text, 10);
        let authority_score = compute_authority_score(chunk_index, total_chunks);

        KnowledgeObject {
            chunk_id,
            entities,
            keywords,
            authority_score,
            entity_types,
        }
    }

    /// Serialize to compact JSON string for database storage.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize from JSON string (from database).
    pub fn from_json(chunk_id: &str, json: &str) -> Result<Self, String> {
        serde_json::from_str::<KnowledgeObject>(json)
            .map_err(|e| format!("Failed to parse knowledge JSON for {}: {}", chunk_id, e))
    }
}

/// Extract named entities using regex patterns.
/// Returns (entities, entity_types) in parallel order.
/// Uses module-level Lazy<Regex> statics — zero per-call compilation cost.
fn extract_entities(text: &str) -> (Vec<String>, Vec<String>) {
    let mut entities = Vec::new();
    let mut types = Vec::new();

    // Extract years (1800-2099)
    for m in YEAR_REGEX.find_iter(text) {
        let y = m.as_str().to_string();
        if !entities.contains(&y) {
            entities.push(y);
            types.push("date".to_string());
        }
    }

    // Extract monetary amounts ($X, $XK, $XM, $XB)
    for m in MONEY_REGEX.find_iter(text) {
        let amt = m.as_str().to_string();
        if !entities.contains(&amt) {
            entities.push(amt);
            types.push("amount".to_string());
        }
    }

    // Extract percentages
    for m in PCT_REGEX.find_iter(text) {
        let pct = m.as_str().to_string();
        if !entities.contains(&pct) {
            entities.push(pct);
            types.push("amount".to_string());
        }
    }

    // Extract multi-word capitalized phrases (likely named entities: people, orgs, places)
    for m in NAME_REGEX.find_iter(text) {
        let name = m.as_str().trim().to_string();
        if name.len() < 4 || is_likely_false_positive(&name) {
            continue;
        }
        if !entities.contains(&name) {
            entities.push(name);
            types.push("person_or_org".to_string());
        }
    }

    // Cap at 20 entities
    entities.truncate(20);
    types.truncate(20);

    (entities, types)
}

/// Determine if a capitalized phrase is likely a false positive
fn is_likely_false_positive(name: &str) -> bool {
    let lower = name.to_lowercase();
    let sentence_starters = [
        "this study", "the study", "this paper", "the paper", "the results",
        "this work", "the work", "the data", "this method", "the model",
        "the system", "the process", "the analysis", "the report", "the document",
        "the following", "the above", "the below", "the first", "the last",
    ];
    sentence_starters.iter().any(|s| lower.starts_with(s))
}

/// Extract top-N keywords from text using term frequency (TF).
/// Filters stopwords and short tokens.
fn extract_keywords(text: &str, top_n: usize) -> Vec<String> {
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for word in text.split(|c: char| !c.is_alphabetic()) {
        let w = word.to_lowercase();
        if w.len() < 4 {
            continue;
        }
        if STOPWORDS.contains(&w.as_str()) {
            continue;
        }
        *freq.entry(w).or_insert(0) += 1;
    }

    let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
    ranked.sort_by_key(|b| std::cmp::Reverse(b.1));
    ranked.into_iter().take(top_n).map(|(w, _)| w).collect()
}

/// Compute authority score based on chunk position in document.
///
/// - First 3 chunks (intro): 1.25x — high authority (definitions, scope)
/// - Last 10% of chunks (conclusion): 1.15x — high authority (findings, summary)
/// - Middle: 1.0x — neutral
fn compute_authority_score(chunk_index: i32, total_chunks: i32) -> f32 {
    if total_chunks <= 0 {
        return 1.0;
    }
    if chunk_index < 3 {
        return 1.25;
    }
    let tail_start = (total_chunks as f32 * 0.9) as i32;
    if chunk_index >= tail_start {
        return 1.15;
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_extraction_years() {
        let text = "In 2023 OpenAI released GPT-4. By 2024 it had millions of users.";
        let (entities, types) = extract_entities(text);
        assert!(entities.contains(&"2023".to_string()));
        assert!(entities.contains(&"2024".to_string()));
        let year_idx = entities.iter().position(|e| e == "2023").unwrap();
        assert_eq!(types[year_idx], "date");
    }

    #[test]
    fn test_entity_extraction_money() {
        let text = "The deal was worth $5.2M and profit grew by 15%.";
        let (entities, _) = extract_entities(text);
        assert!(entities.contains(&"$5.2M".to_string()));
        assert!(entities.contains(&"15%".to_string()));
    }

    #[test]
    fn test_keyword_extraction() {
        let text = "machine learning models require large datasets for training purposes";
        let keywords = extract_keywords(text, 5);
        assert!(keywords.contains(&"machine".to_string()) || keywords.contains(&"learning".to_string()));
    }

    #[test]
    fn test_authority_score() {
        assert!((compute_authority_score(0, 20) - 1.25).abs() < 0.01);
        assert!((compute_authority_score(10, 20) - 1.0).abs() < 0.01);
        assert!((compute_authority_score(18, 20) - 1.15).abs() < 0.01);
    }

    #[test]
    fn test_round_trip_json() {
        let ko = KnowledgeObject::extract("chunk_1", "OpenAI released GPT-4 in 2023 for $20/month.", 0, 10);
        let json = ko.to_json();
        let restored = KnowledgeObject::from_json("chunk_1", &json).unwrap();
        assert_eq!(restored.chunk_id, "chunk_1");
        assert!((restored.authority_score - 1.25).abs() < 0.01);
    }
}
