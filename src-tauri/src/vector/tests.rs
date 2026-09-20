// Test file for RAG enhancements
// Run with: cargo test --package privacythink --lib rag::prompts::tests

#[cfg(test)]
mod integration_tests {
    use crate::vector::types::{SearchFilters, SearchResult};
    use super::*;

    #[test]
    fn test_hybrid_search_integration() {
        // This test would require a full database setup
        // For now, we verify the types compile correctly
        let filters = SearchFilters {
            doc_type: Some("legal".to_string()),
            document_ids: None,
            section_title: None,
        };
        
        assert_eq!(filters.doc_type, Some("legal".to_string()));
    }

    #[test]
    fn test_search_result_has_new_fields() {
        let result = SearchResult {
            chunk_id: "test".to_string(),
            text: "test text".to_string(),
            page_number: Some(1),
            document_id: "doc1".to_string(),
            document_name: "test.pdf".to_string(),
            file_type: "pdf".to_string(),
            similarity: 0.95,
            doc_type: Some("legal".to_string()),
            section_title: Some("Introduction".to_string()),
        };

        assert_eq!(result.doc_type, Some("legal".to_string()));
        assert_eq!(result.section_title, Some("Introduction".to_string()));
    }

    #[test]
    fn test_lkos_e2e_pipeline() {
        use crate::rag::KnowledgeObject;
        use crate::vector::store::VectorStore;
        use crate::document::types::{Document, DocumentChunk, FileType, DocumentMetadata};

        // 1. Initialize global or in-memory store
        let store = VectorStore::global().expect("Failed to get store");
        let store = store.lock();

        // 2. Prepare test document & chunk
        let doc = Document {
            id: "doc_e2e_1".to_string(),
            filename: "Contract_Agreement_2026.pdf".to_string(),
            path: std::path::PathBuf::from("/tmp/test.pdf"),
            file_type: FileType::Pdf,
            pages: vec![],
            total_pages: 5,
            metadata: DocumentMetadata {
                size_bytes: 10240,
                extension: "pdf".to_string(),
                is_code: false,
                requires_ocr: false,
                extraction_time_ms: 50,
            },
            created_at: chrono::Utc::now(),
        };

        let text = "SECTION 1. EXECUTIVE SUMMARY\nBilal signed the contract with Google Deepmind in 2026 for $50,000. \
The agreement covers Advanced Agentic Coding and LKOS Architecture.";

        let chunk = DocumentChunk {
            id: "chunk_e2e_1".to_string(),
            document_id: "doc_e2e_1".to_string(),
            chunk_index: 0,
            text: text.to_string(),
            char_count: text.len(),
            source_page: Some(1),
        };

        let ko = KnowledgeObject::extract(&chunk.id, text, 0, 5);
        assert!(ko.authority_score > 1.0, "Intro chunk should have authority_score > 1.0");

        // 3. Add document + chunk
        let dummy_embedding = vec![0.1f32; 384];
        let _ = store.add_document(&doc, &[chunk], &[dummy_embedding]);

        // 4. Store knowledge and index entities
        let _ = store.store_chunk_knowledge("chunk_e2e_1", &ko.to_json());
        let _ = store.index_chunk_entities("chunk_e2e_1", "doc_e2e_1", &ko.entities, &ko.entity_types);

        // 5. Verify FTS5 keyword search
        let fts_results = store.fts_search_only("Google Deepmind", 5, &SearchFilters::default())
            .expect("FTS search failed");
        assert!(!fts_results.is_empty());

        // 6. Verify Entity Search
        let entity_results = store.search_by_entity("Google", 5).expect("Entity search failed");
        assert!(!entity_results.is_empty());

        // 7. Verify Document Entities lookup
        let doc_entities = store.get_top_entities_for_document("doc_e2e_1", 10).expect("Get doc entities failed");
        assert!(!doc_entities.is_empty(), "Document should have extracted entities");
    }
}
