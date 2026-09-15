// RAG Module - Domain-specific prompts, document type detection, and LKOS knowledge enrichment
//
// Provides intelligent prompt templates based on document content type
// for improved RAG accuracy (+15% improvement).
// Phase 3: KnowledgeObject abstraction for entity/keyword/authority enrichment.

pub mod prompts;
pub mod knowledge_object;

pub use prompts::{detect_document_type, get_prompt, get_prompt_template, PromptTemplate};
pub use knowledge_object::KnowledgeObject;
