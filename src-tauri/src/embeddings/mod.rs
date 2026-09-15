// Embeddings Module - Local embedding generation for PrivacyThink
//
// Uses fastembed with the all-MiniLM-L6-v2 model for generating
// 384-dimensional embeddings locally. The model is downloaded
// automatically on first use (~80MB).
//
// Features:
// - Singleton pattern for model initialization
// - Batch processing for efficient embedding generation
// - Thread-safe access via mutex

pub mod generator;

pub use generator::EmbeddingGenerator;
