// Vector Module - SQLite-based vector store for PrivacyThink
//
// Provides persistent storage for document embeddings with
// cosine similarity search functionality.
//
// Database is stored at %APPDATA%/PrivacyThink/vector.db

pub mod store;
pub mod types;
#[cfg(test)]
pub mod tests;

pub use store::VectorStore;
pub use types::{SearchResult, DocumentInfo, ChunkRecord};
