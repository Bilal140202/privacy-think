// Vector Store - SQLite-based vector database with similarity search
//
// Stores document chunks and their embeddings in SQLite.
// Implements cosine similarity search in Rust for fast retrieval.
//
// Database schema:
// - documents: Document metadata
// - chunks: Chunk text with embeddings stored as binary blobs

use rusqlite::{Connection, params, OptionalExtension};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::collections::HashMap;
use tracing::{info, debug};

use super::types::{SearchResult, DocumentInfo, ChunkRecord, VectorStoreStats, SearchFilters};
use crate::document::types::{Document, DocumentChunk};
use crate::rag::detect_document_type;
use crate::utils::text::safe_slice;


/// Global vector store instance
static VECTOR_STORE: OnceCell<Mutex<VectorStore>> = OnceCell::new();


/// Vector store for document embeddings
pub struct VectorStore {
    conn: Connection,
    db_path: PathBuf,
}

impl VectorStore {
    /// Get the database path
    fn get_db_path() -> Result<PathBuf, String> {
        let app_data = std::env::var("APPDATA")
            .map_err(|_| "APPDATA environment variable not found")?;
        let dir = PathBuf::from(app_data).join("PrivacyThink");
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
        Ok(dir.join("vector.db"))
    }

    /// Create or open the vector database
    pub fn new() -> Result<Self, String> {
        let db_path = Self::get_db_path()?;
        info!("Opening vector database at: {:?}", db_path);
        
        let conn = Connection::open(&db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;
        
        // Enable foreign keys and optimize for performance
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;"
        ).map_err(|e| format!("Failed to set PRAGMAs: {}", e))?;
        
        let store = Self { conn, db_path };
        
        // Initialize schema
        store.init_schema()?;
        
        Ok(store)
    }

    /// Initialize the database schema
    fn init_schema(&self) -> Result<(), String> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                total_pages INTEGER NOT NULL DEFAULT 1,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER,
                language TEXT,
                embedding BLOB NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_number);
            "#
        ).map_err(|e| format!("Failed to initialize schema: {}", e))?;
        
        debug!("Vector database schema initialized");
        
        // Apply RAG enhancements migration (FTS5, metadata columns)
        self.apply_rag_migration()?;
        
        // Apply LKOS Phase 2 migration (summary, readiness_state, section_count)
        self.apply_lkos_migration()?;

        // Apply LKOS Phase 3 migration (knowledge_json per chunk)
        self.apply_knowledge_migration()?;

        // Apply LKOS Phase 5 migration (entity_index table)
        self.apply_entity_index_migration()?;
        
        Ok(())
    }

    /// Check if RAG enhancement migration has been applied
    fn is_rag_migrated(&self) -> bool {
        // Check if migrations table exists and has version 1.1.0
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='migrations')",
            [],
            |row| row.get::<_, bool>(0)
        ).unwrap_or(false) && self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM migrations WHERE version = '1.1.0')",
            [],
            |row| row.get::<_, bool>(0)
        ).unwrap_or(false)
    }

    /// Apply RAG enhancement migration (FTS5, metadata columns)
    fn apply_rag_migration(&self) -> Result<(), String> {
        if self.is_rag_migrated() {
            debug!("RAG migration already applied, skipping");
            return Ok(());
        }

        info!("Applying RAG enhancements migration (v1.1.0)...");

        // Add new columns (ignore errors if they already exist)
        let _ = self.conn.execute("ALTER TABLE chunks ADD COLUMN doc_type TEXT DEFAULT 'general'", []);
        let _ = self.conn.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT", []);
        let _ = self.conn.execute("ALTER TABLE chunks ADD COLUMN char_count INTEGER", []);
        let _ = self.conn.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'general'", []);

        // Update existing records with defaults
        let _ = self.conn.execute(
            "UPDATE chunks SET char_count = LENGTH(text), doc_type = 'general' WHERE char_count IS NULL",
            []
        );
        let _ = self.conn.execute(
            "UPDATE documents SET doc_type = 'general' WHERE doc_type IS NULL",
            []
        );

        // Create FTS5 virtual table
        self.conn.execute_batch(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                tokenize='porter unicode61'
            );
            "#
        ).map_err(|e| format!("Failed to create FTS table: {}", e))?;

        // Populate FTS index with existing data
        self.conn.execute(
            "INSERT OR IGNORE INTO chunks_fts(chunk_id, text) SELECT id, text FROM chunks WHERE id NOT IN (SELECT chunk_id FROM chunks_fts)",
            []
        ).map_err(|e| format!("Failed to populate FTS: {}", e))?;

        // Create sync triggers
        let _ = self.conn.execute_batch(
            r#"
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(chunk_id, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                UPDATE chunks_fts SET text = new.text WHERE chunk_id = old.id;
            END;
            "#
        );

        // Create indexes
        let _ = self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_type)", []);
        let _ = self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_title)", []);
        let _ = self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_char_count ON chunks(char_count)", []);

        // Create migrations table and record this migration
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT OR IGNORE INTO migrations (version, applied_at) VALUES ('1.1.0', datetime('now'));
            "#
        ).map_err(|e| format!("Failed to record migration: {}", e))?;

        info!("✅ RAG enhancements migration (v1.1.0) completed successfully");
        Ok(())
    }

    /// Check if LKOS Phase 2 migration has been applied
    fn is_lkos_migrated(&self) -> bool {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM migrations WHERE version = '1.2.0')",
            [],
            |row| row.get::<_, bool>(0)
        ).unwrap_or(false)
    }

    /// Apply LKOS Phase 2 migration (summary, readiness_state, section_count)
    fn apply_lkos_migration(&self) -> Result<(), String> {
        if self.is_lkos_migrated() {
            debug!("LKOS migration already applied, skipping");
            return Ok(());
        }

        info!("Applying LKOS Phase 2 migration (v1.2.0)...");

        // Add new columns to documents table (ignore errors if they already exist)
        let _ = self.conn.execute("ALTER TABLE documents ADD COLUMN summary TEXT", []);
        let _ = self.conn.execute("ALTER TABLE documents ADD COLUMN readiness_state TEXT DEFAULT 'indexing'", []);
        let _ = self.conn.execute("ALTER TABLE documents ADD COLUMN section_count INTEGER DEFAULT 0", []);
        let _ = self.conn.execute("ALTER TABLE documents ADD COLUMN summary_generated_at TEXT", []);

        // Update any existing documents to be 'complete' since they are already fully indexed
        let _ = self.conn.execute(
            "UPDATE documents SET readiness_state = 'complete' WHERE readiness_state IS NULL",
            []
        );

        // Record this migration
        self.conn.execute(
            "INSERT OR IGNORE INTO migrations (version, applied_at) VALUES ('1.2.0', datetime('now'))",
            []
        ).map_err(|e| format!("Failed to record LKOS migration: {}", e))?;

        info!("✅ LKOS Phase 2 migration (v1.2.0) completed successfully");
        Ok(())
    }

    /// Check if LKOS Phase 3 (knowledge_json) migration has been applied
    fn is_knowledge_migrated(&self) -> bool {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM migrations WHERE version = '1.3.0')",
            [],
            |row| row.get::<_, bool>(0)
        ).unwrap_or(false)
    }

    /// Apply LKOS Phase 3 migration — adds knowledge_json column to chunks
    fn apply_knowledge_migration(&self) -> Result<(), String> {
        if self.is_knowledge_migrated() {
            debug!("Knowledge migration already applied, skipping");
            return Ok(());
        }
        info!("Applying LKOS Phase 3 migration (v1.3.0)...");
        let _ = self.conn.execute("ALTER TABLE chunks ADD COLUMN knowledge_json TEXT", []);
        self.conn.execute(
            "INSERT OR IGNORE INTO migrations (version, applied_at) VALUES ('1.3.0', datetime('now'))",
            []
        ).map_err(|e| format!("Failed to record Phase 3 migration: {}", e))?;
        info!("✅ LKOS Phase 3 migration (v1.3.0) completed");
        Ok(())
    }

    /// Check if LKOS Phase 5 (entity_index) migration has been applied
    fn is_entity_index_migrated(&self) -> bool {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM migrations WHERE version = '1.4.0')",
            [],
            |row| row.get::<_, bool>(0)
        ).unwrap_or(false)
    }

    /// Apply LKOS Phase 5 migration — creates entity_index table
    fn apply_entity_index_migration(&self) -> Result<(), String> {
        if self.is_entity_index_migrated() {
            debug!("Entity index migration already applied, skipping");
            return Ok(());
        }
        info!("Applying LKOS Phase 5 migration (v1.4.0)...");
        self.conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS entity_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_index(entity COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_entity_doc ON entity_index(document_id);
            CREATE INDEX IF NOT EXISTS idx_entity_chunk ON entity_index(chunk_id);
        "#).map_err(|e| format!("Failed to create entity_index: {}", e))?;
        self.conn.execute(
            "INSERT OR IGNORE INTO migrations (version, applied_at) VALUES ('1.4.0', datetime('now'))",
            []
        ).map_err(|e| format!("Failed to record Phase 5 migration: {}", e))?;
        info!("✅ LKOS Phase 5 migration (v1.4.0) completed");
        Ok(())
    }


    /// Get the global vector store instance
    pub fn global() -> Result<&'static Mutex<VectorStore>, String> {
        VECTOR_STORE.get_or_try_init(|| {
            VectorStore::new().map(Mutex::new)
        })
    }

    /// Add a document with its chunks and embeddings
    pub fn add_document(
        &self,
        doc: &Document,
        chunks: &[DocumentChunk],
        embeddings: &[Vec<f32>],
    ) -> Result<(), String> {
        if chunks.len() != embeddings.len() {
            return Err(format!(
                "Chunks and embeddings count mismatch: {} vs {}", 
                chunks.len(), 
                embeddings.len()
            ));
        }

        // Auto-detect document type based on filename and first chunk content
        let content_sample = chunks.first()
            .map(|c| safe_slice(&c.text, 500))
            .unwrap_or("");

        let doc_type_detected = detect_document_type(&doc.filename, content_sample);
        let doc_type_str = doc_type_detected.as_str();
        debug!("Auto-detected document type: {} for {}", doc_type_str, doc.filename);

        // 1. Handle document removal and creation in the first transaction
        {
            let tx = self.conn.unchecked_transaction()
                .map_err(|e| format!("Initial transaction failed: {}", e))?;

            // Check if document already exists, delete if so
            tx.execute("DELETE FROM documents WHERE id = ?1", params![doc.id])
                .map_err(|e| format!("Failed to delete existing document: {}", e))?;

            // Insert document
        tx.execute(
            "INSERT INTO documents (id, filename, path, file_type, total_pages, size_bytes, created_at, doc_type, readiness_state, section_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'indexing', 0)",
            params![
                doc.id,
                doc.filename,
                doc.path.to_str().unwrap_or(""),
                format!("{:?}", doc.file_type),
                doc.total_pages as i32,
                doc.metadata.size_bytes as i64,
                doc.created_at.to_rfc3339(),
                doc_type_str,
            ],
        ).map_err(|e| format!("Failed to insert document: {}", e))?;

            tx.commit().map_err(|e| format!("Initial commit failed: {}", e))?;
        }

        // 2. Insert chunks in batches to avoid long locks
        const BATCH_SIZE: usize = 100;
        for (batch_idx, chunk_batch) in chunks.chunks(BATCH_SIZE).enumerate() {
            let start_idx = batch_idx * BATCH_SIZE;
            let tx = self.conn.unchecked_transaction()
                .map_err(|e| format!("Batch transaction failed: {}", e))?;

            for (i, chunk) in chunk_batch.iter().enumerate() {
                let embedding = &embeddings[start_idx + i];
                
                // Convert embedding Vec<f32> to bytes (little-endian)
                let emb_bytes: Vec<u8> = embedding.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();

                // Estimate token count (~4 chars per token)
                let token_count = (chunk.text.len() / 4) as i32;
                let char_count = chunk.text.len() as i32;

                // Insert chunk with doc_type and char_count for enhanced filtering
                tx.execute(
                    "INSERT INTO chunks (id, document_id, text, page_number, chunk_index, token_count, language, embedding, doc_type, char_count)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        chunk.id,
                        chunk.document_id,
                        chunk.text,
                        chunk.source_page.map(|p| p as i32),
                        chunk.chunk_index as i32,
                        token_count,
                        Option::<String>::None,
                        emb_bytes,
                        doc_type_str,
                        char_count,
                    ],
                ).map_err(|e| format!("Failed to insert chunk: {}", e))?;
            }

            tx.commit().map_err(|e| format!("Batch commit failed: {}", e))?;
            debug!("Committed batch {} for document {}", batch_idx + 1, doc.id);
        }
        
        info!("Added document {} ({}) with {} chunks to vector store", doc.id, doc_type_str, chunks.len());
        Ok(())
    }

    /// Search for similar chunks using cosine similarity
    pub fn search_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        doc_ids: Option<Vec<String>>,
    ) -> Result<Vec<SearchResult>, String> {
        debug!("Searching for similar chunks, top_k: {}, filtered: {}", top_k, doc_ids.is_some());
        
        // Build query with optional filtering (now includes doc_type, section_title)
        let (query, params): (String, Vec<rusqlite::types::Value>) = if let Some(ids) = doc_ids {
            if ids.is_empty() {
                return Ok(Vec::new());
            }
            
            let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            let sql = format!(
                "SELECT c.id, c.text, c.page_number, c.document_id, d.filename, c.embedding, d.file_type, c.doc_type, c.section_title
                 FROM chunks c
                 JOIN documents d ON c.document_id = d.id
                 WHERE c.document_id IN ({})",
                placeholders
            );
            
            let p = ids.into_iter().map(rusqlite::types::Value::Text).collect();
            (sql, p)
        } else {
            let sql = "SELECT c.id, c.text, c.page_number, c.document_id, d.filename, c.embedding, d.file_type, c.doc_type, c.section_title
                       FROM chunks c
                       JOIN documents d ON c.document_id = d.id".to_string();
            (sql, Vec::new())
        };

        let mut stmt = self.conn.prepare(&query)
            .map_err(|e| format!("Query preparation failed: {}", e))?;

        let params_ref: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p as &dyn rusqlite::ToSql).collect();

        let chunk_iter = stmt.query_map(&params_ref[..], |row| {
            let id: String = row.get(0)?;
            let text: String = row.get(1)?;
            let page_number: Option<i32> = row.get(2)?;
            let document_id: String = row.get(3)?;
            let document_name: String = row.get(4)?;
            let embedding_bytes: Vec<u8> = row.get(5)?;
            let file_type: String = row.get(6)?;
            let doc_type: Option<String> = row.get(7)?;
            let section_title: Option<String> = row.get(8)?;
            
            // Convert bytes back to Vec<f32>
            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(bytes)
                })
                .collect();
            
            Ok((id, text, page_number, document_id, document_name, embedding, file_type, doc_type, section_title))
        }).map_err(|e| format!("Query execution failed: {}", e))?;

        // Calculate similarities
        let mut results = Vec::new();
        
        for chunk_result in chunk_iter {
            let (id, text, page_number, document_id, document_name, embedding, file_type, doc_type, section_title) = 
                chunk_result.map_err(|e| format!("Row reading failed: {}", e))?;
            
            let similarity = Self::cosine_similarity(query_embedding, &embedding);
            results.push((id, text, page_number, document_id, document_name, similarity, file_type, doc_type, section_title));
        }

        debug!("Found {} total chunks in database", results.len());

        // Sort by similarity (descending) and take top_k
        results.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_results: Vec<SearchResult> = results
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, text, page_number, document_id, document_name, similarity, file_type, doc_type, section_title)| {
                debug!("Result: doc='{}', file_type='{}', similarity={:.4}, text='{}'", 
                       document_name, file_type, similarity, safe_slice(&text, 100));

                SearchResult {
                    chunk_id,
                    text,
                    page_number,
                    document_id,
                    document_name,
                    file_type,
                    similarity,
                    doc_type,
                    section_title,
                    match_source: Some("vector".to_string()),
                    authority_score: None,
                }
            })
            .collect();

        info!("Search returned {} results", top_results.len());
        Ok(top_results)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    // ==================== HYBRID SEARCH METHODS ====================

    /// Hybrid search combining vector similarity and keyword (FTS5) search
    /// Uses Reciprocal Rank Fusion to merge results for better accuracy
    pub fn hybrid_search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>, String> {
        info!("Performing hybrid search: query='{}', top_k={}", 
              safe_slice(query_text, 50), top_k);


        // Step 1: Vector similarity search (get 2x results for fusion)
        let vector_results = self.vector_search_with_filters(query_embedding, top_k * 2, filters)?;
        debug!("Vector search returned {} results", vector_results.len());

        // Step 2: Keyword search using FTS5 (get 1x results)
        let keyword_results = self.keyword_search(query_text, top_k, filters)?;
        debug!("Keyword search returned {} results", keyword_results.len());

        // Step 3: Merge using Reciprocal Rank Fusion
        let merged = self.reciprocal_rank_fusion(vector_results, keyword_results);
        debug!("RRF merged to {} unique results", merged.len());

        // Step 4: Return top K
        let final_results: Vec<SearchResult> = merged.into_iter().take(top_k).collect();
        info!("Hybrid search returning {} results", final_results.len());

        Ok(final_results)
    }

    /// Vector search with SearchFilters support
    fn vector_search_with_filters(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>, String> {
        // Build SQL with filters
        let mut sql = String::from(
            "SELECT c.id, c.text, c.page_number, c.document_id, d.filename, c.embedding, d.file_type, c.doc_type, c.section_title
             FROM chunks c
             JOIN documents d ON c.document_id = d.id
             WHERE 1=1"
        );
        let mut params: Vec<String> = Vec::new();

        // Apply doc_type filter
        if let Some(doc_type) = &filters.doc_type {
            sql.push_str(&format!(" AND c.doc_type = ?{}", params.len() + 1));
            params.push(doc_type.clone());
        }

        // Apply document_ids filter
        if let Some(doc_ids) = &filters.document_ids {
            if !doc_ids.is_empty() {
                let placeholders: Vec<String> = doc_ids.iter()
                    .enumerate()
                    .map(|(i, _)| format!("?{}", params.len() + i + 1))
                    .collect();
                sql.push_str(&format!(" AND c.document_id IN ({})", placeholders.join(",")));
                params.extend(doc_ids.clone());
            }
        }

        let mut stmt = self.conn.prepare(&sql)
            .map_err(|e| format!("Vector search prep failed: {}", e))?;

        // Build params for rusqlite
        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let chunk_iter = stmt.query_map(&params_refs[..], |row| {
            let id: String = row.get(0)?;
            let text: String = row.get(1)?;
            let page_number: Option<i32> = row.get(2)?;
            let document_id: String = row.get(3)?;
            let document_name: String = row.get(4)?;
            let embedding_bytes: Vec<u8> = row.get(5)?;
            let file_type: String = row.get(6)?;
            let doc_type: Option<String> = row.get(7)?;
            let section_title: Option<String> = row.get(8)?;

            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(bytes)
                })
                .collect();

            Ok((id, text, page_number, document_id, document_name, embedding, file_type, doc_type, section_title))
        }).map_err(|e| format!("Vector search failed: {}", e))?;

        // Calculate similarities and collect results
        let mut results: Vec<(f32, SearchResult)> = Vec::new();
        for chunk_result in chunk_iter {
            let (id, text, page_number, document_id, document_name, embedding, file_type, doc_type, section_title) =
                chunk_result.map_err(|e| format!("Row read failed: {}", e))?;

            let similarity = Self::cosine_similarity(query_embedding, &embedding);
            results.push((similarity, SearchResult {
                chunk_id: id,
                text,
                page_number,
                document_id,
                document_name,
                file_type,
                similarity,
                doc_type,
                section_title,
                match_source: Some("vector".to_string()),
                authority_score: None,
            }));
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results.into_iter().take(top_k).map(|(_, r)| r).collect())
    }

    /// Keyword search using FTS5 full-text search
    fn keyword_search(
        &self,
        query_text: &str,
        top_k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>, String> {
        // Check if FTS table exists
        let fts_exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts')",
            [],
            |row| row.get(0)
        ).unwrap_or(false);

        if !fts_exists {
            debug!("FTS table not found, skipping keyword search");
            return Ok(Vec::new());
        }

        // Sanitize query for FTS5 (escape special characters, join with OR)
        let sanitized_query = query_text
            .replace(['"', '\''], "")
            .split_whitespace()
            .filter(|w| w.len() > 1) // Skip single chars
            .collect::<Vec<_>>()
            .join(" OR ");

        if sanitized_query.is_empty() {
            return Ok(Vec::new());
        }

        // Build FTS query with filters
        let mut sql = String::from(
            "SELECT c.id, c.text, c.page_number, c.document_id, d.filename, d.file_type, 
                    c.doc_type, c.section_title, bm25(chunks_fts) as score
             FROM chunks_fts
             JOIN chunks c ON chunks_fts.chunk_id = c.id
             JOIN documents d ON c.document_id = d.id
             WHERE chunks_fts MATCH ?1"
        );
        let mut params: Vec<String> = vec![sanitized_query];

        // Apply doc_type filter
        if let Some(doc_type) = &filters.doc_type {
            sql.push_str(&format!(" AND c.doc_type = ?{}", params.len() + 1));
            params.push(doc_type.clone());
        }

        // Apply document_ids filter
        if let Some(doc_ids) = &filters.document_ids {
            if !doc_ids.is_empty() {
                let placeholders: Vec<String> = doc_ids.iter()
                    .enumerate()
                    .map(|(i, _)| format!("?{}", params.len() + i + 1))
                    .collect();
                sql.push_str(&format!(" AND c.document_id IN ({})", placeholders.join(",")));
                params.extend(doc_ids.clone());
            }
        }

        sql.push_str(&format!(" ORDER BY score ASC LIMIT ?{}", params.len() + 1));
        params.push(top_k.to_string());

        let mut stmt = self.conn.prepare(&sql)
            .map_err(|e| format!("FTS query prep failed: {}", e))?;

        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let results = stmt.query_map(&params_refs[..], |row| {
            let score: f64 = row.get(8)?;
            // Convert BM25 score to 0-1 range (BM25 returns negative scores, lower is better)
            let similarity = (1.0 / (1.0 - score.min(0.0))).min(1.0) as f32;

            Ok(SearchResult {
                chunk_id: row.get(0)?,
                text: row.get(1)?,
                page_number: row.get(2)?,
                document_id: row.get(3)?,
                document_name: row.get(4)?,
                file_type: row.get(5)?,
                doc_type: row.get(6)?,
                section_title: row.get(7)?,
                similarity,
                match_source: Some("keyword".to_string()),
                authority_score: None,
            })
        }).map_err(|e| format!("FTS query failed: {}", e))?;

        results.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("FTS result collection failed: {}", e))
    }

    /// Reciprocal Rank Fusion: Merge vector and keyword search results
    /// Phase 3 enhancement: batch-loads authority scores to avoid N+1 queries
    pub fn reciprocal_rank_fusion(
        &self,
        vector_results: Vec<SearchResult>,
        keyword_results: Vec<SearchResult>,
    ) -> Vec<SearchResult> {
        const K: f32 = 60.0; // RRF constant (standard value)

        let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

        // Score from vector search
        for (rank, result) in vector_results.into_iter().enumerate() {
            let score = 1.0 / (rank as f32 + K);
            scores.insert(result.chunk_id.clone(), (score, result));
        }

        // Add score from keyword search
        for (rank, result) in keyword_results.into_iter().enumerate() {
            let score = 1.0 / (rank as f32 + K);
            scores.entry(result.chunk_id.clone())
                .and_modify(|(s, _)| *s += score)
                .or_insert((score, result));
        }

        // Batch-load authority scores for all chunk IDs in ONE query (Phase 3 - avoids N+1)
        let chunk_ids: Vec<&str> = scores.keys().map(|s| s.as_str()).collect();
        let authority_map = self.batch_authority_scores(&chunk_ids);

        // Apply authority score weighting and tag with LKOS metadata
        let mut results: Vec<(f32, SearchResult)> = scores.into_values()
            .map(|(score, mut result)| {
                let authority = authority_map.get(&result.chunk_id).copied().unwrap_or(1.0);
                result.match_source = Some("hybrid".to_string());
                result.authority_score = Some(authority);
                (score * authority, result)
            })
            .collect();

        // Sort by combined score (descending)
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        results.into_iter().map(|(_, result)| result).collect()
    }

    /// Batch-fetch authority_scores for a list of chunk_ids in ONE SQL query.
    /// Returns a HashMap<chunk_id, authority_score> (default 1.0 for missing).
    fn batch_authority_scores(&self, chunk_ids: &[&str]) -> HashMap<String, f32> {
        if chunk_ids.is_empty() {
            return HashMap::new();
        }
        // Build IN clause: (?1, ?2, ..., ?N)
        let placeholders: Vec<String> = (1..=chunk_ids.len())
            .map(|i| format!("?{}", i))
            .collect();
        let sql = format!(
            "SELECT id, knowledge_json FROM chunks WHERE id IN ({})",
            placeholders.join(",")
        );
        let mut map = HashMap::with_capacity(chunk_ids.len());
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let params: Vec<&dyn rusqlite::ToSql> = chunk_ids.iter()
                .map(|s| s as &dyn rusqlite::ToSql)
                .collect();
            if let Ok(rows) = stmt.query_map(&params[..], |row| {
                let id: String = row.get(0)?;
                let json: Option<String> = row.get(1)?;
                Ok((id, json))
            }) {
                for row in rows.flatten() {
                    let (id, json_opt) = row;
                    let score = json_opt
                        .and_then(|j| serde_json::from_str::<serde_json::Value>(&j).ok())
                        .and_then(|v| v.get("authority_score").and_then(|s| s.as_f64()))
                        .map(|v| v as f32)
                        .unwrap_or(1.0);
                    map.insert(id, score);
                }
            }
        }
        map
    }

    /// Assemble context from search results with token budget management
    pub fn assemble_context(
        &self,
        chunks: Vec<SearchResult>,
        max_tokens: usize,
        include_metadata: bool,
    ) -> String {
        let mut context = String::new();
        let mut current_tokens = 0;
        let mut used_chunks: std::collections::HashSet<String> = std::collections::HashSet::new();

        for chunk in chunks {
            // Skip duplicates
            if used_chunks.contains(&chunk.chunk_id) {
                continue;
            }

            // Estimate tokens (rough: 1 token ≈ 4 chars)
            let chunk_tokens = chunk.text.len() / 4;

            // Stop if exceeds budget (leave 500 tokens for response)
            if current_tokens + chunk_tokens > max_tokens.saturating_sub(500) {
                break;
            }

            // Add chunk with optional metadata
            if include_metadata {
                let mut header = format!(
                    "\n--- SOURCE: {} (Page {})",
                    chunk.document_name,
                    chunk.page_number.unwrap_or(0)
                );

                if let Some(source) = &chunk.match_source {
                    header.push_str(&format!(" [Match: {}]", source));
                }

                if let Some(auth) = chunk.authority_score {
                    if auth > 1.1 {
                        header.push_str(" [High Authority]");
                    }
                }

                header.push_str(" ---\n");
                context.push_str(&header);

                if let Some(section) = &chunk.section_title {
                    context.push_str(&format!("Section: {}\n", section));
                }
            }

            context.push_str(&chunk.text);
            context.push_str("\n\n");

            current_tokens += chunk_tokens;
            used_chunks.insert(chunk.chunk_id.clone());
        }

        context
    }

    /// Delete a document and all its chunks
    pub fn delete_document(&self, doc_id: &str) -> Result<(), String> {
        self.conn.execute("DELETE FROM documents WHERE id = ?1", params![doc_id])
            .map_err(|e| format!("Delete failed: {}", e))?;
        
        info!("Deleted document {} from vector store", doc_id);
        Ok(())
    }

    /// List all indexed documents
    pub fn list_documents(&self) -> Result<Vec<DocumentInfo>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT d.id, d.filename, d.path, d.file_type, d.total_pages, d.size_bytes, d.created_at,
                    (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as chunk_count,
                    d.doc_type, d.summary, d.readiness_state, d.section_count, d.summary_generated_at
             FROM documents d
             ORDER BY d.created_at DESC"
        ).map_err(|e| format!("Query failed: {}", e))?;

        let docs = stmt.query_map([], |row| {
            Ok(DocumentInfo {
                id: row.get(0)?,
                filename: row.get(1)?,
                path: row.get(2)?,
                file_type: row.get(3)?,
                total_pages: row.get(4)?,
                size_bytes: row.get(5)?,
                created_at: row.get(6)?,
                chunk_count: row.get(7)?,
                doc_type: row.get(8)?,
                summary: row.get(9)?,
                readiness_state: row.get(10)?,
                section_count: row.get(11)?,
                summary_generated_at: row.get(12)?,
            })
        }).map_err(|e| format!("Query failed: {}", e))?;

        docs.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collection failed: {}", e))
    }

    /// Get a specific document by ID
    pub fn get_document(&self, doc_id: &str) -> Result<Option<DocumentInfo>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT d.id, d.filename, d.path, d.file_type, d.total_pages, d.size_bytes, d.created_at,
                    (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as chunk_count,
                    d.doc_type, d.summary, d.readiness_state, d.section_count, d.summary_generated_at
             FROM documents d
             WHERE d.id = ?1"
        ).map_err(|e| format!("Query failed: {}", e))?;

        stmt.query_row(params![doc_id], |row| {
            Ok(DocumentInfo {
                id: row.get(0)?,
                filename: row.get(1)?,
                path: row.get(2)?,
                file_type: row.get(3)?,
                total_pages: row.get(4)?,
                size_bytes: row.get(5)?,
                created_at: row.get(6)?,
                chunk_count: row.get(7)?,
                doc_type: row.get(8)?,
                summary: row.get(9)?,
                readiness_state: row.get(10)?,
                section_count: row.get(11)?,
                summary_generated_at: row.get(12)?,
            })
        }).optional()
        .map_err(|e| format!("Query failed: {}", e))
    }

    /// Get chunks for a document
    pub fn get_chunks(&self, doc_id: &str) -> Result<Vec<ChunkRecord>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT id, document_id, text, page_number, chunk_index, token_count, language
             FROM chunks
             WHERE document_id = ?1
             ORDER BY chunk_index"
        ).map_err(|e| format!("Query failed: {}", e))?;

        let chunks = stmt.query_map(params![doc_id], |row| {
            Ok(ChunkRecord {
                id: row.get(0)?,
                document_id: row.get(1)?,
                text: row.get(2)?,
                page_number: row.get(3)?,
                chunk_index: row.get(4)?,
                token_count: row.get(5)?,
                language: row.get(6)?,
            })
        }).map_err(|e| format!("Query failed: {}", e))?;

        chunks.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collection failed: {}", e))
    }

    /// Get statistics about the vector store
    pub fn get_stats(&self) -> Result<VectorStoreStats, String> {
        let doc_count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents",
            [],
            |row| row.get(0)
        ).map_err(|e| format!("Query failed: {}", e))?;

        let chunk_count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks",
            [],
            |row| row.get(0)
        ).map_err(|e| format!("Query failed: {}", e))?;

        let db_size = std::fs::metadata(&self.db_path)
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(VectorStoreStats {
            document_count: doc_count,
            chunk_count,
            database_size_bytes: db_size,
        })
    }

    /// Check if a document exists
    pub fn document_exists(&self, doc_id: &str) -> Result<bool, String> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE id = ?1",
            params![doc_id],
            |row| row.get(0)
        ).map_err(|e| format!("Query failed: {}", e))?;

        Ok(count > 0)
    }

    /// Get first N chunks of a document (ordered by chunk_index)
    /// Used for summary requests to read document start
    pub fn get_document_chunks(
        &self,
        document_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.text, c.page_number, c.document_id, d.filename, d.file_type, c.doc_type, c.section_title
             FROM chunks c
             JOIN documents d ON c.document_id = d.id
             WHERE c.document_id = ?1
             ORDER BY c.chunk_index ASC
             LIMIT ?2"
        ).map_err(|e| format!("Failed to prepare query: {}", e))?;
        
        let results = stmt.query_map(params![document_id, limit as i64], |row| {
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                text: row.get(1)?,
                page_number: row.get(2)?,
                document_id: row.get(3)?,
                document_name: row.get(4)?,
                file_type: row.get(5)?,
                similarity: 1.0, // Not similarity-based, so set to 100%
                doc_type: row.get(6)?,
                section_title: row.get(7)?,
                match_source: Some("document_start".to_string()),
                authority_score: None,
            })
        }).map_err(|e| format!("Failed to execute query: {}", e))?;
        
        let chunks: Vec<SearchResult> = results.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to get chunks: {}", e))?;
        
        debug!("Retrieved {} chunks from document start", chunks.len());
        Ok(chunks)
    }

    /// Get first N chunks across all documents
    /// Used for summary requests when no specific document is selected
    pub fn get_all_document_starts(
        &self,
        limit: usize,
    ) -> Result<Vec<SearchResult>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.text, c.page_number, c.document_id, d.filename, d.file_type, c.doc_type, c.section_title
             FROM chunks c
             JOIN documents d ON c.document_id = d.id
             ORDER BY d.created_at DESC, c.chunk_index ASC
             LIMIT ?1"
        ).map_err(|e| format!("Failed to prepare query: {}", e))?;
        
        let results = stmt.query_map(params![limit as i64], |row| {
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                text: row.get(1)?,
                page_number: row.get(2)?,
                document_id: row.get(3)?,
                document_name: row.get(4)?,
                file_type: row.get(5)?,
                similarity: 1.0,
                doc_type: row.get(6)?,
                section_title: row.get(7)?,
                match_source: Some("document_start".to_string()),
                authority_score: None,
            })
        }).map_err(|e| format!("Failed to execute query: {}", e))?;
        
        let chunks: Vec<SearchResult> = results.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to get document starts: {}", e))?;
        
        debug!("Retrieved {} chunks from all document starts", chunks.len());
        Ok(chunks)
    }

    /// Update document readiness state
    pub fn update_document_readiness(&self, doc_id: &str, state: &str) -> Result<(), String> {
        self.conn.execute(
            "UPDATE documents SET readiness_state = ?1 WHERE id = ?2",
            params![state, doc_id]
        ).map_err(|e| format!("Failed to update readiness state: {}", e))?;
        Ok(())
    }

    /// Update document summary
    pub fn update_document_summary(&self, doc_id: &str, summary: &str) -> Result<(), String> {
        self.conn.execute(
            "UPDATE documents SET summary = ?1, summary_generated_at = datetime('now') WHERE id = ?2",
            params![summary, doc_id]
        ).map_err(|e| format!("Failed to update summary: {}", e))?;
        Ok(())
    }

    /// Update document section count
    pub fn update_document_section_count(&self, doc_id: &str, count: i32) -> Result<(), String> {
        self.conn.execute(
            "UPDATE documents SET section_count = ?1 WHERE id = ?2",
            params![count, doc_id]
        ).map_err(|e| format!("Failed to update section count: {}", e))?;
        Ok(())
    }

    /// Get document summary directly
    pub fn get_document_summary(&self, doc_id: &str) -> Result<Option<String>, String> {
        self.conn.query_row(
            "SELECT summary FROM documents WHERE id = ?1",
            params![doc_id],
            |row| row.get::<_, Option<String>>(0)
        ).map_err(|e| format!("Failed to query summary: {}", e))
    }

    /// Update the section_title of a specific chunk (used by LKOS section detector)
    pub fn update_chunk_section(&self, chunk_id: &str, section_title: &str) -> Result<(), String> {
        self.conn.execute(
            "UPDATE chunks SET section_title = ?1 WHERE id = ?2",
            params![section_title, chunk_id]
        ).map_err(|e| format!("Failed to update chunk section: {}", e))?;
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // LKOS Phase 3 — KnowledgeObject methods
    // ═══════════════════════════════════════════════════════════════════

    /// Store a KnowledgeObject's JSON into a chunk's knowledge_json column.
    pub fn store_chunk_knowledge(&self, chunk_id: &str, knowledge_json: &str) -> Result<(), String> {
        self.conn.execute(
            "UPDATE chunks SET knowledge_json = ?1 WHERE id = ?2",
            params![knowledge_json, chunk_id]
        ).map_err(|e| format!("Failed to store chunk knowledge: {}", e))?;
        Ok(())
    }

    /// Get all chunks for a document with their chunk_index (needed for KnowledgeObject extraction).
    pub fn get_chunks_for_knowledge(&self, doc_id: &str) -> Result<Vec<ChunkRecord>, String> {
        self.get_chunks(doc_id)
    }

    // ═══════════════════════════════════════════════════════════════════
    // LKOS Phase 4 — Public FTS search arm (for parallel execution)
    // ═══════════════════════════════════════════════════════════════════

    /// Public FTS-only search — used as an independent arm in Phase 4 parallel search.
    pub fn fts_search_only(
        &self,
        query_text: &str,
        top_k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>, String> {
        self.keyword_search(query_text, top_k, filters)
    }

    // ═══════════════════════════════════════════════════════════════════
    // LKOS Phase 5 — Entity Index methods
    // ═══════════════════════════════════════════════════════════════════

    /// Index entities for a chunk into entity_index table.
    pub fn index_chunk_entities(
        &self,
        chunk_id: &str,
        doc_id: &str,
        entities: &[String],
        entity_types: &[String],
    ) -> Result<(), String> {
        // Remove old entries for this chunk first (idempotent)
        let _ = self.conn.execute("DELETE FROM entity_index WHERE chunk_id = ?1", params![chunk_id]);

        let tx = self.conn.unchecked_transaction()
            .map_err(|e| format!("Entity index transaction failed: {}", e))?;

        for (entity, etype) in entities.iter().zip(entity_types.iter()) {
            tx.execute(
                "INSERT INTO entity_index (entity, entity_type, document_id, chunk_id) VALUES (?1, ?2, ?3, ?4)",
                params![entity, etype, doc_id, chunk_id]
            ).map_err(|e| format!("Failed to insert entity: {}", e))?;
        }

        tx.commit().map_err(|e| format!("Entity index commit failed: {}", e))?;
        Ok(())
    }

    /// Search chunks by entity name (case-insensitive) across all documents.
    /// Returns matching SearchResult objects for use in hybrid search merging.
    pub fn search_by_entity(&self, entity_query: &str, top_k: usize) -> Result<Vec<SearchResult>, String> {
        let like_pattern = format!("%{}%", entity_query);
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT c.id, c.text, c.page_number, c.document_id, d.filename,
                    d.file_type, c.doc_type, c.section_title
             FROM entity_index ei
             JOIN chunks c ON ei.chunk_id = c.id
             JOIN documents d ON c.document_id = d.id
             WHERE ei.entity LIKE ?1 COLLATE NOCASE
             LIMIT ?2"
        ).map_err(|e| format!("Entity search prep failed: {}", e))?;

        let results = stmt.query_map(params![like_pattern, top_k as i64], |row| {
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                text: row.get(1)?,
                page_number: row.get(2)?,
                document_id: row.get(3)?,
                document_name: row.get(4)?,
                file_type: row.get(5)?,
                doc_type: row.get(6)?,
                section_title: row.get(7)?,
                similarity: 0.85, // Entity match has high base relevance
                match_source: Some("entity".to_string()),
                authority_score: None,
            })
        }).map_err(|e| format!("Entity search failed: {}", e))?;

        results.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Entity result collection failed: {}", e))
    }

    /// Get top N entities for a document (for Library UI badges).
    pub fn get_top_entities_for_document(&self, doc_id: &str, limit: usize) -> Result<Vec<(String, String)>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT entity, entity_type, COUNT(*) as freq
             FROM entity_index
             WHERE document_id = ?1
             GROUP BY entity, entity_type
             ORDER BY freq DESC
             LIMIT ?2"
        ).map_err(|e| format!("Entity query prep failed: {}", e))?;

        let results = stmt.query_map(params![doc_id, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }).map_err(|e| format!("Entity query failed: {}", e))?;

        results.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Entity collection failed: {}", e))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors should have similarity 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((VectorStore::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity 0.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(VectorStore::cosine_similarity(&a, &b).abs() < 0.001);

        // Opposite vectors should have similarity -1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((VectorStore::cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }
}
