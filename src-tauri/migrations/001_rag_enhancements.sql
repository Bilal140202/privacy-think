-- Migration: RAG Enhancements v1.1.0
-- Adds FTS5 keyword search and metadata columns for improved RAG accuracy
-- Date: 2026-02-09

-- Step 1: Add metadata columns to chunks table
-- Using PRAGMA to check if columns exist first would be ideal, but SQLite
-- ALTERs are idempotent for "ADD COLUMN IF NOT EXISTS" isn't supported.
-- We wrap in a try-catch equivalent via the Rust migration logic.

ALTER TABLE chunks ADD COLUMN doc_type TEXT DEFAULT 'general';
ALTER TABLE chunks ADD COLUMN section_title TEXT;
ALTER TABLE chunks ADD COLUMN char_count INTEGER;

-- Add metadata to documents table too
ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'general';

-- Step 2: Update existing chunks with defaults
UPDATE chunks SET 
    char_count = LENGTH(text),
    doc_type = 'general'
WHERE char_count IS NULL;

UPDATE documents SET doc_type = 'general' WHERE doc_type IS NULL;

-- Step 3: Create Full-Text Search virtual table using FTS5
-- Uses Porter stemmer for English and unicode61 for broad character support
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);

-- Step 4: Populate FTS index with existing chunk data
INSERT INTO chunks_fts(chunk_id, text)
SELECT id, text FROM chunks
WHERE id NOT IN (SELECT chunk_id FROM chunks_fts);

-- Step 5: Create triggers to keep FTS in sync with chunks table

-- Trigger: After INSERT on chunks, add to FTS
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, text) VALUES (new.id, new.text);
END;

-- Trigger: After DELETE on chunks, remove from FTS
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;

-- Trigger: After UPDATE on chunks, update FTS
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    UPDATE chunks_fts SET text = new.text WHERE chunk_id = old.id;
END;

-- Step 6: Add indexes for faster filtering
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_title);
CREATE INDEX IF NOT EXISTS idx_chunks_char_count ON chunks(char_count);

-- Step 7: Migration metadata table
CREATE TABLE IF NOT EXISTS migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Record this migration
INSERT OR IGNORE INTO migrations (version, applied_at) VALUES ('1.1.0', datetime('now'));
