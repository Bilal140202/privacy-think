// Document Chunker - Smart chunking for RAG pipelines
//
// Features:
// - Code-aware chunking (preserve function/class boundaries)
// - Paragraph-based chunking for text
// - Configurable chunk size and overlap
// - Language-specific patterns for code extraction

use crate::document::types::*;
use regex::Regex;
use tracing::debug;

/// Configuration for chunking behavior
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Whether to preserve code block boundaries
    pub preserve_code_blocks: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 800,
            overlap: 100,
            preserve_code_blocks: true,
        }
    }
}

/// Chunk a document into smaller pieces for RAG
pub fn chunk_document(document: &Document, config: &ChunkConfig) -> Vec<DocumentChunk> {
    let mut chunks = Vec::new();
    
    for page in &document.pages {
        let page_chunks = match &document.file_type {
            FileType::Code(language) if config.preserve_code_blocks => {
                chunk_code(&page.text, language, config)
            }
            _ => chunk_text(&page.text, config),
        };
        
        for text in page_chunks {
            let chunk_index = chunks.len();
            chunks.push(DocumentChunk {
                id: format!("{}-chunk-{}", document.id, chunk_index),
                document_id: document.id.clone(),
                chunk_index,
                text: text.clone(),
                char_count: text.len(),
                source_page: Some(page.number),
            });
        }
    }
    
    // If no chunks were created, create at least one from the full text
    if chunks.is_empty() {
        let full_text = document.full_text();
        if !full_text.is_empty() {
            chunks.push(DocumentChunk {
                id: format!("{}-chunk-0", document.id),
                document_id: document.id.clone(),
                chunk_index: 0,
                text: full_text.clone(),
                char_count: full_text.len(),
                source_page: Some(1),
            });
        }
    }
    
    debug!("Created {} chunks from document {}", chunks.len(), document.filename);
    chunks
}

/// Chunk text content by paragraphs
fn chunk_text(text: &str, config: &ChunkConfig) -> Vec<String> {
    let mut chunks = Vec::new();
    
    // Split by paragraphs (double newline)
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();
    
    if paragraphs.is_empty() {
        // No paragraphs, split by single newlines
        let lines: Vec<&str> = text.lines().collect();
        return chunk_by_size(&lines.join("\n"), config);
    }
    
    let mut current_chunk = String::new();
    
    for para in paragraphs {
        let para_trimmed = para.trim();
        
        // If adding this paragraph exceeds chunk size, save current and start new
        if current_chunk.len() + para_trimmed.len() > config.chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            
            // Add overlap from end of previous chunk
            if config.overlap > 0 && current_chunk.len() > config.overlap {
                let overlap_start = current_chunk.len() - config.overlap;
                current_chunk = current_chunk[overlap_start..].to_string();
                current_chunk.push_str("\n\n");
            } else {
                current_chunk = String::new();
            }
        }
        
        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para_trimmed);
    }
    
    // Don't forget the last chunk
    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

/// Chunk code content preserving function/class boundaries
fn chunk_code(text: &str, language: &str, config: &ChunkConfig) -> Vec<String> {
    let blocks = extract_code_blocks(text, language);
    
    if blocks.is_empty() {
        // Fallback to line-based chunking
        return chunk_by_size(text, config);
    }
    
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    for block in blocks {
        // If a single block exceeds chunk size, add it as its own chunk
        if block.len() > config.chunk_size {
            if !current_chunk.trim().is_empty() {
                chunks.push(current_chunk.clone());
                current_chunk = String::new();
            }
            chunks.push(block);
            continue;
        }
        
        // If adding this block exceeds chunk size, save current and start new
        if current_chunk.len() + block.len() > config.chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            current_chunk = String::new();
        }
        
        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(&block);
    }
    
    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

/// Extract code blocks (functions, classes, etc.) based on language
fn extract_code_blocks(text: &str, language: &str) -> Vec<String> {
    let patterns = get_language_patterns(language);
    
    if patterns.is_empty() {
        // No patterns, return the whole text as one block
        return vec![text.to_string()];
    }
    
    let mut blocks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    
    let mut current_block = String::new();
    let mut in_block = false;
    let mut brace_count = 0;
    let mut block_start_indent = 0;
    
    for (i, line) in lines.iter().enumerate() {
        let line_indent = line.len() - line.trim_start().len();
        let trimmed = line.trim();
        
        // Check if this line starts a new block
        let starts_block = patterns.iter().any(|p| {
            if let Ok(re) = Regex::new(p) {
                re.is_match(trimmed)
            } else {
                trimmed.starts_with(p)
            }
        });
        
        if starts_block && !in_block {
            // Save any accumulated non-block code
            if !current_block.trim().is_empty() {
                blocks.push(current_block.clone());
            }
            current_block = String::new();
            in_block = true;
            block_start_indent = line_indent;
            brace_count = 0;
        }
        
        // Add line to current block
        if !current_block.is_empty() {
            current_block.push('\n');
        }
        current_block.push_str(line);
        
        // Track braces for languages that use them
        if in_block {
            brace_count += trimmed.chars().filter(|c| *c == '{').count() as i32;
            brace_count -= trimmed.chars().filter(|c| *c == '}').count() as i32;
            
            // Check if block is complete
            let block_complete = match language.to_lowercase().as_str() {
                "python" | "yaml" => {
                    // Python uses indentation - block ends when we return to or below start indent
                    // Check next line if available
                    if i + 1 < lines.len() {
                        let next_line = lines[i + 1];
                        let next_trimmed = next_line.trim();
                        let next_indent = next_line.len() - next_line.trim_start().len();
                        
                        !next_trimmed.is_empty() && next_indent <= block_start_indent
                    } else {
                        true // Last line
                    }
                }
                _ => {
                    // Brace-based languages
                    brace_count == 0 && trimmed.ends_with('}')
                }
            };
            
            if block_complete && !current_block.trim().is_empty() {
                blocks.push(current_block.clone());
                current_block = String::new();
                in_block = false;
            }
        }
    }
    
    // Don't forget remaining content
    if !current_block.trim().is_empty() {
        blocks.push(current_block);
    }
    
    blocks
}

/// Get regex patterns for detecting function/class definitions by language
fn get_language_patterns(language: &str) -> Vec<&'static str> {
    match language.to_lowercase().as_str() {
        "python" => vec![
            r"^(async\s+)?def\s+\w+",
            r"^class\s+\w+",
        ],
        "javascript" | "typescript" | "react jsx" => vec![
            r"^(export\s+)?(async\s+)?function\s+\w+",
            r"^(export\s+)?class\s+\w+",
            r"^(export\s+)?const\s+\w+\s*=\s*(async\s+)?\(",
            r"^(export\s+)?const\s+\w+\s*=\s*\(",
        ],
        "rust" => vec![
            r"^(pub\s+)?(async\s+)?fn\s+\w+",
            r"^(pub\s+)?struct\s+\w+",
            r"^(pub\s+)?enum\s+\w+",
            r"^(pub\s+)?impl\s+",
            r"^(pub\s+)?trait\s+\w+",
        ],
        "java" | "kotlin" | "scala" => vec![
            r"^(public|private|protected)?\s*(static)?\s*(final)?\s*(class|interface|enum)\s+\w+",
            r"^(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(",
        ],
        "go" => vec![
            r"^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+",
            r"^type\s+\w+\s+(struct|interface)",
        ],
        "c" | "c++" | "c/c++ header" => vec![
            r"^\w+\s+\w+\s*\([^)]*\)\s*\{?",
            r"^(class|struct)\s+\w+",
        ],
        "php" => vec![
            r"^(public|private|protected)?\s*(static)?\s*function\s+\w+",
            r"^class\s+\w+",
        ],
        "ruby" => vec![
            r"^def\s+\w+",
            r"^class\s+\w+",
            r"^module\s+\w+",
        ],
        "swift" => vec![
            r"^(public|private|internal|fileprivate|open)?\s*func\s+\w+",
            r"^(public|private|internal|fileprivate|open)?\s*(class|struct|enum|protocol)\s+\w+",
        ],
        "c#" | "f#" => vec![
            r"^(public|private|protected|internal)?\s*(static)?\s*(class|interface|struct|enum)\s+\w+",
            r"^(public|private|protected|internal)?\s*(static)?\s*(async)?\s*\w+\s+\w+\s*\(",
        ],
        _ => vec![], // No patterns, will use fallback chunking
    }
}

/// Simple size-based chunking as fallback
fn chunk_by_size(text: &str, config: &ChunkConfig) -> Vec<String> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    
    let mut current_chunk = String::new();
    
    for line in lines {
        if current_chunk.len() + line.len() > config.chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            
            // Add overlap
            if config.overlap > 0 && current_chunk.len() > config.overlap {
                let overlap_start = current_chunk.len() - config.overlap;
                current_chunk = current_chunk[overlap_start..].to_string();
                current_chunk.push('\n');
            } else {
                current_chunk = String::new();
            }
        }
        
        if !current_chunk.is_empty() {
            current_chunk.push('\n');
        }
        current_chunk.push_str(line);
    }
    
    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_text() {
        let text = "Paragraph one here.\n\nParagraph two is longer and has more content.\n\nParagraph three.";
        let config = ChunkConfig {
            chunk_size: 50,
            overlap: 10,
            preserve_code_blocks: false,
        };
        
        let chunks = chunk_text(text, &config);
        assert!(!chunks.is_empty());
    }
    
    #[test]
    fn test_chunk_code() {
        let code = r#"
def hello():
    print("Hello")

def world():
    print("World")
"#;
        let config = ChunkConfig {
            chunk_size: 50,
            overlap: 0,
            preserve_code_blocks: true,
        };
        
        let chunks = chunk_code(code, "Python", &config);
        assert!(!chunks.is_empty());
    }
    
    #[test]
    fn test_get_language_patterns() {
        assert!(!get_language_patterns("python").is_empty());
        assert!(!get_language_patterns("rust").is_empty());
        assert!(!get_language_patterns("javascript").is_empty());
    }
}
