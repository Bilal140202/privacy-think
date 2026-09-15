// Code Extractor - Handles source code files with language detection
// Supports: Python, JavaScript, TypeScript, Rust, C/C++, Java, Go, PHP, Ruby, Swift, Kotlin, etc.

use crate::document::types::*;
use chrono::Utc;
use std::fs;
use std::path::Path;
use uuid::Uuid;

pub struct CodeExtractor;

impl CodeExtractor {
    /// Detect programming language from file extension
    pub fn detect_language(extension: &str) -> Option<String> {
        let lang = match extension.to_lowercase().as_str() {
            "py" => "Python",
            "js" | "mjs" | "cjs" => "JavaScript",
            "ts" | "tsx" => "TypeScript",
            "jsx" => "React JSX",
            "rs" => "Rust",
            "cpp" | "cc" | "cxx" => "C++",
            "c" => "C",
            "h" | "hpp" | "hxx" => "C/C++ Header",
            "java" => "Java",
            "go" => "Go",
            "php" => "PHP",
            "rb" => "Ruby",
            "swift" => "Swift",
            "kt" | "kts" => "Kotlin",
            "scala" => "Scala",
            "cs" => "C#",
            "fs" => "F#",
            "dart" => "Dart",
            "lua" => "Lua",
            "r" => "R",
            "sql" => "SQL",
            "sh" | "bash" => "Shell",
            "ps1" => "PowerShell",
            "bat" | "cmd" => "Batch",
            "zig" => "Zig",
            "nim" => "Nim",
            "ex" | "exs" => "Elixir",
            "erl" => "Erlang",
            "hs" => "Haskell",
            "ml" | "mli" => "OCaml",
            "vue" => "Vue",
            "svelte" => "Svelte",
            _ => return None,
        };
        Some(lang.to_string())
    }
    
    /// Check if this extractor supports the given extension
    pub fn supports(extension: &str) -> bool {
        Self::detect_language(extension).is_some()
    }
    
    /// Extract text from a code file with language metadata
    pub fn extract(path: &Path) -> Result<Document, DocumentError> {
        let content = fs::read_to_string(path)
            .map_err(|e| DocumentError::ExtractionError(format!("Failed to read code file: {}", e)))?;
        
        let metadata_fs = fs::metadata(path)
            .map_err(|e| DocumentError::IoError(format!("Cannot read metadata: {}", e)))?;
        
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let language = Self::detect_language(&extension)
            .unwrap_or_else(|| "Unknown".to_string());
        
        let line_count = content.lines().count();
        
        // For code, treat entire file as one "page" but preserve structure
        let page = Page {
            number: 1,
            text: content.clone(),
            char_count: content.len(),
            line_count: Some(line_count),
            language: Some(language.clone()),
        };
        
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        Ok(Document {
            id: Uuid::new_v4().to_string(),
            filename,
            path: path.to_path_buf(),
            file_type: FileType::Code(language),
            pages: vec![page],
            total_pages: 1,
            metadata: DocumentMetadata {
                size_bytes: metadata_fs.len(),
                extension,
                is_code: true,
                requires_ocr: false,
                extraction_time_ms: 0,
            },
            created_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_language_detection() {
        assert_eq!(CodeExtractor::detect_language("py"), Some("Python".to_string()));
        assert_eq!(CodeExtractor::detect_language("rs"), Some("Rust".to_string()));
        assert_eq!(CodeExtractor::detect_language("js"), Some("JavaScript".to_string()));
        assert_eq!(CodeExtractor::detect_language("unknown"), None);
    }
    
    #[test]
    fn test_supports() {
        assert!(CodeExtractor::supports("py"));
        assert!(CodeExtractor::supports("rs"));
        assert!(!CodeExtractor::supports("txt"));
        assert!(!CodeExtractor::supports("pdf"));
    }
}
