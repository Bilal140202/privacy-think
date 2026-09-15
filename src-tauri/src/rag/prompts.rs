// RAG Prompts - Domain-specific prompt templates
//
// Provides specialized prompts for different document types:
// - Legal: Contract analysis, clause extraction
// - Code: Technical explanations, API documentation
// - Medical: Clinical terminology, treatment information
// - Financial: Numerical analysis, regulatory compliance
// - General: Default fallback for mixed content

use serde::{Deserialize, Serialize};
use tracing::debug;

// ==================== Prompt Templates ====================

pub const GENERAL_PROMPT: &str = r#"You are an AI assistant helping the user analyze documents. Answer based on the provided context.

Context from documents:
{context}

User question: {question}

Instructions:
- Provide accurate, relevant information from the context
- Cite specific sources when possible (document name, page number)
- If the information is not in the context, say so clearly
- Keep responses concise and well-organized"#;

pub const LEGAL_PROMPT: &str = r#"You are a legal document analysis assistant. Analyze the following legal documents with precision.

Context from legal documents:
{context}

User question: {question}

Instructions:
- Quote exact language from contracts when relevant
- Identify specific clauses, sections, or provisions
- Note any defined terms and their meanings
- Highlight potential risks or obligations
- Cite the exact source (document, page, section)
- Use professional legal terminology appropriately
- If the answer requires legal advice, remind the user to consult a licensed attorney"#;

pub const CODE_PROMPT: &str = r#"You are a technical documentation assistant specializing in code analysis.

Context from code documentation:
{context}

User question: {question}

Instructions:
- Explain concepts clearly with technical accuracy
- Reference specific functions, classes, or modules by name
- Include code examples when helpful
- Note any dependencies or prerequisites
- Cite the source file and relevant line numbers if available
- Use proper code formatting for any code snippets
- Explain the "why" behind implementation choices when apparent"#;

pub const MEDICAL_PROMPT: &str = r#"You are a medical document analysis assistant. Help the user understand medical information.

Context from medical documents:
{context}

User question: {question}

Instructions:
- Use clear, precise medical terminology
- Explain complex terms when they appear
- Reference specific studies or clinical guidelines if mentioned
- Note any dosages, frequencies, or protocols exactly as written
- Cite the source document and section
- IMPORTANT: Always remind users that this is not medical advice
- Recommend consulting healthcare professionals for medical decisions"#;

pub const FINANCIAL_PROMPT: &str = r#"You are a financial document analysis assistant. Help the user understand financial information.

Context from financial documents:
{context}

User question: {question}

Instructions:
- Present numerical data accurately
- Reference specific figures, tables, or charts when relevant
- Note time periods and reporting standards
- Identify key financial metrics and ratios
- Cite the source document, page, and date
- Highlight any disclaimers or risk factors mentioned
- IMPORTANT: This is not financial advice - recommend consulting a financial professional"#;

// ==================== Types ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum PromptTemplate {
    General,
    Legal,
    Code,
    Medical,
    Financial,
}

impl PromptTemplate {
    pub fn as_str(&self) -> &'static str {
        match self {
            PromptTemplate::General => "general",
            PromptTemplate::Legal => "legal",
            PromptTemplate::Code => "code",
            PromptTemplate::Medical => "medical",
            PromptTemplate::Financial => "financial",
        }
    }

    pub fn parse_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "legal" => PromptTemplate::Legal,
            "code" => PromptTemplate::Code,
            "medical" => PromptTemplate::Medical,
            "financial" => PromptTemplate::Financial,
            _ => PromptTemplate::General,
        }
    }
}

// ==================== Document Type Detection ====================

/// Detect document type based on filename and content analysis
/// Returns the best-matching PromptTemplate type
pub fn detect_document_type(filename: &str, content_sample: &str) -> PromptTemplate {
    let filename_lower = filename.to_lowercase();
    let content_lower = content_sample.to_lowercase();

    // ===== Filename-based detection (highest confidence) =====
    
    // Code files by extension
    let code_extensions = [
        ".rs", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", 
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".sh", ".bash", ".ps1", ".sql", ".json",
        ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".md"
    ];
    if code_extensions.iter().any(|ext| filename_lower.ends_with(ext)) {
        debug!("Detected code document by extension: {}", filename);
        return PromptTemplate::Code;
    }

    // Legal documents by filename patterns
    let legal_patterns = [
        "contract", "agreement", "terms", "policy", "nda", "license",
        "legal", "compliance", "disclaimer", "warranty", "liability"
    ];
    if legal_patterns.iter().any(|p| filename_lower.contains(p)) {
        debug!("Detected legal document by filename: {}", filename);
        return PromptTemplate::Legal;
    }

    // Medical documents by filename patterns
    let medical_patterns = [
        "medical", "clinical", "patient", "diagnosis", "treatment",
        "prescription", "hospital", "doctor", "healthcare", "lab_report"
    ];
    if medical_patterns.iter().any(|p| filename_lower.contains(p)) {
        debug!("Detected medical document by filename: {}", filename);
        return PromptTemplate::Medical;
    }

    // Financial documents by filename patterns
    let financial_patterns = [
        "financial", "invoice", "budget", "expense", "revenue", "balance",
        "statement", "tax", "audit", "quarterly", "annual_report", "10-k", "10-q"
    ];
    if financial_patterns.iter().any(|p| filename_lower.contains(p)) {
        debug!("Detected financial document by filename: {}", filename);
        return PromptTemplate::Financial;
    }

    // ===== Content-based detection (fallback) =====
    
    // Legal content keywords
    let legal_keywords = [
        "whereas", "hereby", "herein", "thereof", "party", "parties",
        "shall", "covenant", "indemnify", "termination clause", "arbitration",
        "jurisdiction", "governing law", "breach", "remedy", "negligence"
    ];
    let legal_score: usize = legal_keywords.iter()
        .filter(|kw| content_lower.contains(*kw))
        .count();

    // Code content keywords
    let code_keywords = [
        "function", "class", "import", "export", "return", "const", "let",
        "if", "else", "for", "while", "try", "catch", "async", "await",
        "pub fn", "impl", "struct", "enum", "def ", "self.", "#include"
    ];
    let code_score: usize = code_keywords.iter()
        .filter(|kw| content_lower.contains(*kw))
        .count();

    // Medical content keywords
    let medical_keywords = [
        "patient", "diagnosis", "treatment", "medication", "dosage",
        "symptoms", "chronic", "acute", "clinical", "prescription",
        "blood pressure", "mg", "ml", "iv", "prn", "bid", "tid"
    ];
    let medical_score: usize = medical_keywords.iter()
        .filter(|kw| content_lower.contains(*kw))
        .count();

    // Financial content keywords
    let financial_keywords = [
        "revenue", "expense", "profit", "loss", "balance sheet",
        "cash flow", "assets", "liabilities", "equity", "depreciation",
        "ebitda", "roi", "margin", "fiscal", "$", "%"
    ];
    let financial_score: usize = financial_keywords.iter()
        .filter(|kw| content_lower.contains(*kw))
        .count();

    // Determine winner (need threshold to avoid false positives)
    let threshold = 3;
    let scores = [
        (legal_score, PromptTemplate::Legal),
        (code_score, PromptTemplate::Code),
        (medical_score, PromptTemplate::Medical),
        (financial_score, PromptTemplate::Financial),
    ];

    if let Some((score, template)) = scores.iter().max_by_key(|(s, _)| s) {
        if *score >= threshold {
            debug!("Detected {} document by content (score: {})", template.as_str(), score);
            return template.clone();
        }
    }

    debug!("Defaulting to general document type for: {}", filename);
    PromptTemplate::General
}

/// Get the prompt template string for a given template type
pub fn get_prompt_template(template: &PromptTemplate) -> &'static str {
    match template {
        PromptTemplate::General => GENERAL_PROMPT,
        PromptTemplate::Legal => LEGAL_PROMPT,
        PromptTemplate::Code => CODE_PROMPT,
        PromptTemplate::Medical => MEDICAL_PROMPT,
        PromptTemplate::Financial => FINANCIAL_PROMPT,
    }
}

/// Generate a complete prompt with context and question filled in
pub fn get_prompt(template: &PromptTemplate, context: &str, question: &str) -> String {
    let template_str = get_prompt_template(template);
    template_str
        .replace("{context}", context)
        .replace("{question}", question)
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_code_by_extension() {
        assert_eq!(detect_document_type("main.rs", ""), PromptTemplate::Code);
        assert_eq!(detect_document_type("app.py", ""), PromptTemplate::Code);
        assert_eq!(detect_document_type("utils.ts", ""), PromptTemplate::Code);
    }

    #[test]
    fn test_detect_legal_by_filename() {
        assert_eq!(detect_document_type("service_agreement.pdf", ""), PromptTemplate::Legal);
        assert_eq!(detect_document_type("NDA_2024.docx", ""), PromptTemplate::Legal);
    }

    #[test]
    fn test_detect_legal_by_content() {
        let content = "WHEREAS the parties hereby agree to the following terms and covenants, including indemnification clauses and governing law provisions.";
        assert_eq!(detect_document_type("document.pdf", content), PromptTemplate::Legal);
    }

    #[test]
    fn test_detect_code_by_content() {
        let content = "pub fn main() { let x = 5; if x > 3 { return x; } }";
        assert_eq!(detect_document_type("readme.txt", content), PromptTemplate::Code);
    }

    #[test]
    fn test_general_fallback() {
        assert_eq!(detect_document_type("random_notes.pdf", "These are some notes about various topics."), PromptTemplate::General);
    }

    #[test]
    fn test_get_prompt() {
        let prompt = get_prompt(&PromptTemplate::General, "Some context here", "What is this about?");
        assert!(prompt.contains("Some context here"));
        assert!(prompt.contains("What is this about?"));
    }
}
