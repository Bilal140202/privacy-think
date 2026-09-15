// Message models - Data structures for chat/AI messages

use serde::{Deserialize, Serialize};

/// Role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// A single message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: String,
    pub document_refs: Vec<String>,
}

/// A conversation/chat session
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<Message>,
    pub created_at: String,
    pub updated_at: String,
    pub document_ids: Vec<String>,
}

/// Response from the local AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiResponse {
    pub message: String,
    pub sources: Vec<DocumentReference>,
    pub processing_time_ms: u64,
}

/// Reference to a document section
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentReference {
    pub document_id: String,
    pub document_name: String,
    pub page_number: Option<u32>,
    pub excerpt: String,
}
