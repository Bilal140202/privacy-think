use serde::Serialize;
use tauri::command;

#[derive(Serialize)]
struct DiscordEmbed {
    title: String,
    description: String,
    color: u32,
    footer: DiscordFooter,
}

#[derive(Serialize)]
struct DiscordFooter {
    text: String,
}

#[derive(Serialize)]
struct DiscordPayload {
    embeds: Vec<DiscordEmbed>,
}

#[command]
pub async fn send_feedback(
    message: String,
    tag: String,
    email: Option<String>,
) -> Result<(), String> {
    // NOTE: In a real app, this should be an environment variable or fetched from a secure config
    // For this beta demo, we'll use a placeholder URL. 
    // The user should replace this with their actual Discord Webhook URL.
    let webhook_url = "https://discord.com/api/webhooks/1462658563066560695/_KHxLWhrBdhEusvcI2bsFpLYqkTYi2_cU3l8Lxa0RJeWqQVieR8KmZpzR-C4AcYm2ROG";

    if webhook_url.contains("YOUR_WEBHOOK_ID") {
        // Fallback to local file logging if webhook is not configured
        let feedback_dir = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join("feedback");
        std::fs::create_dir_all(&feedback_dir).map_err(|e| e.to_string())?;
        
        let timestamp = chrono::Utc::now().to_rfc3339();
        let filename = format!("feedback_{}.json", timestamp.replace(":", "-"));
        
        let feedback = serde_json::json!({
            "message": message,
            "tag": tag,
            "email": email,
            "timestamp": timestamp,
            "os": std::env::consts::OS,
            "version": "0.1.0"
        });
        
        std::fs::write(
            feedback_dir.join(filename),
            serde_json::to_string_pretty(&feedback).unwrap()
        ).map_err(|e| e.to_string())?;

        return Ok(());
    }

    let description = format!(
        "**Feedback:**\n{}\n\n**Email:** {}\n**OS:** {}\n**Version:** 0.1.0",
        message,
        email.unwrap_or_else(|| "Anonymous".to_string()),
        std::env::consts::OS
    );

    let color = match tag.as_str() {
        "Bug" => 15158332,      // Red
        "Feature" => 3066993,    // Green
        "Question" => 3447003,   // Blue
        _ => 7506394,            // Purple
    };

    let payload = DiscordPayload {
        embeds: vec![DiscordEmbed {
            title: format!("📢 Beta Feedback: {}", tag),
            description,
            color,
            footer: DiscordFooter {
                text: format!("PrivacyThink Beta | {}", chrono::Utc::now().to_rfc3339()),
            },
        }],
    };

    let client = reqwest::Client::new();
    client.post(webhook_url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}
