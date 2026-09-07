//! Context estimation and task-preserving compaction helpers.
//!
//! Tool outputs are capped before entering history. Compaction summarizes the
//! intact conversation and retains the original request verbatim; older context
//! is never replaced by a marker before the model has summarized it.

use crate::api::types::{ContentBlock, Message, MessageContent};

/// Maximum characters for a single tool result before truncation.
const TOOL_OUTPUT_MAX_CHARS: usize = 30_000;

use std::collections::HashSet;
use std::sync::LazyLock;
use tiktoken_rs::{cl100k_base, CoreBPE};

/// Global tokenizer (initialized once, thread-safe).
static TOKENIZER: LazyLock<CoreBPE> =
    LazyLock::new(|| cl100k_base().expect("failed to initialize cl100k tokenizer"));

/// Empty set for encode's allowed_special parameter.
static NO_SPECIAL: LazyLock<HashSet<&'static str>> = LazyLock::new(HashSet::new);

/// Count tokens in a string using tiktoken.
fn count_tokens(text: &str) -> usize {
    TOKENIZER.encode(text, &NO_SPECIAL).0.len()
}

/// Estimate the token count of the conversation using tiktoken.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    let mut total = 0;

    for msg in messages {
        match &msg.content {
            MessageContent::Text(text) => {
                total += count_tokens(text);
            }
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            total += count_tokens(text);
                        }
                        // Image token accounting varies by provider and image
                        // dimensions. Reserve a conservative fixed amount
                        // without counting the much larger base64 transport.
                        ContentBlock::Image { .. } => total += 1_024,
                        ContentBlock::Reasoning { text, details } => {
                            if let Some(text) = text {
                                total += count_tokens(text);
                            }
                            if !details.is_empty() {
                                total += count_tokens(
                                    &serde_json::Value::Array(details.clone()).to_string(),
                                );
                            }
                        }
                        ContentBlock::ToolUse { input, name, .. } => {
                            total += count_tokens(name);
                            total += count_tokens(&input.to_string());
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            total += count_tokens(content);
                        }
                    }
                }
            }
        }
    }

    total
}

/// Truncate a tool output string if it exceeds the maximum.
/// Returns the (possibly truncated) string and whether it was truncated.
pub fn truncate_tool_output(output: &str) -> (String, bool) {
    if output.len() <= TOOL_OUTPUT_MAX_CHARS {
        return (output.to_string(), false);
    }

    // Keep first and last portions with a truncation marker
    let keep_start = TOOL_OUTPUT_MAX_CHARS * 2 / 3;
    let keep_end = TOOL_OUTPUT_MAX_CHARS / 6;

    let start = crate::utils::truncate_str(output, keep_start);
    let end = crate::utils::tail_str(output, keep_end);
    let truncated_chars = output.len() - start.len() - end.len();

    let result = format!("{start}\n\n... ({truncated_chars} characters truncated) ...\n\n{end}");

    (result, true)
}

/// Keep the handoff focused on information the next model round needs to act.
/// Later user corrections take precedence over the retained original request.
pub const SUMMARY_PROMPT: &str = "Summarize the conversation into a compact task handoff.
Do not continue the task or call tools. Use these sections:
- Objective: the current user goal, including changes to the original request.
- Constraints: user requirements, prohibitions, preferences, and later corrections.
- Decisions: choices made and the reasons that still matter.
- Progress: completed work, changed file paths, test results, and failures.
- Outstanding work: remaining steps, blockers, and the immediate next action.
Preserve concrete paths, identifiers, and unresolved errors needed to resume.
Distinguish completed work from plans. Later user instructions supersede earlier
ones. Carry forward still-relevant details from any earlier handoff. Do not
invent missing facts; use 'None' for empty sections. Keep it concise.";

/// Pin the first actual user request, including attached images. Because this
/// message remains first after compaction, repeated compaction and session resume
/// retain it without separate metadata or a storage migration.
pub fn original_request(messages: &[Message]) -> Option<Message> {
    messages
        .iter()
        .find(|message| {
            message.role == "user"
                && match &message.content {
                    MessageContent::Text(text) => !text.trim().is_empty(),
                    MessageContent::Blocks(blocks) => {
                        !blocks
                            .iter()
                            .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
                            && blocks.iter().any(|block| {
                                matches!(
                                    block,
                                    ContentBlock::Text { .. } | ContentBlock::Image { .. }
                                )
                            })
                    }
                }
        })
        .cloned()
}

/// Context window sizes for known models.
#[cfg(test)]
pub fn context_window_for_model(model: &str) -> usize {
    crate::model::built_in_metadata(model).context_window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(&[]), 0);
    }

    #[test]
    fn estimate_tokens_text() {
        let msgs = vec![Message::user("hello world")]; // 11 chars ≈ 2-3 tokens
        let tokens = estimate_tokens(&msgs);
        assert!(tokens > 0);
        assert!(tokens < 10);
    }

    #[test]
    fn truncate_short_output_unchanged() {
        let (result, truncated) = truncate_tool_output("short");
        assert_eq!(result, "short");
        assert!(!truncated);
    }

    #[test]
    fn truncate_long_output() {
        let long = "x".repeat(50_000);
        let (result, truncated) = truncate_tool_output(&long);
        assert!(truncated);
        assert!(result.len() < long.len());
        assert!(result.contains("truncated"));
    }

    #[test]
    fn truncate_long_multibyte_output_no_panic() {
        // Regression: byte-indexed slicing panicked when a cut point landed
        // mid-codepoint. 4-byte chars guarantee both cut points do.
        let long = "🦀".repeat(15_000); // 60k bytes
        let (result, truncated) = truncate_tool_output(&long);
        assert!(truncated);
        assert!(result.contains("truncated"));
        // Both kept segments must still be valid crab-only text
        assert!(result.starts_with('🦀'));
        assert!(result.ends_with('🦀'));
    }

    #[test]
    fn original_request_skips_tool_results_and_preserves_images() {
        let request = Message::user_with_images(
            "fix this screenshot",
            vec![crate::api::types::ImageSource {
                source_type: "base64".into(),
                media_type: "image/png".into(),
                data: "image-data".into(),
            }],
        );
        let messages = vec![
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "old".into(),
                content: "old output".into(),
                is_error: None,
            }]),
            request.clone(),
            Message::assistant_text("working"),
        ];
        assert_eq!(
            serde_json::to_value(original_request(&messages).unwrap()).unwrap(),
            serde_json::to_value(request).unwrap()
        );
        assert!(original_request(&[]).is_none());
    }

    #[test]
    fn context_window_known_models() {
        assert_eq!(
            context_window_for_model("claude-sonnet-4-20250514"),
            200_000
        );
        assert_eq!(context_window_for_model("gpt-4o"), 128_000);
        assert_eq!(context_window_for_model("gpt-5.6-sol"), 1_050_000);
        assert_eq!(context_window_for_model("gpt-5.3-codex"), 400_000);
    }
}
