use serde::{Deserialize, Serialize};

use crate::utils::diff::generate_diff;

/// One configured permission rule: `Tool` or `Tool(pattern)`.
///
/// A bare tool name matches every call of that tool. A pattern is matched as
/// a glob against the call's subject: the command for Bash, the path for
/// file tools, the URL for WebFetch, the prompt for Agent. `*` as the tool
/// name matches every tool. Pattern rules never match a tool whose input has
/// no recognizable subject, so they cannot accidentally widen.
#[derive(Debug, Clone)]
pub struct PermissionRule {
    raw: String,
    tool: String,
    pattern: Option<glob::Pattern>,
}

impl PermissionRule {
    pub fn parse(rule: &str) -> Result<Self, String> {
        let raw = rule.trim();
        if raw.is_empty() {
            return Err("permission rule is empty".to_string());
        }
        let (tool, pattern) = match raw.split_once('(') {
            Some((tool, rest)) => {
                let Some(pattern) = rest.strip_suffix(')') else {
                    return Err(format!(
                        "permission rule {raw:?} is missing its closing ')'"
                    ));
                };
                (tool.trim(), Some(pattern.trim()))
            }
            None => (raw, None),
        };
        if tool.is_empty()
            || !tool
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '*'))
        {
            return Err(format!("permission rule {raw:?} has an invalid tool name"));
        }
        let pattern = match pattern {
            Some("") => return Err(format!("permission rule {raw:?} has an empty pattern")),
            Some(pattern) => Some(
                glob::Pattern::new(pattern)
                    .map_err(|error| format!("permission rule {raw:?}: {error}"))?,
            ),
            None => None,
        };
        Ok(Self {
            raw: raw.to_string(),
            tool: tool.to_string(),
            pattern,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn matches(&self, tool_name: &str, input: &serde_json::Value) -> bool {
        if self.tool != "*" && self.tool != tool_name {
            return false;
        }
        match &self.pattern {
            None => true,
            Some(pattern) => rule_subject(tool_name, input)
                .map(|subject| pattern.matches(subject))
                .unwrap_or(false),
        }
    }
}

/// The part of a tool call that pattern rules are matched against.
fn rule_subject<'a>(tool_name: &str, input: &'a serde_json::Value) -> Option<&'a str> {
    let keys: &[&str] = match tool_name {
        "Bash" => &["command"],
        "Read" | "Write" | "Edit" => &["file_path"],
        "Glob" | "Grep" => &["path", "pattern"],
        "WebFetch" => &["url"],
        "Agent" => &["prompt"],
        _ => &[
            "command",
            "file_path",
            "path",
            "url",
            "pattern",
            "prompt",
            "query",
            "name",
        ],
    };
    keys.iter()
        .find_map(|key| input.get(key).and_then(|v| v.as_str()))
}

/// Parsed `[permissions]` rules. Deny wins over ask, ask over allow.
#[derive(Debug, Clone, Default)]
pub struct PermissionRules {
    pub allow: Vec<PermissionRule>,
    pub deny: Vec<PermissionRule>,
    pub ask: Vec<PermissionRule>,
}

impl PermissionRules {
    pub fn parse(allow: &[String], deny: &[String], ask: &[String]) -> Result<Self, String> {
        let parse_all = |rules: &[String]| {
            rules
                .iter()
                .map(|rule| PermissionRule::parse(rule))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            allow: parse_all(allow)?,
            deny: parse_all(deny)?,
            ask: parse_all(ask)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.allow.is_empty() && self.deny.is_empty() && self.ask.is_empty()
    }

    fn first_match<'a>(
        rules: &'a [PermissionRule],
        tool_name: &str,
        input: &serde_json::Value,
    ) -> Option<&'a PermissionRule> {
        rules.iter().find(|rule| rule.matches(tool_name, input))
    }
}

/// How permissions are handled.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    /// Prompt for write operations, auto-allow reads
    #[default]
    Default,
    /// Auto-allow file edits, still prompt for bash
    AcceptEdits,
    /// Allow everything without prompting
    Bypass,
    /// Deny all write operations
    Plan,
}

/// Result of a permission check.
pub enum PermissionResult {
    Allow,
    Deny(String),
    Ask {
        message: String,
        diff: Option<String>,
    },
}

/// User's response to a permission prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionResponse {
    /// Allow this one time
    Allow,
    /// Deny this one time and cancel remaining tools in the batch
    Deny,
    /// Always allow this tool for the rest of the session
    AlwaysAllow,
    /// Always allow this specific command (for Bash tool only)
    AlwaysAllowCommand(String),
    /// Deny and cancel remaining tools, with a message sent to the model
    DenyAndCancel,
}

impl PermissionResponse {
    /// Build an "always allow" response with the narrowest useful scope.
    ///
    /// Bash grants are tied to the exact raw command rather than the
    /// human-readable (and potentially truncated) permission summary.
    pub fn always_allow_for(tool_name: &str, input: &serde_json::Value) -> Self {
        if tool_name == "Bash" {
            return input["command"]
                .as_str()
                .map(|command| Self::AlwaysAllowCommand(command.to_string()))
                .unwrap_or(Self::Allow);
        }
        Self::AlwaysAllow
    }
}

pub struct PermissionChecker {
    mode: PermissionMode,
    /// Configured allow/deny/ask rules, evaluated before the mode.
    rules: PermissionRules,
    /// Tools the user has "always allowed" this session
    session_allows: std::collections::HashSet<String>,
    /// Specific bash commands the user has "always allowed" this session
    bash_command_allows: std::collections::HashSet<String>,
}

impl PermissionChecker {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            rules: PermissionRules::default(),
            session_allows: std::collections::HashSet::new(),
            bash_command_allows: std::collections::HashSet::new(),
        }
    }

    pub fn with_rules(mut self, rules: PermissionRules) -> Self {
        self.rules = rules;
        self
    }

    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    /// The prompt a tool call would show if it needed confirmation.
    fn ask_for(tool_name: &str, input: &serde_json::Value) -> PermissionResult {
        match tool_name {
            "Bash" => {
                let cmd = input["command"].as_str().unwrap_or("");
                PermissionResult::Ask {
                    message: format!("bash: {}", truncate(cmd, 80)),
                    diff: None,
                }
            }
            "Write" => {
                let path = input["file_path"].as_str().unwrap_or("?");
                PermissionResult::Ask {
                    message: format!("write: {path}"),
                    diff: None,
                }
            }
            "Edit" => {
                let path = input["file_path"].as_str().unwrap_or("?");
                let old_string = input["old_string"].as_str().unwrap_or("");
                let new_string = input["new_string"].as_str().unwrap_or("");
                let diff = if !old_string.is_empty() && !new_string.is_empty() {
                    Some(generate_diff(old_string, new_string, path))
                } else {
                    None
                };
                PermissionResult::Ask {
                    message: format!("edit: {path}"),
                    diff,
                }
            }
            "WebFetch" => {
                let url = input["url"].as_str().unwrap_or("?");
                PermissionResult::Ask {
                    message: format!("fetch: {url}"),
                    diff: None,
                }
            }
            _ => PermissionResult::Ask {
                message: tool_name.to_string(),
                diff: None,
            },
        }
    }

    /// Record that the user chose "always allow" for a tool.
    pub fn always_allow(&mut self, tool_name: &str) {
        self.session_allows.insert(tool_name.to_string());
    }

    /// Record that the user chose "always allow" for a specific bash command.
    pub fn always_allow_command(&mut self, cmd: &str) {
        self.bash_command_allows.insert(cmd.to_string());
    }

    /// Clear permissions granted for the previous conversation.
    pub fn reset_session(&mut self) {
        self.session_allows.clear();
        self.bash_command_allows.clear();
    }

    /// Check whether a tool invocation should be allowed.
    pub fn check(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        is_read_only: bool,
    ) -> PermissionResult {
        // Configured deny rules win over everything, including session
        // grants and bypass mode.
        if let Some(rule) = PermissionRules::first_match(&self.rules.deny, tool_name, input) {
            return PermissionResult::Deny(format!(
                "blocked by permission rule `{}`",
                rule.as_str()
            ));
        }

        // Session-level always-allow overrides
        if self.session_allows.contains(tool_name) {
            return PermissionResult::Allow;
        }

        // Command-specific allows for Bash
        if tool_name == "Bash" {
            if let Some(cmd) = input["command"].as_str() {
                if self.bash_command_allows.contains(cmd) {
                    return PermissionResult::Allow;
                }
            }
        }

        // Configured ask rules force a prompt even where the mode would
        // auto-allow; configured allow rules skip the prompt the mode would
        // show. Plan mode's write denial still applies to allow rules.
        if PermissionRules::first_match(&self.rules.ask, tool_name, input).is_some() {
            return Self::ask_for(tool_name, input);
        }
        if PermissionRules::first_match(&self.rules.allow, tool_name, input).is_some()
            && (is_read_only || self.mode != PermissionMode::Plan)
        {
            return PermissionResult::Allow;
        }

        match self.mode {
            PermissionMode::Bypass => PermissionResult::Allow,

            PermissionMode::Plan => {
                if is_read_only {
                    PermissionResult::Allow
                } else {
                    PermissionResult::Deny("Plan mode: write operations are disabled".to_string())
                }
            }

            PermissionMode::AcceptEdits => {
                if is_read_only || tool_name == "Write" || tool_name == "Edit" {
                    PermissionResult::Allow
                } else if tool_name == "Bash" {
                    let cmd = input["command"].as_str().unwrap_or("");
                    PermissionResult::Ask {
                        message: format!("Allow bash: {}?", truncate(cmd, 80)),
                        diff: None,
                    }
                } else {
                    // AcceptEdits grants implicit approval only to file edits.
                    // Agent, MCP, and future mutating tools retain an explicit
                    // permission boundary.
                    PermissionResult::Ask {
                        message: tool_name.to_string(),
                        diff: None,
                    }
                }
            }

            PermissionMode::Default => {
                if is_read_only {
                    // Local inspection is safe and must also work for
                    // non-interactive sub-agents. Network reads retain a
                    // prompt boundary because they disclose request data.
                    if tool_name == "WebFetch" {
                        let url = input["url"].as_str().unwrap_or("?");
                        PermissionResult::Ask {
                            message: format!("fetch: {url}"),
                            diff: None,
                        }
                    } else {
                        PermissionResult::Allow
                    }
                } else {
                    match tool_name {
                        "Bash" => {
                            let cmd = input["command"].as_str().unwrap_or("");
                            PermissionResult::Ask {
                                message: format!("bash: {}", truncate(cmd, 80)),
                                diff: None,
                            }
                        }
                        "Write" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            PermissionResult::Ask {
                                message: format!("write: {path}"),
                                diff: None,
                            }
                        }
                        "Edit" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            let old_string = input["old_string"].as_str().unwrap_or("");
                            let new_string = input["new_string"].as_str().unwrap_or("");

                            let diff = if !old_string.is_empty() && !new_string.is_empty() {
                                Some(generate_diff(old_string, new_string, path))
                            } else {
                                None
                            };

                            PermissionResult::Ask {
                                message: format!("edit: {path}"),
                                diff,
                            }
                        }
                        _ => PermissionResult::Ask {
                            message: tool_name.to_string(),
                            diff: None,
                        },
                    }
                }
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> &str {
    crate::utils::truncate_str(s, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bypass_allows_everything() {
        let checker = PermissionChecker::new(PermissionMode::Bypass);
        let input = json!({"command": "rm -rf /"});
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn plan_denies_writes() {
        let checker = PermissionChecker::new(PermissionMode::Plan);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Deny(_)
        ));
    }

    #[test]
    fn plan_allows_reads() {
        let checker = PermissionChecker::new(PermissionMode::Plan);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(
            checker.check("Read", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_allows_read_only() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "*.rs"});
        assert!(matches!(
            checker.check("Glob", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_asks_for_bash() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn default_asks_for_write() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "/tmp/test", "content": "hello"});
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn accept_edits_allows_write_and_edit() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Edit", &input, false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn accept_edits_asks_for_bash() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"command": "rm -rf /"});
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn accept_edits_asks_for_agent() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"prompt": "run a shell command"});

        assert!(matches!(
            checker.check("Agent", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn accept_edits_asks_for_mcp_tools() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"repository": "example/project"});

        assert!(matches!(
            checker.check("github__delete_repository", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn always_allow_overrides_mode() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});

        // First call should ask
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Ask { .. }
        ));

        // After always_allow, should allow
        checker.always_allow("Bash");
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn always_allow_is_tool_specific() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        checker.always_allow("Bash");

        let input = json!({"file_path": "/tmp/test"});
        // Write should still ask
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn reset_session_clears_all_session_grants() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        let bash_input = json!({"command": "cargo test"});
        let write_input = json!({"file_path": "/tmp/test"});

        checker.always_allow("Write");
        checker.always_allow_command("cargo test");
        assert!(matches!(
            checker.check("Write", &write_input, false),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Bash", &bash_input, false),
            PermissionResult::Allow
        ));

        checker.reset_session();

        assert!(matches!(
            checker.check("Write", &write_input, false),
            PermissionResult::Ask { .. }
        ));
        assert!(matches!(
            checker.check("Bash", &bash_input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn ask_summary_contains_command() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});
        if let PermissionResult::Ask { message, diff: _ } = checker.check("Bash", &input, false) {
            assert!(message.contains("cargo test"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn ask_summary_contains_file_path() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "/home/ducks/important.rs"});
        if let PermissionResult::Ask { message, diff: _ } = checker.check("Edit", &input, false) {
            assert!(message.contains("important.rs"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn edit_permission_includes_diff_when_fields_present() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({
            "file_path": "src/main.rs",
            "old_string": "let x = 1",
            "new_string": "let x = 2"
        });

        if let PermissionResult::Ask { message, diff } = checker.check("Edit", &input, false) {
            assert!(message.contains("src/main.rs"));
            assert!(
                diff.is_some(),
                "Diff should be generated when old_string and new_string are provided"
            );
            let diff_content = diff.unwrap();
            assert!(diff_content.contains("src/main.rs"));
            assert!(diff_content.contains("-let x = 1"));
            assert!(diff_content.contains("+let x = 2"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn edit_permission_no_diff_when_fields_missing() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "src/main.rs"});

        if let PermissionResult::Ask { message, diff } = checker.check("Edit", &input, false) {
            assert!(message.contains("src/main.rs"));
            assert!(
                diff.is_none(),
                "Diff should be None when old_string/new_string are missing"
            );
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn default_auto_allows_read_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "src/secret.rs"});

        assert!(matches!(
            checker.check("Read", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_auto_allows_grep_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "SECRET_KEY", "path": "src/"});

        assert!(matches!(
            checker.check("Grep", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_auto_allows_glob_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "*.rs"});

        assert!(matches!(
            checker.check("Glob", &input, true),
            PermissionResult::Allow
        ));
    }

    fn rules(allow: &[&str], deny: &[&str], ask: &[&str]) -> PermissionRules {
        let owned = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        PermissionRules::parse(&owned(allow), &owned(deny), &owned(ask)).unwrap()
    }

    #[test]
    fn rule_parsing_accepts_bare_and_pattern_forms_and_rejects_malformed() {
        assert!(PermissionRule::parse("Read").is_ok());
        assert!(PermissionRule::parse("Bash(git *)").is_ok());
        assert!(PermissionRule::parse("*").is_ok());
        assert!(PermissionRule::parse("mcp__github__list_issues").is_ok());
        for bad in ["", "Bash(", "Bash()", "Ba sh", "Bash(git [)"] {
            assert!(PermissionRule::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn pattern_rules_match_the_tool_subject() {
        let rule = PermissionRule::parse("Bash(git *)").unwrap();
        assert!(rule.matches("Bash", &json!({"command": "git status"})));
        assert!(!rule.matches("Bash", &json!({"command": "gitk"})));
        assert!(!rule.matches("Write", &json!({"command": "git status"})));
        // No subject: a pattern rule cannot match.
        assert!(!rule.matches("Bash", &json!({})));

        let paths = PermissionRule::parse("Edit(src/**)").unwrap();
        assert!(paths.matches("Edit", &json!({"file_path": "src/a/b.rs"})));
        assert!(!paths.matches("Edit", &json!({"file_path": "tests/a.rs"})));

        let any = PermissionRule::parse("*").unwrap();
        assert!(any.matches("mcp__x__y", &json!({})));
    }

    #[test]
    fn deny_rules_win_over_bypass_and_session_grants() {
        let mut checker = PermissionChecker::new(PermissionMode::Bypass).with_rules(rules(
            &[],
            &["Bash(rm -rf *)"],
            &[],
        ));
        checker.always_allow("Bash");
        checker.always_allow_command("rm -rf /");
        match checker.check("Bash", &json!({"command": "rm -rf /"}), false) {
            PermissionResult::Deny(reason) => assert!(reason.contains("Bash(rm -rf *)")),
            _ => panic!("expected Deny"),
        }
        assert!(matches!(
            checker.check("Bash", &json!({"command": "ls"}), false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn allow_rules_skip_the_prompt_in_default_mode() {
        let checker = PermissionChecker::new(PermissionMode::Default).with_rules(rules(
            &["Bash(cargo *)", "Edit(src/**)"],
            &[],
            &[],
        ));
        assert!(matches!(
            checker.check("Bash", &json!({"command": "cargo test"}), false),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Bash", &json!({"command": "rm x"}), false),
            PermissionResult::Ask { .. }
        ));
        assert!(matches!(
            checker.check("Edit", &json!({"file_path": "src/lib.rs"}), false),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Edit", &json!({"file_path": "README.md"}), false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn allow_rules_do_not_override_plan_mode_writes() {
        let checker =
            PermissionChecker::new(PermissionMode::Plan).with_rules(rules(&["Write"], &[], &[]));
        assert!(matches!(
            checker.check("Write", &json!({"file_path": "x"}), false),
            PermissionResult::Deny(_)
        ));
    }

    #[test]
    fn ask_rules_force_a_prompt_where_the_mode_would_allow() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits).with_rules(rules(
            &[],
            &[],
            &["Edit(Cargo.toml)", "Read(.env*)"],
        ));
        match checker.check("Edit", &json!({"file_path": "Cargo.toml"}), false) {
            PermissionResult::Ask { message, .. } => assert!(message.contains("Cargo.toml")),
            _ => panic!("expected Ask"),
        }
        assert!(matches!(
            checker.check("Edit", &json!({"file_path": "src/main.rs"}), false),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Read", &json!({"file_path": ".env.local"}), true),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn ask_rules_yield_to_a_session_always_allow() {
        let mut checker = PermissionChecker::new(PermissionMode::Default).with_rules(rules(
            &[],
            &[],
            &["Bash(cargo *)"],
        ));
        assert!(matches!(
            checker.check("Bash", &json!({"command": "cargo test"}), false),
            PermissionResult::Ask { .. }
        ));
        checker.always_allow_command("cargo test");
        assert!(matches!(
            checker.check("Bash", &json!({"command": "cargo test"}), false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_prompts_for_webfetch_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"url": "https://example.com/api"});

        if let PermissionResult::Ask { message, diff } = checker.check("WebFetch", &input, true) {
            assert!(message.contains("https://example.com/api"));
            assert!(diff.is_none());
        } else {
            panic!("expected Ask for WebFetch tool");
        }
    }
}
