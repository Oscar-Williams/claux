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
    /// Always allow this conservative Bash command family for the session
    AlwaysAllowCommandType(String),
    /// Deny and cancel remaining tools, with a message sent to the model
    DenyAndCancel,
}

impl PermissionResponse {
    /// Build an "always allow" response with the narrowest useful scope.
    ///
    /// Common development commands receive a conservative reusable family;
    /// everything else is tied to the exact raw command rather than the
    /// human-readable (and potentially truncated) permission summary.
    pub fn always_allow_for(tool_name: &str, input: &serde_json::Value) -> Self {
        if tool_name == "Bash" {
            return input["command"]
                .as_str()
                .map(|command| {
                    bash_command_type(command)
                        .map(|command_type| Self::AlwaysAllowCommandType(command_type.key))
                        .unwrap_or_else(|| Self::AlwaysAllowCommand(command.to_string()))
                })
                .unwrap_or(Self::Allow);
        }
        Self::AlwaysAllow
    }

    pub fn always_allow_label(tool_name: &str, input: &serde_json::Value) -> String {
        if tool_name != "Bash" {
            return "(a)lways allow this tool".to_string();
        }
        input["command"]
            .as_str()
            .and_then(bash_command_type)
            .map(|command_type| format!("(a)lways allow {} commands", command_type.label))
            .unwrap_or_else(|| "(a)lways allow this exact command".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BashCommandType {
    key: String,
    label: String,
}

impl BashCommandType {
    fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
        }
    }
}

/// Return a narrow reusable family for common development commands. Unknown,
/// mutating, compound, redirected, or dynamically expanded commands retain
/// exact-command approval.
fn bash_command_type(command: &str) -> Option<BashCommandType> {
    if command.contains(['\n', ';', '|', '&', '>', '<', '`'])
        || command.contains("$(")
        || command.contains("${")
    {
        return None;
    }
    command_type_from_words(&shlex::split(command)?)
}

fn command_type_from_words(words: &[String]) -> Option<BashCommandType> {
    let executable = words.first()?;
    if is_environment_assignment(executable) || executable.contains(['/', '\\']) {
        return None;
    }

    match executable.as_str() {
        "cargo" => typed_subcommand(
            "cargo",
            words,
            &["bench", "build", "check", "clippy", "doc", "fmt", "test"],
        ),
        "go" => typed_subcommand("go", words, &["build", "test", "vet"]),
        "git" => typed_subcommand("git", words, &["diff", "log", "show", "status"]),
        "gh" => gh_command_type(words),
        "npm" | "pnpm" | "yarn" | "bun" => package_script_type(executable, words),
        "bundle" if words.get(1).map(String::as_str) == Some("exec") => {
            test_runner_type(words.get(2..).unwrap_or_default(), Some("bundle exec"))
        }
        "make" => make_command_type(words),
        "nix-shell" => nested_command_type("nix-shell", words, "--run"),
        "nix" if words.get(1).map(String::as_str) == Some("develop") => {
            nested_command_type("nix develop", words, "-c")
        }
        "pytest" | "rspec" | "rubocop" => Some(BashCommandType::new(executable, executable)),
        "cargo-nextest" if words.get(1).map(String::as_str) == Some("run") => Some(
            BashCommandType::new("cargo-nextest:run", "cargo nextest run"),
        ),
        "cat" | "df" | "du" | "file" | "grep" | "head" | "ls" | "pwd" | "rg" | "stat" | "tail"
        | "type" | "wc" | "which" => Some(BashCommandType::new(executable, executable)),
        _ => None,
    }
}

fn typed_subcommand(
    executable: &str,
    words: &[String],
    allowed: &[&str],
) -> Option<BashCommandType> {
    // Do not skip flags here: many command-level options consume the next word.
    // Mistaking an option value for a subcommand would silently broaden a grant.
    let subcommand = words.get(1)?;
    allowed.contains(&subcommand.as_str()).then(|| {
        BashCommandType::new(
            format!("{executable}:{subcommand}"),
            format!("{executable} {subcommand}"),
        )
    })
}

fn gh_command_type(words: &[String]) -> Option<BashCommandType> {
    let group = words.get(1)?.as_str();
    let action = words.get(2)?.as_str();
    let allowed = match group {
        "pr" => &["checks", "diff", "list", "status", "view"][..],
        "issue" => &["list", "status", "view"][..],
        "run" => &["list", "view", "watch"][..],
        "repo" => &["list", "view"][..],
        _ => return None,
    };
    allowed.contains(&action).then(|| {
        BashCommandType::new(
            format!("gh:{group}:{action}"),
            format!("gh {group} {action}"),
        )
    })
}

fn package_script_type(executable: &str, words: &[String]) -> Option<BashCommandType> {
    match words.get(1)?.as_str() {
        "test" => Some(BashCommandType::new(
            format!("{executable}:test"),
            format!("{executable} test"),
        )),
        "run" => {
            let script = words.get(2)?;
            is_simple_name(script).then(|| {
                BashCommandType::new(
                    format!("{executable}:run:{script}"),
                    format!("{executable} run {script}"),
                )
            })
        }
        _ => None,
    }
}

fn test_runner_type(words: &[String], prefix: Option<&str>) -> Option<BashCommandType> {
    let runner = std::path::Path::new(words.first()?).file_name()?.to_str()?;
    ["pytest", "rspec", "rubocop"].contains(&runner).then(|| {
        let label = prefix
            .map(|prefix| format!("{prefix} {runner}"))
            .unwrap_or_else(|| runner.to_string());
        BashCommandType::new(label.replace(' ', ":"), label)
    })
}

fn make_command_type(words: &[String]) -> Option<BashCommandType> {
    // Like command subcommands, targets after options are ambiguous because
    // flags such as `-f` consume their following word.
    let target = words.get(1)?;
    if target.starts_with('-') || target.contains('=') {
        return None;
    }
    [
        "check", "ci", "fmt", "format", "lint", "spec", "test", "verify",
    ]
    .contains(&target.as_str())
    .then(|| BashCommandType::new(format!("make:{target}"), format!("make {target}")))
}

fn nested_command_type(
    wrapper: &str,
    words: &[String],
    command_flag: &str,
) -> Option<BashCommandType> {
    let command_index = words.iter().position(|word| word == command_flag)?;
    let nested_words = if command_flag == "--run" {
        shlex::split(words.get(command_index + 1)?)?
    } else {
        words.get(command_index + 1..)?.to_vec()
    };
    let nested = command_type_from_words(&nested_words)?;
    let wrapper_context = words.get(..command_index)?.join("\u{1f}");
    Some(BashCommandType::new(
        format!("{wrapper}:{wrapper_context}:{}", nested.key),
        format!("{} via {wrapper}", nested.label),
    ))
}

fn is_environment_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn is_simple_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
}

/// A decision returned by an `on_permission_check` hook.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HookDecision {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Deserialize)]
struct HookVerdict {
    decision: HookDecision,
    #[serde(default)]
    reason: Option<String>,
}

/// Parse a hook's stdout. Empty output is "no opinion"; anything else must
/// be the documented JSON object.
pub fn parse_hook_verdict(output: &str) -> Result<Option<(HookDecision, Option<String>)>, String> {
    let output = output.trim();
    if output.is_empty() {
        return Ok(None);
    }
    let verdict: HookVerdict = serde_json::from_str(output)
        .map_err(|error| format!("hook output is not a decision object: {error}"))?;
    Ok(Some((verdict.decision, verdict.reason)))
}

/// The proposed decision as a hook sees it.
pub fn proposed_decision(result: &PermissionResult) -> &'static str {
    match result {
        PermissionResult::Allow => "allow",
        PermissionResult::Deny(_) => "deny",
        PermissionResult::Ask { .. } => "ask",
    }
}

/// Fold hook verdicts into the checker's result. The most restrictive
/// verdict wins: any deny denies, any ask escalates an allow to a prompt,
/// and allow only clears a prompt. A configured or mode deny is final and
/// no hook can lift it.
pub fn apply_hook_verdicts(
    tool_name: &str,
    input: &serde_json::Value,
    result: PermissionResult,
    verdicts: &[(String, HookDecision, Option<String>)],
) -> PermissionResult {
    if let PermissionResult::Deny(_) = result {
        return result;
    }
    if let Some((name, _, reason)) = verdicts
        .iter()
        .find(|(_, decision, _)| *decision == HookDecision::Deny)
    {
        let reason = reason.as_deref().unwrap_or("no reason given");
        return PermissionResult::Deny(format!("blocked by hook {name}: {reason}"));
    }
    if verdicts
        .iter()
        .any(|(_, decision, _)| *decision == HookDecision::Ask)
    {
        return match result {
            PermissionResult::Ask { .. } => result,
            _ => PermissionChecker::ask_for(tool_name, input),
        };
    }
    if verdicts
        .iter()
        .any(|(_, decision, _)| *decision == HookDecision::Allow)
    {
        return PermissionResult::Allow;
    }
    result
}

/// A mode plus its rules: everything a checker needs, passed as one value
/// so sub-agents inherit both together.
#[derive(Debug, Clone, Default)]
pub struct PermissionPolicy {
    pub mode: PermissionMode,
    pub rules: PermissionRules,
}

impl PermissionPolicy {
    pub fn new(mode: PermissionMode, rules: PermissionRules) -> Self {
        Self { mode, rules }
    }

    pub fn checker(&self) -> PermissionChecker {
        PermissionChecker::new(self.mode).with_rules(self.rules.clone())
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
    /// Conservative Bash command families allowed for this session
    bash_command_type_allows: std::collections::HashSet<String>,
}

impl PermissionChecker {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            rules: PermissionRules::default(),
            session_allows: std::collections::HashSet::new(),
            bash_command_allows: std::collections::HashSet::new(),
            bash_command_type_allows: std::collections::HashSet::new(),
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
    pub(crate) fn ask_for(tool_name: &str, input: &serde_json::Value) -> PermissionResult {
        match tool_name {
            "Bash" => {
                let cmd = input["command"].as_str().unwrap_or("");
                PermissionResult::Ask {
                    message: format!("bash{}: {}", background_hint(input), truncate(cmd, 80)),
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

    /// Record that the user allowed a conservative Bash command family.
    pub fn always_allow_command_type(&mut self, command_type: &str) {
        self.bash_command_type_allows
            .insert(command_type.to_string());
    }

    /// Clear permissions granted for the previous conversation.
    pub fn reset_session(&mut self) {
        self.session_allows.clear();
        self.bash_command_allows.clear();
        self.bash_command_type_allows.clear();
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
                if bash_command_type(cmd).is_some_and(|command_type| {
                    self.bash_command_type_allows.contains(&command_type.key)
                }) {
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
                        message: format!(
                            "Allow bash{}: {}?",
                            background_hint(input),
                            truncate(cmd, 80)
                        ),
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
                                message: format!(
                                    "bash{}: {}",
                                    background_hint(input),
                                    truncate(cmd, 80)
                                ),
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

fn background_hint(input: &serde_json::Value) -> &'static str {
    if input["background"].as_bool() == Some(true) {
        " (background; stops on session close)"
    } else {
        ""
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
    fn always_allow_groups_known_development_command_types() {
        assert_eq!(
            PermissionResponse::always_allow_for("Bash", &json!({"command": "cargo test --lib"})),
            PermissionResponse::AlwaysAllowCommandType("cargo:test".to_string())
        );
        assert_eq!(
            PermissionResponse::always_allow_for(
                "Bash",
                &json!({"command": "bundle exec rspec spec/models/user_spec.rb"})
            ),
            PermissionResponse::AlwaysAllowCommandType("bundle:exec:rspec".to_string())
        );
        assert_eq!(
            PermissionResponse::always_allow_for(
                "Bash",
                &json!({"command": "nix-shell shell.nix --run 'cargo test --lib'"})
            ),
            PermissionResponse::AlwaysAllowCommandType(
                "nix-shell:nix-shell\u{1f}shell.nix:cargo:test".to_string()
            )
        );
    }

    #[test]
    fn command_type_grant_allows_variants_but_not_sibling_types() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        checker.always_allow_command_type("cargo:test");

        assert!(matches!(
            checker.check(
                "Bash",
                &json!({"command": "cargo test permissions -- --nocapture"}),
                false
            ),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Bash", &json!({"command": "cargo build --release"}), false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn nested_command_grants_are_bound_to_the_wrapper_environment() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        let command_type = bash_command_type("nix-shell shell-a.nix --run 'cargo test --lib'")
            .expect("known nested command type");
        checker.always_allow_command_type(&command_type.key);

        assert!(matches!(
            checker.check(
                "Bash",
                &json!({"command": "nix-shell shell-a.nix --run 'cargo test query'"}),
                false
            ),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check(
                "Bash",
                &json!({"command": "nix-shell shell-b.nix --run 'cargo test query'"}),
                false
            ),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn unsafe_or_compound_commands_keep_exact_scope() {
        for command in [
            "rm -rf target",
            "git clean -fd",
            "cargo test && git push",
            "cargo test > results.txt",
            "gh pr merge 42",
            "RUSTC_WRAPPER=/tmp/wrapper cargo test",
            "/tmp/cargo test",
            "cargo --manifest-path test build",
            "make -f test release",
        ] {
            assert_eq!(
                PermissionResponse::always_allow_for("Bash", &json!({"command": command})),
                PermissionResponse::AlwaysAllowCommand(command.to_string()),
                "{command} must not receive a reusable command-type grant"
            );
        }
    }

    #[test]
    fn always_allow_label_explains_the_actual_scope() {
        assert_eq!(
            PermissionResponse::always_allow_label(
                "Bash",
                &json!({"command": "cargo test permissions"})
            ),
            "(a)lways allow cargo test commands"
        );
        assert_eq!(
            PermissionResponse::always_allow_label("Bash", &json!({"command": "rm -rf target"})),
            "(a)lways allow this exact command"
        );
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
    fn hook_verdict_parsing_accepts_empty_and_documented_json_only() {
        assert_eq!(parse_hook_verdict("  \n").unwrap(), None);
        assert_eq!(
            parse_hook_verdict(r#"{"decision":"deny","reason":"nope"}"#).unwrap(),
            Some((HookDecision::Deny, Some("nope".to_string())))
        );
        assert_eq!(
            parse_hook_verdict(r#"{"decision":"allow"}"#).unwrap(),
            Some((HookDecision::Allow, None))
        );
        assert!(parse_hook_verdict("allow").is_err());
        assert!(parse_hook_verdict(r#"{"decision":"maybe"}"#).is_err());
    }

    #[test]
    fn hook_verdicts_fold_to_the_most_restrictive_and_never_lift_a_deny() {
        let input = json!({"command": "ls"});
        let verdict = |d: HookDecision| ("h".to_string(), d, Some("r".to_string()));

        // deny beats allow and ask
        match apply_hook_verdicts(
            "Bash",
            &input,
            PermissionResult::Allow,
            &[verdict(HookDecision::Allow), verdict(HookDecision::Deny)],
        ) {
            PermissionResult::Deny(reason) => assert!(reason.contains("blocked by hook h: r")),
            _ => panic!("expected Deny"),
        }
        // ask escalates allow
        assert!(matches!(
            apply_hook_verdicts(
                "Bash",
                &input,
                PermissionResult::Allow,
                &[verdict(HookDecision::Ask)]
            ),
            PermissionResult::Ask { .. }
        ));
        // allow clears ask
        let ask = PermissionChecker::ask_for("Bash", &input);
        assert!(matches!(
            apply_hook_verdicts("Bash", &input, ask, &[verdict(HookDecision::Allow)]),
            PermissionResult::Allow
        ));
        // a configured deny is final
        assert!(matches!(
            apply_hook_verdicts(
                "Bash",
                &input,
                PermissionResult::Deny("rule".into()),
                &[verdict(HookDecision::Allow)]
            ),
            PermissionResult::Deny(_)
        ));
        // no verdicts: unchanged
        assert!(matches!(
            apply_hook_verdicts("Bash", &input, PermissionResult::Allow, &[]),
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
