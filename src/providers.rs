//! Built-in defaults and capabilities, not overrides for configured transports.
use clap::{builder::PossibleValue, ValueEnum};

use crate::config::{ModelBinding, OpenAIProtocol, ProviderKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthFlow {
    None,
    ApiKey,
    OpenRouterPkce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageIntegration {
    OpenRouter,
    OpenCodeGo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogIntegration {
    OpenRouter,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BuiltinProvider {
    #[default]
    Anthropic,
    Openai,
    OpenRouter,
    OpenCodeGo,
    Vercel,
    Ollama,
}

pub struct ProviderDescriptor {
    pub provider: BuiltinProvider,
    pub id: &'static str,
    pub label: &'static str,
    pub aliases: &'static [&'static str],
    pub kind: ProviderKind,
    pub base_url: &'static str,
    pub api_key_env: &'static str,
    pub legacy_key_envs: &'static [&'static str],
    pub protocol: OpenAIProtocol,
    pub auth: AuthFlow,
    pub usage: Option<(UsageIntegration, &'static str)>,
    pub catalog: Option<CatalogIntegration>,
    pub requires_api_key: bool,
    pub prompt_caching: bool,
    pub default_model: &'static str,
}

pub static PROVIDERS: &[ProviderDescriptor] = &[
    ProviderDescriptor {
        provider: BuiltinProvider::Anthropic,
        id: "anthropic",
        label: "Anthropic",
        aliases: &[],
        kind: ProviderKind::Anthropic,
        base_url: "https://api.anthropic.com/v1",
        api_key_env: "ANTHROPIC_API_KEY",
        legacy_key_envs: &[],
        protocol: OpenAIProtocol::ChatCompletions,
        auth: AuthFlow::None,
        usage: None,
        catalog: None,
        requires_api_key: true,
        prompt_caching: false,
        default_model: "claude-sonnet-5",
    },
    ProviderDescriptor {
        provider: BuiltinProvider::Openai,
        id: "openai",
        label: "OpenAI",
        aliases: &[],
        kind: ProviderKind::Openai,
        base_url: "https://api.openai.com/v1",
        api_key_env: "OPENAI_API_KEY",
        legacy_key_envs: &[],
        protocol: OpenAIProtocol::Responses,
        auth: AuthFlow::None,
        usage: None,
        catalog: None,
        requires_api_key: true,
        prompt_caching: false,
        default_model: "gpt-5.6-sol",
    },
    ProviderDescriptor {
        provider: BuiltinProvider::OpenRouter,
        id: "openrouter",
        label: "OpenRouter",
        aliases: &[],
        kind: ProviderKind::Openai,
        base_url: "https://openrouter.ai/api/v1",
        api_key_env: "OPENROUTER_API_KEY",
        legacy_key_envs: &[],
        protocol: OpenAIProtocol::ChatCompletions,
        auth: AuthFlow::OpenRouterPkce,
        usage: Some((
            UsageIntegration::OpenRouter,
            "https://openrouter.ai/api/v1/key",
        )),
        catalog: Some(CatalogIntegration::OpenRouter),
        requires_api_key: true,
        prompt_caching: true,
        default_model: "anthropic/claude-sonnet-5",
    },
    ProviderDescriptor {
        provider: BuiltinProvider::OpenCodeGo,
        id: "opencode-go",
        label: "OpenCode Go",
        aliases: &["opencode"],
        kind: ProviderKind::Openai,
        base_url: "https://opencode.ai/zen/go/v1",
        api_key_env: "OPENCODE_GO_API_KEY",
        legacy_key_envs: &["OPENCODE_API_KEY"],
        protocol: OpenAIProtocol::ChatCompletions,
        auth: AuthFlow::ApiKey,
        usage: Some((
            UsageIntegration::OpenCodeGo,
            "https://opencode.ai/zen/go/v1/usage",
        )),
        catalog: None,
        requires_api_key: true,
        prompt_caching: false,
        default_model: "glm-5.3",
    },
    ProviderDescriptor {
        provider: BuiltinProvider::Vercel,
        id: "vercel",
        label: "Vercel AI Gateway",
        aliases: &["vercel-ai-gateway"],
        kind: ProviderKind::Openai,
        base_url: "https://ai-gateway.vercel.sh/v1",
        api_key_env: "AI_GATEWAY_API_KEY",
        legacy_key_envs: &[],
        protocol: OpenAIProtocol::ChatCompletions,
        auth: AuthFlow::ApiKey,
        usage: None,
        catalog: None,
        requires_api_key: true,
        prompt_caching: false,
        default_model: "zai/glm-5.3-flash",
    },
    ProviderDescriptor {
        provider: BuiltinProvider::Ollama,
        id: "ollama",
        label: "Ollama",
        aliases: &[],
        kind: ProviderKind::Openai,
        base_url: "http://localhost:11434/v1",
        api_key_env: "OLLAMA_API_KEY",
        legacy_key_envs: &[],
        protocol: OpenAIProtocol::ChatCompletions,
        auth: AuthFlow::None,
        usage: None,
        catalog: None,
        requires_api_key: false,
        prompt_caching: false,
        default_model: "llama3",
    },
];

impl BuiltinProvider {
    pub fn descriptor(self) -> &'static ProviderDescriptor {
        PROVIDERS
            .iter()
            .find(|p| p.provider == self)
            .expect("registered provider")
    }
}

impl ProviderDescriptor {
    pub fn environment_key(&self) -> Option<String> {
        self.key_from_environment(|name| std::env::var(name).ok())
    }

    fn key_from_environment(&self, lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
        std::iter::once(self.api_key_env)
            .chain(self.legacy_key_envs.iter().copied())
            .find_map(|name| lookup(name).filter(|value| !value.trim().is_empty()))
    }

    pub fn api_key(&self) -> anyhow::Result<Option<String>> {
        if let Some(key) = self.environment_key() {
            return Ok(Some(key));
        }
        if self.auth == AuthFlow::None {
            return Ok(None);
        }
        crate::auth::read_provider_key(self.id)
    }
}

impl ValueEnum for BuiltinProvider {
    fn value_variants<'a>() -> &'a [Self] {
        static VALUES: std::sync::OnceLock<Vec<BuiltinProvider>> = std::sync::OnceLock::new();
        VALUES.get_or_init(|| PROVIDERS.iter().map(|p| p.provider).collect())
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        let descriptor = self.descriptor();
        Some(PossibleValue::new(descriptor.id).aliases(descriptor.aliases.iter().copied()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthProvider(pub BuiltinProvider);

impl ValueEnum for AuthProvider {
    fn value_variants<'a>() -> &'a [Self] {
        static VALUES: std::sync::OnceLock<Vec<AuthProvider>> = std::sync::OnceLock::new();
        VALUES.get_or_init(|| {
            PROVIDERS
                .iter()
                .filter(|p| p.auth != AuthFlow::None)
                .map(|p| Self(p.provider))
                .collect()
        })
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        self.0.to_possible_value()
    }
}

pub fn named(name: &str) -> Option<&'static ProviderDescriptor> {
    PROVIDERS.iter().find(|p| {
        p.id.eq_ignore_ascii_case(name) || p.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
    })
}

/// Match a parsed host, never a substring in userinfo, a path or a lookalike.
pub fn identify(
    kind: ProviderKind,
    name: &str,
    profile: &str,
    base_url: Option<&str>,
) -> Option<&'static ProviderDescriptor> {
    [name, profile]
        .into_iter()
        .filter_map(named)
        .find(|p| p.kind == kind)
        .or_else(|| {
            let url = reqwest::Url::parse(base_url?).ok()?;
            let host = url.host_str()?;
            PROVIDERS
                .iter()
                .filter(|p| p.kind == kind && p.provider != BuiltinProvider::Ollama)
                .find(|p| {
                    let base = reqwest::Url::parse(p.base_url).expect("built-in URL");
                    let expected = base.host_str().expect("built-in host");
                    host == expected || host.ends_with(&format!(".{expected}"))
                })
        })
}

pub fn for_binding(binding: &ModelBinding) -> Option<&'static ProviderDescriptor> {
    identify(
        binding.provider_kind,
        &binding.provider_name,
        &binding.provider,
        binding.base_url.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_cli_capabilities_come_from_registry() {
        assert_eq!(
            named("OPENCODE").unwrap().provider,
            BuiltinProvider::OpenCodeGo
        );
        assert_eq!(
            named("vercel-ai-gateway").unwrap().provider,
            BuiltinProvider::Vercel
        );
        assert_eq!(BuiltinProvider::value_variants().len(), PROVIDERS.len());
        for provider in AuthProvider::value_variants() {
            assert_ne!(provider.0.descriptor().auth, AuthFlow::None);
        }
        assert!(AuthProvider::from_str("ollama", false).is_err());
        assert!(AuthProvider::from_str("opencode", false).is_ok());
    }

    #[test]
    fn endpoint_detection_does_not_match_spoofed_hosts_or_paths() {
        for url in [
            "https://openrouter.ai.evil.test/v1",
            "https://evil.test/openrouter.ai",
            "https://openrouter.ai@evil.test/v1",
        ] {
            assert!(identify(ProviderKind::Openai, "gateway", "gateway", Some(url)).is_none());
        }
        assert_eq!(
            identify(
                ProviderKind::Openai,
                "gateway",
                "gateway",
                Some("https://openrouter.ai/api/v1")
            )
            .unwrap()
            .provider,
            BuiltinProvider::OpenRouter
        );
        assert!(identify(ProviderKind::Anthropic, "openrouter", "gateway", None).is_none());
        assert!(identify(
            ProviderKind::Openai,
            "local",
            "local",
            Some("http://localhost:8080/v1")
        )
        .is_none());
    }

    #[test]
    fn environment_key_prefers_primary_and_keeps_legacy_aliases() {
        let descriptor = BuiltinProvider::OpenCodeGo.descriptor();
        assert_eq!(
            descriptor.key_from_environment(|_| Some("primary".into())),
            Some("primary".into())
        );
        assert_eq!(
            descriptor.key_from_environment(|name| {
                Some(
                    if name == descriptor.api_key_env {
                        "  "
                    } else {
                        "legacy"
                    }
                    .into(),
                )
            }),
            Some("legacy".into())
        );
        assert_eq!(descriptor.key_from_environment(|_| None), None);
    }

    #[test]
    fn credential_child() {
        let Ok(config) = std::env::var("CLAUX_PROVIDER_TEST_CONFIG") else {
            return;
        };
        let config: crate::config::Config = toml::from_str(&config).unwrap();
        let resolved = config.resolve_model("test").unwrap();
        assert_eq!(
            resolved.resolve_api_key().as_deref(),
            Some(
                std::env::var("CLAUX_PROVIDER_TEST_EXPECTED")
                    .unwrap()
                    .as_str()
            )
        );
        let descriptor = for_binding(&resolved.binding).unwrap();
        assert_eq!(
            descriptor.api_key().unwrap().as_deref(),
            Some(
                std::env::var("CLAUX_PROVIDER_TEST_SHARED")
                    .unwrap()
                    .as_str()
            )
        );
    }

    #[test]
    fn credential_lookup_uses_aliases_and_preserves_explicit_precedence() {
        for (name, provider) in [
            ("renamed", BuiltinProvider::OpenRouter),
            ("opencode", BuiltinProvider::OpenCodeGo),
            ("vercel-ai-gateway", BuiltinProvider::Vercel),
        ] {
            let descriptor = provider.descriptor();
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(descriptor.id);
            std::fs::write(&path, "saved-key\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            }
            for scenario in [
                "saved",
                "blank_env",
                "environment",
                "explicit_env",
                "literal",
            ] {
                let extra = match scenario {
                    "explicit_env" => "api_key_env = 'CLAUX_PROVIDER_TEST_KEY'",
                    "literal" => "api_key = 'literal-key'",
                    _ => "",
                };
                let config = format!(
                    r#"
                    [providers.{name}]
                    type = "openai"
                    base_url = "{}"
                    {extra}
                    [model_profiles.test]
                    provider = "{name}"
                    model = "test-model"
                "#,
                    descriptor.base_url
                );
                let expected = match scenario {
                    "saved" | "blank_env" => "saved-key",
                    "environment" => "environment-key",
                    "explicit_env" => "custom-key",
                    "literal" => "literal-key",
                    _ => unreachable!(),
                };
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                command
                    .args([
                        "--exact",
                        "providers::tests::credential_child",
                        "--nocapture",
                    ])
                    .env("CLAUX_CREDENTIALS_DIR", dir.path())
                    .env("CLAUX_PROVIDER_TEST_CONFIG", config)
                    .env("CLAUX_PROVIDER_TEST_EXPECTED", expected)
                    .env(
                        "CLAUX_PROVIDER_TEST_SHARED",
                        if matches!(scenario, "saved" | "blank_env") {
                            "saved-key"
                        } else {
                            "environment-key"
                        },
                    );
                for descriptor in PROVIDERS {
                    command.env_remove(descriptor.api_key_env);
                    for alias in descriptor.legacy_key_envs {
                        command.env_remove(alias);
                    }
                }
                if scenario == "blank_env" {
                    command.env(descriptor.api_key_env, "  ");
                } else if scenario != "saved" {
                    command.env(descriptor.api_key_env, "environment-key");
                }
                command.env("CLAUX_PROVIDER_TEST_KEY", "custom-key");
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "{name}/{scenario}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }
}
