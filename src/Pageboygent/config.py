"""
AI Generated Configuration Management System for Pageboygent
Not fully tested, may contain bugs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

APP_NAME = "pageboygent"
ENV_PREFIX = "PAGEBOYGENT_"


@dataclass(frozen=True)
class ConfigItem:
    """Metadata for a configuration item."""
    name: str
    type_: type = str
    default: Any = None
    env_name: str | None = None
    is_secret: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """Validate configuration item."""
        if not self.name:
            raise ValueError("Configuration name cannot be empty")

    def get_env_name(self) -> str:
        """Get environment variable name, defaulting from config name."""
        if self.env_name:
            return self.env_name
        return f"{ENV_PREFIX}{self.name.upper()}"


class ConfigRegistry:
    """Registry for managing configuration items."""

    def __init__(self) -> None:
        self._items: dict[str, ConfigItem] = {}
        self._cache: dict[str, Any] = {}

    def register(
        self,
        name: str,
        type_: type = str,
        default: Any = None,
        env_name: str | None = None,
        is_secret: bool = False,
        description: str = "",
    ) -> ConfigItem:
        """
        Register a configuration item.

        Args:
            name: Configuration name (required, used in code).
            type_: Type of the configuration value (default: str).
            default: Default value if not found anywhere (optional).
            env_name: Environment variable name (optional, defaults to "{ENV_PREFIX}{name.upper()}").
            is_secret: Whether this value should be masked in logs (default: False).
            description: Human-readable description (optional).

        Returns:
            The registered ConfigItem.

        Raises:
            ValueError: If name is empty or already registered.
        """
        if not name:
            raise ValueError("Configuration name cannot be empty")
        if name in self._items:
            raise ValueError(f"Configuration '{name}' is already registered")

        item = ConfigItem(
            name=name,
            type_=type_,
            default=default,
            env_name=env_name,
            is_secret=is_secret,
            description=description,
        )
        self._items[name] = item
        # Clear cache when new item is registered
        self._cache.clear()
        return item

    def get(self, name: str, override: Any = None) -> Any:
        """
        Get configuration value by name.

        Precedence (high -> low):
        1. Override value (if provided, overrides everything)
        2. Environment variables (with ENV_PREFIX)
        3. .env file
        4. pageboygent.toml
        5. User config file (~/.config/pageboygent/)
        6. Default value

        Args:
            name: Configuration name.
            override: Optional override value (highest priority).

        Returns:
            Configuration value or default.

        Raises:
            ValueError: If configuration is not registered.
        """
        if name not in self._items:
            raise ValueError(f"Configuration '{name}' is not registered")

        # Override has highest priority
        if override is not None:
            return override

        # Return cached value if available
        if name in self._cache:
            return self._cache[name]

        item = self._items[name]
        value = self._resolve_value(item)

        # Convert type if needed
        if value is not None and item.type_ != str:
            value = self._convert_type(value, item.type_)

        # Cache and return
        self._cache[name] = value
        return value

    def _resolve_value(self, item: ConfigItem) -> Any:
        """Resolve configuration value from various sources."""
        env_name = item.get_env_name()

        # 1. Check environment variables
        if env_name in os.environ:
            env_value = os.environ[env_name]
            if _is_non_empty(env_value):
                return env_value

        # 2. Check .env file
        dotenv_path = _get_dotenv_path()
        dotenv_map = _read_dotenv(dotenv_path)
        if item.name in dotenv_map:
            dotenv_value = dotenv_map[item.name]
            if _is_non_empty(dotenv_value):
                return dotenv_value
        if env_name in dotenv_map:
            dotenv_value = dotenv_map[env_name]
            if _is_non_empty(dotenv_value):
                return dotenv_value

        # 3. Check project config file (pageboygent.toml)
        project_cfg = _read_toml(_get_project_config_path())
        project_value = _get_from_cfg(project_cfg, item.name, _get_active_profile())
        if project_value is not None:
            return project_value

        # 4. Check user config file
        user_cfg = _read_toml(_get_user_config_path())
        user_value = _get_from_cfg(user_cfg, item.name, _get_active_profile())
        if user_value is not None:
            return user_value

        # 5. Return default
        return item.default

    def _convert_type(self, value: Any, target_type: type) -> Any:
        """Convert value to target type."""
        if target_type == str:
            return str(value)
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes", "on")
        else:
            return value

    def list_all(self, include_defaults: bool = False) -> dict[str, Any]:
        """
        List all registered configurations.

        Args:
            include_defaults: Include default values (default: False).

        Returns:
            Dictionary of {name: value} pairs.
        """
        result = {}
        for name, item in self._items.items():
            value = self.get(name)
            if include_defaults or value is not None:
                result[name] = value
        return result

    def display_all(self, include_defaults: bool = False) -> str:
        """
        Get a formatted display of all configurations, masking secrets.

        Args:
            include_defaults: Include default values (default: False).

        Returns:
            Formatted configuration display.
        """
        configs = self.list_all(include_defaults=include_defaults)
        lines = ["=== Configuration Registry ==="]
        for name, value in sorted(configs.items()):
            item = self._items[name]
            if item.is_secret and value:
                display_value = masked(value)
            else:
                display_value = str(value) if value is not None else "<not set>"
            lines.append(f"  {name}: {display_value}")
        return "\n".join(lines)


# Global configuration registry
_registry = ConfigRegistry()

# Active profile (set by load_settings)
_active_profile: str = "default"


def _get_active_profile() -> str:
    return _active_profile


def register_config(
    name: str,
    type_: type = str,
    default: Any = None,
    env_name: str | None = None,
    is_secret: bool = False,
    description: str = "",
) -> ConfigItem:
    """
    Register a new configuration item.

    Example:
        register_config("zotero_id")
        register_config("ds_api_key", is_secret=True)
    """
    return _registry.register(name, type_, default, env_name, is_secret, description)


def get_config(key: str, override: Any = None) -> Any:
    """
    Get a configuration value by key.

    Args:
        key: Configuration name.
        override: Optional override value (highest priority).

    Returns:
        Configuration value or default.

    Raises:
        ValueError: If configuration is not registered.
    """
    return _registry.get(key, override)


def list_all_configs(include_defaults: bool = False) -> dict[str, Any]:
    """List all registered configurations."""
    return _registry.list_all(include_defaults)


def display_all_configs(include_defaults: bool = False) -> str:
    """Display all configurations with secrets masked."""
    return _registry.display_all(include_defaults)


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


# Register built-in configuration items
def _initialize_default_configs_if_needed() -> None:
    """Ensure default configs are registered (idempotent)."""
    if not _registry._items:
        _initialize_default_configs()


def _initialize_default_configs() -> None:
    """Initialize default configuration items. Called once at module load."""
    register_config(
        "zotero_id",
        type_=str,
        is_secret=True,
        env_name=f"{ENV_PREFIX}ZOTERO_ID",
        description="Zotero user ID for the local API (empty = 0, the library of the running Zotero desktop app)",
    )
    register_config(
        "zotero_local_api_key",
        type_=str,
        is_secret=True,
        default=None,
        env_name=f"{ENV_PREFIX}ZOTERO_LOCAL_API_KEY",
        description=(
            "Pre-authorized Zotero local API key (obtained once via Zotero's "
            "'Always Allow' local-write authorization dialog), used to skip the "
            "interactive authorization prompt in headless/CI environments. If unset, "
            "Pageboygent requests authorization interactively on first write and "
            "caches the granted key under workstation_dir/zotero_local_api_key.txt."
        ),
    )
    register_config(
        "mineru_token",
        type_=str,
        is_secret=True,
        env_name=f"{ENV_PREFIX}MINERU_TOKEN",
        description="MineRU API token for PDF processing",
    )
    register_config(
        "mineru_access_key",
        type_=str,
        is_secret=True,
        env_name=f"{ENV_PREFIX}MINERU_ACCESS_KEY",
        description="OpenXLab Access Key ID; with mineru_secret_key, used as a fallback when mineru_token is missing or rejected (the mineru-worker exchanges it for short-lived MinerU JWTs automatically)",
    )
    register_config(
        "mineru_secret_key",
        type_=str,
        is_secret=True,
        env_name=f"{ENV_PREFIX}MINERU_SECRET_KEY",
        description="OpenXLab Secret Access Key paired with mineru_access_key",
    )
    register_config(
        "ds_api_key",
        type_=str,
        is_secret=True,
        env_name=f"{ENV_PREFIX}DS_API_KEY",
        description="DeepSeek API key",
    )
    register_config(
        "ds_url",
        type_=str,
        default="https://api.deepseek.com",
        env_name=f"{ENV_PREFIX}DS_URL",
        description="DeepSeek API endpoint URL",
    )
    register_config(
        "embedding_url",
        type_=str,
        default="http://127.0.0.1:11434/v1",
        description="URL for embedding service (e.g. Ollama or OpenAI)",
        env_name=f"{ENV_PREFIX}EMBEDDING_URL",
    )
    register_config(
        "embedding_model",
        type_=str,
        default="qwen3-embedding:0.6b",
        description="Model name for generating embeddings",
        env_name=f"{ENV_PREFIX}EMBEDDING_MODEL",
    )
    register_config(
        "embedding_key",
        type_=str,
        is_secret=True,
        description="API key for embedding service (e.g. Ollama or OpenAI)",
        env_name=f"{ENV_PREFIX}EMBEDDING_APIKEY",
    )
    register_config(
        "embedding_dim",
        type_=int,
        default=1024,
        description="Dimension of the embedding vectors produced by the embedding model",
        env_name=f"{ENV_PREFIX}EMBEDDING_DIM",
    )
    register_config(
        "max_chunk_size",
        type_=int,
        default=1024,
        description="Maximum number of characters in a text chunk to be embedded. Longer texts will be split into multiple chunks.",
        env_name=f"{ENV_PREFIX}MAX_CHUNK_SIZE",
    )
    register_config(
    "workstation_dir",
    default=str(Path.home() / "PageboygentData"),
    description="Base directory for Pageboygent data storage",
    )
    register_config(
    "pdf_require_collection",
    default="PDF Requirement",
    description="Zotero collection name to put items that require PDFs",
    )
    register_config(
    "mineru_output_dir",
    default=str(Path.home() / "MinerU"),
    description="Directory for MineRU outputs",
    )
    register_config(
    "mineru_result_glob_pattern",
    default="{pdf_name}.pdf-*/full.md",
    description=r"Glob pattern to find MineRU results, where {pdf_name} is replaced by the original PDF filename without extension",
    )
    register_config(
    "pdf2md_mode",
    default="api",
    description='PDF->MD conversion mode: "api" (MinerU official API via the mineru-worker process) or "manual" (move PDFs to human_upload_pdfs for manual conversion)',
    )
    register_config(
    "mineru_api_base",
    default="https://mineru.net/api/v4",
    description="MinerU official API base URL",
    )
    register_config(
    "mineru_model_version",
    default="vlm",
    description='MinerU API model_version: "pipeline" or "vlm"',
    )
    register_config(
    "mineru_upload_accelerate",
    type_=bool,
    default=True,
    description="Upload PDFs via the Aliyun OSS global acceleration endpoint (much faster outside mainland China); falls back to the regional endpoint on failure",
    )
    register_config(
    "mineru_poll_interval_seconds",
    type_=int,
    default=30,
    description="Interval for the mineru-worker to submit queued tasks and poll results",
    )
    register_config(
    "mineru_max_attempts",
    type_=int,
    default=3,
    description="Max attempts per PDF before a MinerU API task is marked failed",
    )
    register_config(
    "tag_for_ai_notes",
    default="AI Notes",
    description="Zotero tag to mark notes that have been generated by Pageboygent's AI",
    )
    register_config(
    "tag_for_quick_summary",
    default="Simple Notes",
    description="Zotero tag to mark notes that have been generated by quick summary.",
    )
    register_config(
    "tag_for_detailed_reading",
    default="Detailed Reading",
    description="Zotero tag to mark notes that have been generated by detailed reading.",
    )
    register_config(
    "model_name_cheap",
    default="deepseek-v4-flash",
    description="Model name for cheap operations like quick summary",
    )
    register_config(
    "model_name_professional",
    default="deepseek-v4-pro",
    description="Model name for professional operations like detailed analysis",
    )
    register_config(
    "lit_search_max_rounds",
    type_=int,
    default=10,
    description="Max LLM rounds (each may issue several tool calls) for the AI literature search agent before it is forced to submit",
    )
    register_config(
    "lit_search_context_budget_tokens",
    type_=int,
    default=200000,
    description="AI literature search soft context limit (tokens). At the start of a follow-up turn, if the last request exceeded this, earlier turns' tool outputs are folded into one-line stubs (evidence IDs stay valid and recallable); if still too long, earlier turns are compacted to summaries",
    )
    register_config(
    "lit_search_context_hard_tokens",
    type_=int,
    default=500000,
    description="AI literature search hard context limit (tokens). Within a turn, when the estimated prompt approaches this, older tool outputs of the current turn are folded too. Keep well below the model's own limit",
    )
    register_config(
    "lit_search_deep_read_max_chars",
    type_=int,
    default=400000,
    description="Max characters of one paper's full text sent to the deep-reading sub-agent (reference list is stripped first; longer texts are truncated)",
    )
    register_config(
    "review_pipeline_concurrency",
    type_=int,
    default=3,
    description="Review pipeline: number of papers read (detailed notes) in parallel",
    )
    register_config(
    "review_pipeline_max_output_tokens",
    type_=int,
    default=32768,
    description="Review pipeline: max output tokens per LLM call (planning, notes, reviews, final review)",
    )
    register_config(
    "feed_fetch_interval_hours",
    type_=float,
    default=6.0,
    description="Paper tracking (RSS): fetch all feeds every N hours (driven by the storage heartbeat)",
    )
    register_config(
    "feed_retention_days",
    type_=int,
    default=60,
    description="Paper tracking (RSS): RSS items older than this are deleted automatically (ingested items and items you rated are kept as training examples)",
    )
    register_config(
    "feed_root_collection",
    default="RSS_TRACK",
    description="Paper tracking (RSS): Zotero collection for ingested papers; each feed gets a sub-collection",
    )
    register_config(
    "feed_auto_ingest_per_run",
    type_=int,
    default=10,
    description="Paper tracking (RSS): max papers auto-ingested per fetch run",
    )
    register_config(
    "feed_llm_batch_size",
    type_=int,
    default=8,
    description="Paper tracking (RSS): papers judged per LLM call (shortlist only)",
    )
    # LLM 费率（元 / 百万 tokens），用于前端预估费用。默认值为 DeepSeek-V4.1-Flash 高峰时段费率。
    for _tier, _label, _defaults in (
        ("cheap", "model_name_cheap", (0.04, 2.0, 8.0)),
        ("professional", "model_name_professional", (0.30, 9.0, 27.0)),   # DeepSeek V4 Pro 高峰时段价
    ):
        for _kind, _desc, _default in zip(
            ("input_cache_hit", "input_cache_miss", "output"),
            ("input tokens that hit the prompt cache", "input tokens that miss the prompt cache", "output tokens"),
            _defaults,
        ):
            register_config(
                f"price_{_tier}_{_kind}",
                type_=float,
                default=_default,
                description=f"Price in CNY per 1M {_desc} for {_label}; used to estimate cost in the UI (unset = no estimate)",
            )
    register_config(
        "openalex_api_key",
        type_=str,
        is_secret=True,
        description="OpenAlex API key (free at openalex.org/settings/api); required since Feb 2026 for more than a tiny daily allowance. Used to enrich RSS items",
    )
    register_config(
        "ads_api_token",
        type_=str,
        is_secret=True,
        description="NASA ADS API token (free at ui.adsabs.harvard.edu → Settings → API Token); enables ADS in deep search's web scope",
    )
    register_config(
        "web_search_collection",
        type_=str,
        default="WEB_SEARCH",
        description="Zotero collection that papers added from deep search's web results go into",
    )
    register_config(
        "ui_language",
        type_=str,
        default="zh",
        description='Language of the web interface: "zh" (中文) or "en" (English); takes effect on page reload',
    )
    register_config(
        "ai_language",
        type_=str,
        default="",
        description='Language of AI-generated text (answers, reviews, reports, profile notes): "zh", "en", or empty to follow ui_language; takes effect after restart',
    )
    register_config(
        "research_field",
        type_=str,
        default="地球化学、地球物理、宇宙化学与行星科学",
        description="Your research field, written into the AI prompts (e.g. '神经科学' or 'materials chemistry')",
    )
    register_config(
    "storage_heartbeat_interval_seconds",
    type_=int,
    default=60,
    description="Heartbeat interval in seconds for storage service background tasks",
    )

@dataclass(frozen=True)
class Settings:
    """Legacy Settings class for backward compatibility."""
    profile: str
    zotero_id: str | None = None
    mineru_token: str | None = None
    ds_api_key: str | None = None
    ds_url: str = "https://api.deepseek.com"

    def require(self, *keys: str) -> "Settings":
        """Validate that required keys are present and non-empty."""
        missing = [key for key in keys if not getattr(self, key, None)]
        if missing:
            key_list = ", ".join(missing)
            raise ConfigError(
                f"Missing required configuration: {key_list}. "
                f"Set environment variables or config file values."
            )
        return self


def load_settings(
    profile: str = "default",
    *,
    config_file: str | Path | None = None,
    dotenv_file: str | Path | None = None,
    project_config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """
    Load configuration using the new registry system (backward compatible).

    Configuration precedence (high -> low):
    1. overrides argument
    2. Environment variables
    3. .env file
    4. pageboygent.toml
    5. User config file
    6. Built-in defaults

    Args:
        profile: Configuration profile name.
        config_file: Path to user config file (overrides default location).
        dotenv_file: Path to .env file (overrides default location).
        project_config_file: Path to project config file.
        overrides: Dictionary of config values to override all other sources.

    Returns:
        Settings object with loaded configuration.
    """
    profile = profile.strip() or "default"
    overrides = overrides or {}

    # Update active profile so _resolve_value picks the right profile section
    global _active_profile
    _active_profile = profile
    _registry._cache.clear()

    # Build Settings from registry, applying overrides
    return Settings(
        profile=profile,
        zotero_id=_normalize_text(get_config("zotero_id", overrides.get("zotero_id"))),
        mineru_token=_normalize_text(get_config("mineru_token", overrides.get("mineru_token"))),
        ds_api_key=_normalize_text(get_config("ds_api_key", overrides.get("ds_api_key"))),
        ds_url=_normalize_text(get_config("ds_url", overrides.get("ds_url"))) or "https://api.deepseek.com",
    )


def _default_user_config_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            base = Path(appdata) / APP_NAME
            return _find_config_file(base) or base / "config.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg) / APP_NAME
        return _find_config_file(base) or base / "config.toml"
    base = Path.home() / ".config" / APP_NAME
    return _find_config_file(base) or base / "config.toml"


def _find_config_file(directory: Path) -> Path | None:
    """Find config.toml in directory, case-insensitively."""
    if not directory.is_dir():
        return None
    for candidate in directory.iterdir():
        if candidate.name.lower() == "config.toml":
            return candidate
    return None


def _default_project_config_path() -> Path:
    return Path.cwd() / "pageboygent.toml"


def _default_dotenv_path() -> Path:
    return Path.cwd() / ".env"


# Configurable paths (can be overridden by CLI)
_custom_user_config_path: Path | None = None
_custom_project_config_path: Path | None = None
_custom_dotenv_path: Path | None = None


def set_config_paths(
    user_config: str | Path | None = None,
    project_config: str | Path | None = None,
    dotenv: str | Path | None = None,
) -> None:
    """Set custom configuration paths. Call before load_settings()."""
    global _custom_user_config_path, _custom_project_config_path, _custom_dotenv_path
    _custom_user_config_path = Path(user_config) if user_config else None
    _custom_project_config_path = Path(project_config) if project_config else None
    _custom_dotenv_path = Path(dotenv) if dotenv else None
    # Clear cache when paths change
    _registry._cache.clear()


def _get_user_config_path() -> Path:
    """Get user config path (custom or default)."""
    return _custom_user_config_path or _default_user_config_path()


def _get_project_config_path() -> Path:
    """Get project config path (custom or default)."""
    return _custom_project_config_path or _default_project_config_path()


def _get_dotenv_path() -> Path:
    """Get dotenv path (custom or default)."""
    return _custom_dotenv_path or _default_dotenv_path()


def _get_from_cfg(cfg: dict[str, Any], key: str, profile: str) -> Any:
    """Get a key from config dict, checking profile section first, then top-level."""
    # Profile-specific value takes precedence over top-level
    profiles = cfg.get("profiles")
    if isinstance(profiles, dict):
        profile_data = profiles.get(profile)
        if isinstance(profile_data, dict):
            val = profile_data.get(key)
            if _is_non_empty(val):
                return val
    # Fall back to top-level key
    val = cfg.get(key)
    if _is_non_empty(val):
        return val
    return None


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or tomllib is None:
        return {}
    with path.open("rb") as fp:
        data = tomllib.load(fp)
    return data if isinstance(data, dict) else {}


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}

    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
            value = value[1:-1]

        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        env[key] = value
    return env


def _is_non_empty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None




def masked(value: str | None, visible: int = 4) -> str:
    """Mask secrets for logs, keeping the tail visible."""
    if not value:
        return "<empty>"
    if len(value) <= visible:
        return "*" * len(value)
    return "*" * (len(value) - visible) + value[-visible:]


def _cli_export_config_docs() -> None:
    """Entry point for `pageboygent-export-config-docs` CLI command."""
    import sys
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "config_table.md"
    export_config_docs(output_path=dest)
    print(f"Configuration reference written to {dest}")


def export_config_examples(dest_dir: str | Path | None = None) -> None:
    """
    Generate .env.example and pageboygent.toml.example in dest_dir.

    Files are only written when they do not already exist, to avoid
    overwriting a user's customised copies.

    Args:
        dest_dir: Directory to write the example files into.
                  Defaults to the current working directory.
    """
    _initialize_default_configs_if_needed()
    base = Path(dest_dir) if dest_dir else Path.cwd()

    # ── .env.example ────────────────────────────────────────────────────────
    env_lines: list[str] = [
        "# .env.example – copy to .env and fill in your values.",
        "# Secret values should be set here (or as real env vars) rather than",
        "# in pageboygent.toml so they are not accidentally committed to VCS.",
        "",
    ]
    for item in _registry._items.values():
        if item.description:
            env_lines.append(f"# {item.description}")
        default_hint = f"  # default: {item.default}" if item.default is not None else ""
        env_lines.append(f"# {item.get_env_name()}=your_{item.name}_here{default_hint}")
        env_lines.append("")
    env_path = base / ".env.example"
    if not env_path.exists():
        env_path.write_text("\n".join(env_lines), encoding="utf-8")
        print(f"[pageboygent] .env.example written to {env_path}")
    else:
        print(f"[pageboygent] .env.example already exists, skipping.")

    # ── pageboygent.toml.example ─────────────────────────────────────────────
    toml_lines: list[str] = [
        "# pageboygent.toml.example – copy to pageboygent.toml and customise.",
        "# Non-secret defaults and profile-specific overrides go here.",
        "# Secrets are better kept in .env or real environment variables.",
        "",
        "# Top-level keys (apply to all profiles unless overridden below)",
    ]
    for item in _registry._items.values():
        if not item.is_secret:
            if item.description:
                toml_lines.append(f"# {item.description}")
            if item.default is not None:
                toml_lines.append(f'{item.name} = "{item.default}"')
            else:
                toml_lines.append(f'# {item.name} = ""')
            toml_lines.append("")
    toml_lines += [
        "[profiles.default]",
        "# Profile-specific overrides. Keys here take precedence over top-level.",
        "",
    ]
    for item in _registry._items.values():
        toml_lines.append(f'# {item.name} = ""')
    toml_lines.append("")
    toml_path = base / "pageboygent.toml.example"
    if not toml_path.exists():
        toml_path.write_text("\n".join(toml_lines), encoding="utf-8")
        print(f"[pageboygent] pageboygent.toml.example written to {toml_path}")
    else:
        print(f"[pageboygent] pageboygent.toml.example already exists, skipping.")


def export_config_docs(output_path: str | Path | None = None) -> str:
    """
    Export configuration documentation as a Markdown table.

    Generates a table describing all registered configuration items,
    including their name, type, default value, environment variable name,
    whether the value is secret, and a description.

    Args:
        output_path: Optional path to write the Markdown file.
                     Defaults to ``config_table.md`` in the current directory.

    Returns:
        The Markdown string that was written.
    """
    _initialize_default_configs_if_needed()
    rows: list[str] = []
    rows.append("# Configuration Reference")
    rows.append("")
    rows.append(
        "This table is auto-generated by `pageboygent-export-config-docs`.\n"
        "Do not edit manually — re-run the command to refresh it."
    )
    rows.append("")
    header = "| Name | Type | Default | Environment Variable | Secret | Description |"
    sep    = "|------|------|---------|----------------------|--------|-------------|"
    rows.append(header)
    rows.append(sep)
    for item in _registry._items.values():
        type_name = item.type_.__name__
        default   = str(item.default) if item.default is not None else "—"
        default   = default.replace(str(Path.home()), "~")   # 文档里不写出生成者本机的用户目录
        env_name  = item.get_env_name()
        secret    = "✓" if item.is_secret else ""
        desc      = item.description.replace("|", "\\|")
        rows.append(f"| `{item.name}` | `{type_name}` | `{default}` | `{env_name}` | {secret} | {desc} |")
    rows.append("")
    content = "\n".join(rows)

    dest = Path(output_path) if output_path else Path.cwd() / "config_table.md"
    dest.write_text(content, encoding="utf-8")
    return content


# Initialize default configurations when module is loaded
_initialize_default_configs()


def research_field() -> str:
    """提示词里描述用户研究方向的那一句（配置项 research_field）。"""
    return str(get_config("research_field") or "").strip() or "科学研究"
