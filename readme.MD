# Pageboygent

**Pagebodygent** can **Arranging, Gathering and Exploring** literature, help you **Better Organize your Growing Expository Notes and Track research trends**.

## Configuration

This project provides a unified config loader in `src/Pageboygent/config.py`.

For a full reference of all configuration keys, see [config_table.md](config_table.md)
(auto-generated on install by `pageboygent-export-config-docs`).

### Quick start

1. Copy [`.env.example`](.env.example) to `.env` and fill in your secrets.
2. (Optional) Copy [`pageboygent.toml.example`](pageboygent.toml.example) to `pageboygent.toml` for shared non-secret config.

Both files are auto-generated on install if they don't already exist.

### Precedence

Configuration sources are merged with this order (high -> low):

1. `overrides` argument passed to `load_settings()`
2. Environment variables
3. `.env` in current working directory
4. `pageboygent.toml` in current working directory
5. User config file:
   - macOS/Linux: `~/.config/pageboygent/config.toml`
   - Windows: `%APPDATA%/pageboygent/config.toml`
6. Built-in defaults

### CLI usage

Show effective config (secrets are masked):

```bash
pageboygent --show-config
```

Validate required keys before running tasks:

```bash
pageboygent --require ds_api_key zotero_key
```

Use a non-default profile and custom config paths:

```bash
pageboygent \
  --profile work \
  --project-config-file ./pageboygent.toml \
  --dotenv-file ./.env
```

