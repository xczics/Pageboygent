<p align="center">
  <img src="https://raw.githubusercontent.com/xczics/Pageboygent/main/docs/logo.svg" alt="Pageboygent logo: a cat page boy peeking over an open book" width="128">
</p>

# Pageboygent

**Pageboygent** turns your Zotero library into a searchable, conversational research workspace. It extracts the full
text of your PDFs, writes AI summaries back into Zotero, indexes everything for retrieval, and adds agentic literature
search, AI-written reviews, a search-engine-style "deep search" and RSS paper tracking on top, all in a local web app.

> 中文简介：Pageboygent 是一个围绕 Zotero 文献库的本地研究助手。它把 PDF 转成全文，生成 AI 总结并写回 Zotero，
> 建立全文与笔记索引，并提供 AI 对话、AI 文献检索、AI 综述、深度搜索和 RSS 文献追踪。界面支持中文和英文。

The web interface is available in **English and Simplified Chinese** (switch with the 中 / EN toggle in the navigation bar). Pageboygent is alpha software, written for one researcher's daily
workflow and now being opened up; expect rough edges.

## Features

| Page | What it does |
|---|---|
| **文献追踪 · Paper tracking** | Reads the RSS feeds you subscribe to, enriches entries via OpenAlex / Crossref, scores them against your library (hybrid dense + keyword similarity, author affinity, your 👍/👎 feedback, an LLM judge on a shortlist), auto-ingests the best into Zotero and queues borderline ones for you. Includes an RSS reader, a weekly report, a self-grouped interest profile you can edit line by line, and a settings page. |
| **AI 对话 · Chat** | Chat with your library. The agent searches full text, summaries and notes by itself and cites what it used; citations are validated against the evidence it actually retrieved. |
| **AI 文献检索 · Literature search** | Ask for literature in natural language; the agent decides how much to search and returns a short answer, ranked candidates with key passages, and secondary references (works your papers cite but you don't have). |
| **AI 综述 · Reviews** | Pick Zotero collections and a topic; it plans sub-topics, writes per-paper notes, sub-topic reviews and a final review. |
| **深度搜索 · Deep search** | A search engine over your library without any LLM: title/author/DOI, full-text BM25, full-text semantic, AI summaries and notes, and whole-paper semantic search, fused with reciprocal-rank fusion. Supports `"phrases"`, `-exclusions`, `author:`, `year:2015-2024` and pasted DOIs. An opt-in **Web** scope also queries OpenAlex, NASA ADS and Crossref (merged and de-duplicated, papers you already have are marked, others can be added to Zotero in one click); nothing is sent outside unless you pick it. |
| **数据库 · Database** | Dashboard of the processing pipeline plus editable tables (papers, themes, traced collections, PDF/full-text status, MinerU tasks). |

Token usage and estimated cost are tracked for every AI feature (default prices are DeepSeek's peak-hour rates for the flash and pro models; change them in Settings).

## How it works

```
Zotero collections you trace
   └─ PDF ──► MinerU (PDF → Markdown) ──► "Full Text MD" attachment in Zotero
                                          ├─ chunks ──► embeddings (Chroma) + BM25 index
                                          └─ LLM ──► quick summary, saved as a Zotero note
Themes (groups of collections) ──► per-theme summaries
RSS feeds ──► enrich ──► score ──► auto-ingest / confirm ──► Zotero "RSS_TRACK/<feed>" ──► same pipeline
```

A background heartbeat runs the pipeline step by step while the web server is up.

## Requirements

- **Python 3.13+** on macOS or Linux (file locks use `fcntl`, so Windows is not supported yet).
- **Zotero 7** desktop, running, with *Settings → Advanced → "Allow other applications on this computer to communicate
  with Zotero"* enabled. Pageboygent talks only to this local API; no zotero.org API key is needed. On the first write,
  Zotero asks you to authorize Pageboygent: choose **Always Allow**.
- An **OpenAI-compatible chat API** for the AI features. Defaults target DeepSeek (a "flash" model for routine work and a
  "pro" model for reviews); any compatible endpoint can be configured.
- An **OpenAI-compatible embedding endpoint**. The default is a local [Ollama](https://ollama.com) with
  `qwen3-embedding:0.6b` at `http://127.0.0.1:11434/v1` (1024 dimensions).
- For PDF → Markdown, a **[MinerU](https://mineru.net) API token** (`pdf2md_mode = "api"`). Alternatively, use
  `pdf2md_mode = "manual"` and convert PDFs yourself.

## Installation

```bash
pip install pageboygent
```

For development:

```bash
git clone https://github.com/xczics/Pageboygent.git
cd Pageboygent
pip install -e ".[test]"
```

## Quick start

1. Start Zotero, then start the server:

   ```bash
   pageboygent run --open-browser                # http://127.0.0.1:8765 by default
   ```

2. The home page is **Deep search**. If required settings are missing, it shows a link to **Paper tracking → Settings**. Fill in at least your chat API key (`ds_api_key`).
   `zotero_id` can stay empty: it then uses the library of the Zotero app running on this computer.
   Also set your **research field** (`research_field`): it is written into every AI prompt. `ai_language` chooses the
   language of AI-generated text (empty = same as the interface). For RSS enrichment, add a free
   [OpenAlex API key](https://openalex.org/settings/api) (`openalex_api_key`). Restart the server after saving.
3. In **数据库 → 追踪集合 (Traced collections)**, add the Zotero collections you want Pageboygent to process. Optionally
   group collections into **主题 (Themes)** to get per-theme summaries.
4. Leave the server running: PDFs are converted, summarized and indexed in the background. Progress is on the
   **数据库** dashboard.
5. On **Paper tracking**, add RSS feeds and click **分组并提炼说明** on the **画像** tab to build your interest profile from your
   library.

## Configuration

Every setting can be edited on **Paper tracking → Settings**, which saves to your user config file. Settings that only apply to another
mode (for example MinerU API options when `pdf2md_mode = "manual"`) are hidden. Most modules read configuration at
start-up, so restart the server after changing it.

Sources, from highest to lowest precedence:

1. Environment variables (`PAGEBOYGENT_<NAME>`, see the reference)
2. `.env` in the working directory
3. `pageboygent.toml` in the working directory
4. User config file: `~/.config/pageboygent/config.toml` (Windows: `%APPDATA%/pageboygent/config.toml`); secrets live under
   `[profiles.<profile>]`
5. Built-in defaults

The full list of settings, with types, defaults and environment variable names, is in
[docs/configuration.md](docs/configuration.md). Regenerate it with `pageboygent-export-config-docs docs/configuration.md`.

## What Pageboygent writes to Zotero

So you know what to expect in your library:

- A **"Full Text MD"** attachment (the MinerU Markdown) on each processed item.
- **Notes**: quick summaries, theme summaries and review notes, tagged with `AI Notes` and friends (configurable).
- Items without a PDF are added to the **`PDF Requirement`** collection (configurable) so you can find them.
- RSS ingests create **new items** in **`RSS_TRACK/<feed name>`** (configurable). If an item with the same DOI or exactly
  the same title already exists, it is reused instead.
- Metadata of items it created is completed later (DOI, authors, journal, date). Fields you have filled in are never
  overwritten.

## Data and privacy

Local data lives in `~/PageboygentData` (`workstation_dir`): SQLite databases, the Chroma vector store, search indexes,
chat and review sessions, and the RSS database.

Outbound requests:

| Service | What is sent |
|---|---|
| Your chat LLM endpoint | Excerpts of your papers, summaries, notes, your questions, your interest profile |
| MinerU (API mode) | The PDFs being converted |
| Embedding endpoint | Text chunks and queries (local when using Ollama) |
| OpenAlex, Crossref | DOIs and titles of RSS entries, for metadata enrichment (OpenAlex needs a free API key since 2026) |
| OpenAlex, NASA ADS, Crossref | Your query, only when you choose the **Web** scope in deep search (ADS needs a free token) |
| RSS feeds | Ordinary feed requests |

Secrets are never sent to the browser; the settings page shows them masked.

## Command line

```bash
pageboygent run [HOST:PORT] [--open-browser]             # web app + background pipeline
pageboygent --show-config                               # effective config, secrets masked
pageboygent --require ds_api_key zotero_id              # check required settings
pageboygent rag-status                                  # status of the chunk-filtering classifier
pageboygent --profile work --project-config-file ./pageboygent.toml --dotenv-file ./.env run
```

## Development

Interface text is translated through `locales/{zh,en}.json`; see [docs/i18n.md](docs/i18n.md) for the key
conventions.

```bash
pytest                      # unit tests (src/Pageboygent/tests)
pytest experimental/tests   # retired / experimental modules, not part of the package
```

## License

Pageboygent is licensed under the [GNU General Public License v3.0 or later](LICENSE).
