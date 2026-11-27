# Experimental / retired modules

These modules are kept for reference but are **not part of the `pageboygent` package** (they are not installed from PyPI
and nothing in `src/Pageboygent` imports them):

| Module | What it was |
|---|---|
| `bookboy_heartbeat.py`, `knowledge_graph.py`, `mail.py` | An experimental multi-agent "bookboy" system (heartbeat loop, mailboxes, knowledge graph) |
| `paper_query.py` | The first AI chat pipeline, superseded by the agentic literature search (`lit_search.py`) and chat (`chat_agent.py`) |
| `bibimpoter.py` | A one-off script for importing `.bib` files and matching preprints to published versions |
| `zt_smoke_test.py` | A manual smoke test against a live Zotero instance |

They import the main package by absolute name (`from Pageboygent.config import get_config`) and each other as
`experimental.*`, so run them from the repository root, e.g. `python -m pytest experimental/tests`.
