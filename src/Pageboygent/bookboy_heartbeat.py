"""Bookboy Agent — heartbeat loop with knowledge graph persistence.

Each bookboy agent runs a heartbeat cycle:
    1. Load config → check mail → process inbox → extract knowledge → sleep
    
The key difference from a plain agent: bookboys persist structured knowledge
nodes (with content, source, confidence) into a SQLite knowledge graph,
enabling hierarchical querying across sessions.

Config JSON (per bookboy):
    {
        "name": "chem_boy",
        "system_prompt": "You are a chemistry research assistant...",
        "model": "deepseek-v4-flash",
        "heartbeat_interval": 300,  // seconds between beats (default: 300)
        "mail_base_dir": "{workstation}/pageboys"  // {workstation} replaced at runtime
    }
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import get_config
from .knowledge_graph import KnowledgeGraph, extract_knowledge_from_text
from .mail import send as mail_send, check as mail_check


# ===== Config =====


@dataclass
class BookBoyConfig:
    """Runtime configuration for a bookboy agent."""
    name: str
    system_prompt: str
    model: str
    heartbeat_interval: int = 300
    resolved_mail_base: Path = field(default_factory=Path)
    runtime_dir: Path = field(default_factory=Path)


@dataclass
class BookBoyRuntime:
    """Runtime state for a bookboy agent."""
    config: BookBoyConfig
    llm_client: OpenAI | None = None
    kg: KnowledgeGraph | None = None


def load_config(config_path: str | Path, workstation_dir: str | Path) -> BookBoyConfig:
    """Load and validate a bookboy config JSON.

    Args:
        config_path: Path to the JSON config file.
        workstation_dir: The workstation base dir (for {workstation} template).

    Returns:
        A BookBoyConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Bookboy config not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    required = ["name", "system_prompt", "model"]
    for field_name in required:
        if field_name not in data:
            raise ValueError(f"Missing required config field: '{field_name}'")

    # Resolve {workstation} template in mail_base_dir
    raw_mail_base = data.get("mail_base_dir", "{workstation}/pageboys")
    resolved_mail_base = Path(
        raw_mail_base.replace("{workstation}", str(Path(workstation_dir)))
    )

    runtime_dir = path.parent

    return BookBoyConfig(
        name=data["name"],
        system_prompt=data["system_prompt"],
        model=data["model"],
        heartbeat_interval=data.get("heartbeat_interval", 300),
        resolved_mail_base=resolved_mail_base,
        runtime_dir=runtime_dir,
    )


# ===== LLM Interface =====


def _get_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client for DeepSeek."""
    return OpenAI(
        api_key=get_config("ds_api_key"),
        base_url=get_config("ds_url") or "https://api.deepseek.com",
    )


def _call_llm(client: OpenAI, system_prompt: str, user_message: str,
              model: str) -> str:
    """Call the LLM and return the response text.

    Args:
        client: OpenAI-compatible client.
        system_prompt: System prompt for the bookboy.
        user_message: The user's message content.
        model: Model name to use.

    Returns:
        The LLM response text.

    Raises:
        RuntimeError: On API errors.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


# ===== Knowledge Graph Integration =====


def reflect_to_graph(runtime: BookBoyRuntime, llm_response: str,
                     source_paper_id: str | None = None) -> list[str]:
    """Extract structured knowledge from an LLM response and store in the graph.

    Args:
        runtime: The bookboy runtime state (must have .kg initialized).
        llm_response: The LLM response text to extract knowledge from.
        source_paper_id: Optional paper identifier to attach to all extracted nodes.

    Returns:
        List of node IDs that were added.
    """
    if runtime.kg is None:
        raise RuntimeError("KnowledgeGraph not initialized in runtime")

    nodes, relations = extract_knowledge_from_text(
        llm_response, source_paper_id=source_paper_id,
    )

    node_ids: list[str] = []
    for node_data in nodes:
        nid = runtime.kg.add_node(
            content=node_data.get("content", ""),
            type=node_data.get("type", "claim"),
            confidence=node_data.get("confidence", 0.5),
            source=node_data.get("source", {}),
            metadata=node_data.get("metadata", {}),
        )
        node_ids.append(nid)

    # Add relations only if we have LLM-powered extraction in the future
    # For now, simple keyword extraction doesn't detect relations

    return node_ids


def query_knowledge(
    runtime: BookBoyRuntime,
    keyword: str | None = None,
    types: list[str] | None = None,
    min_confidence: float = 0.5,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query the knowledge graph. Can be exposed as a function_call.

    Args:
        runtime: The bookboy runtime.
        keyword: Search keyword in content.
        types: Filter by node type(s).
        min_confidence: Minimum confidence.
        limit: Max results.

    Returns:
        List of matching knowledge nodes.
    """
    if runtime.kg is None:
        return []
    return runtime.kg.query(
        types=types,
        min_confidence=min_confidence,
        keyword=keyword,
        limit=limit,
    )


# ===== Heartbeat Loop =====


def heartbeat(runtime: BookBoyRuntime, dry_run: bool = False) -> dict[str, Any]:
    """Execute one heartbeat cycle.

    Check inbox → process mails with LLM → auto-reply → extract knowledge.

    Args:
        runtime: The bookboy runtime state.
        dry_run: If True, check inbox without processing mails.

    Returns:
        Result dict with:
            - mails_processed: int
            - errors: int
            - knowledge_nodes_added: int
            - dry_run: bool
    """
    result = {
        "mails_processed": 0,
        "errors": 0,
        "knowledge_nodes_added": 0,
        "dry_run": dry_run,
    }

    cfg = runtime.config

    # Initialize knowledge graph if not already set
    if runtime.kg is None:
        db_path = cfg.runtime_dir / "knowledge_graph.db"
        runtime.kg = KnowledgeGraph(db_path)

    mails = mail_check(cfg.resolved_mail_base, cfg.name, folder="inbox")

    if dry_run:
        result["mails_processed"] = len(mails)
        return result

    if not mails:
        return result

    if runtime.llm_client is None:
        runtime.llm_client = _get_llm_client()

    for mail in mails:
        try:
            # Build user message from mail content
            user_msg = (
                f"From: {mail.get('from', 'unknown')}\n"
                f"Subject: {mail.get('subject', '')}\n"
                f"Message: {mail.get('content', '')}"
            )

            # Call LLM
            response = _call_llm(
                runtime.llm_client,
                cfg.system_prompt,
                user_msg,
                cfg.model,
            )

            # Auto-reply to the sender
            mail_send(
                cfg.resolved_mail_base,
                cfg.name,
                mail["from"],
                f"Re: {mail.get('subject', '')}",
                response,
            )

            # Extract structured knowledge into the graph
            source_id = mail.get("source_paper", None) or mail.get("from", None)
            node_ids = reflect_to_graph(runtime, response, source_paper_id=source_id)
            result["knowledge_nodes_added"] += len(node_ids)
            result["mails_processed"] += 1

        except Exception as e:
            result["errors"] += 1

    return result


# ===== CLI Entry Point =====


def run_once(config_path: str | Path, workstation_dir: str | Path | None = None) -> int:
    """Run a single heartbeat for a bookboy agent.

    This is the CLI entry point. Returns 0 on success, 1 on error.

    Args:
        config_path: Path to the bookboy's JSON config file.
        workstation_dir: Override workstation dir (default: from get_config).
    """
    if workstation_dir is None:
        workstation_dir = Path(get_config("workstation_dir"))

    try:
        cfg = load_config(config_path, workstation_dir)
        runtime = BookBoyRuntime(cfg)
        result = heartbeat(runtime)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
