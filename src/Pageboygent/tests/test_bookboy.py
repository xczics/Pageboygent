"""Tests for the bookboy agent heartbeat loop (bookboy_heartbeat.py).

Now uses KnowledgeGraph instead of reflections.md for persistence.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from Pageboygent.bookboy_heartbeat import (
    BookBoyConfig,
    BookBoyRuntime,
    load_config,
    heartbeat,
    reflect_to_graph,
    query_knowledge,
)
from Pageboygent.knowledge_graph import KnowledgeGraph


# ===== Fixtures =====


@pytest.fixture
def test_dir():
    """Create a temporary directory for all bookboy data."""
    tmp = Path(tempfile.mkdtemp(prefix="bookboy_beat_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_config():
    """A minimal valid bookboy config."""
    return {
        "name": "chem_boy",
        "system_prompt": "You are a chemistry assistant.",
        "model": "deepseek-v4-flash",
        "heartbeat_interval": 300,
        "mail_base_dir": "{workstation}/pageboys",
    }


@pytest.fixture
def config_path(test_dir, sample_config):
    """Write a sample config to test_dir and return the path."""
    path = test_dir / "chem_boy.json"
    with open(path, "w") as f:
        json.dump(sample_config, f, ensure_ascii=False)
    return path


# ===== BookBoyConfig tests =====


class TestBookBoyConfig:
    def test_load_valid_config(self, config_path, test_dir):
        cfg = load_config(config_path, test_dir)
        assert isinstance(cfg, BookBoyConfig)
        assert cfg.name == "chem_boy"
        assert cfg.system_prompt == "You are a chemistry assistant."
        assert cfg.model == "deepseek-v4-flash"

    def test_load_missing_file(self, test_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(test_dir / "nonexistent.json", test_dir)

    def test_load_missing_required_field(self, test_dir):
        bad_cfg = {"name": "incomplete"}
        path = test_dir / "bad.json"
        with open(path, "w") as f:
            json.dump(bad_cfg, f)
        with pytest.raises(ValueError, match="system_prompt"):
            load_config(path, test_dir)

    def test_resolves_mail_base_dir(self, config_path, test_dir):
        cfg = load_config(config_path, test_dir)
        expected = test_dir / "pageboys"
        assert cfg.resolved_mail_base == expected

    def test_heartbeat_interval_default(self, test_dir):
        cfg_dict = {
            "name": "test_boy",
            "system_prompt": "Assistant.",
            "model": "deepseek-v4-flash",
        }
        path = test_dir / "test.json"
        with open(path, "w") as f:
            json.dump(cfg_dict, f)
        cfg = load_config(path, test_dir)
        assert cfg.heartbeat_interval == 300  # default


# ===== Knowledge Graph Integration tests =====


class TestKnowledgeGraphIntegration:
    """Tests for reflect_to_graph and query_knowledge."""

    def test_reflect_to_graph_adds_nodes(self, test_dir):
        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assist.",
            model="deepseek-v4-flash",
            heartbeat_interval=300,
            resolved_mail_base=test_dir / "pageboys",
            runtime_dir=test_dir,
        )
        kg = KnowledgeGraph(test_dir / "knowledge_graph.db")
        runtime = BookBoyRuntime(cfg, kg=kg)

        llm_response = """
## Key Results
- Narrow ring model reproduces Mars mass effectively
- Two-source models show better isotopic agreement
"""
        node_ids = reflect_to_graph(runtime, llm_response)
        assert len(node_ids) >= 2
        assert kg.count_nodes() >= 2

    def test_reflect_to_graph_with_source(self, test_dir):
        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assist.",
            model="deepseek-v4-flash",
            runtime_dir=test_dir,
            resolved_mail_base=test_dir / "pageboys",
        )
        kg = KnowledgeGraph(test_dir / "knowledge_graph.db")
        runtime = BookBoyRuntime(cfg, kg=kg)

        response = "## Discussion\n- Key finding about planet formation"
        node_ids = reflect_to_graph(runtime, response, source_paper_id="Hansen_2009")
        assert len(node_ids) > 0

        # Verify source is attached
        node = kg.get_node(node_ids[0])
        assert node["source"].get("paper_id") == "Hansen_2009"

    def test_query_knowledge_returns_results(self, test_dir):
        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assist.",
            model="deepseek-v4-flash",
            runtime_dir=test_dir,
            resolved_mail_base=test_dir / "pageboys",
        )
        kg = KnowledgeGraph(test_dir / "knowledge_graph.db")
        runtime = BookBoyRuntime(cfg, kg=kg)

        # Populate some knowledge
        kg.add_node("Mars is small", type="claim", confidence=0.9)
        kg.add_node("Venus is Earth-like", type="claim", confidence=0.8)

        results = query_knowledge(runtime, keyword="Mars")
        assert len(results) == 1
        assert "Mars" in results[0]["content"]

    def test_query_without_kg_returns_empty(self, test_dir):
        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assist.",
            model="deepseek-v4-flash",
            runtime_dir=test_dir,
            resolved_mail_base=test_dir / "pageboys",
        )
        runtime = BookBoyRuntime(cfg)  # No KG
        results = query_knowledge(runtime, keyword="anything")
        assert results == []

    def test_reflect_to_graph_no_kg_raises(self, test_dir):
        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assist.",
            model="deepseek-v4-flash",
            runtime_dir=test_dir,
            resolved_mail_base=test_dir / "pageboys",
        )
        runtime = BookBoyRuntime(cfg)  # No KG
        with pytest.raises(RuntimeError, match="KnowledgeGraph"):
            reflect_to_graph(runtime, "some response")


# ===== Integration: full heartbeat cycle =====


@patch("Pageboygent.bookboy_heartbeat._call_llm")
class TestHeartbeatCycle:
    def test_heartbeat_no_mail_does_nothing(self, mock_llm, test_dir):
        """No mail in inbox → no LLM call, no knowledge extraction."""
        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assistant.",
            model="deepseek-v4-flash",
            heartbeat_interval=300,
            resolved_mail_base=test_dir / "pageboys",
            runtime_dir=test_dir,
        )
        runtime = BookBoyRuntime(cfg)
        result = heartbeat(runtime, dry_run=True)
        assert result["mails_processed"] == 0
        assert result["dry_run"] is True
        mock_llm.assert_not_called()

    def test_heartbeat_creates_kg_on_first_call(self, mock_llm, test_dir):
        """KG gets initialized during heartbeat if not set."""
        mock_llm.return_value = "Mocked AI response"

        from Pageboygent.mail import send

        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="You are helpful.",
            model="deepseek-v4-flash",
            heartbeat_interval=300,
            resolved_mail_base=test_dir / "pageboys",
            runtime_dir=test_dir,
        )
        send(cfg.resolved_mail_base, "user1", "test_boy", "Question?", "What is chemistry?")

        runtime = BookBoyRuntime(cfg)  # No KG initially
        result = heartbeat(runtime)

        assert result["mails_processed"] == 1
        assert runtime.kg is not None
        # KG file should exist
        assert (test_dir / "knowledge_graph.db").exists()

    def test_heartbeat_processes_inbox(self, mock_llm, test_dir):
        """Mail in inbox → call LLM → auto-reply → extract knowledge."""
        mock_llm.return_value = "## Key Findings\n- Chemistry is the study of matter"

        from Pageboygent.mail import send

        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="You are helpful.",
            model="deepseek-v4-flash",
            heartbeat_interval=300,
            resolved_mail_base=test_dir / "pageboys",
            runtime_dir=test_dir,
        )
        send(cfg.resolved_mail_base, "user1", "test_boy", "Question?", "What is chemistry?")

        runtime = BookBoyRuntime(cfg)
        result = heartbeat(runtime)

        assert result["mails_processed"] == 1
        assert result["errors"] == 0
        # Knowledge was extracted
        assert result["knowledge_nodes_added"] > 0
        mock_llm.assert_called_once()

    def test_heartbeat_handles_llm_error(self, mock_llm, test_dir):
        """LLM failure → error counted, no knowledge extraction."""
        mock_llm.side_effect = RuntimeError("API timeout")

        from Pageboygent.mail import send

        cfg = BookBoyConfig(
            name="test_boy",
            system_prompt="Assistant.",
            model="deepseek-v4-flash",
            heartbeat_interval=300,
            resolved_mail_base=test_dir / "pageboys",
            runtime_dir=test_dir,
        )
        send(cfg.resolved_mail_base, "user1", "test_boy", "Error test", "Trigger error")

        runtime = BookBoyRuntime(cfg)
        result = heartbeat(runtime)

        assert result["mails_processed"] == 0
        assert result["errors"] == 1
        assert result["knowledge_nodes_added"] == 0
