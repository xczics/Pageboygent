"""Tests for knowledge_graph.py"""

import json
import tempfile
from pathlib import Path

import pytest

from experimental.knowledge_graph import (
    KnowledgeGraph,
    extract_knowledge_from_text,
    NODE_TYPES,
    RELATION_TYPES,
)


# ===== Fixtures =====


@pytest.fixture
def kg():
    """Create a temporary KnowledgeGraph for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg.db"
        graph = KnowledgeGraph(db_path)
        yield graph
        graph.close()


# ===== KnowledgeGraph Tests =====


class TestKnowledgeGraphInit:
    """Tests for KnowledgeGraph initialization."""

    def test_creates_db_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            graph = KnowledgeGraph(db_path)
            try:
                assert db_path.exists()
            finally:
                graph.close()

    def test_creates_schema(self, kg):
        assert kg.count_nodes() == 0
        assert kg.count_relations() == 0

    def test_reuses_existing_db(self, kg):
        # Second instance with same path reuses schema
        kg.add_node("test", type="claim", confidence=0.8)
        kg2 = KnowledgeGraph(kg._path)
        try:
            assert kg2.count_nodes() == 1
        finally:
            kg2.close()


class TestKnowledgeGraphAddNode:
    """Tests for adding nodes."""

    def test_add_basic_node(self, kg):
        nid = kg.add_node(
            "Narrow ring solves Mars problem",
            type="claim",
            confidence=0.8,
        )
        assert nid.startswith("kn_")
        assert kg.count_nodes() == 1

    def test_add_node_with_source(self, kg):
        source = {"paper_id": "Hansen_2009", "doi": "10.1088/0004-637X/703/1/1131"}
        nid = kg.add_node(
            "Test content",
            type="observation",
            confidence=0.9,
            source=source,
        )
        node = kg.get_node(nid)
        assert node is not None
        assert node["source"]["paper_id"] == "Hansen_2009"

    def test_add_node_with_metadata(self, kg):
        meta = {"scope": "inner_solar_system", "tags": ["Mars", "formation"]}
        nid = kg.add_node(
            "Test with metadata",
            type="theory",
            confidence=0.5,
            metadata=meta,
        )
        node = kg.get_node(nid)
        assert node["metadata"]["scope"] == "inner_solar_system"

    @pytest.mark.parametrize("ntype", list(NODE_TYPES))
    def test_all_node_types(self, kg, ntype):
        nid = kg.add_node(f"test type {ntype}", type=ntype, confidence=0.5)
        node = kg.get_node(nid)
        assert node["type"] == ntype

    def test_invalid_type_raises(self, kg):
        with pytest.raises(ValueError, match="Invalid node type"):
            kg.add_node("test", type="invalid_type")

    def test_invalid_confidence_raises(self, kg):
        with pytest.raises(ValueError):
            kg.add_node("test", confidence=1.5)
        with pytest.raises(ValueError):
            kg.add_node("test", confidence=-0.1)

    def test_multiple_nodes(self, kg):
        ids = []
        for i in range(5):
            nid = kg.add_node(f"node {i}", type="claim", confidence=0.5 + i * 0.1)
            ids.append(nid)
        assert kg.count_nodes() == 5
        assert len(set(ids)) == 5  # All unique


class TestKnowledgeGraphAddRelation:
    """Tests for adding relations."""

    def test_add_basic_relation(self, kg):
        n1 = kg.add_node("First fact", type="claim", confidence=0.8)
        n2 = kg.add_node("Second fact", type="claim", confidence=0.6)
        rid = kg.add_relation(n1, n2, type="supports", confidence=0.9)
        assert rid.startswith("kr_")
        assert kg.count_relations() == 1

    def test_add_all_relation_types(self, kg):
        n1 = kg.add_node("source", type="claim")
        n2 = kg.add_node("target", type="claim")
        for rtype in RELATION_TYPES:
            kg.add_relation(n1, n2, type=rtype, confidence=0.5)
        assert kg.count_relations() == len(RELATION_TYPES)

    def test_node_not_found_raises(self, kg):
        n1 = kg.add_node("exists", type="claim")
        with pytest.raises(ValueError, match="Node"):
            kg.add_relation(n1, "nonexistent", type="supports")


class TestKnowledgeGraphQuery:
    """Tests for querying nodes."""

    def test_query_all(self, kg):
        for i in range(3):
            kg.add_node(f"node {i}", type="claim", confidence=0.5)
        results = kg.query()
        assert len(results) == 3

    def test_query_by_type(self, kg):
        kg.add_node("a claim", type="claim", confidence=0.8)
        kg.add_node("an observation", type="observation", confidence=0.7)
        kg.add_node("a method", type="method", confidence=0.6)

        claims = kg.query(types=["claim"])
        assert len(claims) == 1
        assert claims[0]["type"] == "claim"

        multi = kg.query(types=["claim", "observation"])
        assert len(multi) == 2

    def test_query_by_confidence(self, kg):
        for i in range(5):
            kg.add_node(f"node {i}", type="claim", confidence=i * 0.25)
        high = kg.query(min_confidence=0.75)
        assert len(high) == 2  # Node 3 (0.75) and node 4 (1.0)

    def test_query_by_keyword(self, kg):
        kg.add_node("Mars is small", type="claim", confidence=0.9)
        kg.add_node("Venus is similar to Earth", type="claim", confidence=0.8)
        kg.add_node("Jupiter is large", type="claim", confidence=0.7)

        results = kg.query(keyword="Mars")
        assert len(results) == 1
        assert "Mars" in results[0]["content"]

    def test_query_by_source(self, kg):
        kg.add_node("fact 1", type="claim", source={"paper_id": "PaperA"})
        kg.add_node("fact 2", type="claim", source={"paper_id": "PaperB"})
        results = kg.query(source_paper="PaperA")
        assert len(results) == 1

    def test_query_pagination(self, kg):
        for i in range(10):
            kg.add_node(f"node {i}", type="claim", confidence=i / 10)
        results = kg.query(limit=3, offset=0)
        assert len(results) == 3


class TestKnowledgeGraphRelations:
    """Tests for querying relations."""

    def test_get_relations_by_node(self, kg):
        n1 = kg.add_node("source", type="claim")
        n2 = kg.add_node("target1", type="claim")
        n3 = kg.add_node("target2", type="claim")
        kg.add_relation(n1, n2, type="supports")
        kg.add_relation(n1, n3, type="refines")

        rels = kg.get_relations(node_id=n1)
        assert len(rels) == 2

        rels_n2 = kg.get_relations(node_id=n2)
        assert len(rels_n2) == 1

    def test_get_relations_by_type(self, kg):
        n1 = kg.add_node("source", type="claim")
        n2 = kg.add_node("target", type="claim")
        n3 = kg.add_node("other", type="claim")
        kg.add_relation(n1, n2, type="supports")
        kg.add_relation(n1, n3, type="contradicts")

        supports = kg.get_relations(types=["supports"])
        assert len(supports) == 1
        contradicts = kg.get_relations(types=["contradicts"])
        assert len(contradicts) == 1


class TestKnowledgeGraphDelete:
    """Tests for deleting nodes."""

    def test_delete_node_removes_relations(self, kg):
        n1 = kg.add_node("node 1", type="claim")
        n2 = kg.add_node("node 2", type="claim")
        kg.add_relation(n1, n2, type="supports")
        assert kg.count_relations() == 1

        deleted = kg.delete_node(n1)
        assert deleted is True
        assert kg.count_nodes() == 1
        assert kg.count_relations() == 0  # Relation was cascade-deleted

    def test_delete_nonexistent_node(self, kg):
        deleted = kg.delete_node("nonexistent")
        assert deleted is False


class TestKnowledgeGraphStats:
    """Tests for statistics."""

    def test_stats(self, kg):
        kg.add_node("claim1", type="claim", confidence=0.9)
        kg.add_node("claim2", type="claim", confidence=0.8)
        kg.add_node("obs1", type="observation", confidence=0.7)
        n1 = kg.add_node("method1", type="method", confidence=0.5)
        n2 = kg.add_node("data1", type="data_point", confidence=0.6)
        kg.add_relation(n1, n2, type="supports")

        stats = kg.stats()
        assert stats["total_nodes"] == 5
        assert stats["total_relations"] == 1
        assert stats["nodes_by_type"]["claim"] == 2
        assert stats["nodes_by_type"]["method"] == 1


# ===== Knowledge Extraction Tests =====


class TestExtractKnowledge:
    """Tests for text-based knowledge extraction."""

    def test_extract_from_bullets(self):
        text = """
## Key Results
- The narrow ring model successfully reproduces Mars mass
- Two-source models show better isotopic agreement
## Methods
- N-body simulations with 1000 planetesimals
"""
        nodes, relations = extract_knowledge_from_text(text)
        assert len(nodes) >= 2  # Should extract bullet points

        # Check types based on section
        node_types = {n["type"] for n in nodes}
        assert "claim" in node_types or "observation" in node_types

    def test_extract_from_numbered(self):
        text = """
## Results
1. The simulation produced Earth-like planets in 80% of runs
2. Mercury analogs formed in only 30% of narrow ring simulations
3. Mars mass was consistently below 0.3 Earth masses
"""
        nodes, relations = extract_knowledge_from_text(text)
        assert len(nodes) >= 2  # Should extract numbered items

    def test_extract_with_source(self):
        text = """
- Key finding about disk formation
"""
        nodes, relations = extract_knowledge_from_text(text, source_paper_id="Test_2024")
        for node in nodes:
            assert node["source"]["paper_id"] == "Test_2024"

    def test_short_lines_skipped(self):
        text = "- hi\n- short\n## Long section\n- This is a sufficiently long finding to be extracted"
        nodes, relations = extract_knowledge_from_text(text)
        assert len(nodes) == 1  # Only the last line is long enough

    def test_relations_always_empty(self):
        """Current extraction doesn't auto-detect relations."""
        text = "- Finding one\n- Finding two\n"
        nodes, relations = extract_knowledge_from_text(text)
        assert len(relations) == 0
