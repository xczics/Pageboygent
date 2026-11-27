"""Knowledge Graph — structured, queryable knowledge for bookboy agents.

Stores extracted knowledge as nodes (facts/claims) and relations (links between facts)
in a SQLite database. Supports hierarchical querying by topic, source, confidence, etc.

Schema:
    knowledge_nodes (id, content, type, confidence, source_json, metadata_json,
                     created_at, updated_at)
    knowledge_relations (id, source_id, target_id, type, confidence, metadata_json,
                         created_at)

Type taxonomy:
    claim        — Direct assertion (e.g., "X causes Y")
    observation  — Empirical finding (e.g., "Simulation produced mass < 0.3 M_Earth")
    method       — Technique/approach description
    theory       — Theoretical framework/model
    data_point   — Specific numerical result
    question     — Open question or unresolved issue

Relation types:
    supports         — Node A supports/confirms Node B
    contradicts      — Node A contradicts Node B
    refines          — Node A refines/extends Node B
    depends_on       — Node A depends on Node B
    example_of       — Node A is an example of Node B (hierarchical)
    leads_to         — Node A leads to/inspires Node B
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ===== Constants =====

DEFAULT_DB_FILENAME = "knowledge_graph.db"

NODE_TYPES = frozenset({
    "claim", "observation", "method", "theory", "data_point", "question",
})

RELATION_TYPES = frozenset({
    "supports", "contradicts", "refines", "depends_on", "example_of", "leads_to",
})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    type        TEXT NOT NULL CHECK(type IN ('claim','observation','method',
                                             'theory','data_point','question')),
    confidence  REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    source_json TEXT DEFAULT '{}',
    metadata_json TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_relations (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES knowledge_nodes(id),
    target_id   TEXT NOT NULL REFERENCES knowledge_nodes(id),
    type        TEXT NOT NULL CHECK(type IN ('supports','contradicts','refines',
                                             'depends_on','example_of','leads_to')),
    confidence  REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    metadata_json TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON knowledge_nodes(confidence);
CREATE INDEX IF NOT EXISTS idx_relations_type ON knowledge_relations(type);
CREATE INDEX IF NOT EXISTS idx_relations_source ON knowledge_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON knowledge_relations(target_id);
"""


# ===== Knowledge Graph Manager =====


class KnowledgeGraph:
    """Thread-safe knowledge graph backed by SQLite.

    Usage:
        kg = KnowledgeGraph("/path/to/knowledge_graph.db")
        nid = kg.add_node("Narrow ring model solves Mars problem",
                          type="claim", confidence=0.8,
                          source={"paper_id": "Hansen_2009", "doi": "10.1088/..."})
        kg.add_relation(nid, other_id, type="supports")
        results = kg.query(types=["claim"], min_confidence=0.5, limit=10)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def add_node(
        self,
        content: str,
        type: str = "claim",
        confidence: float = 0.5,
        source: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a knowledge node.

        Args:
            content: The knowledge statement.
            type: One of NODE_TYPES.
            confidence: 0.0 (speculative) to 1.0 (established fact).
            source: Dict with at least paper_id/doi/evidence.
            metadata: Optional extra fields.

        Returns:
            The generated node ID.

        Raises:
            ValueError: On invalid type or confidence.
        """
        if type not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {type}. Must be one of {NODE_TYPES}")
        if not (0 <= confidence <= 1):
            raise ValueError(f"Confidence must be 0-1, got {confidence}")

        nid = _next_id("kn")
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO knowledge_nodes
                   (id, content, type, confidence, source_json, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    nid,
                    content.strip(),
                    type,
                    confidence,
                    json.dumps(source or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()

        return nid

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        type: str = "supports",
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a relation between two knowledge nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            type: One of RELATION_TYPES.
            confidence: 0.0 to 1.0.
            metadata: Optional extra fields.

        Returns:
            The generated relation ID.

        Raises:
            ValueError: On invalid type, confidence, or missing nodes.
        """
        if type not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {type}")
        if not (0 <= confidence <= 1):
            raise ValueError(f"Confidence must be 0-1, got {confidence}")

        rid = _next_id("kr")
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._get_conn()
            # Verify nodes exist
            existing = conn.execute(
                "SELECT id FROM knowledge_nodes WHERE id IN (?, ?)",
                (source_id, target_id),
            ).fetchall()
            if len(existing) < 2:
                missing = []
                if source_id not in {r["id"] for r in existing}:
                    missing.append(f"source_id={source_id}")
                if target_id not in {r["id"] for r in existing}:
                    missing.append(f"target_id={target_id}")
                raise ValueError(f"Node(s) not found: {', '.join(missing)}")

            conn.execute(
                """INSERT INTO knowledge_relations
                   (id, source_id, target_id, type, confidence, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rid, source_id, target_id, type, confidence,
                 json.dumps(metadata or {}, ensure_ascii=False), now),
            )
            conn.commit()

        return rid

    def query(
        self,
        types: list[str] | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        keyword: str | None = None,
        source_paper: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query knowledge nodes with filters.

        Args:
            types: Filter by node type(s). None = all types.
            min_confidence: Minimum confidence threshold.
            max_confidence: Maximum confidence threshold.
            keyword: Full-text search in content.
            source_paper: Filter by source paper_id (JSON field search).
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of node dicts (each with id, content, type, confidence,
            source, metadata, created_at).
        """
        conditions: list[str] = ["1=1"]
        params: list[Any] = []

        if types:
            placeholders = ",".join("?" for _ in types)
            conditions.append(f"type IN ({placeholders})")
            params.extend(types)

        conditions.append("confidence >= ? AND confidence <= ?")
        params.extend([min_confidence, max_confidence])

        if keyword:
            conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")

        if source_paper:
            conditions.append("source_json LIKE ?")
            params.append(f"%{source_paper}%")

        query = f"""
            SELECT id, content, type, confidence, source_json, metadata_json, created_at
            FROM knowledge_nodes
            WHERE {' AND '.join(conditions)}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": r["id"],
                "content": r["content"],
                "type": r["type"],
                "confidence": r["confidence"],
                "source": json.loads(r["source_json"]),
                "metadata": json.loads(r["metadata_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_relations(
        self,
        node_id: str | None = None,
        types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get relations, optionally filtered by node or type.

        Args:
            node_id: If set, return relations involving this node (as source or target).
            types: Filter by relation type(s).
            limit: Max results.

        Returns:
            List of relation dicts.
        """
        conditions: list[str] = ["1=1"]
        params: list[Any] = []

        if node_id:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([node_id, node_id])

        if types:
            placeholders = ",".join("?" for _ in types)
            conditions.append(f"type IN ({placeholders})")
            params.extend(types)

        query = f"""
            SELECT id, source_id, target_id, type, confidence, metadata_json, created_at
            FROM knowledge_relations
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "type": r["type"],
                "confidence": r["confidence"],
                "metadata": json.loads(r["metadata_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a single node by ID."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT id, content, type, confidence, source_json, metadata_json, created_at "
                "FROM knowledge_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "content": row["content"],
            "type": row["type"],
            "confidence": row["confidence"],
            "source": json.loads(row["source_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def count_nodes(self) -> int:
        """Get the total number of nodes."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM knowledge_nodes").fetchone()
        return row["cnt"]

    def count_relations(self) -> int:
        """Get the total number of relations."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_relations"
            ).fetchone()
        return row["cnt"]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all relations involving it.

        Returns True if a node was actually deleted.
        """
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM knowledge_relations WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )
            cursor = conn.execute("DELETE FROM knowledge_nodes WHERE id = ?", (node_id,))
            conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Get summary statistics about the knowledge graph."""
        with self._lock:
            conn = self._get_conn()
            total_nodes = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_nodes"
            ).fetchone()["cnt"]
            total_relations = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_relations"
            ).fetchone()["cnt"]
            type_dist = conn.execute(
                "SELECT type, COUNT(*) as cnt FROM knowledge_nodes "
                "GROUP BY type ORDER BY cnt DESC"
            ).fetchall()
            node_count_by_type = {r["type"]: r["cnt"] for r in type_dist}

        return {
            "total_nodes": total_nodes,
            "total_relations": total_relations,
            "nodes_by_type": node_count_by_type,
            "db_path": str(self._path),
        }


# ===== Helpers =====


_counter = 0
_counter_lock = threading.Lock()


def _next_id(prefix: str) -> str:
    """Generate a unique ID with prefix and timestamp."""
    global _counter
    with _counter_lock:
        _counter += 1
        ts = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S%f")[:16]
        return f"{prefix}_{ts}_{_counter:04d}"


# ===== LLM-Guided Knowledge Extraction =====


EXTRACTION_PROMPT = """You are a knowledge extraction assistant. Given a research output,
extract the key knowledge statements as structured nodes.

For each finding, identify:
1. CONTENT: The specific knowledge statement (one fact per node)
2. TYPE: One of: claim | observation | method | theory | data_point | question
3. CONFIDENCE: 0.0-1.0 based on how established the finding is
4. SOURCE: The paper or source ID if mentioned

Also identify relationships between findings:
1. SOURCE_ID / TARGET_ID (reference by node index, 0-based)
2. TYPE: supports | contradicts | refines | depends_on | example_of | leads_to

Output ONLY valid JSON in this exact format - no other text:
{
  "nodes": [
    {"content": "...", "type": "claim", "confidence": 0.85, "source": {"paper_id": "..."}}
  ],
  "relations": [
    {"source": 0, "target": 1, "type": "supports", "confidence": 0.8}
  ]
}
"""


def extract_knowledge_from_text(
    llm_response: str,
    source_paper_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract structured knowledge from an LLM response using keyword/pattern matching.

    This is a lightweight extraction that doesn't require an LLM call.
    For production use, pair with EXTRACTION_PROMPT and an LLM call.

    Args:
        llm_response: The text to extract knowledge from.
        source_paper_id: Optional paper identifier to attach to all nodes.

    Returns:
        Tuple of (nodes, relations) dicts ready for KnowledgeGraph.add_node/add_relation.
    """
    nodes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    # Simple heuristic extraction based on section headers and key sentences
    lines = llm_response.split("\n")
    current_section: str | None = None
    section_map = {
        "introduction": "theory",
        "method": "method",
        "result": "observation",
        "discussion": "claim",
        "conclusion": "claim",
        "future": "question",
        "outstanding": "question",
    }

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Detect section headers (markdown-style)
        if line_stripped.startswith("###") or line_stripped.startswith("##"):
            section_lower = line_stripped.lower()
            for keyword, node_type in section_map.items():
                if keyword in section_lower:
                    current_section = node_type
                    break
            continue

        # Extract bullet points as knowledge nodes
        if line_stripped.startswith("- ") or line_stripped.startswith("* "):
            content = line_stripped.lstrip("-* ").strip()
            if len(content) > 20:  # Skip very short bullets
                source = {}
                if source_paper_id:
                    source["paper_id"] = source_paper_id
                node = {
                    "content": content,
                    "type": current_section or "claim",
                    "confidence": 0.6,
                    "source": source,
                    "metadata": {},
                }
                nodes.append(node)

        # Extract numbered findings (1. 2. etc.)
        elif (line_stripped[0].isdigit() and ". " in line_stripped[:4]):
            content = line_stripped.split(". ", 1)[1] if ". " in line_stripped else line_stripped
            if len(content) > 30:
                source = {}
                if source_paper_id:
                    source["paper_id"] = source_paper_id
                node = {
                    "content": content,
                    "type": current_section or "observation",
                    "confidence": 0.7,
                    "source": source,
                    "metadata": {},
                }
                nodes.append(node)

    return nodes, relations
