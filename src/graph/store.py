"""P3-11: graph store — Neo4j backend plus an in-memory backend for tests/offline.

``build_graph`` loads Document/Chunk/Entity nodes and HAS_CHUNK/MENTIONS/TREATS/
HAS_SYMPTOM edges. The Neo4j backend uses MERGE with per-label uniqueness
constraints; the in-memory backend keeps identical query semantics so the
retriever and its tests do not need a running database.
"""
from abc import ABC, abstractmethod
import os

from src.graph.schema import EDGE_TYPES, NODE_TYPES, entity_node_type, validate_edge, validate_node
from src.utils.io import read_json, write_json

ENTITY_NODE_TYPES = ("Disease", "Drug", "Symptom", "Treatment")


class GraphStore(ABC):
    """Minimal contract used by GraphRetriever and build_graph."""

    @abstractmethod
    def upsert_node(self, node):
        raise NotImplementedError

    @abstractmethod
    def upsert_edge(self, edge):
        raise NotImplementedError

    @abstractmethod
    def chunks_for_entity(self, entity_id):
        """Return ``[{chunk_id, doc_id}]`` mentioning the entity."""

    @abstractmethod
    def related_entities(self, entity_id):
        """Return entity IDs linked by TREATS/HAS_SYMPTOM edges."""

    @abstractmethod
    def stats(self):
        raise NotImplementedError

    def close(self):
        return None


class InMemoryGraphStore(GraphStore):
    """Deterministic adjacency store; also used to persist/load a small graph."""

    def __init__(self):
        self.nodes, self.edges = {}, {}
        self._entity_chunks, self._entity_related = {}, {}

    def upsert_node(self, node):
        validate_node(node)
        self.nodes[node["node_id"]] = node
        return node

    def upsert_edge(self, edge):
        validate_edge(edge)
        key = (edge["source"], edge["edge_type"], edge["target"])
        self.edges[key] = edge
        if edge["edge_type"] == "MENTIONS":
            self._entity_chunks.setdefault(edge["target"], {})[edge["source"]] = True
        elif edge["edge_type"] in ("TREATS", "HAS_SYMPTOM"):
            self._entity_related.setdefault(edge["source"], set()).add(edge["target"])
            self._entity_related.setdefault(edge["target"], set()).add(edge["source"])
        return edge

    def chunks_for_entity(self, entity_id):
        rows = []
        for chunk_id in sorted(self._entity_chunks.get(entity_id, {})):
            node = self.nodes.get(chunk_id, {})
            rows.append({"chunk_id": chunk_id, "doc_id": node.get("properties", {}).get("doc_id")})
        return rows

    def related_entities(self, entity_id):
        return sorted(self._entity_related.get(entity_id, set()))

    def stats(self):
        return {"backend": "memory", "node_count": len(self.nodes), "edge_count": len(self.edges),
                "chunk_count": sum(1 for node in self.nodes.values() if node["node_type"] == "Chunk"),
                "entity_count": sum(1 for node in self.nodes.values() if node["node_type"] in ENTITY_NODE_TYPES)}

    def to_dict(self):
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges.values())}

    @classmethod
    def from_dict(cls, payload):
        store = cls()
        for node in payload.get("nodes", []):
            store.upsert_node(node)
        for edge in payload.get("edges", []):
            store.upsert_edge(edge)
        return store


NEO4J_ENVIRONMENT = {"uri": "NEO4J_URI", "user": "NEO4J_USER",
                     "password": "NEO4J_PASSWORD", "database": "NEO4J_DATABASE"}
NEO4J_DEFAULTS = {"uri": "bolt://localhost:7687", "user": "neo4j",
                  "password": "neo4j", "database": "neo4j"}
EDGE_MATCH_LABELS = {"HAS_CHUNK": ("Document", "Chunk"), "MENTIONS": ("Chunk", "Entity"),
                     "TREATS": ("Entity", "Entity"), "HAS_SYMPTOM": ("Entity", "Entity")}


class Neo4jGraphStore(GraphStore):
    """Neo4j backend (P3-11). Requires ``pip install -e '.[graph]'`` + a server."""

    def __init__(self, uri=None, user=None, password=None, database=None, *, driver=None):
        self.uri = uri or os.environ.get(NEO4J_ENVIRONMENT["uri"], NEO4J_DEFAULTS["uri"])
        self.user = user or os.environ.get(NEO4J_ENVIRONMENT["user"], NEO4J_DEFAULTS["user"])
        self.password = password or os.environ.get(NEO4J_ENVIRONMENT["password"], NEO4J_DEFAULTS["password"])
        self.database = database or os.environ.get(NEO4J_ENVIRONMENT["database"], NEO4J_DEFAULTS["database"])
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as error:  # pragma: no cover - depends on optional extra
                raise RuntimeError("neo4j driver is required: pip install -e '.[graph]'") from error
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver = driver
        self._counts = {"nodes": 0, "edges": 0}

    def _run(self, query, **params):
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, **params))

    def create_constraints(self):
        for label in ("Document", "Chunk", "Entity"):
            self._run(f"CREATE CONSTRAINT {label.lower()}_node_id IF NOT EXISTS "
                      f"FOR (n:{label}) REQUIRE n.node_id IS UNIQUE")

    def upsert_node(self, node):
        validate_node(node)
        node_type = node["node_type"]
        labels = f"Entity:{node_type}" if node_type in ENTITY_NODE_TYPES else node_type
        self._run(f"MERGE (n:{labels} {{node_id: $node_id}}) SET n += $properties",
                  node_id=node["node_id"], properties=node.get("properties", {}))
        self._counts["nodes"] += 1
        return node

    def upsert_edge(self, edge):
        validate_edge(edge)
        source_label, target_label = EDGE_MATCH_LABELS[edge["edge_type"]]
        properties = dict(edge.get("provenance", {}))
        if edge.get("evidence") is not None:
            properties["evidence"] = edge["evidence"]
        self._run(f"MERGE (a:{source_label} {{node_id: $source}}) "
                  f"MERGE (b:{target_label} {{node_id: $target}}) "
                  f"MERGE (a)-[r:{edge['edge_type']}]->(b) SET r += $properties",
                  source=edge["source"], target=edge["target"], properties=properties)
        self._counts["edges"] += 1
        return edge

    def chunks_for_entity(self, entity_id):
        rows = self._run(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {node_id: $entity_id}) "
            "RETURN c.node_id AS chunk_id, c.doc_id AS doc_id ORDER BY chunk_id", entity_id=entity_id)
        return [{"chunk_id": row["chunk_id"], "doc_id": row["doc_id"]} for row in rows]

    def related_entities(self, entity_id):
        rows = self._run(
            "MATCH (e:Entity {node_id: $entity_id})-[:TREATS|HAS_SYMPTOM]-(o:Entity) "
            "RETURN DISTINCT o.node_id AS entity_id ORDER BY entity_id", entity_id=entity_id)
        return [row["entity_id"] for row in rows]

    def stats(self):
        return {"backend": "neo4j", "uri": self.uri, "database": self.database,
                "node_count": self._counts["nodes"], "edge_count": self._counts["edges"]}

    def close(self):
        self.driver.close()

    @classmethod
    def from_config(cls, config):
        section = config.get("neo4j", config) if isinstance(config, dict) else {}
        return cls(uri=section.get("uri"), user=section.get("user"),
                   password=section.get("password"), database=section.get("database"))


def build_graph(store, documents=(), chunks=(), entities=(), relations=()):
    """Load Document/Chunk/Entity nodes and HAS_CHUNK/MENTIONS/relation edges.

    Entities must carry ``chunk_id`` provenance (P3-09 output). Relation edges are
    validated by the P3-08 schema, so clinical edges always keep explicit evidence.
    """
    seen_doc_ids = set()
    for document in documents:
        seen_doc_ids.add(document["doc_id"])
        store.upsert_node({"node_id": document["doc_id"], "node_type": "Document",
                           "properties": {"title": document.get("title"),
                                          "metadata": document.get("metadata", {})}})
    for chunk in chunks:
        if chunk["doc_id"] not in seen_doc_ids:
            seen_doc_ids.add(chunk["doc_id"])
            store.upsert_node({"node_id": chunk["doc_id"], "node_type": "Document", "properties": {}})
        store.upsert_node({"node_id": chunk["chunk_id"], "node_type": "Chunk",
                           "properties": {"doc_id": chunk["doc_id"], "text": chunk.get("text")}})
        store.upsert_edge({"source": chunk["doc_id"], "target": chunk["chunk_id"], "edge_type": "HAS_CHUNK",
                           "provenance": {"doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"]}})
    for entity in entities:
        canonical_id = entity.get("canonical_id") or entity.get("canonical_candidate")
        entity_type = entity.get("canonical_type") or entity.get("type")
        if not canonical_id or not entity_type:
            raise ValueError(f"Entity needs canonical id and type: {entity!r}")
        store.upsert_node({"node_id": canonical_id, "node_type": entity_node_type(entity_type),
                           "properties": {"canonical_name": entity.get("canonical_name", canonical_id),
                                          "surface": entity.get("surface")}})
        store.upsert_edge({"source": entity["chunk_id"], "target": canonical_id, "edge_type": "MENTIONS",
                           "provenance": {"chunk_id": entity["chunk_id"], "doc_id": entity.get("doc_id")}})
    for relation in relations:
        store.upsert_edge({**relation})
    return store


def save_graph(path, store):
    """Persist an in-memory graph as JSON."""
    if not isinstance(store, InMemoryGraphStore):
        raise TypeError("Only the in-memory graph can be serialized to JSON")
    payload = store.to_dict()
    write_json(path, payload)
    return payload


def load_graph(path):
    """Load a JSON graph into an in-memory store."""
    return InMemoryGraphStore.from_dict(read_json(path))
