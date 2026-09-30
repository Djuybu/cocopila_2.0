"""P3-08: medical graph schema — node/edge types and provenance validation.

Defines the minimal node set (Document, Chunk, Disease, Drug, Symptom, Treatment)
and edge set (HAS_CHUNK, MENTIONS, TREATS, HAS_SYMPTOM). Clinical relations
(TREATS, HAS_SYMPTOM) are never created from co-occurrence alone: they must carry
explicit textual evidence. P1/P2 modules are not involved here.
"""
from typing import TypedDict

NODE_TYPES = ("Document", "Chunk", "Disease", "Drug", "Symptom", "Treatment")
EDGE_TYPES = ("HAS_CHUNK", "MENTIONS", "TREATS", "HAS_SYMPTOM")
ENTITY_TYPES = ("disease", "drug", "symptom", "treatment")
ENTITY_TO_NODE = {"disease": "Disease", "drug": "Drug", "symptom": "Symptom", "treatment": "Treatment"}
# Edges that require explicit textual evidence, never plain co-occurrence.
EVIDENCE_EDGES = ("TREATS", "HAS_SYMPTOM")
# Edges that must keep the chunk that produced them.
PROVENANCE_EDGES = ("MENTIONS", "TREATS", "HAS_SYMPTOM")


class GraphNode(TypedDict, total=False):
    node_id: str
    node_type: str
    properties: dict


class GraphEdge(TypedDict, total=False):
    source: str
    target: str
    edge_type: str
    evidence: str
    provenance: dict


def entity_node_type(entity_type):
    """Map an extracted entity type to its graph node type."""
    if entity_type not in ENTITY_TO_NODE:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return ENTITY_TO_NODE[entity_type]


def validate_node(node):
    """Validate a node dict; raises ValueError on any violation."""
    if not isinstance(node, dict):
        raise ValueError(f"Node must be an object: {node!r}")
    node_id, node_type = node.get("node_id"), node.get("node_type")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError(f"Node needs a nonempty node_id: {node!r}")
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown node_type {node_type!r}; allowed: {NODE_TYPES}")
    properties = node.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"Node properties must be an object: {node!r}")
    return node


def validate_edge(edge):
    """Validate an edge dict; clinical edges need evidence + chunk provenance."""
    if not isinstance(edge, dict):
        raise ValueError(f"Edge must be an object: {edge!r}")
    source, target, edge_type = edge.get("source"), edge.get("target"), edge.get("edge_type")
    if not isinstance(source, str) or not source:
        raise ValueError(f"Edge needs a nonempty source: {edge!r}")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Edge needs a nonempty target: {edge!r}")
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unknown edge_type {edge_type!r}; allowed: {EDGE_TYPES}")
    if source == target:
        raise ValueError(f"Self edges are not allowed: {edge!r}")
    provenance = edge.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError(f"Edge provenance must be an object: {edge!r}")
    if edge_type in PROVENANCE_EDGES:
        chunk_id = provenance.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"{edge_type} edge requires provenance.chunk_id: {edge!r}")
    if edge_type in EVIDENCE_EDGES:
        evidence = edge.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"{edge_type} edge requires explicit evidence text, not co-occurrence: {edge!r}")
    return edge


def validate_has_chunk_edge(doc_id, chunk_id):
    """HAS_CHUNK must connect an existing document to one of its chunks."""
    return validate_edge({"source": doc_id, "target": chunk_id, "edge_type": "HAS_CHUNK",
                          "provenance": {"doc_id": doc_id, "chunk_id": chunk_id}})


# Direction of clinical edges by entity type, so "X được điều trị bằng Y"
# still yields Drug/Treatment -> Disease.
EDGE_SOURCE_ENTITY_TYPES = {"TREATS": ("drug", "treatment"), "HAS_SYMPTOM": ("disease",)}
EDGE_TARGET_ENTITY_TYPES = {"TREATS": ("disease",), "HAS_SYMPTOM": ("symptom",)}


def orient_edge(edge_type, first, second):
    """Order two entity mentions to match the edge direction, or return ``None``.

    ``None`` means the two mentions do not support this relation, so no edge is
    created. This prevents both reversed edges and unsupported pairings.
    """
    if edge_type not in EDGE_SOURCE_ENTITY_TYPES:
        raise ValueError(f"Edge {edge_type} has no entity-type orientation")
    sources, targets = EDGE_SOURCE_ENTITY_TYPES[edge_type], EDGE_TARGET_ENTITY_TYPES[edge_type]
    if first["type"] in sources and second["type"] in targets:
        return first, second
    if second["type"] in sources and first["type"] in targets:
        return second, first
    return None
