# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# DUAL LICENSED: GNU Affero General Public License v3.0 (AGPL-3.0)
# or Commercial License.
#
# Open-source use is governed by the AGPL-3.0 license. For proprietary
# or commercial applications seeking exemption from copyleft requirements,
# a commercial license must be obtained from INFIDOS LLP.
# Contact: balavignesh@infidos.com | https://infidos.com
# ==============================================================================

"""
Mayon-Cortex Standard Knowledge Schemas
========================================
Standard data models for knowledge ingestion, entity representation,
structured documents, and complete brain knowledge packages.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class StandardKnowledgeItem:
    """
    The atomic unit of structured knowledge (Graph Triplet / Fact).
    Represents an explicit relationship: (source) --[relation]--> (target).
    """
    source: str
    relation: str
    target: str
    sector: str = "general"
    confidence: float = 1.0
    provenance: str = "manual_entry"
    level: str = "FACT"                     # FACT, CONCEPT, ABSTRACT, PHRASE
    node_type: str = "text"                 # text, code, table, image
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.source = str(self.source).strip()
        self.relation = str(self.relation).strip()
        self.target = str(self.target).strip()
        self.sector = str(self.sector).strip().lower()
        self.confidence = float(np_clamp(self.confidence, 0.0, 1.0))
        if self.valid_from is None:
            self.valid_from = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "sector": self.sector,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "level": self.level,
            "node_type": self.node_type,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardKnowledgeItem":
        # Support flexible aliases for source/target/relation
        source = data.get("source") or data.get("subject") or data.get("head") or data.get("entity_a") or ""
        relation = data.get("relation") or data.get("predicate") or data.get("edge_type") or data.get("type") or "connected_to"
        target = data.get("target") or data.get("object") or data.get("tail") or data.get("entity_b") or ""

        if not source or not target:
            raise ValueError(f"Knowledge item missing source or target: {data}")

        return cls(
            source=str(source),
            relation=str(relation),
            target=str(target),
            sector=data.get("sector", "general"),
            confidence=float(data.get("confidence", 1.0)),
            provenance=str(data.get("provenance", "import")),
            level=str(data.get("level", "FACT")),
            node_type=str(data.get("node_type", "text")),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StandardDocumentItem:
    """
    Structured Document container holding narrative text and optional extracted relations.
    """
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    title: Optional[str] = None
    sector: str = "general"
    provenance: str = "document_ingestion"
    entities: List[str] = field(default_factory=list)
    relations: List[StandardKnowledgeItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.text = str(self.text).strip()
        self.sector = str(self.sector).strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "sector": self.sector,
            "provenance": self.provenance,
            "entities": self.entities,
            "relations": [r.to_dict() for r in self.relations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardDocumentItem":
        text = data.get("text") or data.get("content") or data.get("body") or ""
        if not text:
            raise ValueError("Document item missing 'text' or 'content' field")

        relations_raw = data.get("relations", [])
        relations = [
            StandardKnowledgeItem.from_dict(r) if isinstance(r, dict) else r
            for r in relations_raw
        ]

        return cls(
            id=str(data.get("id", str(uuid.uuid4())[:12])),
            title=data.get("title"),
            text=text,
            sector=data.get("sector", "general"),
            provenance=str(data.get("provenance", "document")),
            entities=list(data.get("entities", [])),
            relations=relations,
            metadata=data.get("metadata", {}),
        )


@dataclass
class KnowledgePackage:
    """
    Self-contained portable bundle of facts, documents, and taxonomy metadata.
    Used for brain export, backup, and cross-brain sharing.
    """
    package_id: str = field(default_factory=lambda: f"pkg_{str(uuid.uuid4())[:8]}")
    domain: str = "general"
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    description: str = ""
    items: List[StandardKnowledgeItem] = field(default_factory=list)
    documents: List[StandardDocumentItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "domain": self.domain,
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "total_items": len(self.items),
            "total_documents": len(self.documents),
            "items": [item.to_dict() for item in self.items],
            "documents": [doc.to_dict() for doc in self.documents],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgePackage":
        items_raw = data.get("items", [])
        docs_raw = data.get("documents", [])

        items = [
            StandardKnowledgeItem.from_dict(it) if isinstance(it, dict) else it
            for it in items_raw
        ]
        documents = [
            StandardDocumentItem.from_dict(d) if isinstance(d, dict) else d
            for d in docs_raw
        ]

        return cls(
            package_id=str(data.get("package_id", f"pkg_{str(uuid.uuid4())[:8]}")),
            domain=str(data.get("domain", "general")),
            version=str(data.get("version", "1.0.0")),
            created_at=float(data.get("created_at", time.time())),
            description=str(data.get("description", "")),
            items=items,
            documents=documents,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KnowledgePackage":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def np_clamp(val: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, float(val)))
