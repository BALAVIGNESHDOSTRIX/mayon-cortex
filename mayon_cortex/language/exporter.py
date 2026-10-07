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
Mayon-Cortex Knowledge Exporter
===============================
Exports brain state into standard interchange formats:
- JSON Knowledge Packages (.json)
- Line-delimited JSON Streams (.jsonl)
- Triplet arrays (.json / .csv)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mayon_cortex.core.graph import MayonGraph
from mayon_cortex.language.schema import (
    KnowledgePackage,
    StandardDocumentItem,
    StandardKnowledgeItem,
)


@dataclass
class ExportReport:
    """Summary of a knowledge export operation."""
    total_nodes_exported: int = 0
    total_edges_exported: int = 0
    file_path: str = ""
    file_size_bytes: int = 0
    format: str = "json"
    sectors_included: List[str] = field(default_factory=list)


class BrainKnowledgeExporter:
    """
    Serializes and exports Mayon graph tissue into standard interchange formats.
    """

    @staticmethod
    def export(
        graph: MayonGraph,
        target_path: Union[str, Path],
        format: str = "json",
        sector: Optional[str] = None,
        min_confidence: float = 0.0,
        domain_name: str = "general",
        description: str = "Exported Mayon-Cortex Brain Knowledge",
    ) -> ExportReport:
        """
        Export graph knowledge to disk.
        
        Args:
            graph: MayonGraph instance.
            target_path: Destination file path.
            format: 'json', 'jsonl', 'csv', 'triplets', or 'package'.
            sector: Optional sector filter (e.g. 'medical', 'network').
            min_confidence: Threshold to omit low-confidence edges.
            domain_name: Name of the knowledge domain.
            description: Narrative description of the package.
        """
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower()

        # 1. Filter and gather edges -> StandardKnowledgeItems
        items: List[StandardKnowledgeItem] = []
        sectors_set = set()
        exported_node_ids = set()

        for edge in graph.edges.values():
            if not edge.is_active or edge.confidence < min_confidence:
                continue

            if sector and sector.lower() not in ("all", "any"):
                if edge.sector.lower() != sector.lower() and edge.sector.lower() != "general":
                    continue

            source_node = graph.get_node(edge.source_id)
            target_node = graph.get_node(edge.target_id)
            if not source_node or not target_node:
                continue

            item = StandardKnowledgeItem(
                source=source_node.source_text,
                relation=edge.relation_type,
                target=target_node.source_text,
                sector=edge.sector,
                confidence=edge.confidence,
                provenance=edge.provenance,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
                metadata=edge.metadata,
            )
            items.append(item)
            sectors_set.add(edge.sector)
            exported_node_ids.add(edge.source_id)
            exported_node_ids.add(edge.target_id)

        # 2. Write based on format
        if fmt in ("package", "json", "triplets"):
            if fmt == "triplets":
                payload = [it.to_dict() for it in items]
            else:
                pkg = KnowledgePackage(
                    domain=domain_name,
                    description=description,
                    items=items,
                    metadata={
                        "exported_nodes": len(exported_node_ids),
                        "exported_edges": len(items),
                        "total_graph_nodes": graph.num_nodes,
                        "total_graph_edges": graph.num_edges,
                    },
                )
                payload = pkg.to_dict()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        elif fmt == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it.to_dict()) + "\n")

        elif fmt == "csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "relation", "target", "sector", "confidence", "provenance"])
                for it in items:
                    writer.writerow([
                        it.source, it.relation, it.target,
                        it.sector, it.confidence, it.provenance
                    ])
        else:
            raise ValueError(f"Unsupported export format: {format}")

        file_size = path.stat().st_size if path.exists() else 0
        return ExportReport(
            total_nodes_exported=len(exported_node_ids),
            total_edges_exported=len(items),
            file_path=str(path),
            file_size_bytes=file_size,
            format=fmt,
            sectors_included=list(sectors_set),
        )
