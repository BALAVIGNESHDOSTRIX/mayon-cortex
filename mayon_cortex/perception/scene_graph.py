# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# PROPRIETARY AND CONFIDENTIAL
# This software and associated documentation files contain proprietary and
# confidential information of INFIDOS LLP and BALAVIGNESH M. Unauthorized
# copying, modification, distribution, transmission, or reproduction of this
# material, via any medium, is strictly prohibited without prior written
# permission from INFIDOS LLP.
# ==============================================================================

"""
Scene Graph Engine — Visual Knowledge Grounding & Spatial Relations
===================================================================
Phase 6 of the Brain-Like Intelligence upgrade.

Converts visual features, bounding boxes, object detections, and spatial layouts
into structured knowledge graph representations:
  - Object Nodes: class, bounding box, color, texture, confidence
  - Spatial Edges: left_of, right_of, above, below, inside, contains, adjacent_to
  - Visual Primitive Grounding: links image elements to conceptual knowledge nodes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VisualObject:
    """An object detected or grounded in a visual scene."""
    object_id: str
    label: str
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in [0, 1]
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)  # e.g., {"color": "blue", "shape": "circle"}
    embedding: Optional[np.ndarray] = None


@dataclass
class SpatialRelation:
    """A spatial or semantic relationship between two visual objects."""
    subject_id: str
    relation: str  # left_of, right_of, above, below, inside, contains, near
    object_id: str
    confidence: float = 1.0


class SceneGraph:
    """
    A structured scene graph representing a visual scene's entities and spatial relationships.
    """

    def __init__(self, scene_id: str = "scene_0"):
        self.scene_id = scene_id
        self.objects: Dict[str, VisualObject] = {}
        self.relations: List[SpatialRelation] = []

    def add_object(self, obj: VisualObject):
        """Add a detected object to the scene graph."""
        self.objects[obj.object_id] = obj

    def add_relation(self, rel: SpatialRelation):
        """Add an explicit relation between two objects."""
        self.relations.append(rel)

    def infer_spatial_relations(self, proximity_threshold: float = 0.2):
        """
        Automatically infer spatial edges (left_of, above, inside, near)
        from bounding box coordinates.
        """
        obj_list = list(self.objects.values())
        n = len(obj_list)

        for i in range(n):
            o1 = obj_list[i]
            x1_min, y1_min, x1_max, y1_max = o1.bbox
            cx1, cy1 = (x1_min + x1_max) / 2.0, (y1_min + y1_max) / 2.0

            for j in range(n):
                if i == j:
                    continue
                o2 = obj_list[j]
                x2_min, y2_min, x2_max, y2_max = o2.bbox
                cx2, cy2 = (x2_min + x2_max) / 2.0, (y2_min + y2_max) / 2.0

                # 1. Check inside / contains
                if x1_min >= x2_min and x1_max <= x2_max and y1_min >= y2_min and y1_max <= y2_max:
                    self.add_relation(SpatialRelation(o1.object_id, "inside", o2.object_id, 0.95))
                    self.add_relation(SpatialRelation(o2.object_id, "contains", o1.object_id, 0.95))
                    continue

                # 2. Horizontal relations
                if cx1 < cx2 - 0.1:
                    self.add_relation(SpatialRelation(o1.object_id, "left_of", o2.object_id, 0.85))
                elif cx1 > cx2 + 0.1:
                    self.add_relation(SpatialRelation(o1.object_id, "right_of", o2.object_id, 0.85))

                # 3. Vertical relations
                if cy1 < cy2 - 0.1:
                    self.add_relation(SpatialRelation(o1.object_id, "above", o2.object_id, 0.85))
                elif cy1 > cy2 + 0.1:
                    self.add_relation(SpatialRelation(o1.object_id, "below", o2.object_id, 0.85))

                # 4. Proximity / Near
                dist = np.hypot(cx1 - cx2, cy1 - cy2)
                if dist <= proximity_threshold:
                    self.add_relation(SpatialRelation(o1.object_id, "adjacent_to", o2.object_id, 0.80))

    def to_triples(self) -> List[Tuple[str, str, str]]:
        """Export scene graph into knowledge triples (Subject, Predicate, Object)."""
        triples = []
        # Object class triples
        for o_id, obj in self.objects.items():
            triples.append((o_id, "is_a", obj.label))
            for attr_name, attr_val in obj.attributes.items():
                triples.append((o_id, f"has_{attr_name}", str(attr_val)))

        # Spatial relations
        for rel in self.relations:
            triples.append((rel.subject_id, rel.relation, rel.object_id))

        return triples

    def describe_scene(self) -> str:
        """Generate a natural language description of the visual scene graph."""
        if not self.objects:
            return "The scene contains no recognized visual objects."

        obj_descriptions = []
        for o_id, obj in self.objects.items():
            attrs = [f"{k}: {v}" for k, v in obj.attributes.items()]
            attr_str = f" with {', '.join(attrs)}" if attrs else ""
            obj_descriptions.append(f"{obj.label} ({o_id}){attr_str}")

        text = f"Scene contains {len(self.objects)} entities: " + ", ".join(obj_descriptions) + ". "

        if self.relations:
            rel_strs = [f"{r.subject_id} is {r.relation} {r.object_id}" for r in self.relations[:6]]
            text += "Key spatial configurations: " + "; ".join(rel_strs) + "."

        return text
