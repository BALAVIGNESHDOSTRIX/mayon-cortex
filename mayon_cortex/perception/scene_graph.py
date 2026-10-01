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
from typing import Any, Dict, List, Optional, Set, Tuple

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


class ARCSpatialSceneGraph(SceneGraph):
    """
    ARC 2D Discrete Grid Spatial Scene Graph.
    Converts 2D integer grids into an attributed spatial relation graph:
    - Nodes: Discrete segmented objects with color, area, bbox, centroid, is_solid, num_holes.
    - Edges: Spatial & topological relations (contains, inside, touches, above, below, left_of, right_of, aligned_h, aligned_v, same_color).
    - Global Attributes: shape, palette, background_color, symmetries, delimiters.
    """

    def __init__(self, scene_id: str = "arc_scene", grid: Optional[np.ndarray] = None):
        super().__init__(scene_id=scene_id)
        self.grid_shape: Tuple[int, int] = (0, 0)
        self.background_color: int = 0
        self.palette: Set[int] = set()
        self.symmetries: Dict[str, bool] = {}
        self.delimiters: Dict[str, List[int]] = {"horizontal": [], "vertical": []}
        if grid is not None:
            self.from_grid(grid)

    def from_grid(self, grid: np.ndarray) -> "ARCSpatialSceneGraph":
        """Build the attributed spatial scene graph from a 2D integer numpy matrix."""
        mat = np.array(grid, dtype=int)
        H, W = mat.shape
        self.grid_shape = (H, W)
        self.palette = set(int(c) for c in np.unique(mat))

        # Detect background color: prioritize border color frequencies
        border_pixels = np.concatenate([mat[0, :], mat[-1, :], mat[:, 0], mat[:, -1]])
        unique_border, counts_border = np.unique(border_pixels, return_counts=True)
        if len(unique_border) > 0:
            bg_color = int(unique_border[np.argmax(counts_border)])
        else:
            bg_color = 0
        self.background_color = bg_color

        # Segment objects via 4-connected BFS for each non-background color
        visited = np.zeros((H, W), dtype=bool)
        obj_idx = 0
        extracted_objects: List[VisualObject] = []

        for r in range(H):
            for c in range(W):
                color = int(mat[r, c])
                if color == bg_color or visited[r, c]:
                    continue

                # BFS for connected component of identical color
                q = [(r, c)]
                visited[r, c] = True
                coords: List[Tuple[int, int]] = []

                while q:
                    curr_r, curr_c = q.pop(0)
                    coords.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and mat[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                rows = [p[0] for p in coords]
                cols = [p[1] for p in coords]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)
                area = len(coords)
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                is_solid = (area == height * width)
                centroid = (float(np.mean(rows)), float(np.mean(cols)))

                # Create 2D binary mask
                obj_mask = np.zeros((H, W), dtype=bool)
                for pr, pc in coords:
                    obj_mask[pr, pc] = True

                # Topological hole/cavity analysis inside bounding box
                sub_h, sub_w = height, width
                submask = obj_mask[min_r:max_r + 1, min_c:max_c + 1]
                padded = np.zeros((sub_h + 2, sub_w + 2), dtype=bool)
                padded[1:sub_h + 1, 1:sub_w + 1] = submask

                # Flood fill inverted background from border (0, 0)
                inv_visited = np.zeros_like(padded, dtype=bool)
                bg_q = [(0, 0)]
                inv_visited[0, 0] = True
                while bg_q:
                    br, bc = bg_q.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nbr, nbc = br + dr, bc + dc
                        if 0 <= nbr < sub_h + 2 and 0 <= nbc < sub_w + 2:
                            if not inv_visited[nbr, nbc] and not padded[nbr, nbc]:
                                inv_visited[nbr, nbc] = True
                                bg_q.append((nbr, nbc))

                # Interior holes are unvisited background cells inside original bbox
                interior_mask = (~padded[1:sub_h + 1, 1:sub_w + 1]) & (~inv_visited[1:sub_h + 1, 1:sub_w + 1])
                num_holes = 0
                hole_visited = np.zeros_like(interior_mask, dtype=bool)
                for hr in range(sub_h):
                    for hc in range(sub_w):
                        if interior_mask[hr, hc] and not hole_visited[hr, hc]:
                            num_holes += 1
                            hq = [(hr, hc)]
                            hole_visited[hr, hc] = True
                            while hq:
                                cr, cc = hq.pop(0)
                                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nhr, nhc = cr + dr, cc + dc
                                    if 0 <= nhr < sub_h and 0 <= nhc < sub_w:
                                        if interior_mask[nhr, nhc] and not hole_visited[nhr, nhc]:
                                            hole_visited[nhr, nhc] = True
                                            hq.append((nhr, nhc))

                obj_id = f"obj_{color}_{obj_idx}"
                obj_idx += 1
                v_obj = VisualObject(
                    object_id=obj_id,
                    label=f"color_{color}",
                    bbox=(float(min_r), float(min_c), float(max_r), float(max_c)),
                    confidence=1.0,
                    attributes={
                        "color": color,
                        "area": area,
                        "height": height,
                        "width": width,
                        "is_solid": is_solid,
                        "num_holes": num_holes,
                        "centroid": centroid,
                        "mask": obj_mask,
                        "bbox_int": (min_r, min_c, max_r, max_c),
                    },
                )
                self.add_object(v_obj)
                extracted_objects.append(v_obj)

        # Infer spatial and topological edges between object pairs (cap at top 30 objects if noisy)
        eval_objects = extracted_objects if len(extracted_objects) <= 30 else sorted(
            extracted_objects, key=lambda o: o.attributes["area"], reverse=True
        )[:30]
        n_eval = len(eval_objects)

        for i in range(n_eval):
            o1 = eval_objects[i]
            r1_min, c1_min, r1_max, c1_max = o1.attributes["bbox_int"]
            cr1, cc1 = o1.attributes["centroid"]
            mask1 = o1.attributes["mask"]

            for j in range(n_eval):
                if i == j:
                    continue
                o2 = eval_objects[j]
                r2_min, c2_min, r2_max, c2_max = o2.attributes["bbox_int"]
                cr2, cc2 = o2.attributes["centroid"]
                mask2 = o2.attributes["mask"]

                # 1. Topological Containment / Inside
                if r1_min >= r2_min and r1_max <= r2_max and c1_min >= c2_min and c1_max <= c2_max:
                    self.add_relation(SpatialRelation(o1.object_id, "inside", o2.object_id, 0.95))
                    self.add_relation(SpatialRelation(o2.object_id, "contains", o1.object_id, 0.95))

                # 2. Touch / Contact adjacency (only if bounding boxes are adjacent)
                if (r1_max + 1 >= r2_min and r1_min - 1 <= r2_max and
                    c1_max + 1 >= c2_min and c1_min - 1 <= c2_max):
                    sub_r1 = max(0, min(r1_min, r2_min) - 1)
                    sub_r2 = min(H, max(r1_max, r2_max) + 2)
                    sub_c1 = max(0, min(c1_min, c2_min) - 1)
                    sub_c2 = min(W, max(c1_max, c2_max) + 2)
                    m1_crop = mask1[sub_r1:sub_r2, sub_c1:sub_c2]
                    m2_crop = mask2[sub_r1:sub_r2, sub_c1:sub_c2]
                    pad1 = np.pad(m1_crop, 1, mode="constant")
                    expanded = pad1[:-2, 1:-1] | pad1[2:, 1:-1] | pad1[1:-1, :-2] | pad1[1:-1, 2:]
                    if np.any(expanded & m2_crop):
                        self.add_relation(SpatialRelation(o1.object_id, "touches", o2.object_id, 0.90))

                # 3. Directional Relations
                if r1_max < r2_min:
                    self.add_relation(SpatialRelation(o1.object_id, "above", o2.object_id, 0.85))
                elif r1_min > r2_max:
                    self.add_relation(SpatialRelation(o1.object_id, "below", o2.object_id, 0.85))

                if c1_max < c2_min:
                    self.add_relation(SpatialRelation(o1.object_id, "left_of", o2.object_id, 0.85))
                elif c1_min > c2_max:
                    self.add_relation(SpatialRelation(o1.object_id, "right_of", o2.object_id, 0.85))

                # 4. Alignment
                if abs(cr1 - cr2) < 0.5:
                    self.add_relation(SpatialRelation(o1.object_id, "aligned_h", o2.object_id, 0.80))
                if abs(cc1 - cc2) < 0.5:
                    self.add_relation(SpatialRelation(o1.object_id, "aligned_v", o2.object_id, 0.80))

                # 5. Semantic Properties
                if o1.attributes["color"] == o2.attributes["color"]:
                    self.add_relation(SpatialRelation(o1.object_id, "same_color", o2.object_id, 1.0))
                if o1.attributes["area"] > o2.attributes["area"]:
                    self.add_relation(SpatialRelation(o1.object_id, "larger_than", o2.object_id, 0.80))

        # Global structural symmetries
        self.symmetries = {
            "h_reflection": bool(np.array_equal(mat, np.fliplr(mat))),
            "v_reflection": bool(np.array_equal(mat, np.flipud(mat))),
            "rot180": bool(np.array_equal(mat, np.rot90(mat, 2))),
            "transpose": bool(H == W and np.array_equal(mat, mat.T)),
        }

        # Global delimiter detection (solid rows or cols)
        h_delims = [r for r in range(H) if len(np.unique(mat[r, :])) == 1]
        v_delims = [c for c in range(W) if len(np.unique(mat[:, c])) == 1]
        self.delimiters = {"horizontal": h_delims, "vertical": v_delims}

        return self

    def to_mayon_graph(self, mayon_graph: Any, prefix: str = "arc_") -> None:
        """Inject spatial scene graph nodes and edges directly into MayonGraph substrate."""
        if mayon_graph is None or not hasattr(mayon_graph, "add_node"):
            return

        from mayon_cortex.core.graph import GraphLevel

        # Add Object nodes and track ID mappings
        id_map: Dict[str, str] = {}
        for o_id, obj in self.objects.items():
            vec = np.zeros(getattr(mayon_graph, "concept_dim", 384), dtype=np.float32)
            vec[0] = float(obj.attributes.get("color", 0)) / 10.0
            vec[1] = float(obj.attributes.get("area", 0)) / 100.0
            vec[2] = float(obj.attributes.get("num_holes", 0)) / 5.0
            vec[3] = 1.0 if obj.attributes.get("is_solid", False) else 0.0

            meta = {
                "color": int(obj.attributes.get("color", 0)),
                "area": int(obj.attributes.get("area", 0)),
                "num_holes": int(obj.attributes.get("num_holes", 0)),
                "is_solid": bool(obj.attributes.get("is_solid", False)),
                "bbox": [int(x) for x in obj.attributes.get("bbox_int", (0, 0, 0, 0))],
            }
            node = mayon_graph.add_node(
                vector=vec,
                source_text=f"ARC Object {obj.label} at bbox {obj.attributes.get('bbox_int')}",
                level=GraphLevel.CONCEPT,
                sector="spatial_vision",
                node_type="spatial_object",
                provenance="ARCSpatialSceneGraph",
                confidence=1.0,
                metadata=meta,
            )
            id_map[o_id] = node.id

        # Add Spatial edges
        for rel in self.relations:
            if rel.subject_id in id_map and rel.object_id in id_map:
                mayon_graph.add_edge(
                    source_id=id_map[rel.subject_id],
                    target_id=id_map[rel.object_id],
                    relation_type=rel.relation,
                    confidence=rel.confidence,
                    sector="spatial_vision",
                    provenance="ARCSpatialSceneGraph",
                )

