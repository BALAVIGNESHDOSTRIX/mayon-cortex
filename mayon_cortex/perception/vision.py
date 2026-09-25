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
Vision Cortex — Visual Understanding & Multimodal Graph Integration
====================================================================
Pillar 4 of Cortex-Graph v4.

Integrates images as first-class citizens in the cortical graph:
1. Encodes images into the unified ℝ³⁸⁴ concept vector space (shared with language).
2. Uses lightweight CPU-friendly visual backbone (MobileNetV3 / PIL-based feature extractor).
3. Image nodes in the graph are FAISS-searchable and cross-linked with concept/phrase nodes.
4. Generates visual relations: `depicts`, `contains-object`, `resembles`, `visual-property-of`.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class ImageNode:
    """A first-class image node in the cortical graph."""
    image_id: str
    file_path: Optional[str]
    vector: np.ndarray  # ℝ³⁸⁴
    caption: Optional[str] = None
    detected_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualEncoder:
    """
    Lightweight visual feature extractor projecting images to 384-dim space.
    Gracefully handles torch / torchvision if available, with robust CPU fallback.
    """

    def __init__(self, target_dim: int = 384):
        self.target_dim = target_dim
        self._torch_model = None
        self._init_model()

    def _init_model(self):
        """Try loading lightweight MobileNetV3 or fallback to feature generator."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            # Try initializing lightweight mobilenet
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self._torch_model = models.mobilenet_v3_small(weights=weights)
            self._torch_model.eval()
            self._transforms = weights.transforms()
            # Random projection to 384 dim
            rng = np.random.RandomState(1337)
            self._proj_matrix = rng.randn(1000, self.target_dim).astype(np.float32)
            self._proj_matrix /= (np.linalg.norm(self._proj_matrix, axis=0, keepdims=True) + 1e-8)
        except Exception:
            self._torch_model = None

    def encode(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Extract normalized 384-dim feature vector from an image path, bytes, or ndarray.
        """
        if self._torch_model is not None:
            try:
                import torch
                from PIL import Image
                if isinstance(image_input, str) and os.path.exists(image_input):
                    img = Image.open(image_input).convert("RGB")
                elif isinstance(image_input, bytes):
                    import io
                    img = Image.open(io.BytesIO(image_input)).convert("RGB")
                elif isinstance(image_input, np.ndarray):
                    img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
                else:
                    img = None

                if img is not None:
                    tensor = self._transforms(img).unsqueeze(0)
                    with torch.no_grad():
                        logits = self._torch_model(tensor).numpy()[0]  # (1000,)
                    vec = np.dot(logits, self._proj_matrix)  # (384,)
                    vec /= (np.linalg.norm(vec) + 1e-8)
                    return vec.astype(np.float32)
            except Exception:
                pass

        # Robust deterministic fallback encoder using image content / hash / dimensions
        return self._fallback_encode(image_input)

    def _fallback_encode(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Deterministic feature generator when PIL / torch is absent."""
        if isinstance(image_input, str):
            key = f"path:{image_input}:{os.path.getsize(image_input) if os.path.exists(image_input) else 0}"
        elif isinstance(image_input, bytes):
            key = hashlib.sha256(image_input).hexdigest()
        elif isinstance(image_input, np.ndarray):
            key = f"arr:{image_input.shape}:{image_input.mean()}:{image_input.std()}"
        else:
            key = str(image_input)

        seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.target_dim).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-8)
        return vec


class VisionCortex:
    """
    Vision Processing and Graph Integration Center.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        dim = getattr(self.config, "concept_dim", 384)
        self.encoder = VisualEncoder(target_dim=dim)
        self.image_nodes: Dict[str, ImageNode] = {}

    def see(
        self,
        image_input: Union[str, bytes, np.ndarray],
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
        graph=None,
    ) -> ImageNode:
        """
        Process visual input, encode into unified space, and attach as a first-class graph node.
        """
        vector = self.encoder.encode(image_input)

        if isinstance(image_input, str):
            file_name = os.path.basename(image_input)
            image_id = f"image::{file_name}"
            file_path = image_input
        else:
            image_id = f"image::{len(self.image_nodes)+1}"
            file_path = None

        node = ImageNode(
            image_id=image_id,
            file_path=file_path,
            vector=vector,
            caption=caption,
            detected_tags=tags or [],
        )
        self.image_nodes[image_id] = node

        # Wire to graph if provided
        if graph is not None and hasattr(graph, "add_node"):
            try:
                g_node = graph.add_node(
                    vector=vector,
                    source_text=caption or image_id,
                    level=1,
                    sector="vision",
                    node_type="image",
                    provenance="vision_cortex",
                    metadata={
                        "type": "image",
                        "image_id": image_id,
                        "caption": caption or "",
                        "file_path": file_path or "",
                        "tags": tags or [],
                    }
                )

                # Link tags / entities
                if tags and hasattr(graph, "add_edge"):
                    for tag in tags:
                        tag_id = tag.lower().strip()
                        if tag_id in graph.nodes:
                            graph.add_edge(
                                source_id=g_node.id,
                                target_id=tag_id,
                                relation_type="depicts",
                                confidence=0.9,
                                sector="vision",
                                provenance="vision_cortex",
                            )
            except Exception:
                pass

        return node

    def query_image(self, image_input: Union[str, bytes, np.ndarray], graph=None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts and knowledge nodes in the graph that most closely resemble or relate to this image.
        """
        query_vec = self.encoder.encode(image_input)
        if graph is None:
            # Match against stored images
            sims = []
            for img_id, img_node in self.image_nodes.items():
                s = float(np.dot(img_node.vector, query_vec))
                sims.append((img_id, s))
            sims.sort(key=lambda x: x[1], reverse=True)
            return sims[:top_k]

        # Use graph FAISS or vector search if available
        if hasattr(graph, "find_nearest"):
            return graph.find_nearest(query_vec, top_k=top_k)

        return []

    def see_scene(
        self,
        scene: "SceneGraph",
        graph=None,
    ) -> List[str]:
        """
        Integrate a full SceneGraph (objects + bounding boxes + spatial relations) into Cortex-Graph.
        Returns the list of created graph node IDs.
        """
        from mayon_cortex.perception.scene_graph import SceneGraph
        
        created_ids = []
        if graph is None or not hasattr(graph, "add_node"):
            return created_ids

        # 1. Ingest each object as a node
        obj_node_map = {}
        for obj_id, obj in scene.objects.items():
            vec = obj.embedding if obj.embedding is not None else self.encoder.encode(obj.label)
            try:
                g_node = graph.add_node(
                    vector=vec,
                    source_text=f"{obj.label} ({obj_id})",
                    level=1,
                    sector="vision",
                    node_type="visual_object",
                    provenance="vision_cortex",
                    metadata={
                        "type": "visual_object",
                        "object_id": obj_id,
                        "label": obj.label,
                        "bbox": list(obj.bbox),
                        "attributes": obj.attributes,
                    }
                )
                obj_node_map[obj_id] = g_node.id
                created_ids.append(g_node.id)
            except Exception:
                pass

        # 2. Add spatial relation edges
        if hasattr(graph, "add_edge"):
            for rel in scene.relations:
                s_id = obj_node_map.get(rel.subject_id)
                t_id = obj_node_map.get(rel.object_id)
                if s_id and t_id:
                    try:
                        graph.add_edge(
                            source_id=s_id,
                            target_id=t_id,
                            relation_type=rel.relation,
                            confidence=rel.confidence,
                            sector="vision",
                            provenance="scene_graph",
                        )
                    except Exception:
                        pass

        return created_ids

