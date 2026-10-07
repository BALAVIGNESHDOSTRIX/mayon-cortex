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

import os
import sys

# Ensure proper stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mayon-graph"))

from mayon_cortex import CortexConfig, CortexGraph

brain = CortexGraph()
text = (
    "The sun produces vital light and radiant energy for Earth. "
    "Green plants in the rainforest perform photosynthesis to create oxygen and glucose. "
    "Trees absorb carbon dioxide from the atmosphere and protect the global climate. "
    "Oceans cover most of the planet and regulate global temperatures through water cycles."
)
stats = brain.ingest(text, sector="science")
print(f"Ingested: {stats}")

print("\n--- ALL GRAPH NODES ---")
for nid, n in brain.graph.nodes.items():
    print(f"Node {nid[:8]} [{n.node_type}]: \"{n.source_text}\"")

q = "What do green plants produce during photosynthesis?"
q_vec = brain.embedder.encode_single(q)
act = brain.wave.activate(q_vec, brain.graph, brain.ndu)

print("\n--- SEEDS ---")
for sid in act.seed_node_ids:
    print(f"Seed: {sid[:8]} ({brain.graph.nodes[sid].source_text})")

print("\n--- ALL ACTIVATED PATHS ---")
for p in act.paths:
    path_nodes = [brain.graph.nodes[nid].source_text for nid in p.node_ids]
    print(f"Path: {path_nodes} | Edges: {p.edges}")

scored = brain.ranker.rank_paths(act.paths, q_vec, brain.graph)
print("\n--- SCORED PATHS ---")
for s in scored:
    path_str = " -> ".join([brain.graph.nodes[nid].source_text for nid in s.node_ids])
    print(f"Score: {s.score:.3f} | Endpoint Sim: {s.features[2]:.3f} | Start Sim: {s.features[3]:.3f} | Path: {path_str}")

resp = brain.query(q)
print(f"\nFinal Query Output: \"{resp.text}\"")
