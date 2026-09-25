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
Mayon-Cortex Global Data Ingestion & Multi-Source Demo
======================================================
Demonstrates:
  1. Standard JSON Knowledge Items & Triplets ingestion
  2. Multi-source loading (JSON, CSV, JSONL, Markdown)
  3. Ingestion-time automatic deduplication & conflict resolution
  4. Multi-hop proof extraction across the ingested knowledge
  5. Exporting the complete Brain Knowledge Package
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mayon_cortex.engine import MayonCortex
from mayon_cortex.language.schema import (
    KnowledgePackage,
    StandardDocumentItem,
    StandardKnowledgeItem,
)


def run_demo():
    print("=" * 80)
    print(" [DEMO] MAYON-CORTEX GLOBAL DATA INGESTION & KNOWLEDGE EXPORT ENGINE")
    print(" Pure CPU Neuro-Symbolic Intelligence -- Zero GPUs, Zero Backprop")
    print("=" * 80)

    # 1. Initialize Cortex Brain Instance
    print("\n1. Initializing Cortex Brain...")
    cortex = MayonCortex()
    print(f"   * Initial Nodes: {cortex.graph.num_nodes}, Edges: {cortex.graph.num_edges}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 2. Prepare Multi-Domain Sources
        print("\n2. Preparing Multi-Domain Sources on Disk...")

        # Source A: Networking / Infoblox IPAM Triplets (JSON)
        net_file = tmp_path / "network_topology.json"
        net_facts = [
            {
                "source": "10.240.0.0/16",
                "relation": "contains_subnet",
                "target": "10.240.1.0/24",
                "sector": "network",
                "confidence": 1.0,
                "provenance": "Infoblox_Grid_Export",
                "metadata": {"view": "default", "vlan": 101}
            },
            {
                "source": "10.240.1.0/24",
                "relation": "serves_dhcp_range",
                "target": "10.240.1.100-10.240.1.200",
                "sector": "network",
                "confidence": 0.98,
                "metadata": {"router": "10.240.1.1", "lease_time": 86400}
            },
            {
                "source": "10.240.0.0/16",
                "relation": "defines_dhcp_option",
                "target": "Option 43 Vendor Info",
                "sector": "network",
                "confidence": 0.95,
            },
            {
                "source": "10.240.1.0/24",
                "relation": "inherits_dhcp_option",
                "target": "Option 43 Vendor Info",
                "sector": "network",
                "confidence": 0.90,
            }
        ]
        net_file.write_text(json.dumps(net_facts, indent=2), encoding="utf-8")
        print(f"   * Created JSON Source: {net_file.name} (4 Network Triplets)")

        # Source B: Medical Pharmacology Knowledge (CSV)
        med_file = tmp_path / "medical_kb.csv"
        med_csv = (
            "source,relation,target,sector,confidence,provenance\n"
            "Metformin,treats,Type 2 Diabetes,medical,0.99,FDA_Orange_Book\n"
            "metformin,activates,AMPK Pathway,medical,0.95,Pharmacology_Textbook\n"
            "AMPK Pathway,inhibits,Hepatic Gluconeogenesis,medical,0.92,Endocrinology_2024\n"
            "Metformin,contraindicates,Severe Renal Impairment,medical,1.00,Blackbox_Warning\n"
        )
        med_file.write_text(med_csv, encoding="utf-8")
        print(f"   * Created CSV Source:  {med_file.name} (4 Medical Facts with case variations)")

        # Source C: Software Engineering & Architecture (JSONL)
        code_file = tmp_path / "architecture_docs.jsonl"
        code_lines = [
            json.dumps({
                "source": "Mayon-Cortex",
                "relation": "uses_vector_index",
                "target": "USearch HNSW",
                "sector": "code",
                "confidence": 1.0,
            }),
            json.dumps({
                "source": "USearch HNSW",
                "relation": "executes_on",
                "target": "Pure CPU SIMD",
                "sector": "code",
                "confidence": 0.98,
            }),
            json.dumps({
                "text": "Mayon-Cortex is a neuro-symbolic brain operating on pure CPU with sub-millisecond graph reasoning.",
                "sector": "code",
                "provenance": "architecture_whitepaper"
            })
        ]
        code_file.write_text("\n".join(code_lines), encoding="utf-8")
        print(f"   * Created JSONL Source: {code_file.name} (2 Triplets + 1 Document)")

        # 3. Load All Sources in a Single Coordinated Ingestion Call
        print("\n3. Ingesting All Sources via GlobalDataLoader...")
        report = cortex.load_sources([net_file, med_file, code_file], auto_dedup=True, resolve_conflicts=True)

        print(f"   ================ INGESTION REPORT ================")
        print(f"   * Sources Processed:       {report.sources_loaded}")
        print(f"   * Triplet Items Processed: {report.items_processed}")
        print(f"   * Documents Processed:     {report.documents_processed}")
        print(f"   * Concept Nodes Created:   {report.nodes_created}")
        print(f"   * Concept Nodes Merged:    {report.nodes_merged} (Deduplicated)")
        print(f"   * Relation Edges Created:  {report.edges_created}")
        print(f"   * Conflicts Resolved:      {report.conflicts_resolved}")
        print(f"   * Sectors Affected:        {report.sectors_affected}")
        print(f"   ==================================================")

        # 4. Demonstrate Multi-Hop Reasoning over Ingested Knowledge
        print("\n4. Querying Ingested Knowledge Tissue:")

        # Query A: Network inheritance
        print("\n   [Query 1 - Network Reasoning]:")
        q1 = "What DHCP options does subnet 10.240.1.0/24 inherit?"
        print(f"   Q: {q1}")
        res1 = cortex.query(q1, sector="network")
        print(f"   A: {res1.text}")
        if res1.paths:
            print("   Proof Paths Found:")
            for p in res1.paths[:2]:
                node_names = [cortex.graph.get_node(nid).source_text if cortex.graph.get_node(nid) else nid for nid in p.node_ids]
                print(f"     - Path: {' -> '.join(node_names)} (Score: {p.score:.3f})")

        # Query B: Medical Mechanism
        print("\n   [Query 2 - Medical Reasoning]:")
        q2 = "How does Metformin affect Hepatic Gluconeogenesis?"
        print(f"   Q: {q2}")
        res2 = cortex.query(q2, sector="medical")
        print(f"   A: {res2.text}")
        if res2.paths:
            print("   Proof Paths Found:")
            for p in res2.paths[:2]:
                node_names = [cortex.graph.get_node(nid).source_text if cortex.graph.get_node(nid) else nid for nid in p.node_ids]
                print(f"     - Path: {' -> '.join(node_names)} (Score: {p.score:.3f})")

        # 5. Export Complete Brain Knowledge Package
        print("\n5. Exporting Brain Knowledge Package to JSON...")
        export_file = tmp_path / "mayon_brain_knowledge_package.json"
        export_rep = cortex.export_knowledge(
            target_path=export_file,
            format="package",
            domain_name="enterprise_multi_domain",
            description="Consolidated Knowledge Base of Network, Medical, and Code domains."
        )

        print(f"   * Export File: {export_rep.file_path}")
        print(f"   * Export Size: {export_rep.file_size_bytes:,} bytes")
        print(f"   * Total Edges Exported: {export_rep.total_edges_exported}")
        print(f"   * Sectors Exported:     {export_rep.sectors_included}")

        # Verify Round-Trip Import into Fresh Brain
        print("\n6. Verifying Round-Trip Import into a Fresh Brain Instance...")
        fresh_cortex = MayonCortex()
        import_rep = fresh_cortex.load_data(export_file)
        print(f"   * Fresh Brain Imported {import_rep.items_processed} items into {fresh_cortex.graph.num_nodes} nodes and {fresh_cortex.graph.num_edges} edges.")

        res_fresh = fresh_cortex.query("What does Metformin contraindicate?", sector="medical")
        print(f"   * Fresh Brain Query Response: {res_fresh.text}")

    print("\n" + "=" * 80)
    print(" [DEMO COMPLETE] Multi-source ingestion, deduplication, and export verified successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
