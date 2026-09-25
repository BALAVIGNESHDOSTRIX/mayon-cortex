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
Unit Tests for Mayon-Cortex Global Data Loader & Knowledge Exporter
===================================================================
Tests multi-source ingestion, standard schemas, automatic deduplication,
conflict resolution, and round-trip export/import.
"""

import json
import tempfile
from pathlib import Path
import pytest

from mayon_cortex.engine import MayonCortex
from mayon_cortex.language.schema import (
    StandardKnowledgeItem,
    StandardDocumentItem,
    KnowledgePackage,
)
from mayon_cortex.language.global_loader import GlobalDataLoader, GlobalIngestionReport
from mayon_cortex.language.exporter import BrainKnowledgeExporter, ExportReport


@pytest.fixture
def cortex_instance():
    return MayonCortex()


class TestStandardSchemas:
    """Test data validation and conversion in schemas."""

    def test_standard_knowledge_item(self):
        item = StandardKnowledgeItem(
            source="10.240.0.0/16",
            relation="contains_subnet",
            target="10.240.1.0/24",
            sector="network",
            confidence=0.95,
            provenance="ipam_audit",
            metadata={"vlan": 100},
        )
        d = item.to_dict()
        assert d["source"] == "10.240.0.0/16"
        assert d["confidence"] == 0.95
        assert d["metadata"]["vlan"] == 100

        # Test from_dict with alternate keys (e.g. subject/predicate/object)
        alt = {
            "subject": "Metformin",
            "predicate": "activates",
            "object": "AMPK",
            "sector": "medical",
        }
        item2 = StandardKnowledgeItem.from_dict(alt)
        assert item2.source == "Metformin"
        assert item2.relation == "activates"
        assert item2.target == "AMPK"

    def test_standard_document_item(self):
        doc = StandardDocumentItem(
            title="DHCP Guide",
            text="Subnet 10.240.1.0/24 inherits options from container 10.240.0.0/16.",
            sector="network",
            relations=[
                StandardKnowledgeItem(source="10.240.1.0/24", relation="inherits_from", target="10.240.0.0/16")
            ],
        )
        d = doc.to_dict()
        assert d["title"] == "DHCP Guide"
        assert len(d["relations"]) == 1

        reloaded = StandardDocumentItem.from_dict(d)
        assert reloaded.title == "DHCP Guide"
        assert reloaded.relations[0].source == "10.240.1.0/24"

    def test_knowledge_package_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = KnowledgePackage(
                domain="telecom",
                description="Network topology knowledge pack",
                items=[
                    StandardKnowledgeItem(source="Core-Router-1", relation="connects_to", target="Dist-Switch-1")
                ],
                documents=[
                    StandardDocumentItem(text="Core router connects directly to distribution switch 1.")
                ],
            )
            save_path = Path(tmpdir) / "telecom_pack.json"
            pkg.save(save_path)

            loaded = KnowledgePackage.load(save_path)
            assert loaded.domain == "telecom"
            assert len(loaded.items) == 1
            assert loaded.items[0].source == "Core-Router-1"


class TestGlobalDataLoader:
    """Test multi-format ingestion, deduplication, and conflict resolution."""

    def test_ingest_knowledge_items_and_deduplication(self, cortex_instance):
        cortex = cortex_instance
        records = [
            {"source": "Metformin", "relation": "treats", "target": "Type 2 Diabetes", "sector": "medical"},
            {"source": "metformin", "relation": "activates", "target": "AMPK", "sector": "medical"},
            {"source": "AMPK", "relation": "inhibits", "target": "Hepatic Gluconeogenesis", "sector": "medical"},
        ]

        report = cortex.load_data(records)
        assert report.items_processed == 3
        # 'metformin' and 'Metformin' should be deduplicated to the same node!
        assert report.nodes_merged >= 1
        assert cortex.graph.num_nodes >= 4
        assert cortex.graph.num_edges >= 3

    def test_ingest_structured_document(self, cortex_instance):
        cortex = cortex_instance
        doc = StandardDocumentItem(
            id="doc-net-01",
            title="IPAM Subnet Allocation",
            text="Subnet 192.168.1.0/24 is assigned to VLAN 10. Gateway is 192.168.1.1.",
            sector="network",
            relations=[
                StandardKnowledgeItem(source="192.168.1.0/24", relation="assigned_to", target="VLAN 10", sector="network")
            ]
        )
        report = cortex.load_data(doc)
        assert report.documents_processed == 1
        assert report.edges_created >= 1

    def test_ingest_json_and_jsonl_files(self, cortex_instance):
        cortex = cortex_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create JSON triplet file
            json_file = Path(tmpdir) / "network_facts.json"
            facts = [
                {"source": "VLAN 100", "relation": "serves", "target": "Engineering Subnet", "sector": "network"},
                {"source": "Engineering Subnet", "relation": "has_range", "target": "10.100.1.0/24", "sector": "network"},
            ]
            json_file.write_text(json.dumps(facts), encoding="utf-8")

            # 2. Create JSONL file
            jsonl_file = Path(tmpdir) / "security_facts.jsonl"
            lines = [
                json.dumps({"source": "Firewall-A", "relation": "blocks", "target": "Port 23", "sector": "security"}),
                json.dumps({"source": "Firewall-A", "relation": "permits", "target": "Port 443", "sector": "security"}),
            ]
            jsonl_file.write_text("\n".join(lines), encoding="utf-8")

            # Ingest both sources in one call
            report = cortex.load_sources([json_file, jsonl_file])
            assert report.sources_loaded == 2
            assert report.items_processed == 4
            assert cortex.graph.num_edges >= 4

    def test_ingest_csv_file(self, cortex_instance):
        cortex = cortex_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "database_dump.csv"
            csv_content = (
                "source,relation,target,sector,confidence\n"
                "PostgreSQL,supports,JSONB,tech,0.99\n"
                "JSONB,enables,Indexing,tech,0.95\n"
            )
            csv_file.write_text(csv_content, encoding="utf-8")

            report = cortex.load_data(csv_file, sector="code")
            assert report.items_processed == 2
            assert report.edges_created == 2

    def test_conflict_resolution(self, cortex_instance):
        cortex = cortex_instance
        # Add initial fact with lower confidence
        cortex.load_data({
            "source": "Subnet-A", "relation": "gateway", "target": "10.0.0.1",
            "confidence": 0.5, "sector": "network"
        })

        # Update fact with higher confidence
        report = cortex.load_data({
            "source": "Subnet-A", "relation": "gateway", "target": "10.0.0.1",
            "confidence": 0.95, "sector": "network"
        }, resolve_conflicts=True)

        assert report.conflicts_resolved >= 1
        # Edge confidence should have been updated
        node_a = next(n for n in cortex.graph.nodes.values() if n.source_text == "Subnet-A")
        node_b = next(n for n in cortex.graph.nodes.values() if n.source_text == "10.0.0.1")
        edge = cortex.graph.edges.get(f"{node_a.id}->{node_b.id}:gateway")
        assert edge is not None
        assert edge.confidence > 0.75


class TestKnowledgeExportAndRoundTrip:
    """Test exporting brain knowledge to JSON packages and reloading into a clean instance."""

    def test_round_trip_export_and_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Populate primary brain
            cortex1 = MayonCortex()
            cortex1.load_data([
                {"source": "Router-X", "relation": "routes_to", "target": "Subnet-Y", "sector": "network", "confidence": 1.0},
                {"source": "Subnet-Y", "relation": "contains", "target": "Host-Z", "sector": "network", "confidence": 0.9},
                {"source": "Aspirin", "relation": "treats", "target": "Headache", "sector": "medical", "confidence": 0.95},
            ])

            export_path = Path(tmpdir) / "brain_export.json"
            export_report = cortex1.export_knowledge(
                target_path=export_path,
                format="package",
                domain_name="mixed_knowledge",
            )
            assert export_report.total_edges_exported == 3
            assert export_path.exists()

            # 2. Ingest exported package into a fresh secondary brain
            cortex2 = MayonCortex()
            assert cortex2.graph.num_edges == 0

            ingest_report = cortex2.load_data(export_path)
            assert ingest_report.items_processed == 3
            assert cortex2.graph.num_edges == 3

            # 3. Query the second brain to verify reasoning works on reloaded knowledge
            res = cortex2.query("What does Router-X route to?")
            assert "Subnet-Y" in [n.source_text for n in cortex2.graph.nodes.values()]
