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
Tests for UniversalDataLoader (Generic Format & Language Ingestion)
===================================================================
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex import CortexGraph


@pytest.fixture
def brain():
    return CortexGraph()


def test_ingest_markdown_and_text_file(brain, tmp_path):
    doc_file = tmp_path / "climate_study.md"
    doc_file.write_text(
        "# Climate Study\n\n"
        "Global atmospheric temperature regulates polar ice sheet melting rates.\n"
        "Rising sea levels threaten coastal ecosystems and island communities.\n",
        encoding="utf-8"
    )

    report = brain.ingest_file(doc_file, sector="science")
    assert report.files_processed == 1
    assert report.knowledge_nodes_added > 0
    assert report.total_bytes > 0


def test_ingest_csv_file(brain, tmp_path):
    csv_file = tmp_path / "treatments.csv"
    csv_file.write_text(
        "drug,condition,action\n"
        "Metformin,type 2 diabetes,lowers blood glucose\n"
        "Atorvastatin,hypercholesterolemia,reduces LDL cholesterol\n",
        encoding="utf-8"
    )

    report = brain.ingest_file(csv_file, sector="medical")
    assert report.files_processed == 1
    assert report.knowledge_nodes_added > 0


def test_ingest_json_and_jsonl(brain, tmp_path):
    jsonl_file = tmp_path / "dataset.jsonl"
    jsonl_file.write_text(
        json.dumps({"topic": "astronomy", "fact": "The sun is a main-sequence G-type star."}) + "\n" +
        json.dumps({"topic": "astronomy", "fact": "Jupiter orbits the sun outside the asteroid belt."}) + "\n",
        encoding="utf-8"
    )

    report = brain.ingest_file(jsonl_file, sector="science")
    assert report.files_processed == 1
    assert report.knowledge_nodes_added > 0


def test_ingest_records_stream(brain):
    records = [
        {"entity": "Chlorophyll", "description": "Chlorophyll absorbs blue and red wavelengths during photosynthesis."},
        {"entity": "Mitochondria", "description": "Mitochondria generates ATP cellular energy through respiration."}
    ]

    report = brain.ingest_records(records, default_sector="science")
    assert report.chunks_created == 2
    assert report.knowledge_nodes_added > 0
