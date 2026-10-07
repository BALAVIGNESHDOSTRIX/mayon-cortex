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
Universal Data Loader — Generic, Format-Agnostic Ingestion for Cortex-Graph
===========================================================================
Allows ingesting ANY type of data from ANY source:
- Raw text files (.txt, .md, .rst, .log)
- Structured data (.csv, .tsv, .json, .jsonl, .parquet/tabular)
- Documents & Web (.html, .xml, .pdf)
- Recursive Directory Ingestion (folders of mixed documents)
- In-memory data collections (list of dicts, lists of strings, dataframes)

Performs automatic text normalization, semantic sentence chunking,
open information extraction (entities & relations), and populates
both the Mayon Knowledge Graph and the WordGraph.
"""

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

from mayon_cortex.language.language_ingestion import IngestionStats, LanguageIngester


@dataclass
class IngestionReport:
    """Summary of a universal data ingestion job."""
    files_processed: int = 0
    total_bytes: int = 0
    chunks_created: int = 0
    knowledge_nodes_added: int = 0
    relation_edges_added: int = 0
    phrase_nodes_added: int = 0
    word_nodes_updated: int = 0
    errors: List[str] = field(default_factory=list)

    def merge(self, stats: IngestionStats, file_bytes: int = 0):
        self.total_bytes += file_bytes
        self.chunks_created += 1
        self.knowledge_nodes_added += stats.knowledge_nodes_added
        self.relation_edges_added += stats.knowledge_edges_added
        self.phrase_nodes_added += stats.phrase_nodes_added
        self.word_nodes_updated += stats.word_nodes_added


class UniversalDataLoader:
    """
    Universal, domain-agnostic and format-agnostic data ingester.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".rst", ".log",
        ".csv", ".tsv",
        ".json", ".jsonl",
        ".html", ".htm", ".xml",
    }

    def __init__(self, ingester: LanguageIngester):
        self.ingester = ingester

    def ingest_text(
        self,
        text: str,
        graph,
        sector: str = "general",
        provenance: str = "raw_text",
        chunk_size: int = 5,  # sentences per chunk
    ) -> IngestionStats:
        """Ingest arbitrary text string, chunking long documents."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return IngestionStats()

        total_stats = IngestionStats()

        # Group sentences into digestible chunks
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size]).strip()
            if len(chunk) > 10:
                stats = self.ingester.ingest(
                    chunk, graph, sector=sector, provenance=provenance
                )
                total_stats.knowledge_nodes_added += stats.knowledge_nodes_added
                total_stats.knowledge_edges_added += stats.knowledge_edges_added
                total_stats.phrase_nodes_added += stats.phrase_nodes_added
                total_stats.word_nodes_added += stats.word_nodes_added
                total_stats.sentences_processed += stats.sentences_processed

        return total_stats

    def ingest_file(
        self,
        file_path: Union[str, Path],
        graph,
        sector: Optional[str] = None,
        provenance: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> IngestionReport:
        """Ingest a single file of any supported format."""
        path = Path(file_path)
        report = IngestionReport()

        if not path.exists():
            report.errors.append(f"File not found: {path}")
            return report

        ext = path.suffix.lower()
        prov = provenance or f"file:{path.name}"
        sec = sector or self._infer_sector_from_path(path)

        try:
            file_size = path.stat().st_size
            report.files_processed += 1

            if ext in {".txt", ".md", ".markdown", ".rst", ".log"}:
                text = path.read_text(encoding=encoding, errors="replace")
                stats = self.ingest_text(text, graph, sector=sec, provenance=prov)
                report.merge(stats, file_size)

            elif ext == ".csv":
                self._ingest_csv(path, graph, report, sec, prov, encoding=encoding)

            elif ext == ".tsv":
                self._ingest_csv(path, graph, report, sec, prov, delimiter="\t", encoding=encoding)

            elif ext == ".json":
                self._ingest_json(path, graph, report, sec, prov, encoding=encoding)

            elif ext == ".jsonl":
                self._ingest_jsonl(path, graph, report, sec, prov, encoding=encoding)

            elif ext in {".html", ".htm", ".xml"}:
                raw_html = path.read_text(encoding=encoding, errors="replace")
                clean_text = self._strip_tags(raw_html)
                stats = self.ingest_text(clean_text, graph, sector=sec, provenance=prov)
                report.merge(stats, file_size)

            else:
                # Fallback: attempt plain text reading
                text = path.read_text(encoding=encoding, errors="replace")
                stats = self.ingest_text(text, graph, sector=sec, provenance=prov)
                report.merge(stats, file_size)

        except Exception as e:
            report.errors.append(f"Error reading {path}: {str(e)}")

        return report

    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        graph,
        pattern: str = "*.*",
        recursive: bool = True,
        sector: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> IngestionReport:
        """Recursively scan and ingest all supported files in a directory."""
        dir_path = Path(directory_path)
        overall_report = IngestionReport()

        if not dir_path.is_dir():
            overall_report.errors.append(f"Not a directory: {dir_path}")
            return overall_report

        files = dir_path.rglob(pattern) if recursive else dir_path.glob(pattern)

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                file_report = self.ingest_file(
                    file_path, graph, sector=sector, encoding=encoding
                )
                overall_report.files_processed += file_report.files_processed
                overall_report.total_bytes += file_report.total_bytes
                overall_report.chunks_created += file_report.chunks_created
                overall_report.knowledge_nodes_added += file_report.knowledge_nodes_added
                overall_report.relation_edges_added += file_report.relation_edges_added
                overall_report.phrase_nodes_added += file_report.phrase_nodes_added
                overall_report.word_nodes_updated += file_report.word_nodes_updated
                overall_report.errors.extend(file_report.errors)

        return overall_report

    def ingest_records(
        self,
        records: Iterable[Union[Dict[str, Any], str]],
        graph,
        text_key: Optional[str] = None,
        sector_key: Optional[str] = None,
        default_sector: str = "general",
        provenance: str = "record_stream",
    ) -> IngestionReport:
        """Ingest an iterable stream of dictionaries or strings (e.g. from database or HuggingFace dataset)."""
        report = IngestionReport()

        for idx, item in enumerate(records):
            if isinstance(item, str):
                stats = self.ingest_text(item, graph, sector=default_sector, provenance=f"{provenance}:{idx}")
                report.merge(stats)
            elif isinstance(item, dict):
                # Extract text
                if text_key and text_key in item:
                    content = str(item[text_key])
                else:
                    # Combine all string values
                    content = " ".join(str(v) for v in item.values() if isinstance(v, (str, int, float)))

                sec = item.get(sector_key, default_sector) if sector_key else default_sector
                stats = self.ingest_text(content, graph, sector=str(sec), provenance=f"{provenance}:{idx}")
                report.merge(stats)

        return report

    # ── Internal Helpers ──

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def _strip_tags(self, html_text: str) -> str:
        """Basic regex-based HTML/XML tag stripper."""
        clean = re.sub(r'<script.*?</script>', ' ', html_text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style.*?</style>', ' ', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        return re.sub(r'\s+', ' ', clean).strip()

    def _infer_sector_from_path(self, path: Path) -> str:
        """Infer sector tag from filename or parent folder name."""
        path_str = str(path).lower()
        if "med" in path_str or "clinic" in path_str or "health" in path_str:
            return "medical"
        if "law" in path_str or "legal" in path_str or "court" in path_str:
            return "legal"
        if "fin" in path_str or "econ" in path_str or "bank" in path_str:
            return "finance"
        if "sci" in path_str or "phys" in path_str or "chem" in path_str or "bio" in path_str:
            return "science"
        if "code" in path_str or "prog" in path_str or "dev" in path_str:
            return "code"
        if "story" in path_str or "novel" in path_str or "book" in path_str:
            return "story"
        return "general"

    def _ingest_csv(
        self,
        path: Path,
        graph,
        report: IngestionReport,
        sector: str,
        prov: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ):
        with open(path, mode="r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for idx, row in enumerate(reader):
                # Convert row to human-readable sentence: "key is val, key2 is val2."
                items = [f"{k} is {v}" for k, v in row.items() if v and v.strip()]
                if items:
                    sentence = ". ".join(items) + "."
                    stats = self.ingester.ingest(sentence, graph, sector=sector, provenance=f"{prov}:row_{idx}")
                    report.merge(stats)

    def _ingest_json(
        self,
        path: Path,
        graph,
        report: IngestionReport,
        sector: str,
        prov: str,
        encoding: str = "utf-8",
    ):
        with open(path, mode="r", encoding=encoding, errors="replace") as f:
            data = json.load(f)

        if isinstance(data, list):
            sub_report = self.ingest_records(data, graph, default_sector=sector, provenance=prov)
            report.knowledge_nodes_added += sub_report.knowledge_nodes_added
            report.relation_edges_added += sub_report.relation_edges_added
            report.phrase_nodes_added += sub_report.phrase_nodes_added
            report.word_nodes_updated += sub_report.word_nodes_updated
            report.chunks_created += sub_report.chunks_created
        elif isinstance(data, dict):
            # Flatten or convert key-values
            items = []
            for k, v in data.items():
                if isinstance(v, (str, int, float)):
                    items.append(f"{k}: {v}")
                elif isinstance(v, list):
                    items.append(f"{k}: {', '.join(str(x) for x in v)}")
            if items:
                stats = self.ingest_text(". ".join(items) + ".", graph, sector=sector, provenance=prov)
                report.merge(stats)

    def _ingest_jsonl(
        self,
        path: Path,
        graph,
        report: IngestionReport,
        sector: str,
        prov: str,
        encoding: str = "utf-8",
    ):
        with open(path, mode="r", encoding=encoding, errors="replace") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        content = " ".join(str(v) for v in record.values() if isinstance(v, (str, int, float)))
                    else:
                        content = str(record)
                    stats = self.ingest_text(content, graph, sector=sector, provenance=f"{prov}:line_{idx}")
                    report.merge(stats)
                except Exception:
                    continue
