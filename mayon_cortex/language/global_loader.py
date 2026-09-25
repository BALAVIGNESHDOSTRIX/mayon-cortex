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
Mayon-Cortex Global Multi-Source Data Loader
============================================
The central ingestion engine that accepts data from any source and format,
performs automatic deduplication, resolves conflicts, and populates the graph.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from mayon_cortex.core.deduplication import GraphDeduplicator
from mayon_cortex.core.graph import ConceptNode, GraphLevel, MayonGraph, RelationEdge
from mayon_cortex.core.loader import EmbeddingModel
from mayon_cortex.language.schema import (
    KnowledgePackage,
    StandardDocumentItem,
    StandardKnowledgeItem,
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalIngestionReport:
    """Summary of a multi-source data ingestion operation."""
    sources_loaded: int = 0
    items_processed: int = 0
    documents_processed: int = 0
    nodes_created: int = 0
    nodes_merged: int = 0
    edges_created: int = 0
    conflicts_resolved: int = 0
    sectors_affected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def merge(self, other: "GlobalIngestionReport"):
        self.sources_loaded += other.sources_loaded
        self.items_processed += other.items_processed
        self.documents_processed += other.documents_processed
        self.nodes_created += other.nodes_created
        self.nodes_merged += other.nodes_merged
        self.edges_created += other.edges_created
        self.conflicts_resolved += other.conflicts_resolved
        for sec in other.sectors_affected:
            if sec not in self.sectors_affected:
                self.sectors_affected.append(sec)
        self.errors.extend(other.errors)


class DynamicSectorClassifier:
    """
    Zero-Shot Dynamic Sector Classification via Semantic Vector Prototypes.
    Eliminates static keyword matching. Dynamically determines the best matching domain
    based on semantic embedding proximity.
    """
    DEFAULT_PROTOTYPES = {
        "network": "Computer networking, IPAM, subnets, DHCP options, DNS records, routers, VLANs, and infrastructure.",
        "medical": "Medical pharmacology, clinical trials, pathology, diseases, treatments, dosage, and anatomy.",
        "legal": "Jurisprudence, statutes, court rulings, contracts, legal compliance, and liability.",
        "finance": "Economics, banking, stock market, assets, portfolio management, accounting, and monetary policy.",
        "code": "Software engineering, computer programming, algorithms, APIs, debugging, and source code.",
        "science": "Physics, chemistry, astronomy, biology, empirical research, and mathematics.",
    }

    def __init__(self, embedder: EmbeddingModel, similarity_threshold: float = 0.25):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self._prototypes: Dict[str, str] = dict(self.DEFAULT_PROTOTYPES)
        self._prototype_vectors: Dict[str, np.ndarray] = {}
        self._init_vectors()

    def _init_vectors(self):
        for sector, desc in self._prototypes.items():
            self._prototype_vectors[sector] = self.embedder.encode_single(desc)

    def register_sector(self, sector_name: str, description_or_examples: str):
        """Dynamically add or update a domain sector prototype."""
        sec = sector_name.strip().lower()
        self._prototypes[sec] = description_or_examples
        self._prototype_vectors[sec] = self.embedder.encode_single(description_or_examples)

    def classify(self, text_or_path: Union[str, Path], fallback: str = "general") -> str:
        """
        Dynamically classify content into a domain sector using semantic embedding similarity.
        """
        query_text = Path(text_or_path).stem if isinstance(text_or_path, Path) else str(text_or_path)
        # Clean text
        query_text = query_text.replace("_", " ").replace("-", " ")
        if len(query_text.strip()) == 0:
            return fallback

        q_vec = self.embedder.encode_single(query_text)
        best_sector = fallback
        best_sim = -1.0

        for sec, p_vec in self._prototype_vectors.items():
            norm_q = np.linalg.norm(q_vec)
            norm_p = np.linalg.norm(p_vec)
            if norm_q == 0 or norm_p == 0:
                continue
            sim = float(np.dot(q_vec, p_vec) / (norm_q * norm_p))
            if sim > best_sim:
                best_sim = sim
                best_sector = sec

        if best_sim >= self.similarity_threshold:
            return best_sector
        return fallback


class GlobalDataLoader:
    """
    Universal multi-source ingestion engine for Mayon-Cortex.
    Standardizes all inputs into StandardKnowledgeItem and StandardDocumentItem,
    embeds concepts in batch, and indexes them into the graph tissue.
    """

    def __init__(
        self,
        graph: MayonGraph,
        embedder: Optional[EmbeddingModel] = None,
        deduplicator: Optional[GraphDeduplicator] = None,
        sector_classifier: Optional[DynamicSectorClassifier] = None,
    ):
        self.graph = graph
        self.embedder = embedder or EmbeddingModel(dim=graph.concept_dim)
        self.deduplicator = deduplicator or GraphDeduplicator()
        self.sector_classifier = sector_classifier or DynamicSectorClassifier(self.embedder)

        # Cache of normalized text -> node_id for fast in-memory entity resolution
        self._text_to_node_id: Dict[str, str] = {}
        self._rebuild_node_cache()

    def _rebuild_node_cache(self):
        """Build normalized text index for instant node lookup."""
        self._text_to_node_id.clear()
        for nid, node in self.graph.nodes.items():
            norm_key = self._normalize_label(node.source_text)
            self._text_to_node_id[norm_key] = nid

    @staticmethod
    def _normalize_label(text: str) -> str:
        """Normalized string key for entity matching."""
        return re.sub(r'\s+', ' ', str(text).strip().lower())

    def load_source(
        self,
        source: Union[str, Path, Dict[str, Any], Sequence[Any], KnowledgePackage, StandardKnowledgeItem, StandardDocumentItem],
        format: str = "auto",
        sector: Optional[str] = None,
        default_confidence: float = 1.0,
        provenance: Optional[str] = None,
        auto_dedup: bool = True,
        resolve_conflicts: bool = True,
    ) -> GlobalIngestionReport:
        """
        Load knowledge from any supported source format.
        
        Args:
            source: File path, directory path, Dict, List of Dicts, JSON string, or Schema object.
            format: 'auto', 'json', 'jsonl', 'csv', 'triplets', 'package', 'document', 'text', 'markdown'.
            sector: Override sector domain.
            default_confidence: Default confidence if not specified.
            provenance: Origin tracking tag.
            auto_dedup: Run entity resolution & alias merging.
            resolve_conflicts: Reconcile duplicate/contradicting edge relations.
        """
        report = GlobalIngestionReport()

        try:
            # 1. Schema object instances
            if isinstance(source, KnowledgePackage):
                return self._ingest_package(source, sector, auto_dedup, resolve_conflicts)

            if isinstance(source, StandardKnowledgeItem):
                return self._ingest_triplet_batch([source], sector, auto_dedup, resolve_conflicts)

            if isinstance(source, StandardDocumentItem):
                return self._ingest_document_batch([source], sector, auto_dedup, resolve_conflicts)

            # 2. In-memory Python Data Structures
            if isinstance(source, dict):
                # Check if it's a KnowledgePackage dict
                if "items" in source or "documents" in source or "package_id" in source:
                    pkg = KnowledgePackage.from_dict(source)
                    return self._ingest_package(pkg, sector, auto_dedup, resolve_conflicts)
                elif "source" in source or "subject" in source or "head" in source:
                    item = StandardKnowledgeItem.from_dict(source)
                    return self._ingest_triplet_batch([item], sector, auto_dedup, resolve_conflicts)
                elif "text" in source or "content" in source:
                    doc = StandardDocumentItem.from_dict(source)
                    return self._ingest_document_batch([doc], sector, auto_dedup, resolve_conflicts)
                else:
                    report.errors.append(f"Unrecognized dictionary schema: {list(source.keys())}")
                    return report

            if isinstance(source, (list, tuple)):
                if not source:
                    return report
                first = source[0]
                if isinstance(first, StandardKnowledgeItem):
                    return self._ingest_triplet_batch(list(source), sector, auto_dedup, resolve_conflicts)
                if isinstance(first, StandardDocumentItem):
                    return self._ingest_document_batch(list(source), sector, auto_dedup, resolve_conflicts)
                if isinstance(first, dict):
                    if "source" in first or "subject" in first or "head" in first:
                        items = [StandardKnowledgeItem.from_dict(d) for d in source]
                        return self._ingest_triplet_batch(items, sector, auto_dedup, resolve_conflicts)
                    elif "text" in first or "content" in first:
                        docs = [StandardDocumentItem.from_dict(d) for d in source]
                        return self._ingest_document_batch(docs, sector, auto_dedup, resolve_conflicts)
                if isinstance(first, str):
                    # List of raw sentences or file paths
                    docs = [StandardDocumentItem(text=s, sector=sector or "general") for s in source]
                    return self._ingest_document_batch(docs, sector, auto_dedup, resolve_conflicts)

            # 3. File & Directory Paths
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.is_dir():
                    rep = self._ingest_directory(path, sector, auto_dedup, resolve_conflicts)
                elif path.is_file():
                    rep = self._ingest_file(path, format, sector, default_confidence, provenance, auto_dedup, resolve_conflicts)
                else:
                    # Treat as raw string content (e.g. JSON string or markdown paragraph)
                    str_src = str(source).strip()
                    if str_src.startswith("{") or str_src.startswith("["):
                        try:
                            parsed = json.loads(str_src)
                            rep = self.load_source(parsed, format=format, sector=sector, auto_dedup=auto_dedup, resolve_conflicts=resolve_conflicts)
                        except Exception:
                            parsed = None
                    if not str_src.startswith("{") and not str_src.startswith("["):
                        # Raw text document
                        doc = StandardDocumentItem(text=str_src, sector=sector or "general", provenance=provenance or "raw_text")
                        rep = self._ingest_document_batch([doc], sector, auto_dedup, resolve_conflicts)
                rep.sources_loaded = max(rep.sources_loaded, 1)
                return rep

        except Exception as e:
            report.errors.append(f"Ingestion failure: {str(e)}")

        return report

    def load_sources(
        self,
        sources: Sequence[Any],
        sector: Optional[str] = None,
        auto_dedup: bool = True,
        resolve_conflicts: bool = True,
    ) -> GlobalIngestionReport:
        """Ingest a collection of diverse sources in a single coordinated operation."""
        total_report = GlobalIngestionReport()
        for src in sources:
            r = self.load_source(
                src,
                sector=sector,
                auto_dedup=auto_dedup,
                resolve_conflicts=resolve_conflicts,
            )
            total_report.merge(r)
        return total_report

    # ─────────────────────────────────────────────────────────────────────────
    # File & Format Parsing Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _ingest_file(
        self,
        path: Path,
        format: str,
        sector: Optional[str],
        default_confidence: float,
        provenance: Optional[str],
        auto_dedup: bool,
        resolve_conflicts: bool,
    ) -> GlobalIngestionReport:
        ext = path.suffix.lower()
        prov = provenance or f"file:{path.name}"
        sec = sector or self._infer_sector(path)

        if ext == ".json" or format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.load_source(data, format="json", sector=sec, provenance=prov, auto_dedup=auto_dedup, resolve_conflicts=resolve_conflicts)

        elif ext == ".jsonl" or format == "jsonl":
            triplets = []
            docs = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "source" in obj or "subject" in obj or "head" in obj:
                        triplets.append(StandardKnowledgeItem.from_dict(obj))
                    elif "text" in obj or "content" in obj:
                        docs.append(StandardDocumentItem.from_dict(obj))

            report = GlobalIngestionReport(sources_loaded=1)
            if triplets:
                r1 = self._ingest_triplet_batch(triplets, sec, auto_dedup, resolve_conflicts)
                report.merge(r1)
            if docs:
                r2 = self._ingest_document_batch(docs, sec, auto_dedup, resolve_conflicts)
                report.merge(r2)
            return report

        elif ext in (".csv", ".tsv") or format in ("csv", "tsv"):
            delimiter = "\t" if (ext == ".tsv" or format == "tsv") else ","
            return self._ingest_csv_file(path, delimiter, sec, prov, auto_dedup, resolve_conflicts)

        elif ext in (".txt", ".md", ".markdown", ".rst", ".log") or format in ("text", "markdown"):
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            # Parse YAML frontmatter if present
            front_sec, content = self._extract_frontmatter(raw_text)
            final_sec = sector or front_sec or sec
            doc = StandardDocumentItem(text=content, sector=final_sec, provenance=prov)
            return self._ingest_document_batch([doc], final_sec, auto_dedup, resolve_conflicts)

        else:
            # Fallback text reading
            text = path.read_text(encoding="utf-8", errors="replace")
            doc = StandardDocumentItem(text=text, sector=sec, provenance=prov)
            return self._ingest_document_batch([doc], sec, auto_dedup, resolve_conflicts)

    def _ingest_directory(
        self,
        dir_path: Path,
        sector: Optional[str],
        auto_dedup: bool,
        resolve_conflicts: bool,
    ) -> GlobalIngestionReport:
        overall = GlobalIngestionReport()
        supported = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".markdown"}
        for f in dir_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported:
                r = self._ingest_file(f, "auto", sector, 1.0, None, auto_dedup, resolve_conflicts)
                overall.merge(r)
        return overall

    def _ingest_csv_file(
        self,
        path: Path,
        delimiter: str,
        sector: str,
        provenance: str,
        auto_dedup: bool,
        resolve_conflicts: bool,
    ) -> GlobalIngestionReport:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Find columns
                s = row.get("source") or row.get("subject") or row.get("head") or row.get("entity_a") or ""
                r = row.get("relation") or row.get("predicate") or row.get("type") or "connected_to"
                t = row.get("target") or row.get("object") or row.get("tail") or row.get("entity_b") or ""
                if s and t:
                    conf = float(row.get("confidence", 1.0))
                    sec = row.get("sector", sector)
                    meta = {k: v for k, v in row.items() if k not in ("source", "subject", "relation", "predicate", "target", "object", "confidence", "sector")}
                    items.append(StandardKnowledgeItem(
                        source=s, relation=r, target=t,
                        sector=sec, confidence=conf, provenance=provenance,
                        metadata=meta,
                    ))

        return self._ingest_triplet_batch(items, sector, auto_dedup, resolve_conflicts)

    # ─────────────────────────────────────────────────────────────────────────
    # Ingestion & Graph Graph Assembly Engine
    # ─────────────────────────────────────────────────────────────────────────

    def _ingest_package(
        self,
        pkg: KnowledgePackage,
        sector_override: Optional[str],
        auto_dedup: bool,
        resolve_conflicts: bool,
    ) -> GlobalIngestionReport:
        report = GlobalIngestionReport(sources_loaded=1)
        sec = sector_override or pkg.domain

        if pkg.items:
            r1 = self._ingest_triplet_batch(pkg.items, sec, auto_dedup, resolve_conflicts)
            report.merge(r1)

        if pkg.documents:
            r2 = self._ingest_document_batch(pkg.documents, sec, auto_dedup, resolve_conflicts)
            report.merge(r2)

        return report

    def _ingest_triplet_batch(
        self,
        items: List[StandardKnowledgeItem],
        default_sector: Optional[str],
        auto_dedup: bool,
        resolve_conflicts: bool,
    ) -> GlobalIngestionReport:
        report = GlobalIngestionReport(sources_loaded=0)
        if not items:
            return report

        # 1. Collect unique node labels with intra-batch normalization
        label_to_node_id: Dict[str, str] = {}
        new_labels_to_embed: List[str] = []
        batch_norm_to_primary: Dict[str, str] = {}

        for it in items:
            if default_sector:
                it.sector = default_sector

            for lbl in (it.source, it.target):
                norm_key = self._normalize_label(lbl)

                # Check existing in graph
                if norm_key in self._text_to_node_id:
                    label_to_node_id[lbl] = self._text_to_node_id[norm_key]
                    if auto_dedup and lbl not in label_to_node_id:
                        report.nodes_merged += 1
                # Check intra-batch duplicate
                elif norm_key in batch_norm_to_primary:
                    primary_lbl = batch_norm_to_primary[norm_key]
                    label_to_node_id[lbl] = primary_lbl  # Will be mapped to node_id after creation
                    if auto_dedup:
                        report.nodes_merged += 1
                else:
                    batch_norm_to_primary[norm_key] = lbl
                    new_labels_to_embed.append(lbl)

        # 2. Vectorize and batch insert missing nodes
        if new_labels_to_embed:
            vectors = self.embedder.encode(new_labels_to_embed)
            created_nodes = self.graph.add_nodes_bulk(
                vectors=vectors,
                source_texts=new_labels_to_embed,
                sectors=[default_sector or "general"] * len(new_labels_to_embed),
            )
            for lbl, node in zip(new_labels_to_embed, created_nodes):
                norm_key = self._normalize_label(lbl)
                self._text_to_node_id[norm_key] = node.id
                label_to_node_id[lbl] = node.id
                report.nodes_created += 1

        # Resolve any secondary labels mapped to primary labels
        for norm_key, primary_lbl in batch_norm_to_primary.items():
            if primary_lbl in label_to_node_id:
                primary_id = label_to_node_id[primary_lbl]
                for it in items:
                    if self._normalize_label(it.source) == norm_key:
                        label_to_node_id[it.source] = primary_id
                    if self._normalize_label(it.target) == norm_key:
                        label_to_node_id[it.target] = primary_id

        # 3. Insert Relation Edges with Conflict Resolution
        edges_data = []
        for it in items:
            report.items_processed += 1
            src_id = label_to_node_id.get(it.source)
            tgt_id = label_to_node_id.get(it.target)
            if not src_id or not tgt_id:
                continue

            sec = it.sector or default_sector or "general"
            if sec not in report.sectors_affected:
                report.sectors_affected.append(sec)

            edge_id = f"{src_id}->{tgt_id}:{it.relation}"
            if edge_id in self.graph.edges:
                report.conflicts_resolved += 1

            edge_data = {
                "source_id": src_id,
                "target_id": tgt_id,
                "relation_type": it.relation,
                "confidence": it.confidence,
                "sector": sec,
                "provenance": it.provenance,
                "valid_from": it.valid_from,
                "valid_to": it.valid_to,
                "metadata": it.metadata,
            }
            edges_data.append(edge_data)

        if edges_data:
            created_edges = self.graph.add_edges_bulk(edges_data)
            report.edges_created += len(created_edges)

        return report

    def _ingest_document_batch(
        self,
        docs: List[StandardDocumentItem],
        default_sector: Optional[str],
        auto_dedup: bool,
        resolve_conflicts: bool,
    ) -> GlobalIngestionReport:
        report = GlobalIngestionReport(sources_loaded=0)

        for doc in docs:
            report.documents_processed += 1
            sec = doc.sector or default_sector or "general"
            if sec not in report.sectors_affected:
                report.sectors_affected.append(sec)

            # Ingest explicit relations if provided in document
            if doc.relations:
                r = self._ingest_triplet_batch(doc.relations, sec, auto_dedup, resolve_conflicts)
                report.merge(r)

            # Ingest document text as fact chunks and concept nodes
            sentences = self._split_into_sentences(doc.text)
            for sent in sentences:
                norm_key = self._normalize_label(sent)
                if norm_key in self._text_to_node_id:
                    if auto_dedup:
                        report.nodes_merged += 1
                    continue

                vec = self.embedder.encode_single(sent)
                node = self.graph.add_node(
                    vector=vec,
                    source_text=sent,
                    sector=sec,
                    provenance=doc.provenance,
                    confidence=1.0,
                    metadata=doc.metadata,
                )
                self._text_to_node_id[norm_key] = node.id
                report.nodes_created += 1

        return report

    # ── Helpers ──

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _extract_frontmatter(self, text: str) -> Tuple[Optional[str], str]:
        """Parse YAML frontmatter if exists."""
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                header, body = parts[1], parts[2]
                sec_match = re.search(r'sector:\s*([a-zA-Z0-9_-]+)', header, re.IGNORECASE)
                sector = sec_match.group(1).strip().lower() if sec_match else None
                return sector, body.strip()
        return None, text

    def _infer_sector(self, path: Path) -> str:
        """Dynamically infer domain sector using semantic embedding classification."""
        return self.sector_classifier.classify(path, fallback="general")
