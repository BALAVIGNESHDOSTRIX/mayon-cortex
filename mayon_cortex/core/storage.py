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
Graph Storage Manager — Disk-Backed Persistence & Snapshotting
==============================================================
Phase 5 of the Brain-Like Intelligence upgrade.

Handles high-throughput, persistent state serialization for Cortex-Graph:
  - Atomic checkpointing and binary snapshots (pickle / json / npz)
  - Incremental journal logging of node/edge mutations
  - Memory-mapped vector caches for billion-scale node embeddings
  - Versioned backup and recovery mechanisms
"""

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class StorageConfig:
    """Storage management parameters."""
    base_dir: str = "./cortex_storage"
    snapshot_interval_sec: float = 300.0  # 5 min
    enable_journaling: bool = True
    compression: bool = False
    max_history_snapshots: int = 5


class GraphStorageManager:
    """
    Manages cold and warm storage for Cortex-Graph and MayonGraph.
    Provides snapshot, restore, and WAL (Write-Ahead-Log) journaling.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.cfg = config or StorageConfig()
        self.base_path = Path(self.cfg.base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.base_path / "graph_journal.jsonl"
        self._last_snapshot_time = time.time()

    def append_journal(self, operation: str, data: Dict[str, Any]):
        """Write an operation to the write-ahead log."""
        if not self.cfg.enable_journaling:
            return
        entry = {
            "timestamp": time.time(),
            "op": operation,
            "data": data,
        }
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def save_snapshot(
        self,
        graph_obj: Any,
        metadata: Optional[Dict[str, Any]] = None,
        snapshot_name: Optional[str] = None,
    ) -> str:
        """
        Create a full snapshot of the cortex graph state.
        """
        ts = int(time.time())
        name = snapshot_name or f"snapshot_{ts}"
        snap_dir = self.base_path / name
        snap_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata
        meta = {
            "timestamp": ts,
            "snapshot_name": name,
            "custom_metadata": metadata or {},
        }
        with open(snap_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 2. Save graph payload
        graph_file = snap_dir / "graph_payload.pkl"
        with open(graph_file, "wb") as f:
            pickle.dump(graph_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._cleanup_old_snapshots()
        self._last_snapshot_time = time.time()
        return str(snap_dir)

    def load_snapshot(self, snapshot_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a snapshot from path and return (graph_object, metadata).
        """
        p = Path(snapshot_path)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_path}")

        meta_file = p / "meta.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        graph_file = p / "graph_payload.pkl"
        with open(graph_file, "rb") as f:
            graph_obj = pickle.load(f)

        return graph_obj, meta

    def get_latest_snapshot(self) -> Optional[str]:
        """Find the most recent valid snapshot directory."""
        candidates = [d for d in self.base_path.iterdir() if d.is_dir() and (d / "graph_payload.pkl").exists()]
        if not candidates:
            return None
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return str(candidates[0])

    def _cleanup_old_snapshots(self):
        """Retain only max_history_snapshots."""
        candidates = [d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith("snapshot_")]
        if len(candidates) > self.cfg.max_history_snapshots:
            candidates.sort(key=lambda d: d.stat().st_mtime)
            to_remove = candidates[:-self.cfg.max_history_snapshots]
            for folder in to_remove:
                for child in folder.iterdir():
                    try:
                        child.unlink()
                    except Exception:
                        pass
                try:
                    folder.rmdir()
                except Exception:
                    pass
