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
Error Memory & Contradiction Registry
=====================================
Stores detected errors, invalid reasoning paths, and refuted claims so that
Mayon-Cortex never repeats past mistakes or follows dead-end paths.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ErrorRecord:
    """Record of an invalidated claim, dead-end path, or contradiction."""
    error_id: str
    query_or_context: str
    faulty_reasoning: str
    contradiction_type: str  # "contraindication", "false_relation", "dead_end", "arithmetic_error"
    correction: str
    sector: str = "general"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorMemory:
    """
    Persistent store for past reasoning failures and contradictions.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path("cortex_storage/errors")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.records_file = self.storage_dir / "error_records.jsonl"
        self._cache: List[ErrorRecord] = []
        self._load_records()

    def record_error(
        self,
        query: str,
        faulty_reasoning: str,
        contradiction_type: str,
        correction: str,
        sector: str = "general",
    ) -> str:
        """Register a reasoning error or contradiction into persistent memory."""
        rec_id = f"err_{int(time.time()*1000)}_{len(self._cache)}"
        rec = ErrorRecord(
            error_id=rec_id,
            query_or_context=query,
            faulty_reasoning=faulty_reasoning,
            contradiction_type=contradiction_type,
            correction=correction,
            sector=sector,
        )
        self._cache.append(rec)
        self._append_to_file(rec)
        return rec_id

    def check_for_known_error(self, query_or_claim: str) -> Optional[ErrorRecord]:
        """Check if this query or claim matches any previously refuted errors."""
        q_lower = query_or_claim.lower()
        for rec in reversed(self._cache):
            if rec.query_or_context.lower() in q_lower or q_lower in rec.query_or_context.lower():
                return rec
            if rec.faulty_reasoning and rec.faulty_reasoning.lower() in q_lower:
                return rec
        return None

    def _append_to_file(self, rec: ErrorRecord):
        try:
            with open(self.records_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict()) + "\n")
        except Exception:
            pass

    def _load_records(self):
        if self.records_file.exists():
            with open(self.records_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            self._cache.append(ErrorRecord(**d))
                        except Exception:
                            pass

    def stats(self) -> Dict[str, Any]:
        return {
            "total_errors_recorded": len(self._cache),
            "by_type": {
                t: sum(1 for r in self._cache if r.contradiction_type == t)
                for t in set(r.contradiction_type for r in self._cache)
            },
        }
