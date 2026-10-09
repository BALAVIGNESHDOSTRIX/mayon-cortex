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
Pillar 3: Multi-Agent Cortex Ensemble & Confidence Voting
=========================================================
Coordinates a federated team of specialized Cortex agents (e.g. Medical,
Legal, Financial, Engineering).

Features:
- Parallel or sector-routed multi-agent reasoning.
- ConfidenceVoter: Ranks agent outputs by proof confidence and epistemic certainty.
- Cross-agent contradiction detection (e.g., Clinical risk vs Business rule).
- Persistent saving and loading of entire multi-agent ensembles to disk.
- Prepares verified consensus contexts for LLM verbalization.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from mayon_cortex.core.config import CortexConfig
if TYPE_CHECKING:
    from mayon_cortex.engine import CortexGraph, CortexResponse
from mayon_cortex.integration.cortex_llm_bridge import cortex_think, cortex_validate


@dataclass
class AgentResult:
    """Output from an individual specialized Cortex agent."""
    agent_id: str
    sector: str
    confidence: float
    is_uncertain: bool
    text: str
    proof_paths: List[List[str]] = field(default_factory=list)
    has_contraindication: bool = False
    contraindication_details: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnsembleResponse:
    """Consolidated consensus output from the multi-agent ensemble."""
    query: str
    winner_agent_id: str
    winner_sector: str
    winning_confidence: float
    consensus_reached: bool
    all_agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    contradictions_detected: List[str] = field(default_factory=list)
    consensus_text: str = ""
    llm_output: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "winner_agent_id": self.winner_agent_id,
            "winner_sector": self.winner_sector,
            "winning_confidence": self.winning_confidence,
            "consensus_reached": self.consensus_reached,
            "contradictions_detected": self.contradictions_detected,
            "consensus_text": self.consensus_text,
            "llm_output": self.llm_output,
            "execution_time_ms": self.execution_time_ms,
            "all_agent_results": {k: v.to_dict() for k, v in self.all_agent_results.items()},
        }


class ConfidenceVoter:
    """
    Ranks agent proposals, checks cross-agent contradictions,
    and determines winning consensus.
    """

    @staticmethod
    def vote(
        agent_results: Dict[str, AgentResult],
    ) -> Tuple[str, float, bool, List[str]]:
        """
        Returns: (winner_id, winning_confidence, consensus_reached, contradictions)
        """
        if not agent_results:
            return "none", 0.0, False, ["No agents provided results"]

        # Check for cross-agent contradictions
        contradictions = []
        contraindicated_agents = set()

        for a_id, res in agent_results.items():
            if res.has_contraindication:
                contraindicated_agents.add(a_id)
                contradictions.extend(res.contraindication_details)

        # Filter eligible non-uncertain candidates
        candidates = [
            res for a_id, res in agent_results.items()
            if not res.is_uncertain and a_id not in contraindicated_agents
        ]

        if not candidates:
            # Fall back to highest confidence regardless
            best = max(agent_results.values(), key=lambda r: r.confidence)
            return best.agent_id, best.confidence, False, contradictions

        # Sort by confidence descending
        candidates.sort(key=lambda r: r.confidence, reverse=True)
        winner = candidates[0]

        # Check consensus: if multiple valid agents have conf > 0.6
        valid_count = sum(1 for c in candidates if c.confidence >= 0.6)
        consensus = (valid_count >= 2) and (len(contradictions) == 0)

        return winner.agent_id, winner.confidence, consensus, contradictions


class CortexEnsemble:
    """
    Federation of specialized Mayon-Cortex cognitive instances.
    Provides storage persistence and cross-domain ensemble voting.
    """

    def __init__(
        self,
        agents: Optional[Dict[str, CortexGraph]] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.agents: Dict[str, CortexGraph] = agents or {}
        self.agent_sectors: Dict[str, str] = {}
        self.storage_dir = Path(storage_dir) if storage_dir else Path("cortex_storage/ensembles")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        for name, brain in self.agents.items():
            sectors = list(brain.graph.sector_indices.keys())
            self.agent_sectors[name] = sectors[0] if sectors else "general"

    def register_agent(self, agent_id: str, brain: CortexGraph, sector: str = "general"):
        """Add a specialized agent to the ensemble."""
        self.agents[agent_id] = brain
        self.agent_sectors[agent_id] = sector

    def query_ensemble(
        self,
        query: str,
        target_sectors: Optional[List[str]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> EnsembleResponse:
        """
        Execute parallel reasoning across all candidate agents and vote on the solution.
        """
        t0 = time.time()
        results: Dict[str, AgentResult] = {}

        # 1. Query each agent
        for a_id, brain in self.agents.items():
            sector = self.agent_sectors.get(a_id, "general")
            if target_sectors and sector not in target_sectors:
                continue

            ta0 = time.time()
            think_data = cortex_think(brain, query=query, focus_sectors=[sector], max_hops=3)
            val_data = cortex_validate(brain, claim_or_decision=query)
            elapsed_a = (time.time() - ta0) * 1000.0

            has_contra = len(val_data.get("contradictions", [])) > 0
            contra_list = [f"Agent [{a_id}] contradiction: {c}" for c in val_data.get("contradictions", [])]

            res_obj = AgentResult(
                agent_id=a_id,
                sector=sector,
                confidence=think_data.get("confidence", 0.0),
                is_uncertain=think_data.get("is_uncertain", False),
                text=think_data.get("answer") or " ".join(f.get("text", "") for f in think_data.get("facts", [])[:3]),
                proof_paths=think_data.get("proof_paths", []),
                has_contraindication=has_contra,
                contraindication_details=contra_list,
                execution_time_ms=elapsed_a,
            )
            results[a_id] = res_obj

        # 2. Vote
        winner_id, win_conf, consensus, contradictions = ConfidenceVoter.vote(results)
        winner_res = results.get(winner_id)
        win_sector = winner_res.sector if winner_res else "unknown"

        # 3. Assemble consensus context
        summary_lines = [f"=== Multi-Agent Cortex Ensemble Voting ==="]
        for a_id, r in results.items():
            status = "CONTRAINDICATION" if r.has_contraindication else ("UNCERTAIN" if r.is_uncertain else "VALID")
            summary_lines.append(f"• Agent [{a_id}] ({r.sector}): conf={r.confidence:.2f} [{status}]")
            if r.text:
                summary_lines.append(f"  Proof summary: {r.text[:120]}...")

        if contradictions:
            summary_lines.append("\n=== Detected Contradictions ===")
            for c in contradictions:
                summary_lines.append(f"  ⚠️ {c}")

        consensus_text = "\n".join(summary_lines)

        # 4. Optional LLM Verbalization
        llm_out = ""
        if llm_fn:
            prompt = (
                f"You are the executive communicator for a Multi-Agent Cortex Ensemble.\n"
                f"{consensus_text}\n\n"
                f"Winner Agent: [{winner_id}] with confidence {win_conf:.2f}\n"
                f"Query: {query}\n"
                f"Synthesize the final authoritative answer representing this ensemble consensus."
            )
            llm_out = llm_fn(prompt)
        else:
            llm_out = (winner_res.text if winner_res else "Ensemble consensus completed.")

        total_elapsed = (time.time() - t0) * 1000.0
        return EnsembleResponse(
            query=query,
            winner_agent_id=winner_id,
            winner_sector=win_sector,
            winning_confidence=win_conf,
            consensus_reached=consensus,
            all_agent_results=results,
            contradictions_detected=contradictions,
            consensus_text=consensus_text,
            llm_output=llm_out,
            execution_time_ms=total_elapsed,
        )

    def save_ensemble(self, ensemble_name: str = "default_ensemble"):
        """Save all member agents and ensemble manifest to disk."""
        target_dir = self.storage_dir / ensemble_name
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "ensemble_name": ensemble_name,
            "timestamp": time.time(),
            "agents": {},
        }

        for a_id, brain in self.agents.items():
            agent_path = target_dir / a_id
            brain.save(str(agent_path))
            manifest["agents"][a_id] = {
                "sector": self.agent_sectors.get(a_id, "general"),
                "path": str(agent_path),
                "nodes": brain.graph.num_nodes,
                "edges": brain.graph.num_edges,
            }

        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load_ensemble(
        cls,
        ensemble_dir: Union[str, Path],
        config: Optional[CortexConfig] = None,
    ) -> "CortexEnsemble":
        """Load an ensemble from disk."""
        path = Path(ensemble_dir)
        manifest_file = path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found in {path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        from mayon_cortex.engine import CortexGraph
        agents: Dict[str, CortexGraph] = {}
        for a_id, meta in manifest.get("agents", {}).items():
            agent_path = Path(meta["path"])
            if agent_path.exists():
                brain = CortexGraph.load(str(agent_path), config=config)
                agents[a_id] = brain

        ensemble = cls(agents=agents, storage_dir=path.parent)
        for a_id, meta in manifest.get("agents", {}).items():
            ensemble.agent_sectors[a_id] = meta.get("sector", "general")

        return ensemble
