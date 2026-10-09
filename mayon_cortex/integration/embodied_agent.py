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
Pillar 4: Embodied Cortex Agent & Action Registry
=================================================
Implements the real-world Perception → Reason → Act → Learn cognitive cycle.

Perceive: Ingests real environment signals (telemetry, files, structured JSON, sensor data).
Reason:   Formulates deterministic goal-directed plans with contraindication gating.
Act:      Executes registered real-world tools via ActionRegistry.
Learn:    Reinforces successful decision paths via Hebbian plasticity and records
          precedents into CaseMemory — zero gradient descent or GPU required.
Storage:  Persists all action episodes to disk in cortex_storage/embodied/.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from mayon_cortex.engine import CortexGraph
from mayon_cortex.integration.cortex_llm_bridge import (
    cortex_decide,
    cortex_remember,
    cortex_validate,
)


@dataclass
class ActionResult:
    """Outcome of a tool/action execution."""
    action_name: str
    parameters: Dict[str, Any]
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmbodiedEpisode:
    """Full record of one Perceive-Reason-Act-Learn cycle."""
    episode_id: str
    goal: str
    perception_summary: str
    decision_steps: List[Dict[str, Any]]
    actions_executed: List[ActionResult]
    learned_precedent_id: Optional[str]
    success: bool
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "goal": self.goal,
            "perception_summary": self.perception_summary,
            "decision_steps": self.decision_steps,
            "actions_executed": [a.to_dict() for a in self.actions_executed],
            "learned_precedent_id": self.learned_precedent_id,
            "success": self.success,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
        }


class ActionRegistry:
    """Registry of executable environment actions."""

    def __init__(self):
        self._actions: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._register_default_actions()

    def register_action(self, name: str, fn: Callable[[Dict[str, Any]], Any]):
        """Register a new custom tool or environment action."""
        self._actions[name] = fn

    def execute(self, name: str, params: Dict[str, Any]) -> ActionResult:
        """Execute a registered action with exception isolation."""
        t0 = time.time()
        if name not in self._actions:
            return ActionResult(
                action_name=name,
                parameters=params,
                success=False,
                output=None,
                error=f"Action '{name}' is not registered in ActionRegistry",
                execution_time_ms=(time.time() - t0) * 1000.0,
            )

        try:
            res = self._actions[name](params)
            elapsed = (time.time() - t0) * 1000.0
            return ActionResult(
                action_name=name,
                parameters=params,
                success=True,
                output=res,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000.0
            return ActionResult(
                action_name=name,
                parameters=params,
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=elapsed,
            )

    def _register_default_actions(self):
        """Register core environment tools."""
        # 1. File Write
        def _file_write(p: Dict[str, Any]) -> str:
            path = Path(p["filepath"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(p.get("content", ""))
            return f"Written {len(p.get('content', ''))} bytes to {path}"

        # 2. File Read
        def _file_read(p: Dict[str, Any]) -> str:
            path = Path(p["filepath"])
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        # 3. Math Expression Solver
        def _compute_math(p: Dict[str, Any]) -> str:
            expr = p.get("expression", "")
            # Safe evaluation for basic math
            # Using ast/math or simple parser
            import ast
            import operator as op
            operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                raise ValueError("Unsupported operator")
            res = eval_expr(ast.parse(expr, mode='eval').body)
            return str(res)

        # 4. Dispatch Alert
        def _alert(p: Dict[str, Any]) -> str:
            level = p.get("level", "INFO")
            msg = p.get("message", "")
            return f"[{level.upper()}_ALERT] {msg}"

        self._actions["file_write"] = _file_write
        self._actions["file_read"] = _file_read
        self._actions["compute_math"] = _compute_math
        self._actions["dispatch_alert"] = _alert


class EmbodiedCortexAgent:
    """
    Perception-Reason-Act-Learn embodied cognitive loop.
    Persistent storage in cortex_storage/embodied/.
    """

    def __init__(
        self,
        cortex: CortexGraph,
        action_registry: Optional[ActionRegistry] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.cortex = cortex
        self.registry = action_registry or ActionRegistry()
        self.storage_dir = Path(storage_dir) if storage_dir else Path("cortex_storage/embodied")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.storage_dir / "episodes.jsonl"
        self._episode_count = 0

    def perceive(
        self,
        observation: Union[str, Dict[str, Any]],
        sector: str = "environment",
    ) -> str:
        """
        Perceive state from the environment and integrate into MayonGraph.
        """
        summary = ""
        if isinstance(observation, dict):
            # Ingest key-value telemetry
            statements = []
            for k, v in observation.items():
                stmt = f"Sensor {k} current value is {v}."
                statements.append(stmt)
                self.cortex.ingest(stmt, sector=sector)
            summary = "; ".join(statements)
        else:
            obs_text = str(observation)
            self.cortex.ingest(obs_text, sector=sector)
            summary = obs_text

        return summary

    def reason_and_plan(
        self,
        goal: str,
        sector: str = "environment",
    ) -> Dict[str, Any]:
        """
        Formulate goal-directed action steps with contraindication checks.
        """
        return cortex_decide(self.cortex, situation=goal, sector=sector)

    def execute_cycle(
        self,
        goal: str,
        observation: Optional[Union[str, Dict[str, Any]]] = None,
        required_actions: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
        sector: str = "environment",
    ) -> EmbodiedEpisode:
        """
        Execute full Perceive → Reason → Act → Learn loop.
        """
        t0 = time.time()
        self._episode_count += 1
        ep_id = f"ep_{int(time.time()*1000)}_{self._episode_count}"

        # 1. Perceive
        percep_summary = ""
        if observation:
            percep_summary = self.perceive(observation, sector=sector)

        # 2. Reason & Plan
        decision = self.reason_and_plan(goal, sector=sector)
        steps = decision.get("steps", [])
        conflicts = decision.get("conflicts", [])

        # Safety Gate: Abort if serious contradiction
        val = cortex_validate(self.cortex, claim_or_decision=goal)
        all_conflicts = conflicts + val.get("contradictions", [])

        actions_executed: List[ActionResult] = []
        overall_success = True

        if all_conflicts:
            # Cannot act safely
            overall_success = False
            actions_executed.append(
                ActionResult(
                    action_name="safety_abort",
                    parameters={"conflicts": all_conflicts},
                    success=False,
                    output="Action aborted due to detected contraindication or conflict",
                    error="; ".join(all_conflicts),
                )
            )
        else:
            # 3. Act
            # Execute supplied required actions or mapped decision actions
            to_run = required_actions or []
            if not to_run and steps:
                # Default: dispatch alert of the planned action
                top_step = steps[0]
                to_run = [("dispatch_alert", {"level": "info", "message": top_step.get("action", goal)})]

            for act_name, act_params in to_run:
                act_res = self.registry.execute(act_name, act_params)
                actions_executed.append(act_res)
                if not act_res.success:
                    overall_success = False

        # 4. Learn (Online Hebbian + CaseMemory)
        learned_case_id = None
        if overall_success:
            learned_case_id = cortex_remember(
                brain=self.cortex,
                situation=f"Goal: {goal} | State: {percep_summary[:100]}",
                judgment=f"Executed: {[a.action_name for a in actions_executed]}",
                reasoning="Goal satisfied safely without contradictions",
                sector=sector,
            )

        elapsed = (time.time() - t0) * 1000.0
        episode = EmbodiedEpisode(
            episode_id=ep_id,
            goal=goal,
            perception_summary=percep_summary,
            decision_steps=steps,
            actions_executed=actions_executed,
            learned_precedent_id=learned_case_id,
            success=overall_success,
            total_latency_ms=elapsed,
        )

        self._record_episode(episode)
        return episode

    def _record_episode(self, episode: EmbodiedEpisode):
        """Persist episode log to disk."""
        try:
            with open(self.episodes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode.to_dict()) + "\n")
        except Exception:
            pass

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Read back episode records from disk."""
        episodes = []
        if self.episodes_file.exists():
            with open(self.episodes_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            episodes.append(json.loads(line))
                        except Exception:
                            pass
        return episodes
