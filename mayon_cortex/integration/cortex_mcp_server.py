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
Mayon-Cortex Model Context Protocol (MCP) Server
=================================================
Exposes Mayon-Cortex's 5 neuro-symbolic reasoning tools via the standard MCP protocol.
Compatible with Claude Desktop, Antigravity, Cursor, and any MCP client.

Tools Provided:
  - cortex_think: Multi-hop graph reasoning & symbolic proofs
  - cortex_recall: Case-based precedent memory retrieval
  - cortex_remember: Store successful reasoning case into long-term memory
  - cortex_decide: Deterministic priority decision planning & conflict resolution
  - cortex_validate: Post-generation hallucination & contradiction detection
  - cortex_plan: Strategic query decomposition & complexity routing
"""

import json
import sys
from typing import Any, Dict, List, Optional

from mayon_cortex.engine import CortexGraph
from mayon_cortex.integration.cortex_llm_bridge import (
    cortex_think,
    cortex_recall,
    cortex_remember,
    cortex_decide,
    cortex_validate,
    cortex_plan,
)

# Standard MCP Tool Definitions
TOOLS_SCHEMA = [
    {
        "name": "cortex_think",
        "description": "Performs symbolic multi-hop graph reasoning, spreading activation, and path ranking with exact mathematical/logical proofs and zero hallucinations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The complex query or assertion to reason about."},
                "focus_sectors": {"type": "array", "items": {"type": "string"}, "description": "Optional domain sectors (e.g. medical, legal, code)."},
                "max_hops": {"type": "integer", "default": 4, "description": "Maximum hops for activation wave traversal."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cortex_recall",
        "description": "Retrieves historical solved cases, precedents, and analogies from episodic CaseMemory to guide new reasoning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "situation": {"type": "string", "description": "The current problem or situation to find precedents for."},
                "top_k": {"type": "integer", "default": 3, "description": "Number of precedent cases to retrieve."},
            },
            "required": ["situation"],
        },
    },
    {
        "name": "cortex_remember",
        "description": "Stores a solved case (situation, judgment, reasoning rationale) into persistent episodic CaseMemory for continuous learning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "situation": {"type": "string", "description": "The problem or situation that was solved."},
                "judgment": {"type": "string", "description": "The verified decision or answer."},
                "reasoning": {"type": "string", "description": "The step-by-step rationale for why this judgment was made."},
                "sector": {"type": "string", "default": "general", "description": "Domain sector."},
            },
            "required": ["situation", "judgment", "reasoning"],
        },
    },
    {
        "name": "cortex_decide",
        "description": "Computes deterministic, priority-ordered decision chains and detects contraindications or opposing evidence across multiple domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "situation": {"type": "string", "description": "The decision situation or dilemma to evaluate."},
                "sector": {"type": "string", "default": "general", "description": "The domain sector."},
            },
            "required": ["situation"],
        },
    },
    {
        "name": "cortex_validate",
        "description": "Post-generation verification gate that checks LLM claims against knowledge graph ground truth for hallucinations, contraindications, or missing preconditions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim_or_decision": {"type": "string", "description": "The statement, claim, or action proposed by the LLM."},
                "proposed_reasoning": {"type": "string", "default": "", "description": "The supporting reasoning provided."},
                "strict": {"type": "boolean", "default": False, "description": "If true, treats ungrounded claims as unsafe."},
            },
            "required": ["claim_or_decision"],
        },
    },
    {
        "name": "cortex_plan",
        "description": "Executive query complexity classifier that analyzes questions (trivial vs complex vs multi-part) and selects optimal thinking strategy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The raw user query."},
            },
            "required": ["query"],
        },
    },
]


class MayonCortexMCPServer:
    """JSON-RPC / MCP Server for Mayon-Cortex."""

    def __init__(self, brain: Optional[CortexGraph] = None):
        self.brain = brain or CortexGraph()

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool calls to appropriate Mayon-Cortex subsystem."""
        if name == "cortex_think":
            return cortex_think(
                self.brain,
                query=arguments["query"],
                focus_sectors=arguments.get("focus_sectors"),
                max_hops=arguments.get("max_hops", 4),
            )
        elif name == "cortex_recall":
            return cortex_recall(
                self.brain,
                situation=arguments["situation"],
                top_k=arguments.get("top_k", 3),
            )
        elif name == "cortex_remember":
            case_id = cortex_remember(
                self.brain,
                situation=arguments["situation"],
                judgment=arguments["judgment"],
                reasoning=arguments["reasoning"],
                sector=arguments.get("sector", "general"),
            )
            return {"success": True, "case_id": case_id, "message": f"Successfully stored precedent case {case_id}"}
        elif name == "cortex_decide":
            return cortex_decide(
                self.brain,
                situation=arguments["situation"],
                sector=arguments.get("sector", "general"),
            )
        elif name == "cortex_validate":
            return cortex_validate(
                self.brain,
                claim_or_decision=arguments["claim_or_decision"],
                proposed_reasoning=arguments.get("proposed_reasoning", ""),
                strict=arguments.get("strict", False),
            )
        elif name == "cortex_plan":
            return cortex_plan(
                self.brain,
                query=arguments["query"],
            )
        else:
            raise ValueError(f"Unknown MCP tool: {name}")

    def run_stdio(self):
        """Run the server reading from stdin and writing to stdout (standard MCP mode)."""
        sys.stderr.write("Mayon-Cortex MCP Server running on stdio...\n")
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                method = req.get("method")
                req_id = req.get("id")

                if method == "tools/list":
                    res = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SCHEMA}}
                elif method == "tools/call":
                    params = req.get("params", {})
                    name = params.get("name")
                    args = params.get("arguments", {})
                    out = self.handle_tool_call(name, args)
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(out, indent=2)}]
                        },
                    }
                else:
                    res = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_res = {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}}
                sys.stdout.write(json.dumps(err_res) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    server = MayonCortexMCPServer()
    server.run_stdio()
