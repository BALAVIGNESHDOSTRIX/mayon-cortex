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
Mayon-Cortex × AI Agent (Ollama & OpenAI) Integration Demo
===========================================================
Demonstrates how to connect Mayon-Cortex as a neuro-symbolic brain co-processor
to real LLM backends:
  1. Local LLM via Ollama (Llama 3, Mistral, Qwen 2.5, DeepSeek-R1)
  2. Cloud LLM via OpenAI API (GPT-4o, GPT-4o-mini, or any OpenAI-compatible API like Groq, vLLM, DeepSeek)

Includes 2 Integration Patterns:
  Pattern 1: Autonomous Pre-Prompt Brain Loop (Co-Processor Pattern)
             cortex_plan -> cortex_recall -> cortex_think -> LLM -> cortex_validate
  Pattern 2: Dynamic Agent Tool-Calling (Function Calling Pattern)
             The LLM Agent itself autonomously decides when to invoke Cortex tools.

Prerequisites (Choose either):
  - For Ollama: `ollama run llama3` (running on http://localhost:11434)
  - For OpenAI: `export OPENAI_API_KEY="sk-..."` (or pass api_key parameter)

NOTE: This file is a ready-to-use template. Do not run without active Ollama or OpenAI credentials.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure mayon_cortex is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.engine import CortexGraph
from mayon_cortex.integration.cortex_llm_bridge import (
    CortexLLMBridge,
    cortex_think,
    cortex_recall,
    cortex_remember,
    cortex_decide,
    cortex_validate,
    cortex_plan,
)


# ==============================================================================
# 1. LLM CALLABLE ADAPTERS (Ollama & OpenAI)
# ==============================================================================

class OllamaClient:
    """Zero-dependency HTTP client for local Ollama instances."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"[Ollama Connection Error: Is Ollama running on {self.base_url}? Error: {e}]"


class OpenAIClient:
    """Zero-dependency HTTP client for OpenAI and OpenAI-compatible APIs (Groq, DeepSeek, vLLM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI /chat/completions endpoint."""
        if not self.api_key:
            return "[OpenAI Error: OPENAI_API_KEY environment variable is not set]"

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return "[No response choices returned]"
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            return f"[OpenAI API HTTP Error {e.code}: {err_body}]"
        except urllib.error.URLError as e:
            return f"[OpenAI Connection Error: {e}]"


# ==============================================================================
# 2. PATTERN 1: NEURO-SYMBOLIC CO-PROCESSOR AGENT
# ==============================================================================

class MayonCortexAgent:
    """
    An AI Agent powered by Mayon-Cortex neuro-symbolic substrate.
    
    Before calling the LLM, the agent:
      1. Classifies complexity & strategy (cortex_plan)
      2. Recalls past episodic precedents (cortex_recall)
      3. Traverses multi-hop relational graph & solves exact math (cortex_think)
      4. Generates decision action steps & detects conflicts (cortex_decide)
      5. Calls the LLM with grounded evidence
      6. Validates the output against graph truth & safety (cortex_validate)
      7. Persists verified result into lifelong case memory (cortex_remember)
    """

    def __init__(
        self,
        llm_type: str = "ollama",       # 'ollama' or 'openai'
        model_name: Optional[str] = None,
        brain: Optional[CortexGraph] = None,
        openai_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.brain = brain or CortexGraph(CortexConfig())
        self.bridge = CortexLLMBridge(cortex=self.brain)
        self.llm_type = llm_type.lower()

        # Initialize LLM backend
        if self.llm_type == "ollama":
            self.llm = OllamaClient(
                model=model_name or "llama3",
                base_url=ollama_url,
            )
        elif self.llm_type == "openai":
            self.llm = OpenAIClient(
                api_key=openai_api_key,
                model=model_name or "gpt-4o-mini",
            )
        else:
            raise ValueError("llm_type must be either 'ollama' or 'openai'")

    def seed_knowledge(self, facts: List[Dict[str, Any]]):
        """Seed domain knowledge into Mayon-Cortex graph."""
        for item in facts:
            n_src = self.brain.graph.add_node(
                vector=self.brain.embedder.encode_single(item["source"]),
                source_text=item["source"],
                sector=item.get("sector", "general"),
            )
            n_tgt = self.brain.graph.add_node(
                vector=self.brain.embedder.encode_single(item["target"]),
                source_text=item["target"],
                sector=item.get("sector", "general"),
            )
            self.brain.graph.add_edge(
                source_id=n_src.id,
                target_id=n_tgt.id,
                relation_type=item["relation"],
                confidence=item.get("confidence", 0.95),
                sector=item.get("sector", "general"),
            )

    def ask(self, query: str, sector: str = "general") -> Dict[str, Any]:
        """
        Execute full neuro-symbolic reasoning loop with real LLM backend.
        """
        print(f"\n==================================================")
        print(f"[*] Processing Query: '{query}'")
        print(f"[*] Backend LLM:      {self.llm_type.upper()} ({self.llm.model})")
        print(f"==================================================")

        # Call bridge enhanced_reason with the LLM generate function
        result = self.bridge.enhanced_reason(
            query=query,
            llm_fn=lambda prompt: self.llm.generate(prompt),
            sector=sector,
            auto_remember=True,
        )

        return {
            "query": query,
            "response": result.llm_response,
            "is_grounded": result.is_grounded,
            "validation_verdict": result.validation["verdict"],
            "plan_strategy": result.plan["strategy"],
            "precedents_count": len(result.recall["precedents"]),
            "paths_count": len(result.think["paths"]),
            "latency_ms": result.execution_time_ms,
        }


# ==============================================================================
# 3. PATTERN 2: FUNCTION / TOOL DEFINITIONS FOR OPENAI & OLLAMA
# ==============================================================================

# Standard OpenAI tool format for agents that support native function calling
OPENAI_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cortex_think",
            "description": "Fires a multi-hop spreading activation wave across knowledge graph to retrieve verified proof paths and exact math calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The complex assertion or question."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cortex_recall",
            "description": "Retrieves historical solved cases, precedents, and analogies from episodic CaseMemory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "situation": {"type": "string", "description": "The problem situation."},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["situation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cortex_decide",
            "description": "Computes deterministic priority action steps and checks for contraindications across domains.",
            "parameters": {
                "type": "object",
                "properties": {
                    "situation": {"type": "string", "description": "The clinical, legal, or technical decision situation."},
                },
                "required": ["situation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cortex_validate",
            "description": "Validates a draft response or claim against knowledge graph ground truth for hallucinations and contradictions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_or_decision": {"type": "string", "description": "The proposed claim or decision."},
                },
                "required": ["claim_or_decision"],
            },
        },
    },
]


# ==============================================================================
# 4. EXAMPLE USAGE WORKFLOW (HOW TO RUN)
# ==============================================================================

def example_ollama_workflow():
    """Example workflow showing how to initialize and use with Ollama."""
    # 1. Initialize Cortex Agent with local Ollama
    agent = MayonCortexAgent(
        llm_type="ollama",
        model_name="llama3",  # or mistral, qwen2.5, deepseek-r1
        ollama_url="http://localhost:11434",
    )

    # 2. Seed domain facts into Mayon graph
    agent.seed_knowledge([
        {
            "source": "Metformin is a first-line oral medication for Type 2 Diabetes.",
            "relation": "treats",
            "target": "Type 2 Diabetes mellitus",
            "sector": "medical",
        },
        {
            "source": "Metformin is a first-line oral medication for Type 2 Diabetes.",
            "relation": "contraindicated_for",
            "target": "Chronic Kidney Disease Stage 4 and severe renal impairment (eGFR < 30).",
            "sector": "medical",
        },
        {
            "source": "Linagliptin is a DPP-4 inhibitor excreted through bile.",
            "relation": "safe_in",
            "target": "Chronic Kidney Disease Stage 4 and severe renal impairment (eGFR < 30).",
            "sector": "medical",
        },
    ])

    # 3. Store past clinical precedent
    cortex_remember(
        brain=agent.brain,
        situation="Type 2 Diabetes with CKD Stage 4",
        judgment="Do NOT prescribe Metformin. Switch to Linagliptin 5mg daily.",
        reasoning="Metformin accumulation causes lactic acidosis in renal clearance failure.",
        sector="medical",
    )

    # 4. Ask complex clinical query
    # Cortex will automatically plan, retrieve precedent, traverse graph, and validate LLM output
    result = agent.ask("Can a patient with Type 2 Diabetes and CKD Stage 4 take Metformin?")
    print("\n--- AGENT RESULT ---")
    print(f"Grounded & Verified: {result['is_grounded']}")
    print(f"Validation Status:   {result['validation_verdict']}")
    print(f"LLM Response:\n{result['response']}")


def example_openai_workflow():
    """Example workflow showing how to initialize and use with OpenAI / Groq / DeepSeek."""
    # 1. Initialize Cortex Agent with OpenAI (or any OpenAI-compatible API)
    agent = MayonCortexAgent(
        llm_type="openai",
        model_name="gpt-4o-mini",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    )

    # 2. Ask question
    result = agent.ask("How should we treat Type 2 Diabetes in severe renal impairment?")
    print(result)


if __name__ == "__main__":
    print("Mayon-Cortex × LLM Agent Integration Template Ready.")
    print("To run with Ollama: Ensure 'ollama serve' is running and uncomment example_ollama_workflow().")
    print("To run with OpenAI: Set OPENAI_API_KEY environment variable and uncomment example_openai_workflow().")
