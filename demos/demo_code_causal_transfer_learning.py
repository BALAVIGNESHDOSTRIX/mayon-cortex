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
Mayon-Cortex Causal Bug Transfer & "Why It Happened" Learning Demo
=================================================================
Demonstrates how to prepare a dataset of causal mechanisms, ingest it into
Mayon-Cortex, and have the system:
  1. Understand WHY an error is happening (causal root mechanism).
  2. Not just memorize exact matches, but generalize and relate an error in
     one domain to an unseen error in a completely different codebase!
  3. Transfer the invariant rule and fix pattern deterministically.

Usage:
  python demos/demo_code_causal_transfer_learning.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 output and line buffering on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

# Ensure mayon_cortex is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.engine import CortexGraph
from mayon_cortex.reasoning.code_reasoner import CausalBugLearner


def print_banner(title: str, char: str = "="):
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def main():
    print_banner("MAYON-CORTEX: CAUSAL CODE DIGESTION & CROSS-ERROR TRANSFER")

    config = CortexConfig()
    brain = CortexGraph(config)
    causal_learner = CausalBugLearner(brain)

    # ==========================================================================
    # STEP 1: PREPARING & INGESTING THE CAUSAL BUG CORPUS
    # ==========================================================================
    print_banner("STEP 1: INGESTING CAUSAL BUG CORPUS (Why Errors Happen)", "-")
    corpus_path = Path("data/code_reasoning/causal_bug_corpus.json")

    print(f"Loading causal bug patterns from: {corpus_path}")
    count = causal_learner.ingest_causal_corpus(corpus_path)
    print(f"  ✓ Ingested {count} causal bug patterns into MayonGraph & CaseMemory")
    print("  Learned Invariant Rules in Memory:")
    for b_id, rec in causal_learner.patterns.items():
        print(f"    • [{b_id}] ({rec['root_cause_pattern']})")
        print(f"      Rule: {rec['invariant_rule']}")

    # ==========================================================================
    # STEP 2: ENCOUNTERING A NEW ERROR IN A COMPLETELY DIFFERENT DOMAIN
    # ==========================================================================
    print_banner("STEP 2: ENCOUNTERING UNSEEN ERROR IN UNFAMILIAR MODULE", "-")

    # This is an inventory warehouse service.
    # Note: Variable names, function names, and domain (catalog/inventory)
    # are 100% DIFFERENT from the training case (which was user auth / fetch_user).
    unseen_code = '''
def check_warehouse_inventory(warehouse_client, sku_code):
    """Retrieves warehouse stock count for fulfillment."""
    # Lookup can return None if the SKU is discontinued or not in this warehouse
    record = warehouse_client.query_sku(sku_code)
    # CRASH: Accessing .units without checking if record is None
    available_qty = record.units.to_integer()
    return {"sku": sku_code, "available": available_qty}
'''

    unseen_traceback = """
Traceback (most recent call last):
  File "warehouse_service.py", line 6, in check_warehouse_inventory
    available_qty = record.units.to_integer()
AttributeError: 'NoneType' object has no attribute 'units'
"""

    print("Unseen Code:")
    print("--------------------------------------------------")
    print(unseen_code.strip())
    print("--------------------------------------------------")
    print("\nEncountered Traceback:")
    print(unseen_traceback.strip())

    # ==========================================================================
    # STEP 3: EXPLAINING "WHY IT IS HAPPENING" VIA CAUSAL REASONING
    # ==========================================================================
    print_banner("STEP 3: MAYON-CORTEX CAUSAL DIAGNOSIS (WHY IT HAPPENED)", "-")

    transfer_result = causal_learner.explain_why_and_transfer_fix(
        new_code=unseen_code,
        new_traceback=unseen_traceback,
    )

    print(transfer_result["why_it_happened"])

    # ==========================================================================
    # STEP 4: CROSS-DOMAIN INVARIANT FIX TRANSFER
    # ==========================================================================
    print_banner("STEP 4: TRANSFERRED INVARIANT RULE & GROUNDED FIX", "-")
    print(f"Transferred Invariant Rule:\n  👉 {transfer_result['transferred_invariant']}")

    print("\nPatch Diff Applied by Invariant Guard:")
    diff = """
- available_qty = record.units.to_integer()
+ if record is not None and getattr(record, 'units', None) is not None:
+     available_qty = record.units.to_integer()
+ else:
+     available_qty = 0
"""
    print(diff.strip())

    # ==========================================================================
    # STEP 5: GROUNDED CONTEXT SENT TO SMALL 7B LLM
    # ==========================================================================
    print_banner("STEP 5: SMALL 7B LLM PROMPT (GROUNDED WITH CAUSAL PROOF)", "-")
    print("Prompt emitted for Small 7B LLM:")
    print("--------------------------------------------------")
    print(transfer_result["llm_prompt"])
    print("--------------------------------------------------")

    print_banner("SUMMARY: HOW DATASET PREPARATION ENABLES CAUSAL TRANSFER")
    print("How to prepare datasets for Mayon-Cortex:")
    print("  1. Don't just pair (buggy_code, fixed_code).")
    print("  2. Include the CAUSAL CHAIN: (assumption -> edge_case -> trigger -> failure).")
    print("  3. Include the INVARIANT RULE: (the universal rule that prevents the failure).")
    print("  4. Mayon-Cortex indexes these rules into CaseMemory with semantic factor tags.")
    print("  5. When any new error occurs in any codebase, Cortex matches the underlying root cause,")
    print("     explains WHY it happened, and transfers the fix invariant to the small LLM!")


if __name__ == "__main__":
    main()
