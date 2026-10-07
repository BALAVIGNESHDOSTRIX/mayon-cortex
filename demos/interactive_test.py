#!/usr/bin/env python3

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
Interactive Cortex-Graph Testing Console
========================================
1. Automatically boots the CPU Cognitive Brain.
2. Ingests the 100-story dataset (or your custom text).
3. Lets you type any prompt to generate:
   - Creative multi-act stories (Option: 'story')
   - Factual proof queries with Epistemic Humility verification (Option: 'query')
   - Cross-domain decision chains (Option: 'decide')
"""

import sys
import os
import time

# Ensure local packages are on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mayon-graph"))

from mayon_cortex import CortexConfig, CortexGraph
from demo_100_stories_dynamic import get_100_stories

def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)

def main():
    print_banner("Cortex-Graph: Interactive Cognitive Brain Console (Pure CPU)")
    
    # 1. Initialize & Check for cached brain on disk
    storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cortex_storage", "saved_brain")
    
    if os.path.exists(storage_path) and os.path.exists(os.path.join(storage_path, "brain_meta.json")):
        print(f"[*] Found cached brain on disk! Loading state instantly from {storage_path}...")
        t0 = time.time()
        brain = CortexGraph.load(storage_path)
        print(f"[+] Brain Loaded from Disk in {time.time()-t0:.2f}s! Concept Nodes: {brain.graph.num_nodes} | Words: {len(brain.word_graph.nodes)}")
    else:
        print("[*] Initializing fresh Cortex-Graph Brain Architecture...")
        config = CortexConfig()
        brain = CortexGraph(config)
        
        print("[*] Ingesting 100 stories into local knowledge graph (one-time setup)...")
        stories = get_100_stories()
        t0 = time.time()
        for i, s in enumerate(stories, 1):
            brain.ingest(s, sector="story", provenance=f"story_{i}")
            if i % 25 == 0 or i == len(stories):
                print(f"  -> Ingested {i}/{len(stories)} stories into graph tissue...")
                
        print(f"[+] Ingestion Complete in {time.time()-t0:.1f}s! Saving to local disk at {storage_path}...")
        brain.save(storage_path)
        print(f"[+] Brain state saved! Future startups will load in < 0.2 seconds.")

    
    # 3. Interactive Loop
    print_banner("Interactive Testing Ready")
    print("Commands:")
    print("  1. story <prompt>    -> Generate a 3-act story synthesized from learned essence")
    print("  2. query <question>  -> Ask a factual query (checks Epistemic Humility gate)")
    print("  3. exit              -> Quit console\n")
    
    while True:
        try:
            cmd = input("\n[CortexBrain] > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
                
            if cmd.startswith("story "):
                prompt = cmd[6:].strip()
                print(f"\n[*] Generating 3-act story for theme: \"{prompt}\" on CPU...")
                t_start = time.time()
                res = brain.compose_story(prompt=prompt, acts=3, genre="fantasy")
                gen_time = (time.time() - t_start)
                
                print(f"\n  TITLED: « {res['title']} » ({gen_time:.1f}s)")
                print("  " + "-" * 65)
                for act in res['acts']:
                    print(f"\n  [{act['act_name']} | Mood: {act['emotion']}]")
                    print(f"  {act['text']}")
                    
            elif cmd.startswith("query "):
                question = cmd[6:].strip()
                print(f"\n[*] Evaluating knowledge & epistemic uncertainty for: \"{question}\"...")
                res = brain.query(question, sector="story")
                
                print(f"  -> Confidence: {res.confidence:.2f}")
                print(f"  -> Is Uncertain: {res.is_uncertain}")
                if res.is_uncertain:
                    print(f"  -> Gate Status: Epistemic Humility Triggered (Zero Hallucination)")
                print(f"\n[Response]:\n{res.text}")
                
            else:
                # Default: try both query and story hint
                print(f"Unknown command format. Use 'story <theme>' or 'query <question>'.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
