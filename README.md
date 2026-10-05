# Mayon-Cortex

> **A biologically-inspired, neuro-symbolic cognitive architecture.**  
> Zero GPU. Zero backpropagation. 100% deterministic reasoning.  
> Pure CPU. Python 3.9+. NumPy only.

---

## What Is Mayon-Cortex?

Mayon-Cortex is a **cognitive reasoning engine** — not a language model. It is a graph-native substrate that implements the core computational machinery of biological reasoning: activation spreading, proof chain ranking, logical inference, analogical induction, case-based memory, and metacognitive self-monitoring.

It is designed to sit **underneath** large language models as a reasoning and memory layer, or to operate **standalone** for formal tasks such as mathematical constraint solving, abstract visual pattern induction (ARC-AGI), and multi-hop knowledge graph reasoning.

The core thesis is:

> *Language models are excellent at fluency. They are poor at formal reasoning, persistent memory, and self-correction. A deterministic graph substrate can provide exactly what is missing — without GPU, without gradient descent, and with 100% traceable proof chains.*

---

## Architecture Overview

Mayon-Cortex is organized into **10 biologically-motivated subsystems**, each corresponding to a functional region of the neocortex:

```
mayon_cortex/
├── core/          Brain tissue — MayonGraph, ConceptNode, RelationEdge,
│                  NodeDecisionUnit (NDU), VectorIndex (FAISS/USearch/NumPy)
├── dynamics/      Activation physics — ActivationWave, SalienceWeighter,
│                  FeedforwardCascade, EnergyCanvas (MRF), MorphogeneticGenerator
├── memory/        Working memory, CaseMemory (precedent store),
│                  HyperdimensionalMemory (10k-dim HDC), ThoughtScratchpad
├── reasoning/     PathRanker, LogicEngine, ThinkingEngine, ProblemSolver,
│                  AnalogyEngine, CrossDomainResolver, FractalAbstractionEngine,
│                  KnowledgeCorrelator, CorticalARCReasoner, ARCProgramSynthesizer,
│                  SMT/CSP Solver, Graph-MCTS, SymbolicAlgebra, CalculusEngine
├── executive/     ExecutiveController, MetaCognitionMonitor, UncertaintyGate,
│                  SelfCorrector, DecisionPlanner, CognitiveModeEngine,
│                  CuriosityGapResolver
├── learning/      OnlineLearner (Hebbian plasticity), ForwardForwardLearner,
│                  PredictiveCoder, Teacher, KnowledgeBootstrapper
├── affect/        EmotionEngine (Valence-Arousal-Dominance model)
├── language/      UniversalDataLoader, LanguageIngester, HierarchicalProseEngine,
│                  WordGraph, BrocaDecoder, NGramModel, GrammarInducer,
│                  DiscourseCoherenceTracker, BrocaSelfTrainer, HyperParser
├── perception/    VisionCortex (image→graph), ARCSpatialSceneGraph,
│                  KnowledgeBoundary
└── imagination/   ImagineEngine, MentalSandbox, ConceptBlender,
                   DreamConsolidator, CuriosityEngine
```

---

## Core Design Principles

### 1. Graph as Brain Tissue
All knowledge is stored as a typed relational graph (`MayonGraph`). Nodes are concept embeddings (R^384). Edges carry semantic relation types (`is-a`, `causes`, `treats`, `contraindicates`, `enables`, `part-of`, etc.) with confidence weights and salience scores.

### 2. Activation Wave — Replacing Attention
Instead of quadratic self-attention, `ActivationWave` fires a **parallel wavefront** through the graph. Each node runs a `NodeDecisionUnit` (100-tree gradient-boosted ensemble, 52 features) to decide: `FOLLOW | HALT | BACKTRACK`. This is O(1) local decision at each node, parallelizable across the wavefront. Max depth: 5 hops.

### 3. PathRanker — Replacing Cosine-Only Retrieval
Discovered paths are ranked by a **15-feature gradient-boosted ranker**:
- Path confidence (cumulative edge product)
- Symbolic validity (relation chain deductive soundness)
- Temporal coherence
- Sector consistency
- Salience weighting
- NDU relevance across all hops
- Cycle-free constraint
- Phrase node coverage

### 4. Hebbian Online Learning — No Backprop
`OnlineLearner` runs after every query:
1. **Hebbian reinforce** — strengthen edge weights on used paths (lr = 0.05)
2. **Word-level reinforce** — strengthen bigram transitions in WordGraph
3. **NDU outcome recording** — accumulate training examples, retrain every 500 queries
4. **Graph growth** — add new nodes when novel knowledge is confirmed (threshold = 0.7)

The graph becomes more accurate with every interaction. No retraining. No GPU.

### 5. Hyperdimensional Computing (10k-dim)
`HyperdimensionalMemory` encodes concepts as 10,000-bit binary vectors with three operations:
- **BIND**: A XOR B — combine two concepts into a holographic representation
- **BUNDLE**: majority(A, B, C, ...) — create a set of concepts
- **PERMUTE**: shift(A, k) — encode position and sequence

All operations are CPU-native bit manipulations. No matrix multiply.

### 6. Formal Logic Engine
`LogicEngine` implements 5 domain-agnostic inference rules directly on graph edges:
1. **Transitivity** — A→B, B→C ⟹ A→C (for `is-a`, `part-of`, `causes`, `implies`)
2. **Contrapositive** — A causes B ⟹ not-B implies not-A
3. **Inheritance** — A is-a B, B has-property P ⟹ A has-property P
4. **Exclusion** — A contraindicates B, B treats C ⟹ A may prevent treatment of C
5. **Symmetry** — A similar-to B ⟹ B similar-to A

All proofs are deterministic and fully traceable.

---

## ARC-AGI Integration

`CorticalARCReasoner` connects the Mayon-Cortex cognitive architecture to Francois Chollet's ARC-AGI benchmark — a test of abstract visual pattern induction that current LLMs solve at near-zero accuracy without massive test-time compute.

**Pipeline:**
1. **Perception** — `ARCSpatialSceneGraph` converts 2D discrete color grids into attributed spatial relation graphs (objects, cavities, contacts, alignments, symmetry axes)
2. **Analogy** — `AnalogyEngine` induces the transformation delta (TrainIn → TrainOut) and maps it to (TestIn → ?)
3. **DSL Synthesis** — `ARCProgramSynthesizer` runs heuristic-guided beam search over 60+ DSL primitives including geometric transforms, morphological operations, color mappings, object-level filters, and physics (gravity)
4. **Verification** — hypothesize-and-verify loop checks candidates against ALL demonstration pairs before accepting

**DSL primitive categories:**
- Geometric: rot90/180/270, flip_h/v, transpose, diagonal_flip_anti
- Morphological: dilate, erode, extract_outline, fill_interior, hollow_objects, flood_fill_corners
- Object-level: keep/remove_smallest/largest, sort_by_size, recolor_by_size, area_filter
- Symmetry: complete_rotational/horizontal/vertical_symmetry, mirror_left/right/top/bottom
- Physics: cellular_gravity (4 directions)
- Pattern: tile_2x2/3x3, scale_2x/3x, pattern_extension, downscale_majority

**Run the benchmark:**
```bash
python benchmarks/benchmark_arc_400_eval.py --num-tasks 400
```

---

## Formal Solvers

### MathSolver + CSPSolver (SMT)
- Solves linear equations, multi-variable systems, kinematics, physics, finance formulas
- Constraint Satisfaction via AC-3 + Forward Checking
- Resolution-refutation theorem proving
- 100% exact — no probability, no approximation

### Graph-MCTS
- Monte Carlo Tree Search over MayonGraph tissue
- UCB1 exploration/exploitation over knowledge paths
- Contradiction and dead-end pruning with backtracking
- < 5ms CPU execution per proof search

### Symbolic Algebra + Calculus
- `UniversalEquationSolver` — polynomial and transcendental equations
- `CalculusEngine` — symbolic differentiation and integration
- `NumberTheoryEngine` — primality, GCD, modular arithmetic
- `MatrixEngine` — determinant, inverse, eigenvalues

---

## Metacognition and Epistemic Awareness

`MetaCognitionMonitor` tracks the health of the reasoning process in real time:

| State | Meaning | Action Taken |
|---|---|---|
| PROGRESSING | New conclusions at each step | CONTINUE |
| SLOWING | Progress decelerating | BROADEN or SWITCH_STRATEGY |
| STALLED | No new conclusions | DECOMPOSE or BACKTRACK |
| CIRCULAR | Revisiting same states | BACKTRACK |
| CONFIDENT | High-quality conclusion reached | CONCLUDE |
| EXHAUSTED | Time budget spent | CONCLUDE |

`UncertaintyGate` returns `is_uncertain=True` when:
- Fewer than 1 activated path (no graph evidence)
- Path confidence below 0.50
- Query coverage below 0.55

The system says **"I don't know"** instead of hallucinating.

---

## Integration with LLM Agents and MCP

Mayon-Cortex is designed as a **reasoning backend** for LLM agents. It exposes 5 logical integration points:

| Layer | Cortex Module | What the LLM Receives |
|---|---|---|
| **Plan** | ExecutiveController | Complexity class + thinking strategy + decomposed sub-questions |
| **Recall** | CaseMemory | Top-K most similar past interactions with precedent judgments |
| **Think** | ActivationWave + PathRanker + SelfCorrector | Verified proof chains + confidence + emotion VAD |
| **Validate** | UncertaintyGate + JudgmentEngine | Approved / contradiction detected + suggested correction |
| **Remember** | OnlineLearner + CaseMemory | Graph updated, interaction stored, edges strengthened |

This architecture allows a **small LLM (7B parameters)** to outperform a larger baseline (70B) on tasks where formal reasoning, contradiction detection, and persistent memory are required — because the LLM is no longer responsible for reasoning. It verbalizes results. Cortex reasons.

---

## Quick Start

```python
from mayon_cortex import CortexGraph, CortexConfig

# Initialize
config = CortexConfig()
brain = CortexGraph(config)

# Ingest domain knowledge
brain.ingest("Metformin treats type 2 diabetes.", sector="medical")
brain.ingest("Metformin contraindicates renal failure.", sector="medical")
brain.ingest("Hypertension is a risk factor for renal failure.", sector="medical")

# Query — returns verified proof chains
response = brain.query("Can we use Metformin for a hypertensive patient?")
print(response.text)           # graph-assembled answer
print(response.confidence)     # numerical certainty
print(response.is_uncertain)   # epistemic flag
for path in response.paths:
    print(path.node_ids)       # traceable proof chain

# Decision planning
decision = brain.decide("Should we prescribe Metformin to a patient with renal failure?")
for step in decision.decision_chain.steps:
    print(f"Step {step.rank} [{step.sector}]: {step.action}")
```

```python
# ARC-AGI solving
from mayon_cortex.reasoning.cortical_arc_reasoner import CorticalARCReasoner
from mayon_cortex.reasoning.arc_solver import ARCTask

reasoner = CorticalARCReasoner()
task = ARCTask.from_json_file("data/arc_tasks/00576224.json")
solution = reasoner.solve(task)
print(f"Solved: {solution.is_solved}")
print(f"Rule:   {solution.explanation}")
print(f"Time:   {solution.latency_ms:.1f}ms")
```

```python
# Unified pipeline (simplest entry point)
from mayon_cortex import CortexPipeline

brain = CortexPipeline.from_files("my_domain_docs/*.txt")
print(brain.ask("What are the conditions for contract termination?"))
print(brain.solve("If 3x + 5 = 20, what is x?"))
print(brain.judge("Employee shared confidential data externally."))
```

---

## Key Technical Parameters

| Component | Parameter | Value |
|---|---|---|
| Concept embedding | Dimension | 384 (R^384) |
| HDC memory | Vector dimension | 10,000 bits |
| NodeDecisionUnit | Trees / depth / features | 100 / 6 / 52 |
| PathRanker | Trees / depth / features | 200 / 8 / 15 |
| ActivationWave | Max hops | 5 |
| ActivationWave | Wavefront cap | 200 nodes |
| Working memory | Buffer size | 10 items |
| CaseMemory | Capacity | 2,000 precedents |
| Hebbian learning | Rate | 0.05 |
| Deduplication | Vector threshold | 0.92 cosine |
| Graph-MCTS | Max latency | < 5ms (CPU) |
| ARC DSL | Primitive count | 60+ |

---

## LLM & Agent Integration Suite (The 5 Priority Tools)

Mayon-Cortex serves as the high-speed neuro-symbolic co-processor for traditional Large Language Models (GPT-4, Claude, Gemini, Llama 3, DeepSeek).

```
┌────────────────────────────────────────────────────────────────┐
│                   TRADITIONAL LLM / AGENT                      │
│             (Verbalization, Grammar, Fluency)                 │
└───────────────────────────▲────────────────────────────────────┘
                            │ Structured Context / MCP Protocol
┌───────────────────────────▼────────────────────────────────────┐
│                  MAYON-CORTEX INTEGRATION                      │
│                                                                │
│  [Priority 5] cortex_plan     → Complexity & Strategy Routing  │
│  [Priority 2] cortex_recall   → Episodic Precedents (Lifelong) │
│  [Priority 1] cortex_think    → Multi-Hop Symbolic Proofs      │
│  [Priority 3] cortex_decide   → Priority Chains & Conflicts    │
│  [Priority 4] cortex_validate → Hallucination & Safety Gate    │
└────────────────────────────────────────────────────────────────┘
```

### The 5 Priority Tools:
1. **`cortex_think`**: Traverses multi-hop relational paths with spreading activation waves, ranks proof chains deterministically, and offloads arithmetic/SMT equations without hallucinations.
2. **`cortex_recall` & `cortex_remember`**: Retrieves precedent cases from episodic `CaseMemory` and working context, adapting past solved situations to new queries without retraining.
3. **`cortex_decide`**: Formulates structured, priority-ranked decision steps and resolves cross-domain contradictions (e.g., drug therapy vs clinical contraindication).
4. **`cortex_validate`**: Post-generation hallucination filter that verifies LLM assertions against graph relations, catching contraindications and ungrounded statements.
5. **`cortex_plan`**: Prefrontal executive controller that classifies query complexity (`TRIVIAL`, `MODERATE`, `COMPLEX`, `MULTI_PART`) and directs reasoning pipelines.

### Quick Start & Demo:
```bash
# Run the complete 5-layer integration test suite
pytest tests/test_llm_integration.py -v

# Run the interactive live demo
python demos/demo_cortex_llm_integration.py
```

### Model Context Protocol (MCP) Server:
Start the standard MCP JSON-RPC server on stdio for Claude Desktop, Cursor, and Antigravity:
```bash
python -m mayon_cortex.integration.cortex_mcp_server
```

---

## Dependencies

```
numpy >= 1.20.0        # only hard dependency
scipy                  # optional: ARC spatial operations (ndimage)
faiss-cpu              # optional: FAISS vector index (falls back to NumPy)
usearch                # optional: USearch vector index
```

No PyTorch. No TensorFlow. No CUDA. No transformer weights.

---

## Relation to Existing Research

| Concept | Reference | Mayon-Cortex Implementation |
|---|---|---|
| Spreading activation | Collins & Loftus (1975) | ActivationWave — parallel wavefront with NDU gating |
| Hyperdimensional computing | Kanerva (1988), VS-Graph (2024) | HyperdimensionalMemory — 10k-dim XOR/majority/shift |
| Case-based reasoning | Kolodner (1993) | CaseMemory — graph subpattern precedent store |
| System 1 / System 2 | Kahneman (2011) | ExecutiveController — routes to direct lookup or full deliberation |
| Forward-Forward learning | Hinton (2022) | ForwardForwardLearner — goodness-based layer training |
| Predictive coding | Rao & Ballard (1999) | PredictiveCoder — top-down prediction error minimization |
| ARC-AGI benchmark | Chollet (2019) | CorticalARCReasoner + ARCProgramSynthesizer |
| Monte Carlo Tree Search | Coulom (2006) | GraphMCTSEngine — UCB1 over knowledge graph paths |
| Morphogenetic computation | Turing (1952) | MorphogeneticGenerator — reaction-diffusion on graph |

---

## What This Is Not

- **Not a language model.** Mayon-Cortex does not contain learned weights over natural language.
- **Not a RAG system.** Retrieval is multi-hop graph traversal with typed relations and formal ranking — not flat cosine similarity over text chunks.
- **Not a fine-tuning framework.** The system learns online via Hebbian plasticity on the graph, not gradient descent over neural parameters.
- **Not production-complete standalone.** Text generation quality is bounded by the graph WordGraph bigram model. For fluent output, integrate with any LLM as the verbalization layer.

---

## Status

`v1.0.0` — Research prototype. All 10 subsystems implemented and tested.  
Benchmark harness for ARC-AGI 400 evaluation tasks is included.

---

## License

Proprietary — INFIDOS LLP / BALAVIGNESH M (c) 2026.  
All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

---

## Author

**BALAVIGNESH M**  
INFIDOS LLP  
balavignesh@infidos.com
