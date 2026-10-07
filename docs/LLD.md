# 📐 Mayon-Cortex: Low-Level Design (LLD) & Mathematical Specification

```
Document Version: 1.0.0
Classification: Dual Licensed (AGPL-3.0 / Commercial) Deep-Tech Algorithm Specification
Engine Name: Mayon-Cortex (mayon_cortex)
Developer: BALAVIGNESH M
Company: INFIDOS LLP
Copyright: (c) 2026 INFIDOS LLP & BALAVIGNESH M. All Rights Reserved.
Runtime: Pure CPU in-memory execution (NumPy / Native Graph Adjacency)
```

---

# PART I: MATHEMATICAL FOUNDATIONS & EQUATIONS

---

## 1. Node Decision Unit (NDU) — Neuron Firing Calculus

The **Node Decision Unit (NDU)** is the fundamental atomic computational unit of Mayon-Cortex, replacing Transformer attention matrices with localized $\mathcal{O}(1)$ edge decisions.

```
       Incoming Wave (Energy = E_u)
                  │
                  ▼
         ┌─────────────────┐
         │  ConceptNode u  │
         └────────┬────────┘
                  │ Semantic Synapse (Weight = w_uv, Relation = r)
                  ▼
         ┌─────────────────┐
         │  ConceptNode v  │ ◄─── Evaluates: E_v = E_u · w_uv · sim(q, v) · γ
         └────────┬────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
   E_v ≥ θ_v               E_v < θ_v
[EMIT WAVEFRONT]         [HALT PROPAGATION]
```

### 1.1 Energy Transmission Equation
When a wavefront with incoming energy $E_u \in [0, 1]$ reaches node $u$, the transmitted energy to neighbor node $v$ across directed edge $e = (u, r, v)$ with synaptic weight $w_{uv} \in [0, 1]$ is governed by:

$$E_v = E_u \cdot w_{uv} \cdot \left( \frac{\mathbf{x}_v \cdot \mathbf{q}}{\|\mathbf{x}_v\| \|\mathbf{q}\|} \right) \cdot \gamma_{\text{decay}}$$

Where:
* $\mathbf{x}_v \in \mathbb{R}^D$ is the embedding vector of concept node $v$.
* $\mathbf{q} \in \mathbb{R}^D$ is the user query vector.
* $\gamma_{\text{decay}} \in (0, 1)$ is the spatial attenuation constant (default $\gamma = 0.85$).

### 1.2 Threshold Firing Step Function
Node $v$ possesses an internal firing threshold $\theta_v \in [0.1, 0.9]$ (default $\theta_v = 0.40$). The decision state $\mathcal{D}(v)$ is evaluated as:

$$\mathcal{D}(v) = \begin{cases} 
\text{EMIT} & \text{if } E_v \ge \theta_v \text{ and } \text{Conflict}(v) = \emptyset \\
\text{SUPPRESS} & \text{if } \text{Conflict}(v) \neq \emptyset \text{ (Opposing relation detected)} \\
\text{HALT} & \text{if } E_v < \theta_v \text{ (Sub-threshold dissipation)}
\end{cases}$$

---

## 2. Spreading Activation Wavefront Dynamics

Unlike single-path depth-first graph search, `ActivationWave` expands a continuous parallel energy field across the graph tissue.

### 2.1 Discrete-Time Wavefront Recurrence Relation
Let $A_i^{(t)} \in \mathbb{R}^+$ denote the activation state of node $i$ at discrete time step $t$. The activation field propagates according to:

$$A_i^{(t+1)} = \alpha A_i^{(t)} + (1 - \alpha) \sum_{j \in \text{Pred}(i)} A_j^{(t)} \cdot w_{ji} \cdot \phi(r_{ji}) \cdot e^{-\lambda \cdot k}$$

Where:
* $\alpha \in [0, 1]$ is the temporal persistence inertia constant (default $\alpha = 0.15$).
* $\text{Pred}(i) = \{j \in V \mid (j, r, i) \in E\}$ is the set of in-bound predecessors.
* $\phi(r_{ji}) \in \mathbb{R}^+$ is the relation salience multiplier (e.g., `contraindicates` $= 10.0$, `causes` $= 7.0$, `is-a` $= 1.0$).
* $\lambda \in \mathbb{R}^+$ is the hop decay parameter (default $\lambda = 0.22$).
* $k$ is the current path depth ($k = 1, 2, \dots, K_{\max}$).

### 2.2 Temporal Salience Half-Life Decay
Nodes decay over elapsed operational cycles $\Delta \tau$ via an exponential half-life:

$$S(\tau) = S_0 \cdot 2^{-\frac{\Delta \tau}{\tau_{1/2}}}$$

Where $\tau_{1/2}$ is the configurable sector half-life (e.g. $\tau_{1/2} = 500$ queries).

---

## 3. PathRanker — Multi-Hop Mathematical Proof Calculus

The `PathRanker` extracts and scores acyclic proof trajectories $P = (n_0, e_1, n_1, e_2, n_2, \dots, e_k, n_k)$.

### 3.1 Composite Proof Scoring Function
The global score $\mathcal{S}(P)$ of path $P$ of length $k$ is computed as:

$$\mathcal{S}(P) = \underbrace{\left( \ prod_{i=1}^k w_{e_i} \right)^{\frac{1}{k}}}_{\text{Geometric Mean Synaptic Weight}} \cdot \underbrace{\left( \frac{1}{1 + \mu \cdot k} \right)}_{\text{Length Regularization}} \cdot \underbrace{\left( \frac{\max_{e \in P} \mathcal{P}(e)}{100} \right)}_{\text{Normalized Triage Priority}} \cdot \underbrace{\left( \frac{\mathbf{x}_{n_k} \cdot \mathbf{q}}{\|\mathbf{x}_{n_k}\| \|\mathbf{q}\|} \right)}_{\text{Endpoint Target Relevance}}$$

Where:
* $w_{e_i}$ is the weight of the $i$-th traversed edge.
* $\mu$ is the length penalty coefficient (default $\mu = 0.05$).
* $\mathcal{P}(e) \in [1, 100]$ is the `DecisionPriority` integer (e.g. `SAFETY` $= 100$, `LEGAL` $= 90$, `FINANCIAL` $= 60$).

---

## 4. Real-Time Hebbian Synaptic Plasticity

Mayon-Cortex implements biological **Hebbian Learning ("Neurons that fire together, wire together")** with anti-Hebbian homeostatic decay without backpropagation:

$$\Delta w_{ij} = \eta \cdot a_i \cdot a_j - \lambda_{\text{decay}} \cdot w_{ij}$$

$$w_{ij}^{(t+1)} = \text{clip}\left(w_{ij}^{(t)} + \Delta w_{ij}, w_{\min}, w_{\max}\right)$$

Where:
* $\eta = 0.08$ is the learning rate.
* $a_i, a_j \in [0, 1]$ are the concurrent node activations during verified reasoning.
* $\lambda_{\text{decay}} = 0.005$ prevents runaway synaptic saturation.
* $w_{\min} = 0.05, w_{\max} = 0.99$.

---

## 5. Hyperdimensional Computing (HDC) Binary Algebra

The `HyperdimensionalMemory` module uses $D = 10,000$-dimensional bipolar hypervectors $\mathbf{V} \in \{-1, +1\}^D$ for symbolic binding and zero-parameter associative recall.

```
Binding (A ⊗ B):      Hadamard Product / XOR Algebra (Associative, Commutative)
Bundling (A ⊕ B ⊕ C): Majority Vote Thresholding (Preserves similarity to all constituents)
Permutation (Π^k(A)): Cyclic Coordinate Bit Rotation (Encodes sequence & syntax order)
```

### 5.1 Formal HDC Operations:
1. **Binding (Role-Filler Association):**
   $$\mathbf{C} = \mathbf{A} \otimes \mathbf{B} \iff c_d = a_d \cdot b_d \quad \forall d \in \{1, \dots, D\}$$
   *Orthogonality property:* $\text{sim}(\mathbf{A} \otimes \mathbf{B}, \mathbf{A}) \approx 0$.

2. **Bundling (Set Superposition / Memory Accumulation):**
   $$\mathbf{M} = \text{sgn}\left( \sum_{i=1}^N \mathbf{V}_i \right) \quad \text{where } \text{sgn}(x) = \begin{cases} +1 & x \ge 0 \\ -1 & x < 0 \end{cases}$$

3. **Permutation (Grammatical & Sequence Encoding):**
   $$\Pi^k(\mathbf{V}) = \left[ v_{(1+k)\bmod D}, v_{(2+k)\bmod D}, \dots, v_{(D+k)\bmod D} \right]$$

4. **Normalized Cosine / Hamming Similarity Metric:**
   $$\text{sim}_{\text{HDC}}(\mathbf{A}, \mathbf{B}) = \frac{1}{D} \sum_{d=1}^D a_d \cdot b_d \in [-1, +1]$$

---

## 6. Energy Canvas — Markov Random Field (MRF) Energy Minimization

The `EnergyCanvas` synthesizes 2D visual and symbolic configurations via Gibbs energy optimization on a discrete lattice graph:

$$E(\mathbf{x}) = \sum_{i \in V} \Phi_1(x_i; \mathbf{q}) + \sum_{(i,j) \in \mathcal{E}} \Phi_2(x_i, x_j)$$

Where:
* $\Phi_1(x_i; \mathbf{q}) = \|\mathbf{v}_{x_i} - \mathbf{q}\|^2$ is the singleton unary constraint.
* $\Phi_2(x_i, x_j) = \beta \cdot (1 - \delta(x_i, x_j))$ is the pairwise smoothness / Potts potential.

### Metropolis-Hastings Annealing Transition:
At temperature $T_k = T_0 \cdot \rho^k$, a proposed pixel/node state flip $\mathbf{x} \to \mathbf{x}'$ is accepted with probability:

$$\mathcal{A}(\mathbf{x} \to \mathbf{x}') = \min\left(1, \exp\left(-\frac{E(\mathbf{x}') - E(\mathbf{x})}{T_k}\right)\right)$$

---

## 7. Morphogenetic Reaction-Diffusion (Turing Pattern Genesis)

The `MorphogeneticGenerator` simulates non-linear biological pattern formation (stripes, spots, labyrinths) via coupled reaction-diffusion PDEs:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u - u v^2 + F(1 - u)$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + u v^2 - (F + k)v$$

Where:
* $u(x,y), v(x,y)$ are morphogen chemical concentrations.
* $D_u = 0.16, D_v = 0.08$ are spatial diffusion coefficients.
* $F = 0.035$ is the feed rate; $k = 0.060$ is the kill rate.
* $\nabla^2$ is the discrete 5-point Laplacian stencil: $\nabla^2 u_{i,j} = u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}$.

---

## 8. Forward-Forward Local Contrastive Learning

The `ForwardForwardLearner` implements Geoffrey Hinton’s 2022 algorithm, replacing global backpropagation with local layer-wise goodness updates:

### 8.1 Layer Goodness Function
For hidden activation layer $\mathbf{h} \in \mathbb{R}^M$:

$$\mathcal{G}(\mathbf{h}) = \sum_{j=1}^M h_j^2$$

### 8.2 Contrastive Objective & Local Loss
Given positive data $\mathbf{h}_{\text{pos}}$ and negative/corrupted data $\mathbf{h}_{\text{neg}}$ with goodness threshold $\theta_{\text{good}}$:

$$p(\text{positive}) = \sigma\left(\mathcal{G}(\mathbf{h}_{\text{pos}}) - \theta_{\text{good}}\right) = \frac{1}{1 + e^{-(\mathcal{G}(\mathbf{h}_{\text{pos}}) - \theta_{\text{good}})}}$$

$$\mathcal{L}_{\text{FF}} = \log\left(1 + e^{\theta_{\text{good}} - \mathcal{G}(\mathbf{h}_{\text{pos}})}\right) + \log\left(1 + e^{\mathcal{G}(\mathbf{h}_{\text{neg}}) - \theta_{\text{good}}}\right)$$

---

## 9. Hierarchical Predictive Coding (Friston Free Energy Principle)

The `PredictiveCoder` minimizes surprise and reconstruction error across hierarchical levels:

$$\boldsymbol{\epsilon}_l = \mathbf{r}_l - \mathbf{g}(\mathbf{r}_{l+1})$$

$$\Delta \mathbf{r}_l = \eta_{\text{pc}} \left( \boldsymbol{\Pi}_l \boldsymbol{\epsilon}_l - \mathbf{W}_l^T \boldsymbol{\Pi}_{l-1} \boldsymbol{\epsilon}_{l-1} \right)$$

Where $\boldsymbol{\epsilon}_l$ is the prediction error, $\mathbf{r}_l$ is the representational state, and $\boldsymbol{\Pi}_l$ is the precision weighting matrix.

---

## 10. Hierarchical Prose Engine — Boltzmann Path Sampler

The `HierarchicalProseEngine` articulates multi-act narratives by sampling word transitions over the `WordGraph` using a Boltzmann energy distribution:

$$P(w_{t} \mid w_{t-1}, \mathbf{q}_{\text{theme}}) = \frac{\exp\left(-\frac{\mathcal{E}(w_t, w_{t-1}, \mathbf{q}_{\text{theme}})}{T_{\text{novelty}}}\right)}{\sum_{w' \in \text{Adj}(w_{t-1})} \exp\left(-\frac{\mathcal{E}(w', w_{t-1}, \mathbf{q}_{\text{theme}})}{T_{\text{novelty}}}\right)}$$

Where the composite energy functional $\mathcal{E}$ is defined as:

$$\mathcal{E}(w_t, w_{t-1}, \mathbf{q}) = -\left[ w_{\text{transition}}(w_{t-1}, w_t) + \lambda_{\text{coh}} \cdot \text{sim}(\mathbf{x}_{w_t}, \mathbf{q}) - \lambda_{\text{recency}} \cdot \mathbb{I}(w_t \in \text{RecencyBuffer}) \right]$$

## 11. Quadratic Bayesian Edge Confidence Fusion (Conflict Resolution)

When duplicate or conflicting edges $(u, r, v)$ are ingested across multiple data sources or batches with confidences $c_1, c_2 \in [0, 1]$, the unified confidence $c_{\text{fused}}$ is computed via quadratic Bayesian reinforcement:

$$c_{\text{fused}} = \min\left(0.999, \frac{c_1^2 + c_2^2}{c_1 + c_2} + 0.05 \cdot (1 - \max(c_1, c_2))\right)$$

Where:
* Metadata payloads are shallow-merged: $\mathcal{M}_{\text{fused}} = \mathcal{M}_1 \cup \mathcal{M}_2$.
* Reinforcement access counter increments: $\text{count}_{\text{fused}} = \text{count}_1 + 1$.
* In cases of contradictory relations (e.g. `treats` vs `contraindicates`), the system preserves both edges while flagging epistemic ambiguity for the `UncertaintyGate` and `SelfCorrector`.

---

## 12. Dynamic Zero-Shot Prototype Sector Classification

When incoming items lack an explicit `sector` tag, the `DynamicSectorClassifier` computes cosine similarity between the semantic embedding of the input $\mathbf{e}_{\text{input}}$ and learned/registered domain prototype vectors $\mathbf{p}_s$ for each domain $s \in \mathcal{S}$:

$$\hat{s} = \begin{cases}
\arg\max_{s \in \mathcal{S}} \left( \frac{\mathbf{e}_{\text{input}} \cdot \mathbf{p}_s}{\|\mathbf{e}_{\text{input}}\| \|\mathbf{p}_s\|} \right) & \text{if } \max_{s \in \mathcal{S}} \text{sim}(\mathbf{e}_{\text{input}}, \mathbf{p}_s) \ge \theta_{\text{sector}} \\
\text{"general"} & \text{otherwise}
\end{cases}$$

Where default threshold $\theta_{\text{sector}} = 0.25$.

---

# PART II: MODULE-BY-MODULE CLASS & CODE SCHEMAS

---

## 1. `mayon_cortex.core`

```python
class MayonGraph:
    nodes: Dict[str, ConceptNode]
    _out_edges: Dict[str, Dict[str, Dict[str, Any]]] # adjacency {u: {v: {rel: Edge}}}
    _in_edges: Dict[str, Dict[str, Dict[str, Any]]]
    sectors: Dict[str, Set[str]]
    
    def add_node(self, node_id: str, label: str, sector: str, vector: np.ndarray, metadata: Dict) -> ConceptNode: ...
    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float, bidirectional: bool) -> RelationEdge: ...
    def get_successors(self, node_id: str) -> List[str]: ...
    def get_out_edges_data(self, node_id: str, data: bool = True) -> List[Tuple[str, str, Dict]]: ...

class NodeDecisionUnit:
    def evaluate(self, current_node: ConceptNode, incoming_energy: float, target_edge: RelationEdge, target_node: ConceptNode, query_vector: np.ndarray) -> Tuple[NDUDecision, float]: ...
```

---

## 2. `mayon_cortex.dynamics`

```python
class ActivationWave:
    config: CortexConfig
    def activate(self, query_vec: np.ndarray, graph: MayonGraph, ndu: NodeDecisionUnit, start_nodes: Optional[List[str]] = None, max_hops: int = 4) -> WaveResult: ...

class SalienceWeighter:
    def compute_salience(self, path: List[ConceptNode], edges: List[RelationEdge]) -> float: ...
```

---

## 3. `mayon_cortex.reasoning`

```python
class PathRanker:
    def rank_paths(self, activated_paths: List[ActivatedPath], query_vec: np.ndarray, graph: MayonGraph) -> List[ScoredPath]: ...

class LogicEngine:
    def deduce(self, premises: List[str], rules: List[Rule]) -> FormalProof: ...
    def check_contradiction(self, path: ScoredPath) -> List[Conflict]: ...

class ThinkingEngine:
    def think(self, prompt: str, graph: MayonGraph, mode: CognitiveMode, timeout_ms: int = 5000) -> ThoughtResult: ...
```

---

## 4. `mayon_cortex.executive`

```python
class ExecutiveController:
    def plan(self, question: str, graph: MayonGraph, embedder: EmbeddingModel) -> ExecutivePlan: ...

class UncertaintyGate:
    def assess_confidence(self, top_paths: List[ScoredPath]) -> Tuple[bool, float, str]: ...
```

---

## 5. `mayon_cortex.learning`

```python
class OnlineLearner:
    def learn_from_path(self, path: ScoredPath, graph: MayonGraph, reward: float = 1.0) -> None: ...

class ForwardForwardLearner:
    def train_layer(self, layer_idx: int, positive_data: np.ndarray, negative_data: np.ndarray) -> float: ...
```

---

## 6. `mayon_cortex.language`

```python
@dataclass
class StandardKnowledgeItem:
    source: str
    relation: str
    target: str
    sector: Optional[str] = None
    confidence: float = 1.0
    provenance: str = "global_ingestion"
    level: str = "PROPOSITION"
    node_type: str = "concept"
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StandardDocumentItem:
    text: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    sector: Optional[str] = None
    provenance: str = "document_ingestion"
    entities: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgePackage:
    domain_name: str
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    items: List[StandardKnowledgeItem] = field(default_factory=list)
    documents: List[StandardDocumentItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicSectorClassifier:
    def classify(self, text: str, fallback: str = "general") -> str: ...
    def register_sector(self, sector_name: str, prototype_phrases: List[str]) -> None: ...

class GlobalDataLoader:
    def load(self, source: Any, default_sector: Optional[str] = None, provenance_override: Optional[str] = None, strict: bool = False) -> IngestionReport: ...
    def load_sources(self, sources: List[Any], default_sector: Optional[str] = None) -> IngestionReport: ...

class BrainKnowledgeExporter:
    def export(self, target_path: Union[str, Path], format: str = "package", sector: Optional[str] = None, min_confidence: float = 0.0, **metadata) -> ExportReport: ...
    def to_package(self, domain_name: str = "cortex_brain", sector: Optional[str] = None) -> KnowledgePackage: ...

class HierarchicalProseEngine:
    def compose_paragraph(self, theme_vector: np.ndarray, config: ProseConfig, act: Optional[NarrativeAct] = None, recency_memory: Optional[List[str]] = None, num_sentences: int = 3) -> Tuple[str, List[str]]: ...
    def compose_story(self, prompt: str, acts_count: int = 3, config: Optional[ProseConfig] = None) -> Dict[str, Any]: ...
```

---

## 7. `mayon_cortex.engine` (Master Brain Class)

```python
class MayonCortex:
    graph: MayonGraph
    embedder: EmbeddingModel
    ndu: NodeDecisionUnit
    wave: ActivationWave
    ranker: PathRanker
    prose_engine: HierarchicalProseEngine
    executive: ExecutiveController
    uncertainty: UncertaintyGate
    learner: OnlineLearner
    global_loader: GlobalDataLoader
    exporter: BrainKnowledgeExporter
    
    def query(self, question: str, sector: str = "general") -> CortexResponse: ...
    def decide(self, question: str) -> DecisionResponse: ...
    def think(self, prompt: str, mode: str = "balanced", timeout_ms: int = 5000) -> ThoughtResult: ...
    def ingest(self, text: str, sector: str = "general", provenance: str = "ingestion") -> IngestionStats: ...
    def ingest_file(self, file_path: str, sector: Optional[str] = None) -> IngestionReport: ...
    def load_data(self, source: Any, default_sector: Optional[str] = None, provenance: Optional[str] = None, strict: bool = False) -> IngestionReport: ...
    def load_sources(self, sources: List[Any], default_sector: Optional[str] = None) -> IngestionReport: ...
    def export_knowledge(self, target_path: Union[str, Path], format: str = "package", sector: Optional[str] = None, min_confidence: float = 0.0, **meta) -> ExportReport: ...
    def save(self, save_dir: str) -> None: ...
    @classmethod
    def load(cls, save_dir: str, config: Optional[CortexConfig] = None) -> "MayonCortex": ...
```
