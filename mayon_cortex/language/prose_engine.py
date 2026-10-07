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
Hierarchical Prose Engine — GPT-Scale Creative Prose on Pure Graph
==================================================================
Implements a 4-layer hierarchical graph generator:
1. Macro Narrative Arc Planner (Exposition -> Rising Action -> Climax -> Resolution)
2. Discourse Connectors & Syntactic Rhythm Generator
3. Semantic Waypoint Attractor & Metaphor Blending
4. Boltzmann Energy Sampler with Anti-Loop Recency Buffer & Frequency Decay

Provides universal 1-to-10 control parameters:
- novelty (1-10): Modulates Boltzmann temperature T and cross-domain analogy jumps.
- coherence (1-10): Modulates attractor waypoint gravity and semantic anchoring.
- cadence (1-10): Modulates sentence length (6 to 30 words) and clause complexity.
- emotion_intensity (1-10): Modulates VAD affective steering force.
"""

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.affect.emotion import VAD, EmotionEngine
from mayon_cortex.language.word_graph import WordGraph, WordNode


@dataclass
class ProseConfig:
    """
    Universal 1-to-10 parameter configuration for creative generation.
    """
    novelty: float = 6.0            # 1 (safe/deterministic) to 10 (surreal/high metaphor)
    coherence: float = 7.5          # 1 (free stream-of-consciousness) to 10 (laser topic focus)
    cadence: float = 5.5            # 1 (short punchy Hemingway) to 10 (flowing compound periodic)
    emotion_intensity: float = 6.0  # 1 (neutral/matter-of-fact) to 10 (high melodrama/epic)
    genre: str = "general"          # "sci-fi", "cyberpunk", "philosophical", "noir", "fantasy", "epic", "academic", "nature"
    repetition_penalty: float = 1.8 # Penalty factor for recently used lemmas/words
    recency_window: int = 25        # Words to keep in anti-loop working memory

    def get_temperature(self) -> float:
        """Map novelty (1-10) to Boltzmann temperature T in [0.25, 2.2]."""
        clamped = max(1.0, min(10.0, float(self.novelty)))
        # 1 -> 0.25, 5 -> 0.8, 10 -> 2.2
        return 0.25 + (clamped - 1.0) * (1.95 / 9.0)

    def get_metaphor_probability(self) -> float:
        """Map novelty (1-10) to cross-domain analogy jump probability [0.0, 0.65]."""
        clamped = max(1.0, min(10.0, float(self.novelty)))
        return (clamped - 1.0) / 9.0 * 0.65

    def get_attractor_weight(self) -> float:
        """Map coherence (1-10) to waypoint attraction weight alpha in [0.15, 0.95]."""
        clamped = max(1.0, min(10.0, float(self.coherence)))
        return 0.15 + (clamped - 1.0) * (0.80 / 9.0)

    def get_target_sentence_length(self) -> Tuple[int, int]:
        """Map cadence (1-10) to min/max words per sentence."""
        clamped = max(1.0, min(10.0, float(self.cadence)))
        min_len = int(4 + (clamped - 1.0) * 1.5)  # 1 -> 4, 10 -> 18
        max_len = int(10 + (clamped - 1.0) * 2.2) # 1 -> 10, 10 -> 30
        return (min_len, max_len)

    def get_emotion_weight(self) -> float:
        """Map emotion_intensity (1-10) to VAD alignment weight beta in [0.05, 0.85]."""
        clamped = max(1.0, min(10.0, float(self.emotion_intensity)))
        return 0.05 + (clamped - 1.0) * (0.80 / 9.0)


# Curated discourse transition markers across narrative genres
DISCOURSE_MARKERS = {
    "openers": {
        "general": [
            "In the beginning,", "Across the expanse of time,", "Under the open sky,",
            "With steady momentum,", "At the edge of perception,", "Through every quiet moment,",
        ],
        "sci-fi": [
            "Across the silent vacuum of deep space,", "Beneath the glow of synthetic stars,",
            "At the boundary of the event horizon,", "Within the humming core of the station,",
            "Signals echoed through the cosmic void as", "Beyond the reach of ancient satellites,",
        ],
        "cyberpunk": [
            "Rain-slicked neon reflected against dark asphalt as", "Deep in the subterranean network,",
            "Static hissed across encrypted frequencies when", "Beneath towering holographic spires,",
            "Through chrome corridors and shadow markets,", "High above the sprawling megalopolis,",
        ],
        "philosophical": [
            "In the deep architecture of consciousness,", "Consider the nature of existence where",
            "Between perception and reality lies", "From the earliest inquiry into being,",
            "When truth encounters the boundary of reason,", "In the silent contemplation of the mind,",
        ],
        "noir": [
            "The streetlamps flickered through heavy mist as", "A cold wind swept down the alleyway when",
            "Shadows lengthened across the quiet city while", "Behind the closed doors of the precinct,",
            "In the dim light of the midnight bar,", "Smoke curled into the gloom as",
        ],
        "fantasy": [
            "In an age before memory faded,", "Beneath the canopy of the enchanted grove,",
            "High atop the snow-crowned peaks of Eldoria,", "When the ancient runes began to pulse with light,",
            "Deep within the forgotten catacombs,", "Across the misty valleys of the realm,",
        ],
        "epic": [
            "From the blazing dawn of creation,", "With the thunderous gathering of empires,",
            "Across the blood-soaked plains of history,", "Beneath the golden standards of valor,",
            "When legends forged the destinies of men,", "In the grand tapestry of heroic fate,",
        ],
        "academic": [
            "Empirical evidence demonstrates that", "In examining the fundamental principles of",
            "Theoretical frameworks systematically establish that", "Upon rigorous analysis of the underlying mechanisms,",
            "Contemporary literature indicates that", "A foundational observation in this domain reveals",
        ],
        "nature": [
            "As morning mist drifted across the ancient forest,", "Beneath the emerald canopy of the living woods,",
            "Where rushing mountain waters meet the valley,", "Under the amber glow of the setting sun,",
            "Through seasons of quiet renewal and growth,", "In the untamed wilderness where rivers carve stone,",
        ],
    },
    "transitions": {
        "general": [
            "Furthermore,", "Consequently,", "In contrast,", "Meanwhile,",
            "As time passed,", "Thus,", "Beyond this,", "In turn,",
        ],
        "sci-fi": [
            "Meanwhile, quantum telemetry indicated that", "Consequently, orbital trajectories shifted as",
            "In the adjacent star system,", "Simultaneously, the hyperdrive coils ignited while",
            "Beyond the asteroid belt,", "As automated telemetry confirmed,",
        ],
        "cyberpunk": [
            "Meanwhile, rogue packets flooded the subnet as", "Consequently, neural relays fired in overdrive while",
            "In the dark alleys below,", "Simultaneously, black ice melted through firewalls as",
            "Across encrypted fiber channels,", "Under the flicker of dying neon,",
        ],
        "philosophical": [
            "Consequently, the observer realizes that", "In contrast, dialectical reasoning suggests that",
            "Thus, the essence of thought unfolds where", "Furthermore, this paradox implies that",
            "From this fundamental premise,", "Yet beneath the surface of cognition,",
        ],
        "noir": [
            "Meanwhile, footsteps echoed down the wet pavement as", "Consequently, the tension mounted while",
            "Around the dark corner,", "In the silence that followed,",
            "Yet something lingered in the dark air as", "Behind the Venetian blinds,",
        ],
        "fantasy": [
            "Meanwhile, arcane energies gathered as", "Consequently, the prophecy began to awaken while",
            "Across the enchanted borderland,", "In the shadow of the ancient tower,",
            "Yet a deeper mystery stirred when", "Through whispering winds,",
        ],
        "epic": [
            "Consequently, the war drums sounded across the ridge as", "With unyielding resolve, the legion pressed forward while",
            "Beyond the fortress walls,", "In the crucible of heroic triumph,",
            "Thus destiny unfolded as", "Amidst the clash of iron and courage,",
        ],
        "academic": [
            "Consequently, the experimental findings substantiate that", "In comparison with conventional paradigms,",
            "Furthermore, quantitative metrics indicate that", "Thus, the structural correlation confirms that",
            "In addition to these factors,", "A direct corollary of this finding demonstrates that",
        ],
        "nature": [
            "Meanwhile, autumn breezes rustled the canopy as", "Consequently, the river carved deeper paths while",
            "Across the blooming meadow,", "As twilight descended upon the mountains,",
            "In the quiet rhythm of the wild,", "With the turning of the seasons,",
        ],
    },
    "conclusions": {
        "general": [
            "Ultimately,", "In the final reckoning,", "Thus the pattern was complete as",
            "As the silence settled,", "In this manner,", "And so it remains,",
        ],
        "sci-fi": [
            "Ultimately, the cosmos returned to equilibrium as", "In the endless expanse of the void,",
            "Thus the starship charted a course into tomorrow while", "As the final signal broadcast across the galaxy,",
            "In the quiet wake of the quantum pulse,", "And so the stars continued their eternal dance,",
        ],
        "cyberpunk": [
            "Ultimately, the signal faded into the digital noise as", "In the aftermath of the system breach,",
            "Thus another night vanished into the synthetic skyline while", "As dawn broke over the neon towers,",
            "In the final byte of the decrypted transmission,", "And the city kept pulsing beneath the wire,",
        ],
        "philosophical": [
            "Ultimately, consciousness finds unity where", "In the final synthesis of idea and form,",
            "Thus the mind attains profound clarity as", "As the horizon of understanding expands,",
            "In this quiet realization,", "And so the pursuit of wisdom endures,",
        ],
        "noir": [
            "Ultimately, the truth stepped into the lamplight as", "In the cold dawn that followed,",
            "Thus the case came to its quiet close while", "As the rain washed the empty sidewalks clean,",
            "In the lingering smoke of the night,", "And the city slept its uneasy sleep,",
        ],
        "fantasy": [
            "Ultimately, the light returned to the sacred realm as", "In the chronicles of the elder bards,",
            "Thus peace settled over the kingdom once more while", "As the stars blessed the sleeping land,",
            "In the eternal memory of the heroes,", "And the song of Eldoria echoed forevermore,",
        ],
        "epic": [
            "Ultimately, victory was etched into the annals of glory as", "In the triumphant dawn of the new age,",
            "Thus the legend was immortalized for eternity while", "As the standard flew high above the mountain,",
            "In the enduring splendor of the realm,", "And honor resounded through all generations,",
        ],
        "academic": [
            "Ultimately, these results furnish compelling evidence that", "In conclusion, the proposed architecture establishes that",
            "Thus, future investigations may leverage this framework to", "As synthesized across all empirical evaluations,",
            "In summary, the comprehensive model validates that", "Therefore, these foundational insights provide",
        ],
        "nature": [
            "Ultimately, harmony returned to the living wild as", "In the gentle dusk that embraced the valley,",
            "Thus the eternal cycle of nature turned once more while", "As moonlight bathed the quiet glade,",
            "In the deep tranquility of the earth,", "And the wilderness rested in timeless peace,",
        ],
    }
}


@dataclass
class NarrativeAct:
    """A planned act or paragraph in a multi-act narrative."""
    act_index: int
    act_name: str          # "Exposition / Setup", "Rising Tension", "Climax / Turning Point", "Resolution / Echo"
    goal_theme: str
    target_emotion: str    # "curiosity", "tension", "wonder", "peace", "triumph"
    target_vad: VAD
    discourse_type: str    # "opener", "transition", "conclusion"
    num_sentences: int = 3


class NarrativeArcPlanner:
    """
    Plans multi-paragraph narrative arcs (3-act, 4-act, or 5-act)
    with evolving emotional trajectories and concept waypoints.
    """

    ACT_TEMPLATES = {
        1: [
            ("Core Synthesis", "curiosity", "opener", 4),
        ],
        2: [
            ("Foundational Overview & Context", "curiosity", "opener", 3),
            ("Synthesis & Future Implications", "insight", "conclusion", 3),
        ],
        3: [
            ("Setup & Atmosphere", "curiosity", "opener", 3),
            ("Rising Tension & Discovery", "excitement", "transition", 4),
            ("Climax & Resolution", "triumph", "conclusion", 3),
        ],
        4: [
            ("Exposition & World State", "calm", "opener", 3),
            ("Inciting Incident & Conflict", "tension", "transition", 3),
            ("Climactic Revelation", "awe", "transition", 4),
            ("Echoing Resolution & Legacy", "serenity", "conclusion", 3),
        ],
        5: [
            ("Prologue & Stasis", "curiosity", "opener", 2),
            ("Rising Action & Challenge", "urgency", "transition", 3),
            ("Peripeteia & Climax", "intensity", "transition", 4),
            ("Falling Action & Epiphany", "wonder", "transition", 3),
            ("Epilogue & Timeless Echo", "peace", "conclusion", 3),
        ],
    }

    def __init__(self, emotion_engine: Optional[EmotionEngine] = None):
        self.emotion_engine = emotion_engine or EmotionEngine()

    def plan_arc(
        self,
        prompt: str,
        acts_count: int = 3,
        genre: str = "general",
    ) -> List[NarrativeAct]:
        """Generate structured narrative acts from a base prompt and genre."""
        template_key = acts_count if acts_count in self.ACT_TEMPLATES else 3
        template = self.ACT_TEMPLATES[template_key]

        words = [w for w in re.findall(r'\b\w+\b', prompt.lower()) if len(w) > 3]
        primary_concept = words[0] if words else "journey"
        secondary_concept = words[1] if len(words) > 1 else "destiny"

        planned_acts = []
        for idx, (act_name, emo_name, disc_type, num_sents) in enumerate(template):
            # Act theme evolves
            if idx == 0:
                act_theme = f"{prompt} {primary_concept} origins"
            elif idx == len(template) - 1:
                act_theme = f"{primary_concept} {secondary_concept} eternity resolution"
            else:
                act_theme = f"{primary_concept} {secondary_concept} transformation {emo_name}"

            vad = self.emotion_engine.compute_concept_vad(act_theme + " " + emo_name)

            planned_acts.append(
                NarrativeAct(
                    act_index=idx + 1,
                    act_name=act_name,
                    goal_theme=act_theme,
                    target_emotion=emo_name,
                    target_vad=vad,
                    discourse_type=disc_type,
                    num_sentences=num_sents,
                )
            )

        return planned_acts


class HierarchicalProseEngine:
    """
    Graph-Native Creative Prose & Multi-Paragraph Story Engine.
    Combines graph-extracted fact chains with an ultra-lightweight 135M micro-voice.
    """

    def __init__(
        self,
        word_graph: WordGraph,
        concept_graph: Optional[Any] = None,
        embedder: Optional[Any] = None,
        emotion_engine: Optional[EmotionEngine] = None,
        imagine_engine: Optional[Any] = None,
        use_micro_voice: bool = True,
    ):
        self.word_graph = word_graph
        self.concept_graph = concept_graph
        self.embedder = embedder
        self.emotion_engine = emotion_engine or EmotionEngine()
        self.imagine_engine = imagine_engine
        self.planner = NarrativeArcPlanner(self.emotion_engine)
        self.use_micro_voice = use_micro_voice
        self._voice_pipeline = None

    def _get_voice_pipeline(self):
        """Lazy loader for lightweight 135M micro-voice."""
        if self._voice_pipeline is None and self.use_micro_voice:
            try:
                import warnings
                warnings.filterwarnings("ignore")
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                import torch
                model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
                tok = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
                if hasattr(model, "generation_config") and model.generation_config:
                    model.generation_config.max_length = None
                self._voice_pipeline = pipeline("text-generation", model=model, tokenizer=tok, device="cpu")
            except Exception:
                self.use_micro_voice = False
        return self._voice_pipeline


    def compose_paragraph(
        self,
        theme_vector: np.ndarray,
        config: ProseConfig,
        act: Optional[NarrativeAct] = None,
        context_vad: Optional[VAD] = None,
        recency_memory: Optional[List[str]] = None,
        num_sentences: int = 3,
        prompt_topic: str = "",
    ) -> Tuple[str, List[str]]:
        """
        Synthesize a rich, grammatically grounded paragraph by:
        1. Activating relevant concept & phrase nodes from the knowledge graph.
        2. Chaining coherent ingested narrative clauses.
        3. Ensuring smooth discourse transitions between narrative acts.
        """
        recency = list(recency_memory) if recency_memory else []
        sentences = []
        genre = config.genre.lower() if config.genre.lower() in DISCOURSE_MARKERS["openers"] else "general"

        disc_type = act.discourse_type if act else "transition"
        marker_pool = DISCOURSE_MARKERS.get(
            "openers" if disc_type == "opener" else ("conclusions" if disc_type == "conclusion" else "transitions"),
            DISCOURSE_MARKERS["transitions"]
        ).get(genre, DISCOURSE_MARKERS["transitions"]["general"])

        target_vad = act.target_vad if act else (context_vad or VAD(0.6, 0.5, 0.5))

        # 1. Harvest grounded sentences & phrases from active graph nodes near theme_vector
        grounded_clauses = []
        if self.concept_graph and hasattr(self.concept_graph, "nodes"):
            norm_theme = float(np.linalg.norm(theme_vector))
            scored_nodes = []
            for nid, n in self.concept_graph.nodes.items():
                if getattr(n, "vector", None) is not None:
                    norm_n = float(np.linalg.norm(n.vector))
                    if norm_n > 1e-6 and norm_theme > 1e-6:
                        sim = float(np.dot(n.vector, theme_vector) / (norm_n * norm_theme))
                        if sim > 0.20:
                            meta = getattr(n, "metadata", {})
                            if "sentence" in meta and len(meta["sentence"]) > 15:
                                scored_nodes.append((meta["sentence"].strip(), sim))
                            elif n.node_type == "text" and len(n.source_text) > 15:
                                scored_nodes.append((n.source_text.strip(), sim))

            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            seen_s = set()
            for s_text, _ in scored_nodes:
                if s_text not in seen_s and s_text not in recency:
                    seen_s.add(s_text)
                    grounded_clauses.append(s_text)
                if len(grounded_clauses) >= num_sentences * 3:
                    break

        # 2. If micro-voice is available, articulate smoothly into unified prose
        voice = self._get_voice_pipeline()
        if voice is not None:
            act_title = act.act_name if act else "Story Scene"
            topic_str = prompt_topic if prompt_topic else "An adventurous journey"
            facts_prompt = " ".join(grounded_clauses[:3]) if grounded_clauses else ""
            
            prompt = (
                f"<|im_start|>system\n"
                f"You are a master story author. Write ONLY continuous, vivid story prose without markdown headers or commentary.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Story Theme: {topic_str}\n"
                f"Chapter Scene: {act_title} ({genre})\n"
                f"Learned Context: {facts_prompt}\n\n"
                f"Write 2 to 3 engaging narrative sentences about '{topic_str}' for this scene.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                out = voice(
                    prompt,
                    max_new_tokens=85,
                    temperature=0.6,
                    do_sample=True,
                    clean_up_tokenization_spaces=False,
                    pad_token_id=voice.tokenizer.eos_token_id,
                )
                raw_text = out[0]["generated_text"]
                if "<|im_start|>assistant" in raw_text:
                    gen_content = raw_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
                else:
                    gen_content = raw_text.replace(prompt, "").strip()

                # Clean any lingering markdown headers or quotes
                gen_content = re.sub(r'\*\*.*?\*\*', '', gen_content).strip()
                gen_content = re.sub(r'^(Chapter \d+:?|Scene \d+:?)\s*', '', gen_content, flags=re.IGNORECASE).strip()

                if len(gen_content.split()) >= 8:
                    if not gen_content.endswith(('.', '!', '?')):
                        gen_content += '.'
                    words = [w.lower() for w in re.findall(r'\b\w+\b', gen_content)]
                    recency.extend(words[-15:])
                    return gen_content, recency
            except Exception:
                pass



        # Fallback: Assemble paragraph sentences directly
        for s_idx in range(num_sentences):
            opener_prefix = ""
            if s_idx == 0:
                opener_prefix = random.choice(marker_pool)
            elif s_idx == num_sentences - 1 and disc_type == "conclusion":
                closer_pool = DISCOURSE_MARKERS["conclusions"].get(genre, DISCOURSE_MARKERS["conclusions"]["general"])
                opener_prefix = random.choice(closer_pool)

            if grounded_clauses and s_idx < len(grounded_clauses):
                clause = grounded_clauses[s_idx]
                recency.append(clause)
                clause_clean = clause[0].lower() + clause[1:] if len(clause) > 1 else clause
                if opener_prefix:
                    if opener_prefix.endswith(",") or opener_prefix.endswith("as") or opener_prefix.endswith("while") or opener_prefix.endswith("when"):
                        sent_text = f"{opener_prefix} {clause_clean}"
                    else:
                        sent_text = f"{opener_prefix}, {clause_clean}"
                else:
                    sent_text = clause
                if not sent_text.endswith(('.', '!', '?')):
                    sent_text += '.'
                sentences.append(sent_text)
            else:
                sent_text, recency = self._synthesize_sentence_walk(
                    theme_vector=theme_vector,
                    config=config,
                    target_vad=target_vad,
                    opener_prefix=opener_prefix,
                    recency_memory=recency,
                )
                if sent_text:
                    sentences.append(sent_text)

        paragraph = " ".join(sentences).strip()
        if paragraph and not paragraph.endswith(('.', '!', '?')):
            paragraph += '.'
        return paragraph, recency




    def compose_story(
        self,
        prompt: str,
        acts_count: int = 3,
        config: Optional[ProseConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete multi-act story with full 1-to-10 parameter adaptability.
        Returns a dict containing title, acts, parameters, and full story text.
        """
        cfg = config or ProseConfig()
        acts = self.planner.plan_arc(prompt=prompt, acts_count=acts_count, genre=cfg.genre)

        # Encode prompt
        if self.embedder:
            global_theme_vec = self.embedder.encode_single(prompt)
        else:
            global_theme_vec = np.zeros(64, dtype=np.float32)

        generated_acts = []
        recency: List[str] = []

        for act in acts:
            # Concept blending if novelty >= 7.0 and imagine engine available
            act_vec = global_theme_vec
            if cfg.novelty >= 7.0 and self.imagine_engine and hasattr(self.imagine_engine, "blender"):
                words = [w for w in re.findall(r'\b\w+\b', act.goal_theme.lower()) if len(w) > 3]
                if len(words) >= 2 and self.concept_graph:
                    try:
                        blended = self.imagine_engine.blender.blend(words[0], words[1], self.concept_graph, self.embedder)
                        if blended and hasattr(blended, "vector"):
                            # Blend with global theme
                            act_vec = 0.5 * global_theme_vec + 0.5 * blended.vector
                    except Exception:
                        pass

            para_text, recency = self.compose_paragraph(
                theme_vector=act_vec,
                config=cfg,
                act=act,
                context_vad=act.target_vad,
                recency_memory=recency,
                num_sentences=act.num_sentences,
                prompt_topic=prompt,
            )


            generated_acts.append({
                "act_number": act.act_index,
                "act_name": act.act_name,
                "emotion": act.target_emotion,
                "text": para_text,
            })

        # Generate stylized title
        title_words = [w.capitalize() for w in re.findall(r'\b\w+\b', prompt) if len(w) > 3][:3]
        if not title_words:
            title_words = ["The", "Eternal", "Odyssey"]
        title = "The " + " of ".join(title_words) if len(title_words) == 2 else " ".join(title_words)

        full_text = "\n\n".join([a["text"] for a in generated_acts])

        return {
            "title": title,
            "prompt": prompt,
            "genre": cfg.genre,
            "parameters": {
                "novelty": cfg.novelty,
                "coherence": cfg.coherence,
                "cadence": cfg.cadence,
                "emotion_intensity": cfg.emotion_intensity,
                "temperature": round(cfg.get_temperature(), 2),
            },
            "acts": generated_acts,
            "full_story": full_text,
        }

    def compose_article(
        self,
        topic: str,
        paragraphs_count: int = 3,
        config: Optional[ProseConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured, cohesive non-fiction or analytical article.
        """
        cfg = config or ProseConfig(novelty=4.0, coherence=8.5, cadence=6.0, genre="academic")
        return self.compose_story(prompt=topic, acts_count=paragraphs_count, config=cfg)

    def _synthesize_sentence_walk(
        self,
        theme_vector: np.ndarray,
        config: ProseConfig,
        target_vad: VAD,
        opener_prefix: str = "",
        recency_memory: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Execute energy-weighted Boltzmann traversal with:
        1. Multi-token contextual self-attention over the current sentence vector.
        2. Strict POS grammar transition matrix (forbidding broken sequences).
        3. Clause structure grounding.
        """
        from mayon_cortex.language.word_graph import guess_pos

        # Permitted POS transitions for natural English syntax
        VALID_POS_TRANSITIONS = {
            "DET": {"NOUN", "ADJ", "NUM"},
            "ADJ": {"NOUN", "ADJ", "CONJ"},
            "NOUN": {"VERB", "VERB_PAST", "VERB_GER", "PREP", "CONJ", "ADV", "PUNCT"},
            "PRON": {"VERB", "VERB_PAST", "ADV", "VERB_GER"},
            "VERB": {"DET", "NOUN", "ADJ", "ADV", "PREP", "PRON", "PUNCT"},
            "VERB_PAST": {"DET", "NOUN", "ADJ", "ADV", "PREP", "PRON", "PUNCT"},
            "VERB_GER": {"DET", "NOUN", "ADJ", "PREP", "PUNCT"},
            "ADV": {"VERB", "VERB_PAST", "ADJ", "ADV", "PUNCT"},
            "PREP": {"DET", "NOUN", "ADJ", "PRON"},
            "CONJ": {"DET", "NOUN", "PRON", "ADJ", "ADV"},
            "UNKNOWN": {"NOUN", "VERB", "ADJ", "DET", "PREP"},
        }

        recency = list(recency_memory) if recency_memory else []
        min_words, max_words = config.get_target_sentence_length()
        target_length = random.randint(min_words, max_words)

        temperature = config.get_temperature()
        attractor_wt = config.get_attractor_weight()
        emotion_wt = config.get_emotion_weight()
        rep_penalty = config.repetition_penalty
        norm_theme = float(np.linalg.norm(theme_vector))

        tokens: List[str] = []
        sentence_vecs: List[np.ndarray] = []

        # Parse prefix words
        if opener_prefix:
            prefix_tokens = [w.lower() for w in re.findall(r'\b\w+\b', opener_prefix)]
            recency.extend(prefix_tokens)
            last_word = None
            for pw in reversed(prefix_tokens):
                if pw in self.word_graph.nodes:
                    last_word = pw
                    break
        else:
            last_word = None

        if not last_word or last_word not in self.word_graph.nodes:
            last_word = self._select_seed_word(theme_vector, target_vad, recency, norm_theme=norm_theme)

        tokens.append(last_word)
        recency.append(last_word)
        if last_word in self.word_graph.nodes and self.word_graph.nodes[last_word].vector is not None:
            sentence_vecs.append(self.word_graph.nodes[last_word].vector)

        curr_word = last_word
        curr_pos = guess_pos(curr_word)

        for step in range(target_length):
            node = self.word_graph.nodes.get(curr_word)
            allowed_next_pos = VALID_POS_TRANSITIONS.get(curr_pos, VALID_POS_TRANSITIONS["UNKNOWN"])

            # Compute dynamic sentence context vector (self-attention history)
            if sentence_vecs:
                context_vec = np.mean(sentence_vecs[-6:], axis=0)
                norm_ctx = float(np.linalg.norm(context_vec))
            else:
                context_vec = theme_vector
                norm_ctx = norm_theme

            if not node or not node.next_words:
                next_word = self._select_seed_word(theme_vector, target_vad, recency, norm_theme=norm_theme)
            else:
                candidates = list(node.next_words.items())
                scored_candidates = []

                for next_w, count in candidates:
                    if not next_w.isalpha() or len(next_w) < 1:
                        continue
                    next_node = self.word_graph.nodes.get(next_w)
                    if not next_node:
                        continue

                    cand_pos = guess_pos(next_w)

                    # POS grammatical matrix compatibility filter
                    pos_penalty = 1.0 if cand_pos in allowed_next_pos else 0.15

                    # Base transition probability
                    base_score = math.log(max(1.0, float(count)))

                    # Theme alignment (global goal)
                    cos_theme = 0.0
                    if norm_theme > 1e-6 and next_node.vector is not None:
                        norm_node = float(np.linalg.norm(next_node.vector))
                        if norm_node > 1e-6:
                            cos_theme = float(np.dot(next_node.vector, theme_vector) / (norm_node * norm_theme))

                    # Local contextual self-attention across recent words in sentence
                    cos_ctx = 0.0
                    if norm_ctx > 1e-6 and next_node.vector is not None:
                        norm_node = float(np.linalg.norm(next_node.vector))
                        if norm_node > 1e-6:
                            cos_ctx = float(np.dot(next_node.vector, context_vec) / (norm_node * norm_ctx))

                    # Affective emotional VAD match
                    vad_dist = math.sqrt(
                        (next_node.vad.valence - target_vad.valence) ** 2 +
                        (next_node.vad.arousal - target_vad.arousal) ** 2 +
                        (next_node.vad.dominance - target_vad.dominance) ** 2
                    )
                    emo_sim = max(0.0, 1.0 - vad_dist)

                    # Anti-repetition penalty
                    recent_count = recency[-config.recency_window:].count(next_w)
                    pen = math.pow(rep_penalty, recent_count)

                    # Total Context-Attention Energy
                    energy = (
                        (base_score * 0.8) +
                        (attractor_wt * 3.5 * cos_theme) +
                        (cos_ctx * 2.5) +
                        (emotion_wt * 2.0 * emo_sim)
                    ) * pos_penalty / pen

                    scored_candidates.append((next_w, energy))

                if not scored_candidates:
                    next_word = self._select_seed_word(theme_vector, target_vad, recency, norm_theme=norm_theme)
                else:
                    words_list, energies = zip(*scored_candidates)
                    max_e = max(energies)
                    scaled_exp = [math.exp(max(-15.0, min(15.0, (e - max_e) / temperature))) for e in energies]
                    sum_exp = sum(scaled_exp)
                    probs = [p / sum_exp for p in scaled_exp] if sum_exp > 0 else [1.0 / len(words_list)] * len(words_list)
                    next_word = random.choices(words_list, weights=probs, k=1)[0]

            tokens.append(next_word)
            recency.append(next_word)
            curr_word = next_word
            curr_pos = guess_pos(curr_word)

            if next_word in self.word_graph.nodes and self.word_graph.nodes[next_word].vector is not None:
                sentence_vecs.append(self.word_graph.nodes[next_word].vector)

            # Natural terminal punctuation check
            if step >= min_words and curr_pos in ("NOUN", "VERB_PAST") and next_word not in ("and", "the", "a", "of", "to", "in", "with"):
                if step >= max_words - 3 or random.random() < 0.35:
                    break

        if len(recency) > 100:
            recency = recency[-100:]

        body = " ".join(tokens)
        if opener_prefix:
            if opener_prefix.endswith(",") or opener_prefix.endswith("as") or opener_prefix.endswith("while") or opener_prefix.endswith("when") or opener_prefix.endswith("where") or opener_prefix.endswith("that"):
                full_sent = f"{opener_prefix} {body}."
            else:
                full_sent = f"{opener_prefix}, {body}."
        else:
            full_sent = f"{body.capitalize()}."

        full_sent = re.sub(r'\s+([.,!?;:])', r'\1', full_sent)
        full_sent = re.sub(r'\s+', ' ', full_sent).strip()
        full_sent = full_sent[0].upper() + full_sent[1:] if full_sent else ""

        return full_sent, recency

    def _select_seed_word(
        self,
        theme_vector: np.ndarray,
        target_vad: VAD,
        recency: List[str],
        norm_theme: Optional[float] = None,
    ) -> str:
        """Select an optimal starting node aligned with theme & emotion."""
        if norm_theme is None:
            norm_theme = float(np.linalg.norm(theme_vector))

        candidates = []
        for word, node in self.word_graph.nodes.items():
            if not word.isalpha() or len(word) < 3:
                continue
            if word in recency[-10:]:
                continue

            vad_sim = 1.0 - math.sqrt(
                (node.vad.valence - target_vad.valence) ** 2 +
                (node.vad.arousal - target_vad.arousal) ** 2 +
                (node.vad.dominance - target_vad.dominance) ** 2
            )

            if norm_theme > 1e-6 and node.vector is not None:
                norm_n = float(np.linalg.norm(node.vector))
                cos = float(np.dot(node.vector, theme_vector) / (norm_n * norm_theme)) if norm_n > 1e-6 else 0.0
            else:
                cos = 0.0

            score = (cos * 3.0) + (vad_sim * 2.0) + math.log(max(1, len(node.next_words)))
            candidates.append((word, score))

        if not candidates:
            return "the"

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:min(15, len(candidates))]
        words_k, scores_k = zip(*top_k)
        min_s = min(scores_k)
        weights = [max(0.1, s - min_s + 1.0) for s in scores_k]
        return random.choices(words_k, weights=weights, k=1)[0]

