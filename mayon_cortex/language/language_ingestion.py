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
Language Ingestion — Text → Knowledge + Language + Word Nodes
=============================================================
Parses raw text into THREE unified tiers:
1. Knowledge nodes: WHAT is true (facts, entities, relations)
2. Language phrase nodes: Multi-word phrase bridges
3. Compositional Word nodes: Every word token, sequential bigram edges,
   skip-grams, POS tags, and emotional VAD ratings.

This powers Cortex-Graph's unified reasoning + graph-native language generation.
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.affect.emotion import VAD, EmotionEngine
from mayon_cortex.language.word_graph import WordGraph


# Expanded entity patterns covering real-world stories, nature, science, and everyday life
ENTITY_PATTERNS = {
    "general": [
        r'\b(?:sun|earth|moon|mars|planet|star|galaxy|space|gravity|orbit)\b',
        r'\b(?:water|ocean|river|mountain|forest|tree|plant|rain|climate)\b',
        r'\b(?:dog|cat|bird|lion|tiger|elephant|animal|mammal|fish)\b',
        r'\b(?:human|person|child|family|friend|teacher|doctor|king|queen|hero|villain)\b',
        r'\b(?:love|joy|happiness|sorrow|grief|anger|fear|peace|courage|hope)\b',
        r'\b(?:computer|phone|internet|engine|car|airplane|ship|book|wheel)\b',
        r'\b(?:village|city|country|castle|house|palace|school|library)\b',
    ],
    "story": [
        r'\b(?:hero|dragon|princess|prince|knight|kingdom|sword|treasure|wizard|forest)\b',
        r'\b(?:journey|quest|battle|victory|defeat|curse|magic|secret|tome)\b',
    ],
    "science": [
        r'\b(?:atom|molecule|cell|gene|protein|electron|photon|quantum|dna|rna)\b',
        r'\b(?:gravity|energy|force|mass|velocity|acceleration|entropy|thermodynamics)\b',
        r'\b(?:photosynthesis|evolution|mitochondria|respiration|ecosystem)\b',
    ],
    "medical": [
        r'\b(?:diabetes|hypertension|asthma|cancer|infection|disease|condition|syndrome|disorder)\b',
        r'\b(?:metformin|lisinopril|aspirin|warfarin|insulin|ibuprofen|amoxicillin|penicillin)\b',
        r'\b(?:blood pressure|heart rate|glucose|cholesterol|hemoglobin|platelet)\b',
        r'\b(?:ACE inhibitor|beta blocker|SSRI|NSAID|antibiotic|antiviral|vaccine)\b',
    ],
    "legal": [
        r'\b(?:GDPR|regulation|statute|article|section|act|law|compliance)\b',
        r'\b(?:contract|liability|indemnity|jurisdiction|penalty|prohibition)\b',
    ],
    "finance": [
        r'\b(?:interest rate|inflation|GDP|bond|equity|derivative|liquidity)\b',
        r'\b(?:central bank|Fed|ECB|monetary policy|fiscal policy|revenue|profit)\b',
    ],
    "code": [
        r'\b(?:function|class|method|variable|loop|iterator|generator|decorator)\b',
        r'\b(?:list|dict|set|tuple|string|array|hash|queue|stack|algorithm)\b',
    ],
}

# Relation indicator phrases
RELATION_INDICATORS = {
    "treats":          [r'\btreats?\b', r'\bprescribed\s+for\b', r'\beffective\s+(?:for|against)\b'],
    "causes":          [r'\bcauses?\b', r'\bresults?\s+in\b', r'\bleads?\s+to\b'],
    "prevents":        [r'\bprevents?\b', r'\bprotects?\s+against\b', r'\breduces?\s+risk\b'],
    "is-a":            [r'\bis\s+(?:a|an)\b', r'\bare\s+(?:a|an)\b', r'\btype\s+of\b', r'\bkind\s+of\b'],
    "part-of":         [r'\bpart\s+of\b', r'\bcomponent\s+of\b', r'\bbelongs?\s+to\b', r'\binside\b'],
    "contraindicates": [r'\bcontraindicated?\b', r'\bshould\s+not\b', r'\bavoid\b'],
    "interacts-with":  [r'\binteracts?\s+with\b', r'\bcombined?\s+with\b'],
    "regulates":       [r'\bregulates?\b', r'\bgoverns?\b', r'\bcontrols?\b'],
    "inhibits":        [r'\binhibits?\b', r'\bblocks?\b', r'\bsuppresses?\b'],
    "uses":            [r'\buses?\b', r'\butilizes?\b', r'\bemploys?\b'],
    "calls":           [r'\bcalls?\b', r'\binvokes?\b'],
    "inherits":        [r'\binherits?\b', r'\bextends?\b', r'\bsubclass\b'],
    "orbits":          [r'\borbits?\b', r'\brevolves?\s+around\b'],
    "defeats":         [r'\bdefeats?\b', r'\bvanquishes?\b', r'\bovercomes?\b'],
    "protects":        [r'\bprotects?\b', r'\bguards?\b', r'\bdefends?\b'],
    "produces":        [r'\bproduces?\b', r'\bgenerates?\b', r'\bcreates?\b', r'\byields?\b'],
    "lives-in":        [r'\blives?\s+in\b', r'\binhabits?\b', r'\bdwells?\s+in\b'],
    "rules":           [r'\brules?\b', r'\bgoverns?\b'],
    "commands":        [r'\bcommands?\b', r'\bleads?\b', r'\bcaptains?\b'],
}


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    text: str
    start: int
    end: int
    sector: str


@dataclass
class ExtractedRelation:
    """A relation extracted between two entities."""
    source: str
    target: str
    relation_type: str
    confidence: float


@dataclass
class IngestionStats:
    """Statistics from an ingestion operation."""
    knowledge_nodes_added: int = 0
    phrase_nodes_added: int = 0
    word_nodes_added: int = 0
    knowledge_edges_added: int = 0
    phrase_edges_added: int = 0
    sentences_processed: int = 0


class LanguageIngester:
    """
    Parses text into knowledge + language pattern nodes + word-level graph nodes.
    """

    def __init__(
        self,
        embedder,
        config: Optional[CortexConfig] = None,
        word_graph: Optional[WordGraph] = None,
        emotion_engine: Optional[EmotionEngine] = None,
    ):
        self.embedder = embedder
        self.config = config or CortexConfig()
        self.word_graph = word_graph
        self.emotion_engine = emotion_engine or EmotionEngine()

    def ingest(
        self,
        text: str,
        graph,  # MayonGraph
        sector: str = "general",
        provenance: str = "ingestion",
    ) -> IngestionStats:
        """
        Ingest text into the graph as knowledge, phrases, and word transitions.
        """
        stats = IngestionStats(0, 0, 0, 0, 0, 0)
        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            stats.sentences_processed += 1
            sent_vad = self.emotion_engine.compute_concept_vad(sentence)

            # 1. Tokenize into individual words for the Word Graph
            tokens = self._tokenize_words(sentence)
            if self.word_graph is not None and tokens:
                token_vectors = [self.embedder.encode_single(t) for t in tokens]
                self.word_graph.add_sentence_tokens(
                    tokens=tokens,
                    token_vectors=token_vectors,
                    sector=sector,
                    sentence_vad=sent_vad,
                )
                stats.word_nodes_added += len(tokens)

            # 2. Extract entities
            entities = self._extract_entities(sentence, sector)

            # 3. Extract relations
            relations = self._extract_relations(sentence, entities)

            # 4. Add knowledge nodes
            entity_node_map = {}
            for entity in entities:
                vec = self.embedder.encode_single(entity.text)
                entity_vad = self.emotion_engine.compute_concept_vad(entity.text)
                node = graph.add_node(
                    vector=vec,
                    source_text=entity.text,
                    level=1,  # CONCEPT
                    sector=sector,
                    node_type="text",
                    provenance=provenance,
                    confidence=0.9,
                    metadata={
                        "entity": True,
                        "vad_valence": entity_vad.valence,
                        "vad_arousal": entity_vad.arousal,
                        "vad_dominance": entity_vad.dominance,
                    },
                )
                entity_node_map[entity.text.lower()] = node.id
                stats.knowledge_nodes_added += 1

            # 5. Add knowledge edges
            for relation in relations:
                src_id = entity_node_map.get(relation.source.lower())
                tgt_id = entity_node_map.get(relation.target.lower())
                if src_id and tgt_id:
                    try:
                        graph.add_edge(
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=relation.relation_type,
                            confidence=relation.confidence,
                            sector=sector,
                            provenance=provenance,
                        )
                        stats.knowledge_edges_added += 1
                    except ValueError:
                        pass

            # 5b. Open-Domain Fallback to SemanticRoleExtractor for any domain / vocabulary
            if len(relations) == 0 or stats.knowledge_edges_added == 0:
                from mayon_cortex.language.hyper_parser import SemanticRoleExtractor
                if not hasattr(self, "_hyper_extractor"):
                    self._hyper_extractor = SemanticRoleExtractor()
                hyper_edges = self._hyper_extractor.parse_sentence(sentence, sector=sector, provenance=provenance)
                for he in hyper_edges:
                    src_low = he.source.lower()
                    tgt_low = he.target.lower()
                    if src_low not in entity_node_map:
                        vec = self.embedder.encode_single(he.source)
                        n = graph.add_node(vector=vec, source_text=he.source, level=1, sector=sector, provenance=provenance)
                        entity_node_map[src_low] = n.id
                        stats.knowledge_nodes_added += 1
                    if tgt_low not in entity_node_map:
                        vec = self.embedder.encode_single(he.target)
                        n = graph.add_node(vector=vec, source_text=he.target, level=1, sector=sector, provenance=provenance)
                        entity_node_map[tgt_low] = n.id
                        stats.knowledge_nodes_added += 1

                    try:
                        graph.add_edge(
                            source_id=entity_node_map[src_low],
                            target_id=entity_node_map[tgt_low],
                            relation_type=he.relation,
                            confidence=he.confidence,
                            sector=sector,
                            provenance=provenance,
                        )
                        stats.knowledge_edges_added += 1
                    except Exception:
                        pass

            # 6. Extract and add language phrase nodes
            phrases = self._extract_phrases(sentence, entities)
            phrase_nodes = []
            sentence_id = f"sent_{stats.sentences_processed}_{int(time.time()*1000)}"
            for phrase_text in phrases:
                words = phrase_text.split()
                if len(words) < self.config.min_phrase_len or len(words) > self.config.max_phrase_len:
                    continue

                vec = self.embedder.encode_single(phrase_text)
                pnode = graph.add_node(
                    vector=vec,
                    source_text=phrase_text,
                    level=3,  # PHRASE
                    sector=sector,
                    node_type="phrase",
                    provenance=provenance,
                    confidence=0.85,
                    metadata={"sentence": sentence, "sentence_id": sentence_id},
                )
                phrase_nodes.append(pnode)
                stats.phrase_nodes_added += 1

            # 7. Connect phrase nodes with followed-by edges within THIS sentence only
            for i in range(len(phrase_nodes) - 1):
                try:
                    graph.add_edge(
                        source_id=phrase_nodes[i].id,
                        target_id=phrase_nodes[i + 1].id,
                        relation_type="followed-by",
                        confidence=0.8,
                        sector=sector,
                        provenance=provenance,
                    )
                    stats.phrase_edges_added += 1
                except ValueError:
                    pass

            # 8. Connect entities to their adjacent phrases (expressed-as)
            self._connect_entities_to_phrases(
                sentence, entities, entity_node_map, phrase_nodes, graph,
                sector, provenance, stats, sentence_id,
            )

        return stats

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def _tokenize_words(self, sentence: str) -> List[str]:
        """Tokenize sentence into clean word tokens preserving punctuation tokens."""
        raw_tokens = re.findall(r"\w+|[^\w\s]", sentence)
        return [t.strip() for t in raw_tokens if t.strip()]

    def _extract_entities(self, sentence: str, sector: str) -> List[ExtractedEntity]:
        """Extract open-domain entities and noun concepts from a sentence."""
        entities = []
        seen_spans = set()

        # 1. Domain-specific patterns
        patterns = (
            ENTITY_PATTERNS.get(sector, []) +
            ENTITY_PATTERNS.get("general", []) +
            ENTITY_PATTERNS.get("science", []) +
            ENTITY_PATTERNS.get("story", [])
        )

        for pattern in patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append(ExtractedEntity(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        sector=sector,
                    ))

        # 2. Capitalized named entities
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence):
            span = (match.start(), match.end())
            if span not in seen_spans and len(match.group(0)) > 2:
                seen_spans.add(span)
                entities.append(ExtractedEntity(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    sector=sector,
                ))

        # 3. Open-domain noun chunks and key concept words
        # (e.g. "green plants", "photosynthesis", "oxygen", "glucose", "carbon dioxide", "needle", "cherry tree")
        noun_chunk_pattern = r'\b(?:(?:green|noble|brave|magical|golden|curious|shiny|little|big|old|young)\s+)?([a-z]{3,}(?:s|es|ing|tion|ment|ity)?)\b'
        for match in re.finditer(noun_chunk_pattern, sentence, re.IGNORECASE):
            word = match.group(0).lower().strip()
            # Ignore common stop words
            if word in {
                "the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "been",
                "have", "has", "had", "they", "them", "their", "what", "when", "where", "which",
                "there", "then", "into", "over", "some", "such", "than", "very", "about", "after",
                "once", "upon", "time", "because", "could", "would", "should", "other", "every"
            }:
                continue
            span = (match.start(), match.end())
            # Ensure no overlapping spans
            overlaps = any(s <= match.start() < e or s < match.end() <= e for s, e in seen_spans)
            if not overlaps:
                seen_spans.add(span)
                entities.append(ExtractedEntity(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    sector=sector,
                ))

        entities.sort(key=lambda e: e.start)
        return entities

    def _extract_relations(
        self, sentence: str, entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations between entities using indicator phrases or verb bridges."""
        relations = []
        sent_lower = sentence.lower()

        # 1. Check known relation indicators
        matched_indicators = []
        for rel_type, patterns in RELATION_INDICATORS.items():
            for pattern in patterns:
                m = re.search(pattern, sent_lower)
                if m:
                    matched_indicators.append((m.start(), rel_type))
                    break

        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                src = entities[i]
                tgt = entities[i + 1]

                # Find if an indicator falls between or near these entities
                rel_found = None
                for pos, rtype in matched_indicators:
                    if src.start <= pos <= tgt.end + 10:
                        rel_found = rtype
                        break

                if rel_found:
                    relations.append(ExtractedRelation(
                        source=src.text,
                        target=tgt.text,
                        relation_type=rel_found,
                        confidence=0.88,
                    ))
                else:
                    # Look for verb between entities
                    between_text = sentence[src.end:tgt.start].strip()
                    verb_match = re.search(r'\b(creates?|absorbs?|performs?|produces?|found|finds?|played|helped|visited|drank|grew|guarded|forged|defeated|walked|saw)\b', between_text, re.IGNORECASE)
                    if verb_match:
                        verb_rel = verb_match.group(0).lower()
                        relations.append(ExtractedRelation(
                            source=src.text,
                            target=tgt.text,
                            relation_type=verb_rel,
                            confidence=0.82,
                        ))
                    else:
                        relations.append(ExtractedRelation(
                            source=src.text,
                            target=tgt.text,
                            relation_type="relates-to",
                            confidence=0.70,
                        ))

        return relations

    def _extract_phrases(
        self, sentence: str, entities: List[ExtractedEntity]
    ) -> List[str]:
        """Extract language phrases between entities."""
        if not entities:
            words = sentence.split()
            if self.config.min_phrase_len <= len(words) <= self.config.max_phrase_len:
                return [sentence]
            return []

        phrases = []
        if entities[0].start > 0:
            prefix = sentence[:entities[0].start].strip()
            if prefix and len(prefix.split()) >= self.config.min_phrase_len:
                phrases.append(prefix)

        for i in range(len(entities) - 1):
            between = sentence[entities[i].end:entities[i + 1].start].strip().strip(' ,;:')
            if between and len(between.split()) >= self.config.min_phrase_len:
                phrases.append(between)

        if entities[-1].end < len(sentence):
            suffix = sentence[entities[-1].end:].strip().strip(' ,;:.')
            if suffix and len(suffix.split()) >= self.config.min_phrase_len:
                phrases.append(suffix)

        return phrases

    def _connect_entities_to_phrases(
        self, sentence, entities, entity_map, phrase_nodes, graph,
        sector, provenance, stats, sentence_id: str = "",
    ):
        """Connect entity nodes to their adjacent phrase nodes via expressed-as edges."""
        if not entities or not phrase_nodes:
            return

        for entity in entities:
            eid = entity_map.get(entity.text.lower())
            if not eid:
                continue

            if phrase_nodes:
                best_phrase = phrase_nodes[0]
                best_sim = -1.0

                entity_node = graph.nodes.get(eid)
                if entity_node is not None:
                    for pnode in phrase_nodes:
                        e_norm = np.linalg.norm(entity_node.vector)
                        p_norm = np.linalg.norm(pnode.vector)
                        if e_norm > 0 and p_norm > 0:
                            sim = float(np.dot(entity_node.vector, pnode.vector) / (e_norm * p_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_phrase = pnode

                try:
                    graph.add_edge(
                        source_id=eid,
                        target_id=best_phrase.id,
                        relation_type="expressed-as",
                        confidence=0.75,
                        sector=sector,
                        provenance=provenance,
                    )
                    stats.phrase_edges_added += 1
                except ValueError:
                    pass
