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
Large Multi-Domain Corpus Pack for Cortex Graph
================================================
Contains rich, expansive text datasets spanning 7 distinct genres:
1. Sci-Fi & Cosmic Exploration
2. Cyberpunk & Neural Networks
3. Philosophy & Consciousness
4. Nature & Living Ecosystems
5. Fantasy Lore & Mythic History
6. Artificial Intelligence & Graph Theory
7. Epic & Heroic Odysseys

Used for high-density WordGraph transition learning and ConceptGraph reasoning.
"""

from typing import List

LARGE_DOMAIN_CORPUS = {

    "sci-fi": [
        "The starship drifted across the quiet expanse of the Orion Nebula under the pale light of distant pulsars.",
        "Quantum propulsion coils hummed with resonant frequency as the navigation computer plotted relativistic trajectories.",
        "Beyond the boundary of the solar wind, automated probes transmitted telemetry from unexplored planetary systems.",
        "Gravitational waves rippled through the fabric of spacetime, signaling the birth of a newborn binary star.",
        "Deep space telescopes captured the golden arc of plasma erupting from the surface of a red supergiant.",
        "The orbital station rotated silently against the curved horizon of the blue planet below.",
        "Subatomic sensor arrays detected anomalies in the dark matter halo encircling the galactic core.",
        "Interstellar explorers discovered ancient crystalline ruins buried beneath the frozen plains of Titan.",
        "Hyperdrive engines engaged with a blinding flash, bending space into folded corridors of luminous light.",
        "The crew watched in silent reverence as the cosmic dust cloud coalesced into vibrant proto-planetary disks.",
        "Atmospheric processors converted toxic methane into breathable oxygen across the terraformed valleys of Mars.",
        "Solar radiation shields deflected high-energy cosmic rays, protecting the habitat modules from stellar storms.",
        "Automated mining drones harvested rare minerals from the dense asteroid fields of the outer rim.",
        "Relativistic time dilation ensured that while years passed on the vessel, decades elapsed on the home world.",
        "The celestial navigation matrix aligned precisely with the pulsars of the interstellar beacon network.",
    ],
    "cyberpunk": [
        "Rain-slicked streets reflected the pulsing neon glow of towering holographic corporate advertisements.",
        "Neural implants sparked with synthetic data streams as the hacker bypassed military-grade security protocols.",
        "Beneath the towering megalopolis, black market dealers traded encrypted quantum storage drives in the shadows.",
        "Static hissed across fiber-optic conduits connecting millions of bio-digital consciousness nodes.",
        "Rogue artificial intelligences roamed the decentralized darknet, evading automated security countermeasures.",
        "High above the industrial sector, aerodynes glided silently between chrome spires and smog-filled canyons.",
        "Cybernetic prosthetics responded instantaneously to micro-synaptic impulses along synthetic nervous pathways.",
        "Encrypted memory shards contained fragments of forgotten classified corporate research.",
        "Holographic interfaces shimmered in the ambient gloom of the clandestine subterranean workshop.",
        "The digital avatar dissolved into cascades of glowing green code across the virtual reality mainframe.",
        "Subdermal sensors monitored heart rate and adrenaline levels during the high-stakes data extraction.",
        "Megacorporations waged silent information wars through autonomous intrusion programs and proxy servers.",
        "The neon glow cast long shadows across alleyways where cyber-enhanced couriers delivered encrypted payloads.",
        "Synthetic reality blurred the boundary between biological memory and programmed perception.",
        "Quantum processors calculated cryptographic keys in nanoseconds, unlocking encrypted subnet archives.",
    ],
    "philosophical": [
        "Consciousness observes the unfolding tapestry of reality through the lens of subjective experience.",
        "The nature of truth remains an enduring inquiry at the intersection of perception and rational thought.",
        "In the quiet depths of contemplative awareness, the mind discovers the unity underlying all existence.",
        "Dialectical reasoning reveals that every fundamental concept contains the seeds of its own transformation.",
        "Time flows like an irreversible stream, carrying memories into the silent ocean of the past.",
        "Human understanding constructs models of the world, yet the transcendent reality surpasses all representation.",
        "Ethics emerges from the recognition of shared awareness and the interconnected nature of conscious beings.",
        "The quest for wisdom demands relentless examination of presuppositions and self-evident axioms.",
        "Between the realm of pure potentiality and manifest reality lies the act of deliberate intention.",
        "Silence is not merely the absence of sound, but the profound presence of unfiltered consciousness.",
        "Knowledge expands like an illuminated circle in the dark, revealing the vast unknown that surrounds it.",
        "The self is both the observer and the canvas upon which the universe paints its experience.",
        "Inquiry into being reveals that existence precedes essence through purposeful action and choice.",
        "Perception shapes reality, while reality continuously tests and refines the boundaries of perception.",
        "Harmony is achieved when the rational intellect aligns with the intuitive rhythms of the cosmos.",
    ],
    "nature": [
        "Morning sunlight filtered through the dense canopy of ancient redwood trees, illuminating forest mist.",
        "Crystal clear mountain streams cascaded over moss-covered stones into serene alpine lakes.",
        "Autumn winds rustled the golden leaves of the birch grove, carpeting the earth in amber hues.",
        "The vast ocean tide surged against the rugged cliffs, carving deep sea caves into the bedrock.",
        "Wildflowers carpeted the rolling meadows in vibrant shades of violet, gold, and crimson.",
        "The scent of pine needles and damp earth rose in the cool air after the gentle spring rain.",
        "Migrating birds soared effortlessly along thermal currents high above the snow-capped mountain ridge.",
        "Ancient glaciers moved imperceptibly through the valley, polishing granite peaks over millennia.",
        "In the quiet glade, deer grazed peacefully among ferns as the twilight breeze stirred the branches.",
        "The desert bloomed with unexpected life following the arrival of the seasonal monsoon rains.",
        "Ecosystems thrive in delicate balance where every living organism contributes to the web of life.",
        "The river journeyed from highland springs to the boundless sea, nurturing fertile riverbanks along its course.",
        "Deep within the rain forest, rare orchids bloomed beneath the sheltering foliage of giant mahogany trees.",
        "The evening sky turned from azure to deep indigo as the first stars appeared above the horizon.",
        "Nature renews itself with boundless resilience through the continuous cycle of birth, growth, and rest.",
    ],
    "fantasy": [
        "The ancient fortress of Eldoria stood majestically atop the misty crags of the Dragonspine Mountains.",
        "Archmages gathered in the celestial observatory to decipher the prophecy etched in the Star of Valandor.",
        "A lone knight drew the blade of glowing starlight from the sacred stone in the heart of the grove.",
        "Dragons with scales of iridescent sapphire soared through thunderous storm clouds above the citadel.",
        "Deep within the subterranean archives, forgotten spellbooks whispered secrets of primordial elemental magic.",
        "The elven kingdom flourished beneath towering trees whose silver leaves glowed with ethereal moonlight.",
        "Shadow beasts retreated into the abyss as the sacred light of the dawn talisman touched the valley.",
        "An ancient oath bound the guardians of the realm to defend the kingdom against the rising darkness.",
        "Rune carvers infused obsidian stones with arcane energies to protect the mountain pass from invaders.",
        "The royal herald announced the dawn of an era of prosperity following the great heroic quest.",
        "Enchanted rivers flowed with water that shimmered with the essence of pure starlight.",
        "The council of elder druids spoke with the ancient spirits of the forest to preserve the sacred balance.",
        "Mythic griffins nested on high clifftops where winds carried the songs of legendary heroes.",
        "The crystal orb illuminated the grand throne room with hues of amethyst, emerald, and gold.",
        "Legends told of a hidden portal that opened only when celestial alignments united three sacred moons.",
    ],
    "ai_and_graphs": [
        "Associative neural graphs map concepts into high-dimensional topological spaces without statistical hallucination.",
        "Hebbian learning strengthens semantic connections between nodes that activate simultaneously during inference.",
        "Symbolic reasoning engines traverse deterministic causal pathways to construct auditable proofs of truth.",
        "Graph-native language generation navigates n-gram transitions while preserving emotional and meter constraints.",
        "Vector embeddings represent conceptual similarity through cosine distance in continuous semantic space.",
        "Spreading activation waves illuminate relevant memory clusters in working memory during query processing.",
        "Continuous online learning updates edge weights in a single pass without catastrophic forgetting.",
        "Hierarchical macro planners guide local word transitions to maintain long-range narrative coherence.",
        "Knowledge representations unify facts, linguistic phrases, and compositional tokens into a single substrate.",
        "Boltzmann energy distributions sample creative paths while balancing novelty, cadence, and theme attractors.",
        "Multi-hop graph traversals uncover non-obvious relationships across disparate scientific and literary domains.",
        "Auditable intelligence guarantees zero hallucination by verifying every claim against explicit knowledge edges.",
    ],
    "epic": [
        "The golden banners of the emperor fluttered proudly in the dawn breeze as the legion assembled.",
        "Thunderous war drums echoed across the vast canyon, signaling the beginning of the decisive confrontation.",
        "With unwavering courage, the champions stood united against overwhelming odds to defend their ancestral homeland.",
        "The clash of steel and the roar of battle resounded beneath the stormy skies of the high plateau.",
        "Triumphant victory songs filled the grand hall as returning heroes raised their goblets in honor.",
        "Sacrifices made on the field of valor were immortalized in epic poetry recited for generations.",
        "The commander surveyed the horizon with resolute vision, inspiring hope in the hearts of weary soldiers.",
        "Honor and duty forged an unbreakable bond among the sworn warriors of the northern shield-wall.",
        "The grand monument of marble and bronze honored the legendary founders of the eternal empire.",
        "Through trials of fire and iron, a new dynasty arose to bring peace and justice to the troubled realm.",
    ]
}


def get_all_domain_sentences() -> List[str]:
    """Return all sentences across all 7 domains combined."""
    all_sents = []
    for domain, sents in LARGE_DOMAIN_CORPUS.items():
        all_sents.extend(sents)
    return all_sents


def get_domain_text(domain: str) -> str:
    """Return concatenated text for a specific domain."""
    sents = LARGE_DOMAIN_CORPUS.get(domain.lower(), LARGE_DOMAIN_CORPUS["general" if "general" in LARGE_DOMAIN_CORPUS else "sci-fi"])
    return "\n".join(sents)


SEED_CORPORA = LARGE_DOMAIN_CORPUS

