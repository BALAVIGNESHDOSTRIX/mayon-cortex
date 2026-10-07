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
Bootstrap — Initial Brain Population
=======================================
Feeds the brain its first knowledge and language patterns.
Reuses existing mayon-graph loaders for knowledge, then adds
language pattern sentences for fluency.

Usage:
    from mayon_cortex.learning.bootstrap import bootstrap_brain
    brain = CortexGraph()
    bootstrap_brain(brain, verbose=True)
"""

from typing import Optional

from mayon_cortex.core.config import CortexConfig


# ── Fluent language pattern sentences per sector ──
# These provide the initial language patterns (phrase nodes)
# so the brain can express facts fluently from day 1.
BOOTSTRAP_LANGUAGE = {

    # ── Everyday Life & Common Sense ──
    "general": [
        "When making important decisions, it is essential to consider all relevant factors carefully.",
        "Safety considerations should always take priority over convenience or cost.",
        "Cross-referencing information from multiple sources improves the reliability of conclusions.",
        "Understanding the relationships between concepts is fundamental to effective reasoning.",
        "Structured decision-making involves identifying options, evaluating trade-offs, and selecting the best course of action.",
        "Drinking enough water throughout the day is essential for maintaining good health and energy levels.",
        "A balanced diet includes a variety of fruits, vegetables, proteins, and whole grains.",
        "Regular exercise improves both physical health and mental well-being significantly.",
        "Getting seven to eight hours of quality sleep helps the body recover and the mind stay sharp.",
        "Kindness and empathy strengthen relationships and build a sense of community.",
        "Learning a new skill takes practice, patience, and consistent effort over time.",
        "Reading books expands vocabulary, improves focus, and deepens understanding of the world.",
        "Effective communication involves listening carefully and expressing thoughts clearly.",
        "Time management helps people balance work, personal life, and leisure activities.",
        "Cooking at home is generally healthier and more affordable than eating out regularly.",
        "Maintaining personal hygiene is important for preventing the spread of illness.",
        "Saving money regularly provides financial security and freedom for future needs.",
        "Travel broadens perspectives by exposing people to different cultures and ways of life.",
        "Mistakes are natural and serve as valuable learning opportunities when reflected upon.",
        "Helping others creates a positive ripple effect that benefits entire communities.",
    ],

    # ── Nature & Environment ──
    "science": [
        "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "DNA contains the genetic instructions used in the development of all living organisms.",
        "Gravity is the fundamental force of attraction between objects with mass.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
        "Trees absorb carbon dioxide and release oxygen, making them vital for life on Earth.",
        "The Earth revolves around the Sun once every 365.25 days, which defines a year.",
        "The Moon influences ocean tides through its gravitational pull on Earth's water.",
        "Rainforests are home to more than half of the world's plant and animal species.",
        "Earthquakes occur when tectonic plates beneath the Earth's surface shift and collide.",
        "The human body is composed of approximately 60 percent water.",
        "Sound travels faster through water than through air because water molecules are closer together.",
        "Lightning is a massive electrical discharge caused by imbalanced charges in storm clouds.",
        "Volcanoes form at tectonic plate boundaries where magma rises to the Earth's surface.",
        "The oceans cover approximately 71 percent of the Earth's surface and contain 97 percent of its water.",
        "Seasons change because the Earth is tilted at 23.5 degrees on its axis as it orbits the Sun.",
        "The human brain contains roughly 86 billion neurons connected by trillions of synapses.",
        "Stars produce energy through nuclear fusion, converting hydrogen into helium in their cores.",
        "Evolution is the process by which species change over generations through natural selection.",
        "Ice is less dense than liquid water, which is why it floats on the surface of lakes and oceans.",
        "Cells are the basic building blocks of all living organisms on Earth.",
        "The speed of sound in dry air at 20 degrees Celsius is approximately 343 meters per second.",
        "Bees play a crucial role in pollinating flowers, which enables plants to reproduce.",
        "The Amazon River is the largest river in the world by volume of water flow.",
        "Diamonds are formed from carbon under extreme heat and pressure deep within the Earth.",
    ],

    # ── Human Body & Health ──
    "medical": [
        "Lisinopril is commonly prescribed as a first-line treatment for hypertension.",
        "Metformin is the gold-standard first-line medication for type 2 diabetes mellitus.",
        "Warfarin is an anticoagulant prescribed for the prevention of blood clots.",
        "Aspirin is widely used as both a pain reliever and an anti-platelet agent.",
        "Hypertension is a chronic cardiovascular condition characterized by persistently elevated blood pressure.",
        "Type 2 diabetes is a metabolic disorder that impairs insulin sensitivity and glucose regulation.",
        "Drug interactions can lead to serious adverse effects and must be carefully evaluated.",
        "ACE inhibitors work by blocking the angiotensin-converting enzyme to reduce blood pressure.",
        "The human heart pumps approximately five liters of blood through the body every minute.",
        "The lungs are responsible for exchanging oxygen and carbon dioxide during the process of breathing.",
        "Bones provide structural support, protect internal organs, and produce blood cells in the marrow.",
        "The digestive system breaks down food into nutrients that the body uses for energy and growth.",
        "Vitamins and minerals are essential micronutrients that the body needs in small quantities.",
        "Fever is a natural immune response that helps the body fight infections by raising its temperature.",
        "Antibiotics are effective against bacterial infections but do not work against viral infections.",
        "Vaccines train the immune system to recognize and fight specific diseases before they cause illness.",
        "The skin is the largest organ of the human body and serves as a protective barrier against the environment.",
        "Muscles work in pairs to produce movement, with one contracting while the other relaxes.",
        "Stretching before exercise helps prevent injuries and improves flexibility over time.",
        "Mental health is just as important as physical health, and both require regular attention and care.",
    ],

    # ── Food & Cooking ──
    "general_food": [
        "Rice is a staple food for more than half of the world's population.",
        "Boiling vegetables for too long can destroy essential vitamins and nutrients.",
        "Salt enhances the flavor of food by suppressing bitterness and amplifying other tastes.",
        "Bread is made by mixing flour, water, yeast, and salt, then baking the dough in an oven.",
        "Spices like turmeric, cumin, and coriander are widely used in Indian cooking for both flavor and health benefits.",
        "Fermentation is a natural process used to make foods like yogurt, cheese, kimchi, and bread.",
        "Fruits contain natural sugars, fiber, and vitamins that provide energy and support overall health.",
        "Coffee is one of the most widely consumed beverages in the world and contains caffeine.",
        "Chocolate is made from roasted and ground cacao beans, which originate from tropical regions.",
        "Olive oil is a healthy fat commonly used in Mediterranean cooking for frying and salads.",
    ],

    # ── History & Culture ──
    "general_history": [
        "The wheel was one of the most important inventions in human history, enabling transportation and trade.",
        "Ancient Egypt built the pyramids as tombs for pharaohs using massive stone blocks.",
        "The printing press, invented by Gutenberg around 1440, revolutionized the spread of knowledge.",
        "The Industrial Revolution transformed economies from agricultural to manufacturing-based systems.",
        "World War Two was the deadliest conflict in human history, lasting from 1939 to 1945.",
        "The Renaissance was a period of cultural rebirth in Europe that emphasized art, science, and learning.",
        "Democracy originated in ancient Athens, where citizens participated directly in government decisions.",
        "The internet was developed in the late 20th century and fundamentally changed how people communicate.",
        "Music has been a part of human culture for thousands of years, used for celebration, worship, and expression.",
        "Language is the primary tool humans use to share ideas, emotions, and knowledge with one another.",
    ],

    # ── Technology & Computing ──
    "code": [
        "A Python list comprehension provides a concise way to create and transform lists.",
        "Dictionaries in Python store data as key-value pairs with constant-time lookup.",
        "Functions are defined using the def keyword and can accept parameters and return values.",
        "Exception handling in Python uses try-except blocks to gracefully manage runtime errors.",
        "Object-oriented programming organizes code around classes that encapsulate data and behavior.",
        "Computers process information using binary digits, which are represented as zeros and ones.",
        "The internet connects billions of devices worldwide, enabling instant communication and data sharing.",
        "Artificial intelligence enables machines to learn from data and make predictions or decisions.",
        "A database stores organized collections of data that can be accessed and managed efficiently.",
        "Smartphones combine the functionality of a phone, camera, computer, and navigation system.",
        "Cloud computing allows users to access storage and processing power over the internet.",
        "Cybersecurity protects systems and data from unauthorized access, theft, and damage.",
        "Machine learning is a subset of AI where algorithms improve automatically through experience.",
        "An operating system manages computer hardware and provides services for running applications.",
        "Version control systems like Git track changes in code and enable collaboration among developers.",
    ],

    # ── Emotions & Relationships ──
    "general_emotions": [
        "Happiness is an emotion that arises from positive experiences, achievements, or connections with others.",
        "Sadness is a natural response to loss, disappointment, or difficult situations that everyone experiences.",
        "Anger is a powerful emotion that signals when something feels unfair or threatening.",
        "Fear is the body's natural response to perceived danger, triggering a fight-or-flight reaction.",
        "Love is a deep feeling of affection and care that forms the foundation of meaningful relationships.",
        "Trust is built gradually through consistent honesty, reliability, and mutual respect.",
        "Loneliness can affect both mental and physical health, making social connection essential.",
        "Gratitude is the practice of recognizing and appreciating the good things in life.",
        "Forgiveness is the process of letting go of resentment, which brings peace to both parties.",
        "Empathy is the ability to understand and share the feelings of another person.",
    ],

    # ── Geography & Weather ──
    "general_geo": [
        "Mountains are formed over millions of years through tectonic plate collisions that push rock upward.",
        "Rivers carry water from high elevations to the sea, shaping the landscape along their path.",
        "Deserts receive very little rainfall and are characterized by extreme temperatures and sparse vegetation.",
        "Clouds form when water vapor in the atmosphere cools and condenses into tiny water droplets.",
        "Wind is caused by differences in atmospheric pressure, flowing from high-pressure to low-pressure areas.",
        "Snow forms when water vapor freezes into ice crystals in clouds at temperatures below freezing.",
        "The equator is an imaginary line around the middle of the Earth, dividing it into north and south.",
        "Islands are landmasses surrounded by water, ranging from tiny atolls to large continental islands.",
        "Glaciers are massive bodies of ice that move slowly over land, carving valleys and shaping terrain.",
        "The atmosphere is a layer of gases surrounding the Earth that protects life from solar radiation.",
    ],

    # ── Math & Logic ──
    "math": [
        "Addition combines two or more numbers to find their total sum.",
        "Multiplication is repeated addition, making it faster to calculate large quantities.",
        "A circle is a shape where every point on its edge is the same distance from the center.",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "Probability measures how likely an event is to occur, expressed as a number between zero and one.",
        "An average is calculated by adding all values together and dividing by the number of values.",
        "Geometry studies the properties and relationships of shapes, angles, and spatial figures.",
        "Algebra uses letters and symbols to represent numbers and express mathematical relationships.",
        "Pi is approximately 3.14159 and represents the ratio of a circle's circumference to its diameter.",
        "Statistics is the science of collecting, analyzing, and interpreting numerical data.",
    ],

    # ── Finance & Economics ──
    "finance": [
        "Central bank interest rate decisions directly influence commercial lending rates and monetary supply.",
        "Inflation measures the rate at which the general price level of goods and services increases over time.",
        "Generic medications are significantly more affordable than their brand-name equivalents.",
        "Insurance formularies determine which medications and treatments are covered under a health plan.",
        "A bond is a fixed-income financial instrument representing a loan made by an investor to a borrower.",
        "Saving money in a bank account earns interest, allowing savings to grow over time.",
        "A budget helps track income and expenses, ensuring money is spent wisely.",
        "Taxes fund public services like roads, schools, hospitals, and emergency services.",
        "Supply and demand determine the price of goods in a free market economy.",
        "Investing involves putting money into assets like stocks or property with the goal of earning a return.",
    ],

    # ── Sports & Fitness ──
    "general_sports": [
        "Football is the most popular sport in the world, played and watched by billions of people.",
        "Running is one of the simplest forms of exercise and improves cardiovascular health.",
        "Swimming works nearly every muscle in the body and is gentle on the joints.",
        "Cricket is a bat-and-ball game widely played in countries like India, Australia, and England.",
        "Yoga combines physical postures, breathing exercises, and meditation for overall well-being.",
        "Team sports teach cooperation, discipline, and the importance of working toward a common goal.",
        "Stretching after exercise helps muscles recover and reduces soreness the following day.",
        "Basketball requires agility, speed, and coordination to dribble, pass, and shoot effectively.",
        "Walking for thirty minutes a day can significantly improve heart health and reduce stress.",
        "Chess is a strategic board game that sharpens critical thinking and problem-solving skills.",
    ],

    # ── Animals & Wildlife ──
    "general_animals": [
        "Dogs are loyal companions that have been domesticated by humans for thousands of years.",
        "Cats are independent animals known for their agility, curiosity, and grooming habits.",
        "Elephants are the largest land animals and are known for their intelligence and strong social bonds.",
        "Birds are the only living animals with feathers, and most species are capable of flight.",
        "Fish breathe through gills, which extract dissolved oxygen from water as it flows over them.",
        "Ants live in highly organized colonies where each member has a specific role and responsibility.",
        "Dolphins are intelligent marine mammals that communicate using clicks, whistles, and body language.",
        "Butterflies undergo metamorphosis, transforming from caterpillars into winged adults.",
        "Owls are nocturnal predators with excellent night vision and nearly silent flight.",
        "Whales are the largest animals on Earth, with blue whales reaching up to 30 meters in length.",
    ],

    # ── Art & Music ──
    "general_art": [
        "Painting is a form of visual expression that uses colors and brushstrokes on a canvas or surface.",
        "Music is composed of melody, harmony, and rhythm, creating sounds that evoke emotion.",
        "Dance is a physical art form that expresses ideas and emotions through movement and gesture.",
        "Photography captures moments in time using light and a camera, preserving memories visually.",
        "Sculpture creates three-dimensional art by carving, molding, or assembling materials.",
        "Poetry uses language rhythmically and imaginatively to express feelings and ideas.",
        "Film combines visual storytelling, dialogue, sound, and music to create immersive narratives.",
        "Architecture designs buildings and spaces that are functional, safe, and aesthetically pleasing.",
        "Theater brings stories to life through live performance, combining acting, sets, and dialogue.",
        "Drawing is the foundation of visual art, using lines and shading to represent objects and scenes.",
    ],

    # ── Poetic & Lyrical Expressions (Pillar 1 Rhyme & Meter) ──
    "poetry": [
        "The golden light of morning shines through silent skies so bright.",
        "Across the endless ocean waves the ancient stars bring light.",
        "In quiet sleep the dreaming heart will find its peace so deep.",
        "Through stormy winds and darkest cold the brave their honor hold.",
        "A burning fire of pure desire will lift the spirit higher.",
        "The flowing stream reflects the dream in every silver gleam.",
        "Beneath the skies the wise will rise where gentle silence lies.",
        "Through gentle rain and quiet pain new hope and joy remain.",
        "With noble heart and sacred art true heroes play their part.",
        "Across all time the mountains climb in rhythm and in rhyme.",
        "The peaceful soul attains its goal where sacred waters roll.",
        "The whisper of the wandering wind leaves sorrow far behind.",
        "A shining star from realms afar heals every battle scar.",
        "In timeless grace across all space the silent shadows race.",
        "The glowing moon will whisper soon beneath the desert dune.",
        "With courage true and spirit new the brave will journey through.",
        "The fiery glow begins to grow where sacred rivers flow.",
    ],

    # ── Legal (kept minimal but present) ──
    "legal": [
        "Laws are rules established by governments to maintain order and protect the rights of citizens.",
        "A contract is a legally binding agreement between two or more parties.",
        "Every person has fundamental rights including the right to life, liberty, and equal treatment.",
        "Intellectual property protects creative works such as inventions, designs, and written content.",
        "Privacy laws regulate how personal information is collected, stored, and used by organizations.",
    ],
}


def bootstrap_brain(brain, verbose: bool = True):
    """
    Feed the brain its first knowledge and language patterns.

    Steps:
    1. Load sector knowledge from existing mayon-graph loaders (if available)
    2. Ingest fluent language pattern sentences
    3. Run Phase 1 + Phase 2 curriculum
    """
    if verbose:
        print("[*] Bootstrapping Cortex-Graph Brain...")
        print(f"   Graph: {brain.graph.num_nodes} nodes, {brain.graph.num_edges} edges")

    # Step 1: Try to load sector knowledge from mayon-graph loaders
    try:
        from mayon_cortex.core.loader import (
            PythonDocsLoader,
            MultiSectorKnowledgeLoader,
            populate_mayon_from_knowledge,
        )

        if verbose:
            print("\n[*] Loading domain knowledge...")

        # Python docs
        python_loader = PythonDocsLoader(brain.embedder)
        python_facts = python_loader.extract_all()

        # Multi-sector knowledge
        sector_loader = MultiSectorKnowledgeLoader(brain.embedder)
        sector_facts = sector_loader.get_all_sector_knowledge()

        all_facts = python_facts + sector_facts
        if all_facts:
            populate_mayon_from_knowledge(
                brain.graph, all_facts, brain.embedder, verbose=False
            )

    except ImportError:
        if verbose:
            print("   [!] mayon-graph loaders not available, using bootstrap sentences only")
    except Exception as e:
        if verbose:
            print(f"   [!] Loader error: {e}, using bootstrap sentences only")

    # Step 2: Ingest fluent language patterns
    if verbose:
        print("\n[*] Ingesting language patterns...")

    total_ingested = 0
    for sector, sentences in BOOTSTRAP_LANGUAGE.items():
        for sentence in sentences:
            stats = brain.ingest(sentence, sector=sector)
            total_ingested += stats.knowledge_nodes_added + stats.phrase_nodes_added

    if verbose:
        print(f"   [+] Ingested {total_ingested} nodes from {sum(len(v) for v in BOOTSTRAP_LANGUAGE.values())} sentences")

    # Step 3: Run curriculum (Phase 1 + 2)
    if verbose:
        print("\n[*] Running curriculum (Phase 1: Infant + Phase 2: Toddler)...")

    try:
        brain.teach(phase=1)
        if verbose:
            print("   [+] Phase 1 (Infant categories) complete")

        brain.teach(phase=2)
        if verbose:
            print("   [+] Phase 2 (Toddler domain foundations) complete")
    except Exception as e:
        if verbose:
            print(f"   [!] Curriculum error: {e}")

    if verbose:
        print(f"\n[OK] Brain bootstrapped!")
        print(f"   {brain.graph.num_nodes} nodes, {brain.graph.num_edges} edges")
        print(f"   Sectors: {list(brain.graph.sector_indices.keys())}")
        print(f"   Ready for queries.\n")


def bootstrap_minimal(brain, verbose: bool = True):
    """
    Minimal bootstrap — just language patterns, no external loaders.
    Fastest way to get a working brain.
    """
    if verbose:
        print("[*] Minimal bootstrap...")

    for sector, sentences in BOOTSTRAP_LANGUAGE.items():
        for sentence in sentences:
            brain.ingest(sentence, sector=sector)

    if verbose:
        print(f"   [+] {brain.graph.num_nodes} nodes, {brain.graph.num_edges} edges")
        print("   Ready.\n")



def bootstrap_large_corpus(brain, verbose: bool = True):
    """
    Large multi-domain corpus bootstrap — loads high-density narrative,
    sci-fi, cyberpunk, philosophical, fantasy, nature, and AI domains.
    Supercharges WordGraph transitions and ConceptGraph reasoning for GPT-scale prose.
    """
    from mayon_cortex.language.corpus_data import LARGE_DOMAIN_CORPUS

    if verbose:
        print("[+] Ingesting Large Multi-Domain Corpus...")

    total_sentences = 0
    total_words = 0
    for domain, sentences in LARGE_DOMAIN_CORPUS.items():
        for sentence in sentences:
            brain.ingest(sentence, sector=domain)
            total_sentences += 1
            total_words += len(sentence.split())

    if verbose:
        print(f"   [OK] Ingested {total_sentences} multi-domain sentences (~{total_words} words)")
        print(f"   [OK] WordGraph now contains {len(brain.word_graph.nodes)} unique word nodes")
        print(f"   [OK] KnowledgeGraph now contains {brain.graph.num_nodes} concept nodes, {brain.graph.num_edges} edges\n")


# Aliases
bootstrap_all_sectors = bootstrap_brain


class KnowledgeBootstrapper:
    """Helper class to bootstrap graph memory and seed corpora."""
    @staticmethod
    def bootstrap(brain, verbose: bool = True):
        return bootstrap_brain(brain, verbose=verbose)

    @staticmethod
    def bootstrap_large_corpus(brain, verbose: bool = True):
        return bootstrap_large_corpus(brain, verbose=verbose)



