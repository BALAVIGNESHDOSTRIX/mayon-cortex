#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# System: Mayon-Cortex Cognitive Architecture
# ==============================================================================

"""
train_fluency_and_vision.py
============================
Complete training pipeline for Mayon-Cortex fluency and vision upgrades.

What this script does:
  1. Downloads real English text from internet (Wikipedia, Project Gutenberg)
  2. Downloads real reference images (via public URLs)
  3. Ingests all text into MayonGraph (brain learns by understanding)
  4. Trains GrammarPatternInducer on ingested sentences
  5. Trains BrocaDecoder (self-supervised next-token prediction)
  6. Tests DiscourseCoherenceTracker on sample outputs
  7. Tests SemanticSceneMapper + RaymarchRenderer → saves rendered images
  8. Runs before/after fluency comparison
  9. Saves all model weights to cortex_storage/

Usage:
  python scripts/train_fluency_and_vision.py
  python scripts/train_fluency_and_vision.py --epochs 10 --text-only
  python scripts/train_fluency_and_vision.py --quick   # fast test run
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple

# ── Add project root to path ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Fix Windows console encoding with immediate unbuffered flushing
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True, write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True, write_through=True)

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data Sources
# ─────────────────────────────────────────────────────────────────────────────

WIKIPEDIA_ARTICLES = [
    # Science
    "https://en.wikipedia.org/w/index.php?action=raw&title=Photosynthesis",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Nervous_system",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Evolution",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Quantum_mechanics",
    "https://en.wikipedia.org/w/index.php?action=raw&title=DNA",
    # Medicine
    "https://en.wikipedia.org/w/index.php?action=raw&title=Aspirin",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Hypertension",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Diabetes_mellitus",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Cancer",
    # Technology
    "https://en.wikipedia.org/w/index.php?action=raw&title=Artificial_intelligence",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Machine_learning",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Neural_network",
    # History
    "https://en.wikipedia.org/w/index.php?action=raw&title=World_War_II",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Renaissance",
    # General
    "https://en.wikipedia.org/w/index.php?action=raw&title=Philosophy",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Mathematics",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Physics",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Chemistry",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Biology",
    "https://en.wikipedia.org/w/index.php?action=raw&title=Economics",
]

# Project Gutenberg plain text books (public domain)
GUTENBERG_BOOKS = [
    # Pride and Prejudice — Jane Austen
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    # The Adventures of Sherlock Holmes — Arthur Conan Doyle
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    # Moby Dick — Herman Melville
    "https://www.gutenberg.org/files/2701/2701-0.txt",
]

# Public domain / CC-0 reference images
IMAGE_URLS = [
    ("cat",    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=320"),
    ("dog",    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=320"),
    ("bird",   "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=320"),
    ("flower", "https://images.unsplash.com/photo-1560717789-0ac7c58ac90a?w=320"),
    ("tree",   "https://images.unsplash.com/photo-1448375240586-882707db888b?w=320"),
    ("car",    "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=320"),
    ("house",  "https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=320"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def download_text(url: str, timeout: int = 15) -> Optional[str]:
    """Download text from URL. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MayonCortex/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  [WARN] Failed to download {url[:60]}... : {e}")
        return None


def download_image_bytes(url: str, timeout: int = 15) -> Optional[bytes]:
    """Download image bytes from URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"  [WARN] Failed to download image {url[:60]}... : {e}")
        return None


def clean_wikipedia_text(raw: str) -> str:
    """Strip Wikipedia markup and return clean plain text."""
    import re
    # Remove templates {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", " ", raw)
    # Remove wiki links [[File:...]] and [[Image:...]]
    text = re.sub(r"\[\[(?:File|Image|Media):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Unwrap wiki links [[label|text]] → text or [[text]] → text
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove reference markers [1], [2], etc.
    text = re.sub(r"\[\d+\]", " ", text)
    # Remove section headers == Section ==
    text = re.sub(r"={2,}[^=]+=+", " ", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,}", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sentences(text: str, min_words: int = 6, max_words: int = 50) -> List[str]:
    """Split text into clean sentences."""
    import re
    # Split on sentence boundaries
    raw_sents = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw_sents:
        s = s.strip()
        word_count = len(s.split())
        if min_words <= word_count <= max_words:
            # Basic quality filter: must contain a verb-like word
            if any(c.isalpha() for c in s):
                sentences.append(s)
    return sentences


def print_header(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_section(title: str):
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Download Text Data
# ──────────────────────────────────────────────────────────────────────────────

def download_all_text(quick: bool = False) -> List[str]:
    """Download and clean text from Wikipedia and Gutenberg."""
    print_section("Step 1: Downloading Text Data")

    all_sentences: List[str] = []
    sources = WIKIPEDIA_ARTICLES[:5] if quick else WIKIPEDIA_ARTICLES
    books = GUTENBERG_BOOKS[:1] if quick else GUTENBERG_BOOKS

    # Wikipedia
    print(f"  Fetching {len(sources)} Wikipedia articles...")
    for url in sources:
        title = url.split("title=")[-1].replace("_", " ")
        raw = download_text(url)
        if raw:
            clean = clean_wikipedia_text(raw)
            sents = extract_sentences(clean)
            all_sentences.extend(sents)
            print(f"  ✓ {title}: {len(sents)} sentences")
        time.sleep(0.3)  # Be polite

    # Gutenberg books
    print(f"\n  Fetching {len(books)} Gutenberg books...")
    for url in books:
        raw = download_text(url)
        if raw:
            # For Gutenberg, skip header/footer
            lines = raw.split("\n")
            start = next((i for i, l in enumerate(lines) if "*** START" in l.upper()), 0)
            end = next((i for i, l in enumerate(lines) if "*** END" in l.upper()), len(lines))
            body = "\n".join(lines[start + 1:end])
            sents = extract_sentences(body)
            # Limit per book to avoid dominance
            sents = sents[:500 if quick else 2000]
            all_sentences.extend(sents)
            print(f"  ✓ Gutenberg book: {len(sents)} sentences")
        time.sleep(0.5)

    # Deduplicate
    all_sentences = list(dict.fromkeys(all_sentences))
    print(f"\n  Total sentences collected: {len(all_sentences)}")
    return all_sentences


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Ingest Text into MayonGraph
# ──────────────────────────────────────────────────────────────────────────────

def ingest_into_brain(brain, sentences: List[str], quick: bool = False) -> None:
    """Ingest all sentences into the Cortex brain."""
    print_section("Step 2: Ingesting Text into MayonGraph")

    ingest_sents = sentences[:200] if quick else sentences
    print(f"  Ingesting {len(ingest_sents)} sentences into brain...")

    batch_size = 50
    for i in range(0, len(ingest_sents), batch_size):
        batch = ingest_sents[i:i + batch_size]
        for sent in batch:
            try:
                brain.ingest(sent)
            except Exception:
                pass
        pct = min(100, int((i + batch_size) / len(ingest_sents) * 100))
        print(f"  [{pct:3d}%] Ingested {min(i + batch_size, len(ingest_sents))}/{len(ingest_sents)} sentences", end="\r")

    print(f"\n  ✓ Brain graph: {brain.graph.num_nodes} nodes, {len(brain.graph.edges)} edges")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Train Grammar Inducer
# ──────────────────────────────────────────────────────────────────────────────

def train_grammar_inducer(sentences: List[str]) -> "GrammarPatternInducer":
    """Train the GrammarPatternInducer on all sentences."""
    print_section("Step 3: Grammar Pattern Induction")

    from mayon_cortex.language.grammar_inducer import GrammarPatternInducer

    inducer = GrammarPatternInducer(min_pattern_freq=2, max_pattern_len=8)
    t0 = time.time()
    stats = inducer.induce_from_sentences(sentences)
    elapsed = time.time() - t0

    print(f"  Sentences processed: {stats.sentences_processed}")
    print(f"  Unique patterns found: {stats.unique_patterns}")
    print(f"  Avg sentence length: {stats.avg_sentence_length:.1f} words")
    print(f"  Top POS frequencies: {dict(list(stats.pos_frequency.items())[:6])}")
    print(f"  Training time: {elapsed:.2f}s")
    print(f"\n  Top 5 patterns:")
    for p in stats.top_patterns[:5]:
        ex = f' e.g. "{p.examples[0][:60]}"' if p.examples else ""
        print(f"    [{p.frequency:4d}x] {p.pattern_str}{ex}")

    # Test grammar validation
    good = "The cat sat on the mat."
    bad = "sat cat the mat on the"
    is_good, score_good = inducer.is_grammatical(good)
    is_bad, score_bad = inducer.is_grammatical(bad)
    print(f"\n  Grammar validation test:")
    print(f"    \"{good}\" → {'✓ VALID' if is_good else '✗ INVALID'} (score: {score_good:.2f})")
    print(f"    \"{bad}\" → {'✓ VALID' if is_bad else '✗ INVALID'} (score: {score_bad:.2f})")

    return inducer


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Train BrocaDecoder
# ──────────────────────────────────────────────────────────────────────────────

def train_broca(brain, sentences: List[str], epochs: int = 5, save_dir: str = "cortex_storage/broca", quick: bool = False) -> None:
    """Train the BrocaDecoder self-supervised on ingested text."""
    print_section("Step 4: BrocaDecoder Self-Supervised Training")

    from mayon_cortex.language.broca import BrocaDecoder, BrocaConfig
    from mayon_cortex.language.broca_trainer import BrocaSelfTrainer

    train_sents = sentences[:400] if quick else sentences
    print(f"  Sentences available: {len(train_sents)} (of {len(sentences)})")
    print(f"  Epochs: {epochs}")

    # Before training — generate sample (random weights)
    trainer = BrocaSelfTrainer(
        seq_len=48 if quick else 64,
        batch_size=16,
        epochs=epochs,
        lr=5e-4 if quick else 3e-4,
        max_vocab=4096 if quick else 8192,
        device="cpu",
        verbose=True,
    )

    broca = BrocaDecoder(BrocaConfig())
    print(f"  BrocaDecoder size: {broca.param_size_mb():.2f} MB")

    # Before training sample
    trainer.vocab.learn(train_sents)
    trainer.vocab.build()
    pre_sample = trainer.generate_sample(broca, prompt="the", max_tokens=20)
    print(f"\n  BEFORE training: \"{pre_sample}\"")

    # Train
    print(f"\n  Starting training...")
    stats = trainer.train_from_sentences(
        sentences=train_sents,
        broca_decoder=broca,
        save_path=save_dir,
    )

    # After training sample
    test_prompts = ["the brain", "aspirin", "machine learning", "the cat"]
    print(f"\n  AFTER training samples:")
    for prompt in test_prompts:
        sample = trainer.generate_sample(broca, prompt=prompt, max_tokens=25, temperature=0.6)
        print(f"    [{prompt}] → \"{sample}\"")

    print(f"\n  Training Stats:")
    print(f"    Final Loss:       {stats.final_loss:.4f}")
    print(f"    Final Perplexity: {stats.final_perplexity:.2f}")
    print(f"    Vocab Size:       {stats.vocab_size}")
    print(f"    Training Time:    {stats.training_time_sec:.1f}s")
    print(f"    Model saved to:   {save_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Test Discourse Coherence
# ──────────────────────────────────────────────────────────────────────────────

def test_discourse_coherence(brain) -> None:
    """Demonstrate discourse coherence tracker on real brain outputs."""
    print_section("Step 5: Discourse Coherence Tracker Test")

    from mayon_cortex.language.discourse_coherence import DiscourseCoherenceTracker

    tracker = DiscourseCoherenceTracker(window_size=5)

    # Test with sample sentences about aspirin
    test_sentences = [
        "Aspirin is a widely used medication for pain relief.",
        "It belongs to the class of drugs known as NSAIDs.",
        "Aspirin reduces inflammation by inhibiting prostaglandins.",
        "However, it can cause stomach irritation in some patients.",
        "Therefore, it is often taken with food or milk.",
        "Unrelated topic: The Eiffel Tower is located in Paris.",
        "Furthermore, aspirin has blood-thinning properties.",
    ]

    print(f"  Input sentences ({len(test_sentences)}):")
    for i, s in enumerate(test_sentences):
        print(f"    {i+1}. {s}")

    # Build coherent paragraph
    tracker.start_paragraph("aspirin pain relief medication")
    plan = tracker.build_paragraph(test_sentences, max_sentences=5, min_coherence=0.05)

    print(f"\n  Selected {len(plan.sentences)} coherent sentences:")
    for i, (s, c, score) in enumerate(zip(plan.sentences, plan.connectors, plan.coherence_scores)):
        conn_str = f"[{c}]" if c else "[no connector]"
        print(f"    {i+1}. {conn_str} (score: {score:.2f}) {s[:80]}")

    rendered = plan.render()
    print(f"\n  Final paragraph:\n")
    print(f"  \"{rendered}\"")

    # Score coherence
    score = tracker.score_paragraph(rendered)
    print(f"\n  Paragraph coherence score: {score:.3f} / 1.0")

    # Test improve_sentence_flow
    disjointed = [
        "Aspirin treats headaches.",
        "It was developed in 1897.",
        "The drug reduces fever effectively.",
    ]
    improved = tracker.improve_sentence_flow(disjointed)
    print(f"\n  Improved sentence flow:")
    print(f"    Before: {' '.join(disjointed)}")
    print(f"    After:  {improved}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Test Semantic Scene Mapper + Render Images
# ──────────────────────────────────────────────────────────────────────────────

def test_semantic_scene_mapper(output_dir: str, brain=None, quick: bool = False) -> None:
    """Test SemanticSceneMapper (Deprecated — image generation removed)."""
    print_section("Step 6: SemanticSceneMapper (Skipped — image generation modules removed)")
    try:
        from mayon_cortex.perception.semantic_scene_mapper import SemanticSceneMapper
    except ImportError:
        print("  [NOTE] Image generation modules removed for pure neuro-symbolic LLM reasoning.")
        return

    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        print("  [NOTE] PIL not available — images will be saved as numpy arrays only")
        pil_available = False

    mapper = SemanticSceneMapper(cortex_graph=brain.graph if brain else None)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_prompts = [
        ("orange cat on grass",         "orange_cat_grass.png"),
        ("red car on marble floor",      "red_car_marble.png"),
        ("blue robot in space",          "blue_robot_space.png"),
        ("yellow flower on grass",       "yellow_flower_grass.png"),
        ("white house in sky",           "white_house_sky.png"),
        ("green tree on grass",          "green_tree.png"),
        ("golden planet in space",       "golden_planet_space.png"),
    ]

    test_prompts = all_prompts[:3] if quick else all_prompts
    render_res = 96 if quick else 256

    print(f"  Rendering {len(test_prompts)} scenes ({render_res}×{render_res}, pure CPU raymarcher)...")
    print(f"  Output: {out_path}/\n")

    results = []
    for prompt, filename in test_prompts:
        t0 = time.time()

        # Map semantic prompt → scene spec
        scene = mapper.map(prompt)
        print(f"  '{prompt}'")
        print(f"    → Subject: {scene.subject}, Env: {scene.background_type}, Primitives: {len(scene.primitives)}")

        # Render
        img_array = mapper.render(prompt, width=render_res, height=render_res)
        elapsed = time.time() - t0

        # Save
        fpath = out_path / filename
        if pil_available:
            pil_img = Image.fromarray(img_array.astype(np.uint8), "RGB")
            pil_img.save(str(fpath))
            print(f"    → Saved: {fpath} ({elapsed:.2f}s)")
        else:
            np.save(str(fpath.with_suffix(".npy")), img_array)
            print(f"    → Saved numpy: {fpath.with_suffix('.npy')} ({elapsed:.2f}s)")

        results.append((prompt, filename, elapsed))
        print()

    print(f"  Summary: {len(results)} images rendered")
    for p, f, t in results:
        print(f"    {f:<35} {t:.2f}s")

    print(f"\n  ✓ All images saved to {out_path}/")


# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Download Reference Images and Feed to VisionCortex
# ──────────────────────────────────────────────────────────────────────────────

def download_and_ingest_images(brain, output_dir: str, quick: bool = False) -> None:
    """Download real images and feed to VisionCortex for learning."""
    print_section("Step 7: Downloading & Ingesting Reference Images")

    try:
        from PIL import Image
        import io
    except ImportError:
        print("  [SKIP] PIL not available — skipping image ingestion")
        return

    out_path = Path(output_dir) / "downloaded_images"
    out_path.mkdir(parents=True, exist_ok=True)

    image_sources = IMAGE_URLS[:3] if quick else IMAGE_URLS
    print(f"  Downloading {len(image_sources)} reference images...")

    ingested = 0
    for label, url in image_sources:
        img_bytes = download_image_bytes(url)
        if img_bytes is None:
            continue

        try:
            # Save to disk
            img_path = out_path / f"{label}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # Feed to VisionCortex
            result = brain.vision_cortex.see(
                str(img_path),
                caption=f"A {label}",
                tags=[label],
                graph=brain.graph,
            )
            print(f"  ✓ {label}: learned {result.num_objects if hasattr(result, 'num_objects') else '?'} features")
            ingested += 1

        except Exception as e:
            print(f"  ✗ {label}: {e}")

    print(f"\n  Vision learning complete: {ingested}/{len(image_sources)} images ingested")


# ──────────────────────────────────────────────────────────────────────────────
# Step 8: Before/After Fluency Test
# ──────────────────────────────────────────────────────────────────────────────

def fluency_comparison_test(brain_before_text: str, brain_after, trainer, broca) -> None:
    """Run before/after comparison of brain outputs."""
    print_section("Step 8: Before/After Fluency Comparison")

    from mayon_cortex.language.discourse_coherence import DiscourseCoherenceTracker

    test_queries = [
        "What is aspirin?",
        "What is photosynthesis?",
        "How does the nervous system work?",
        "What is machine learning?",
    ]

    tracker = DiscourseCoherenceTracker()

    print(f"  {'Query':<35} {'Score':>6}")
    print(f"  {'─'*35} {'─'*6}")

    for query in test_queries:
        try:
            result = brain_after.ask(query)
            raw_text = result.text if hasattr(result, "text") else str(result)

            # Apply coherence improvement
            sentences_raw = raw_text.split(". ")
            improved = tracker.improve_sentence_flow(sentences_raw)

            score = tracker.score_paragraph(improved)
            print(f"  {query:<35} {score:>6.3f}")
            print(f"    Output: {improved[:120]}...")
            print()
        except Exception as e:
            print(f"  {query:<35} ERROR: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mayon-Cortex Fluency & Vision Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=5, help="BrocaDecoder training epochs (default: 5)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer articles, fewer epochs")
    parser.add_argument("--text-only", action="store_true", help="Skip image rendering step")
    parser.add_argument("--no-download", action="store_true", help="Skip internet downloads (use cached data)")
    parser.add_argument("--output-dir", type=str, default="cortex_storage/train_output", help="Output directory")
    parser.add_argument("--save-dir", type=str, default="cortex_storage/broca", help="Broca model save directory")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 3)

    print_header("Mayon-Cortex Fluency & Vision Training")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Text-only: {args.text_only}")
    print(f"  Output: {args.output_dir}")

    t_total = time.time()

    # ── Initialize Brain ──
    print_section("Initializing Mayon-Cortex Brain")
    try:
        from mayon_cortex import CortexPipeline
        from mayon_cortex.learning.bootstrap import bootstrap_all_sectors
        brain = CortexPipeline()
        print("  Bootstrapping brain with multi-sector seed knowledge...")
        bootstrap_all_sectors(brain.brain)
        print(f"  Brain ready: {brain.brain.graph.num_nodes} nodes")
    except Exception as e:
        print(f"  ERROR initializing brain: {e}")
        sys.exit(1)

    # ── Step 1: Download / Load Text ──
    cache_path = Path(args.output_dir) / "sentences_cache.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.no_download and cache_path.exists():
        print_section("Loading Cached Text Data")
        with open(cache_path) as f:
            sentences = json.load(f)
        print(f"  Loaded {len(sentences)} cached sentences")
    else:
        sentences = download_all_text(quick=args.quick)
        with open(cache_path, "w") as f:
            json.dump(sentences, f)
        print(f"  Cached sentences to {cache_path}")

    if not sentences:
        print("  [ERROR] No sentences available. Check internet connection.")
        sys.exit(1)

    # ── Step 2: Ingest into Brain ──
    ingest_into_brain(brain.brain, sentences, quick=args.quick)

    # ── Step 3: Grammar Induction ──
    inducer = train_grammar_inducer(sentences)
    # Attach to brain for future use
    brain.brain._grammar_inducer = inducer

    # ── Step 4: Train BrocaDecoder ──
    from mayon_cortex.language.broca import BrocaDecoder, BrocaConfig
    from mayon_cortex.language.broca_trainer import BrocaSelfTrainer

    trainer = BrocaSelfTrainer(
        seq_len=64,
        batch_size=16,
        epochs=args.epochs,
        lr=3e-4,
        max_vocab=8192,
        device="cpu",
        verbose=True,
    )
    broca = BrocaDecoder(BrocaConfig())
    train_broca(brain.brain, sentences, epochs=args.epochs, save_dir=args.save_dir, quick=args.quick)

    # ── Step 5: Discourse Coherence ──
    test_discourse_coherence(brain.brain)

    # ── Step 6: SemanticSceneMapper ──
    if not args.text_only:
        test_semantic_scene_mapper(
            output_dir=str(Path(args.output_dir) / "rendered_images"),
            brain=brain.brain,
            quick=args.quick,
        )
        # ── Step 7: Ingest real images ──
        download_and_ingest_images(
            brain.brain,
            output_dir=args.output_dir,
            quick=args.quick,
        )

    # ── Step 8: Fluency comparison ──
    fluency_comparison_test(
        brain_before_text="[not trained]",
        brain_after=brain,
        trainer=trainer,
        broca=broca,
    )

    # ── Done ──
    total_time = time.time() - t_total
    print_header(f"Training Complete in {total_time:.1f}s")
    print(f"  Sentences trained:   {len(sentences)}")
    print(f"  Brain nodes:         {brain.brain.graph.num_nodes}")
    print(f"  Brain edges:         {len(brain.brain.graph.edges)}")
    print(f"  Broca model saved:   {args.save_dir}/")
    print(f"  Rendered images:     {args.output_dir}/rendered_images/")
    print(f"\n  Run with --quick for a fast test run.")
    print(f"  Run with --no-download to use cached data.")


if __name__ == "__main__":
    main()
