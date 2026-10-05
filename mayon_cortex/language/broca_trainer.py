# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# PROPRIETARY AND CONFIDENTIAL
# ==============================================================================

"""
Broca Self-Trainer — Self-Supervised Articulation Learning
===========================================================
Trains BrocaDecoder on text the brain has already ingested.

Human brain analogy:
  A child learns to SPEAK by hearing language around them.
  They don't need a teacher — just exposure to sentences.
  After enough exposure, speech becomes fluent.

Same here:
  1. Mayon-Cortex ingests text → WordGraph grows
  2. BrocaSelfTrainer extracts sequences from WordGraph
  3. Trains BrocaDecoder on next-word prediction (self-supervised)
  4. Broca learns to articulate graph knowledge fluently
  5. The more text ingested → the more fluent the output

Zero GPU. Zero external data. Pure self-improvement.
"""

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}


class BrocaVocabulary:
    """
    Dynamic vocabulary built from ingested text.
    Grows as new words are seen — like the brain's mental lexicon.
    """

    def __init__(self, max_vocab: int = 8192):
        self.max_vocab = max_vocab
        self.word2idx: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.idx2word: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self._freq: Counter = Counter()
        self._frozen = False

    def learn(self, sentences: List[str]):
        """Count word frequencies from sentences (before freezing)."""
        for sent in sentences:
            for w in self._tokenize(sent):
                self._freq[w] += 1

    def build(self):
        """Freeze vocabulary from top-frequency words."""
        top_words = [w for w, _ in self._freq.most_common(self.max_vocab - len(SPECIAL_TOKENS))]
        for word in top_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self._frozen = True

    def encode(self, sentence: str) -> List[int]:
        """Encode sentence to token IDs."""
        unk = self.word2idx["<UNK>"]
        return [self.word2idx.get(w, unk) for w in self._tokenize(sentence)]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to words."""
        words = []
        for i in ids:
            w = self.idx2word.get(i, "<UNK>")
            if w in ("<BOS>", "<PAD>", "<EOS>"):
                continue
            words.append(w)
        return self._detokenize(words)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = re.sub(r"([.,!?;:\"'()])", r" \1 ", text.lower())
        return [t for t in text.split() if t.strip()]

    @staticmethod
    def _detokenize(words: List[str]) -> str:
        text = " ".join(words)
        # Fix spacing around punctuation
        text = re.sub(r" ([.,!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @property
    def size(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": {str(k): v for k, v in self.idx2word.items()}}, f)

    def load(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self._frozen = True


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BrocaTextDataset(Dataset):
    """
    Sliding window next-token prediction dataset.
    Input: [w1, w2, ..., wN]  Target: [w2, w3, ..., wN+1]
    """

    def __init__(self, sentences: List[str], vocab: BrocaVocabulary, seq_len: int = 64):
        self.vocab = vocab
        self.seq_len = seq_len
        self.samples: List[Tuple[List[int], List[int]]] = []
        self._build(sentences)

    def _build(self, sentences: List[str]):
        bos = self.vocab.word2idx["<BOS>"]
        eos = self.vocab.word2idx["<EOS>"]
        pad = self.vocab.word2idx["<PAD>"]

        for sent in sentences:
            ids = self.vocab.encode(sent)
            if len(ids) < 3:
                continue
            ids = [bos] + ids + [eos]

            # Sliding windows
            for start in range(0, len(ids) - 1, self.seq_len // 2):
                chunk = ids[start: start + self.seq_len + 1]
                if len(chunk) < 4:
                    continue
                inp = chunk[:-1]
                tgt = chunk[1:]
                # Pad to seq_len
                inp = inp + [pad] * (self.seq_len - len(inp))
                tgt = tgt + [pad] * (self.seq_len - len(tgt))
                self.samples.append((inp[:self.seq_len], tgt[:self.seq_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingStats:
    """Statistics from a BrocaDecoder training run."""
    epochs_completed: int = 0
    final_loss: float = 0.0
    final_perplexity: float = 0.0
    sentences_trained: int = 0
    vocab_size: int = 0
    model_size_mb: float = 0.0
    training_time_sec: float = 0.0
    loss_history: List[float] = field(default_factory=list)


class BrocaSelfTrainer:
    """
    Self-supervised trainer for BrocaDecoder.

    Trains entirely on text the brain has already ingested.
    No external data, no GPU required, no backprop through billions of params.

    Workflow:
        trainer = BrocaSelfTrainer()
        trainer.train_from_sentences(sentences, broca_decoder)
        # Now BrocaDecoder produces fluent text!
    """

    def __init__(
        self,
        seq_len: int = 64,
        batch_size: int = 32,
        epochs: int = 5,
        lr: float = 3e-4,
        max_vocab: int = 8192,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_vocab = max_vocab
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.verbose = verbose
        self.vocab = BrocaVocabulary(max_vocab=max_vocab)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[BrocaTrainer] {msg}")

    def extract_sentences_from_word_graph(self, word_graph) -> List[str]:
        """
        Pull all phrase sequences from the existing WordGraph.
        These are sentences already understood by the brain.
        """
        sentences = []
        if word_graph is None:
            return sentences

        # Traverse phrase nodes in the WordGraph
        for node_id, node in word_graph.nodes.items():
            text = getattr(node, "source_text", "")
            if text and len(text.split()) >= 4:
                sentences.append(text)

        self._log(f"Extracted {len(sentences)} sentences from WordGraph")
        return sentences

    def extract_sentences_from_cortex_graph(self, graph) -> List[str]:
        """
        Pull phrase/sentence metadata from MayonGraph nodes.
        """
        sentences = []
        if graph is None:
            return sentences

        for node_id, node in graph.nodes.items():
            meta = getattr(node, "metadata", {}) or {}
            sent = meta.get("sentence", "")
            if sent and len(sent.split()) >= 5:
                sentences.append(sent)
            # Also use source_text directly
            text = getattr(node, "source_text", "")
            if text and len(text.split()) >= 5 and text not in sentences:
                sentences.append(text)

        self._log(f"Extracted {len(sentences)} sentences from CortexGraph")
        return sentences

    def train_from_sentences(
        self,
        sentences: List[str],
        broca_decoder,
        save_path: Optional[str] = None,
    ) -> TrainingStats:
        """
        Train BrocaDecoder on a list of sentences.
        Self-supervised: no labels needed — text IS the training signal.

        Args:
            sentences: List of text sentences (from ingested knowledge)
            broca_decoder: BrocaDecoder instance (with random weights)
            save_path: Optional path to save trained weights

        Returns:
            TrainingStats with loss history and final metrics
        """
        if not sentences:
            self._log("No sentences to train on — ingest some text first.")
            return TrainingStats()

        t_start = time.time()

        # ── 1. Build Vocabulary ──
        self._log(f"Building vocabulary from {len(sentences)} sentences...")
        self.vocab.learn(sentences)
        self.vocab.build()
        self._log(f"Vocabulary size: {self.vocab.size} words")

        # Sync vocabulary size with BrocaDecoder
        actual_vocab = self.vocab.size
        if hasattr(broca_decoder, "cfg"):
            broca_decoder.cfg.vocab_size = actual_vocab

        # Rebuild token embeddings and lm_head to match actual vocab
        orig_dim = broca_decoder.token_emb.embedding_dim
        broca_decoder.token_emb = nn.Embedding(actual_vocab, orig_dim)
        broca_decoder.lm_head = nn.Linear(orig_dim, actual_vocab, bias=False)
        broca_decoder.lm_head.weight = broca_decoder.token_emb.weight
        broca_decoder = broca_decoder.to(self.device)

        # ── 2. Build Dataset ──
        self._log("Building training dataset...")
        dataset = BrocaTextDataset(sentences, self.vocab, seq_len=self.seq_len)
        if len(dataset) == 0:
            self._log("Dataset is empty — sentences may be too short.")
            return TrainingStats()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        self._log(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

        # ── 3. Optimizer + LR Schedule ──
        optimizer = optim.AdamW(broca_decoder.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        pad_idx = self.vocab.word2idx["<PAD>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # ── 4. Training Loop ──
        stats = TrainingStats(sentences_trained=len(sentences), vocab_size=self.vocab.size)
        broca_decoder.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for inp, tgt in loader:
                inp = inp.to(self.device)
                tgt = tgt.to(self.device)

                optimizer.zero_grad()
                logits = broca_decoder(inp)  # (B, T, vocab)
                # Reshape for loss: (B*T, vocab) vs (B*T,)
                loss = criterion(
                    logits.reshape(-1, actual_vocab),
                    tgt.reshape(-1),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(broca_decoder.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            perplexity = math.exp(min(avg_loss, 20))
            scheduler.step()

            stats.loss_history.append(avg_loss)
            self._log(
                f"Epoch {epoch}/{self.epochs} | Loss: {avg_loss:.4f} | "
                f"Perplexity: {perplexity:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # ── 5. Final Stats ──
        stats.epochs_completed = self.epochs
        stats.final_loss = stats.loss_history[-1] if stats.loss_history else 0.0
        stats.final_perplexity = math.exp(min(stats.final_loss, 20))
        stats.model_size_mb = broca_decoder.param_size_mb()
        stats.training_time_sec = time.time() - t_start

        self._log(
            f"\n✅ Training complete in {stats.training_time_sec:.1f}s\n"
            f"   Final Loss: {stats.final_loss:.4f}\n"
            f"   Final Perplexity: {stats.final_perplexity:.2f}\n"
            f"   Model Size: {stats.model_size_mb:.2f} MB"
        )

        # ── 6. Save ──
        if save_path:
            self.save(broca_decoder, save_path)

        return stats

    def generate_sample(
        self,
        broca_decoder,
        prompt: str = "",
        max_tokens: int = 40,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:
        """
        Generate a sample sentence from the trained Broca decoder.
        Used to verify fluency after training.
        """
        broca_decoder.eval()
        bos = self.vocab.word2idx["<BOS>"]
        eos = self.vocab.word2idx["<EOS>"]
        unk = self.vocab.word2idx["<UNK>"]

        # Encode prompt
        if prompt:
            ids = [bos] + self.vocab.encode(prompt)
        else:
            ids = [bos]

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = list(ids[1:])  # skip BOS for display

        with torch.no_grad():
            for _ in range(max_tokens):
                # Truncate to max_seq_len
                inp = input_ids[:, -self.seq_len:]
                logits = broca_decoder(inp)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)

                # Top-K filtering
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, topk_idx, topk_vals)
                    next_logits = filtered

                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()

                if next_tok == eos:
                    break
                generated.append(next_tok)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_tok]], dtype=torch.long, device=self.device)],
                    dim=1,
                )

        return self.vocab.decode(generated)

    def save(self, broca_decoder, path: str):
        """Save trained model weights and vocabulary."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(broca_decoder.state_dict(), path / "broca_weights.pt")
        self.vocab.save(str(path / "broca_vocab.json"))
        self._log(f"Saved model to {path}")

    def load(self, broca_decoder, path: str):
        """Load trained model weights and vocabulary."""
        path = Path(path)
        self.vocab.load(str(path / "broca_vocab.json"))
        weights = torch.load(str(path / "broca_weights.pt"), map_location=self.device)

        # Resize embeddings to match saved vocab
        saved_vocab_size = weights["token_emb.weight"].shape[0]
        orig_dim = broca_decoder.token_emb.embedding_dim
        broca_decoder.token_emb = nn.Embedding(saved_vocab_size, orig_dim)
        broca_decoder.lm_head = nn.Linear(orig_dim, saved_vocab_size, bias=False)
        broca_decoder.lm_head.weight = broca_decoder.token_emb.weight

        broca_decoder.load_state_dict(weights)
        broca_decoder = broca_decoder.to(self.device)
        self._log(f"Loaded model from {path} (vocab: {self.vocab.size})")
        return broca_decoder
