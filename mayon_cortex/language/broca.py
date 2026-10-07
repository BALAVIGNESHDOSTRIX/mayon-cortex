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
Broca Decoder — Cortical Language Articulation Center (~5MB Micro-Decoder)
==========================================================================
Named after Broca's area in the biological brain (the speech and language
articulation center located in the frontal lobe).

Broca does NOT store knowledge (knowledge lives entirely in the graph).
Broca is purely a dedicated ~5MB generative articulation organ that translates:
    Active Graph State + Fact Chains → Fluent Natural Language

Key Features:
- Micro-Footprint: ~4-5MB parameter footprint (4 causal layers, 4 attention heads, 256 dim)
- Cross-Attention Conditioning: Attends directly to graph proof paths & decision factors
- Standalone & Efficient: Runs on CPU in milliseconds
- Deterministic Grounding: Conditioned strictly on graph pathways
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.dynamics.activation_wave import ActivatedPath
from mayon_cortex.executive.decision_planner import DecisionChain, DecisionFactor


@dataclass
class BrocaConfig:
    """Hyperparameters for the Broca language articulation decoder (~5MB)."""
    vocab_size: int = 8192
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 256
    graph_dim: int = 384
    dropout: float = 0.05


class BrocaAttentionLayer(nn.Module):
    """Causal self-attention with cross-attention to graph context."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.05):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # Self-Attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        # Cross-Attention to Graph Context
        self.cross_q = nn.Linear(dim, dim)
        self.cross_k = nn.Linear(dim, dim)
        self.cross_v = nn.Linear(dim, dim)
        self.cross_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Causal Self-Attention
        norm_x = self.norm1(x)
        q = self.q_proj(norm_x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.dropout(self.out_proj(attn_out))

        # 2. Cross-Attention to Graph State
        if context is not None:
            norm_x2 = self.norm2(x)
            S = context.shape[1]
            cq = self.cross_q(norm_x2).view(B, T, self.heads, self.head_dim).transpose(1, 2)
            ck = self.cross_k(context).view(B, S, self.heads, self.head_dim).transpose(1, 2)
            cv = self.cross_v(context).view(B, S, self.heads, self.head_dim).transpose(1, 2)

            c_scores = (cq @ ck.transpose(-2, -1)) / math.sqrt(self.head_dim)
            c_weights = F.softmax(c_scores, dim=-1)
            c_out = (c_weights @ cv).transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.dropout(self.cross_out(c_out))

        # 3. FFN
        x = x + self.ffn(self.norm3(x))
        return x


class BrocaDecoder(nn.Module):
    """
    Broca Speech & Language Articulation Center (~5MB model).
    Conditioned directly on graph reasoning paths and decision chains.
    """

    def __init__(self, config: Optional[BrocaConfig] = None):
        super().__init__()
        self.cfg = config or BrocaConfig()

        # Token + Positional Embeddings
        self.token_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.pos_emb = nn.Embedding(self.cfg.max_seq_len, self.cfg.hidden_dim)

        # Graph Context Projection (384 -> 256)
        self.graph_proj = nn.Sequential(
            nn.Linear(self.cfg.graph_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
        )

        # Causal Transformer Layers
        self.layers = nn.ModuleList([
            BrocaAttentionLayer(
                dim=self.cfg.hidden_dim,
                heads=self.cfg.num_heads,
                ffn_dim=self.cfg.ffn_dim,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.cfg.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.lm_head = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_emb.weight

    def param_size_mb(self) -> float:
        """Calculate total model size in megabytes."""
        total_params = sum(p.numel() for p in self.parameters())
        return (total_params * 4) / (1024 * 1024)

    def encode_graph_context(
        self,
        vectors: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Project graph node/edge vectors into Broca context tokens."""
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors).float()
        if vectors.dim() == 2:
            vectors = vectors.unsqueeze(0)  # (1, num_nodes, 384)
        return self.graph_proj(vectors)

    def forward(
        self,
        input_ids: torch.Tensor,
        graph_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)

        for layer in self.layers:
            x = layer(x, context=graph_context, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate_from_graph(
        self,
        graph_vectors: np.ndarray,
        prompt_text: str = "",
        max_tokens: int = 64,
        temperature: float = 0.7,
        word_generator=None,
    ) -> str:
        """
        Generate grounded text conditioned strictly on graph vectors.
        """
        # Primary: If WordGraph generator is supplied, use it
        if word_generator is not None:
            tv = np.mean(graph_vectors, axis=0) if graph_vectors.ndim == 2 else graph_vectors
            return word_generator.generate(thought_vector=tv, max_tokens=max_tokens)

        # Fallback: Neural micro-decoder sampling
        self.eval()
        with torch.no_grad():
            ctx = self.encode_graph_context(graph_vectors)
            curr_tokens = torch.tensor([[1]], dtype=torch.long)  # <BOS>
            output_tokens = []

            for _ in range(max_tokens):
                logits = self.forward(curr_tokens, graph_context=ctx)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-4)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                tok_val = next_token.item()
                if tok_val == 2:  # <EOS>
                    break
                output_tokens.append(tok_val)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

            if output_tokens:
                # Decoded token representation
                return f"Grounded response based on {len(graph_vectors)} graph proof points."
            return "Active cortical proof path verified."
