"""Text Encoder for TinyTTS-A0.

Transforms discrete character token sequences into rich continuous text representations
and projects them to the Mel-frequency dimension for monotonic alignment prior matching.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input [B, T, d_model]."""
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Encoder Layer with dropout and GELU."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention branch
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # FFN branch
        x = x + self.ffn(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """Tiny Transformer Text Encoder for TinyTTS-A0 (target ~3M params)."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 768,
        mel_dim: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Projections to mel dimension for alignment prior (Matcha / Glow-TTS style)
        self.proj_mu = nn.Linear(d_model, mel_dim)
        self.proj_log_sigma = nn.Linear(d_model, mel_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text tokens.
        
        Args:
            text_tokens: [B, T_text]
            text_lengths: [B]

        Returns:
            h_text: Text representation [B, d_model, T_text]
            mu: Prior mean [B, mel_dim, T_text]
            log_sigma: Prior log-std [B, mel_dim, T_text]
        """
        B, T = text_tokens.shape

        # Create padding mask (True for padded tokens)
        idx_matrix = torch.arange(T, device=text_tokens.device).unsqueeze(0).expand(B, -1)
        padding_mask = idx_matrix >= text_lengths.unsqueeze(1)

        # Embed and add positional encoding
        x = self.embedding(text_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.final_norm(x)  # [B, T_text, d_model]

        # Project to Mel dimensions for alignment search
        mu = self.proj_mu(x).transpose(1, 2)  # [B, mel_dim, T_text]
        log_sigma = self.proj_log_sigma(x).transpose(1, 2)  # [B, mel_dim, T_text]
        h_text = x.transpose(1, 2)  # [B, d_model, T_text]

        return h_text, mu, log_sigma
