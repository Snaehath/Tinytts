"""Convolutional Conditional Flow Matching (Conv-CFM) vector field for TinyTTS-A0.

Implements an efficient Conv1D residual velocity network with sinusoidal time embedding
and frame-aligned text conditioning, trained with Optimal Transport Conditional Flow Matching (OT-CFM).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embedding for continuous time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ConvResBlock(nn.Module):
    """1D Dilated Residual Block with Time and Conditioning Injection."""

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        dropout: float = 0.1,
        use_adaln: bool = False,
        cond_channels: int = 320,
    ):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = nn.GroupNorm(8, channels)

        # Zero-initialized AdaLN projection: W=0, b=0 (strictly preserves norm1 identity at step 0)
        if use_adaln:
            self.adaln_proj = nn.Conv1d(cond_channels, 2 * channels, kernel_size=1)
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)
        else:
            self.adaln_proj = None

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        raw_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, channels, T]
        t_emb: [B, channels]
        cond: [B, channels, T] (additive projected conditioning)
        raw_cond: [B, cond_channels, T] (raw text conditioning for AdaLN modulation)
        """
        res = x
        norm_x = self.norm1(x)

        # Adaptive GroupNorm modulation hooked into existing norm1
        if self.use_adaln and self.adaln_proj is not None and raw_cond is not None:
            gamma_beta = self.adaln_proj(raw_cond)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            norm_x = norm_x * (1.0 + gamma) + beta

        h = self.conv1(self.act(norm_x))
        # Inject time embedding
        h = h + t_emb.unsqueeze(-1)
        # Inject frame conditioning
        if cond is not None:
            h = h + cond

        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return res + h


class AcousticCTCHead(nn.Module):
    """Training-only acoustic CTC projection head operating directly on the 80-channel generated Mel.
    
    Eliminates text-shortcut leakage by operating strictly on acoustic spectrogram features.
    Forces the CFM vector field to produce distinguishable phonetic formants.
    """

    def __init__(self, in_channels: int = 80, hidden_dim: int = 128, vocab_size: int = 70):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.act2 = nn.GELU()

        self.out_proj = nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 80, T_mel] generated mel spectrogram
        Returns:
            logits: [T_mel, B, vocab_size] transposed for PyTorch CTCLoss
        """
        h = self.act1(self.norm1(self.conv1(mel)))
        h = self.act2(self.norm2(self.conv2(h)))
        logits = self.out_proj(h)  # [B, vocab_size, T_mel]
        return logits.permute(2, 0, 1)  # [T_mel, B, vocab_size]


class CTCAuxiliaryHead(nn.Module):
    """Training-only auxiliary CTC projection head on intermediate CFM representations."""

    def __init__(self, in_channels: int = 192, hidden_dim: int = 128, vocab_size: int = 70):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.proj1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, in_channels, T_mel] intermediate CFM vector field state
        Returns:
            logits: [T_mel, B, vocab_size] transposed for PyTorch CTCLoss
        """
        # [B, in_channels, T] -> [B, T, in_channels]
        h_t = h.transpose(1, 2)
        h_norm = self.norm(h_t)
        hidden = self.act(self.proj1(h_norm))
        logits = self.proj2(hidden)  # [B, T_mel, vocab_size]
        return logits.transpose(0, 1)  # [T_mel, B, vocab_size]


class ConvFlowNetwork(nn.Module):
    """Lightweight Conv-based Vector Field Network for Conditional Flow Matching."""

    def __init__(
        self,
        mel_channels: int = 80,
        cond_channels: int = 192,
        hidden_channels: int = 192,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_adaln: bool = False,
    ):
        super().__init__()
        self.mel_channels = mel_channels
        self.hidden_channels = hidden_channels
        self.use_adaln = use_adaln

        # Initial projections
        self.in_proj = nn.Conv1d(mel_channels, hidden_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Conv1d(cond_channels, hidden_channels, kernel_size=1)

        # Time embedding MLP
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Dilated residual blocks (dilations 1, 2, 4, 1, 2, 4)
        dilations = [2 ** (i % 3) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            ConvResBlock(
                hidden_channels,
                dilation=d,
                dropout=dropout,
                use_adaln=use_adaln,
                cond_channels=cond_channels,
            )
            for d in dilations
        ])

        # Final projection to mel velocity field
        self.final_norm = nn.GroupNorm(8, hidden_channels)
        self.final_act = nn.SiLU()
        self.out_proj = nn.Conv1d(hidden_channels, mel_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """Predict velocity field v_t(x, t, cond).
        
        Args:
            x: Noisy mel [B, mel_channels, T]
            t: Continuous time [B] in [0, 1]
            cond: Frame-level text conditioning [B, cond_channels, T]
            mask: Optional boolean or float mask [B, 1, T]
            return_hidden: Whether to return intermediate hidden state h [B, hidden_channels, T]

        Returns:
            v: Velocity vector field [B, mel_channels, T]
            h (optional): Intermediate hidden state [B, hidden_channels, T]
        """
        # Time and conditioning projections
        t_emb = self.time_emb(t)  # [B, hidden_channels]
        c_proj = self.cond_proj(cond)  # [B, hidden_channels, T]

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb, c_proj, raw_cond=cond if self.use_adaln else None)

        v = self.out_proj(self.final_act(self.final_norm(h)))

        if mask is not None:
            v = v * mask

        if return_hidden:
            return v, h
        return v


class ConditionalFlowMatching(nn.Module):
    """Optimal Transport Conditional Flow Matching (OT-CFM) module."""

    def __init__(
        self,
        mel_channels: int = 80,
        cond_channels: int = 192,
        hidden_channels: int = 192,
        num_layers: int = 6,
        sigma_min: float = 1e-4,
        use_adaln: bool = False,
    ):
        super().__init__()
        self.mel_channels = mel_channels
        self.sigma_min = sigma_min
        self.use_adaln = use_adaln
        self.estimator = ConvFlowNetwork(
            mel_channels=mel_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            use_adaln=use_adaln,
        )

    def compute_loss(
        self,
        x_1: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_mel_pred: bool = False,
    ):
        """Compute OT-CFM loss.
        
        Args:
            x_1: Ground truth target Mel [B, mel_channels, T]
            cond: Frame-level conditioning [B, cond_channels, T]
            mask: Frame validity mask [B, 1, T]
            return_hidden: Whether to return intermediate hidden state h
            return_mel_pred: Whether to return instantaneous generated Mel x_1_hat

        Returns:
            loss: Mean squared error between predicted and target velocity field.
            v_pred: Predicted velocity field.
            mel_pred (optional): Instantaneous generated Mel [B, mel_channels, T]
            h_cfm (optional): Intermediate CFM representation [B, hidden_channels, T]
        """
        B, C, T = x_1.shape
        device = x_1.device

        # Prior sample x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # Sample uniform time t in [0, 1]
        t = torch.rand(B, device=device)

        # OT path: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_exp) * x_0 + t_exp * x_1

        # Target vector field: u_t = x_1 - (1 - sigma_min) * x_0
        u_t = x_1 - (1.0 - self.sigma_min) * x_0

        # Predict velocity
        if return_hidden:
            v_pred, h_cfm = self.estimator(x_t, t, cond, mask=mask, return_hidden=True)
        else:
            v_pred = self.estimator(x_t, t, cond, mask=mask)
            h_cfm = None

        # Instantaneous generated mel projection: x_1_hat = x_t + (1 - t) * v_pred
        mel_pred = x_t + (1.0 - t_exp) * v_pred if return_mel_pred else None

        # MSE Loss over valid frames
        diff_sq = (v_pred - u_t) ** 2
        if mask is not None:
            loss = torch.sum(diff_sq * mask) / (torch.sum(mask) * C + 1e-6)
        else:
            loss = torch.mean(diff_sq)

        if return_mel_pred and return_hidden:
            return loss, v_pred, mel_pred, h_cfm
        if return_mel_pred:
            return loss, v_pred, mel_pred
        if return_hidden:
            return loss, v_pred, h_cfm
        return loss, v_pred

    @torch.no_grad()
    def solve_euler(
        self,
        cond: torch.Tensor,
        num_steps: int = 16,
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate Mel spectrogram by integrating the velocity field using Euler method.
        
        Args:
            cond: Frame conditioning [B, cond_channels, T]
            num_steps: Number of integration steps (e.g. 16 or 32)
            temperature: Sampling noise temperature
            mask: Optional mask [B, 1, T]

        Returns:
            mel: Generated Mel spectrogram [B, mel_channels, T]
        """
        B, _, T = cond.shape
        device = cond.device

        # Initial sample at t=0
        x = torch.randn(B, self.mel_channels, T, device=device) * temperature
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), fill_value=t_val, device=device, dtype=torch.float32)
            v = self.estimator(x, t, cond, mask=mask)
            x = x + dt * v

        return x
