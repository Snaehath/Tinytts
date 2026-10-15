"""Post-Length-Regulation Conditioning Adapter for TinyTTS-20 (Stage A2 & A2.1).

Applies depthwise temporal convolution and pointwise projection
with residual connections across expanded phoneme frames.
Smooths abrupt piecewise-constant step boundaries (e.g. AAAAAABBBBB -> AAAaaabbbbbbcccccCCC)
to provide continuous coarticulatory context to the acoustic CFM vector field.

Supports:
- "convnext": Single-scale ConvNeXt block (A2, default kernel_size=7)
- "multiscale": Parallel multi-scale depthwise branches (A2.1, kernel_sizes=(3, 7, 15))
               with exact zero-initialized fusion projection for an identity start.
"""

from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn


class ConvNeXtConditioningBlock(nn.Module):
    """Single ConvNeXt-style temporal conditioning refinement block (Stage A2)."""

    def __init__(
        self,
        channels: int = 320,
        kernel_size: int = 7,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = channels * expansion_factor  # e.g. 320 * 2 = 640
        self.norm = nn.GroupNorm(8, channels)
        # 1x1 projection
        self.proj_in = nn.Conv1d(channels, hidden_dim, kernel_size=1)
        # Depthwise Conv1D for temporal context across frame boundaries
        self.dw_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Pointwise Conv1D back to channel dimension
        self.proj_out = nn.Conv1d(hidden_dim, channels, kernel_size=1)

        # Initialize out projection with small weights for smooth identity start
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.norm(x)
        h = self.proj_in(h)
        h = self.dw_conv(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.proj_out(h)
        return res + h


class MultiScaleConditioningBlock(nn.Module):
    """Multi-Scale temporal conditioning refinement block (Stage A2.1).

    Processes expanded phoneme conditioning vectors concurrently across three
    distinct temporal receptive fields:
      - k=3  (pad 1): transient bursts, stop-releases (~35 ms)
      - k=7  (pad 3): coarticulatory transitions (~81 ms)
      - k=15 (pad 7): syllabic nuclei and vowel phrasing (~174 ms)

    The branches are concatenated (channels * len(kernels), e.g. 320 * 3 = 960)
    and projected back to channels (320) via a 1x1 pointwise convolution.

    Crucially, the output projection is initialized to exact zero (weights and bias),
    ensuring adapter(h) == 0 at initialization, preserving strict identity at step 0.
    """

    def __init__(
        self,
        channels: int = 320,
        kernel_sizes: Sequence[int] = (3, 7, 15),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)

        # Parallel depthwise convolutional branches
        self.branches = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=k,
                padding=k // 2,
                groups=channels,
            )
            for k in kernel_sizes
        ])

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 pointwise fusion projection: (channels * num_branches) -> channels
        fusion_dim = channels * len(kernel_sizes)
        self.proj_out = nn.Conv1d(fusion_dim, channels, kernel_size=1)

        # Strict zero-init: adapter(h) == 0 at initialization for seamless identity start
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.norm(x)
        # Apply parallel multi-scale depthwise convolutions
        branch_feats = [branch(h) for branch in self.branches]
        # Concatenate multi-scale representations along channel dimension
        fused = torch.cat(branch_feats, dim=1)
        fused = self.act(fused)
        fused = self.dropout(fused)
        delta = self.proj_out(fused)
        return res + delta


class ConditioningAdapter(nn.Module):
    """Post-Length-Regulation Conditioning Adapter.

    Transforms piecewise-constant frame-repeated phoneme vectors into
    a smoothly coarticulated contextual representation for the CFM generator.
    Supports single-scale ConvNeXt (Stage A2) and Multi-Scale (Stage A2.1).
    """

    def __init__(
        self,
        channels: int = 320,
        adapter_type: str = "convnext",
        kernel_size: int = 7,
        kernel_sizes: Sequence[int] = (3, 7, 15),
        expansion_factor: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.adapter_type = adapter_type

        if adapter_type == "multiscale":
            self.blocks = nn.ModuleList([
                MultiScaleConditioningBlock(
                    channels=channels,
                    kernel_sizes=kernel_sizes,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                ConvNeXtConditioningBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])

    def forward(
        self,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            cond: [B, channels, T_mel] frame-expanded phoneme representation
            mask: Optional [B, 1, T_mel] frame mask
        Returns:
            smoothed_cond: [B, channels, T_mel] contextually smoothed conditioning
        """
        h = cond
        for block in self.blocks:
            h = block(h)
        if mask is not None:
            h = h * mask
        return h
