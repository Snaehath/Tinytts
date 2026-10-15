"""Duration utilities for MAS alignment extraction and representation expansion."""

from typing import Optional
import torch


def durations_from_alignment(alignment: torch.Tensor) -> torch.Tensor:
    """Extract token durations from binary MAS alignment matrix.
    
    Args:
        alignment: Binary alignment matrix of shape [B, T_text, T_mel].

    Returns:
        durations: Number of frames assigned to each token [B, T_text].
    """
    return torch.sum(alignment, dim=-1)


def expand_by_alignment(features: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    """Expand token-level features to frame-level features using alignment matrix.
    
    Args:
        features: [B, D, T_text]
        alignment: [B, T_text, T_mel]

    Returns:
        expanded: Frame-level features [B, D, T_mel]
    """
    # Matrix multiplication achieves exact sequence expansion: [B, D, T_text] @ [B, T_text, T_mel] = [B, D, T_mel]
    return torch.bmm(features, alignment)


def expand_by_duration(
    features: torch.Tensor,
    durations: torch.Tensor,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """Expand token-level features [B, D, T_text] by integer durations [B, T_text].
    
    Args:
        features: [B, D, T_text]
        durations: [B, T_text] (integer or float durations)
        max_len: Optional max length to truncate or pad to.

    Returns:
        expanded: [B, D, T_mel]
    """
    B, D, T_text = features.shape
    device = features.device
    int_durations = durations.round().long().clamp(min=0)

    expanded_list = []
    for b in range(B):
        dur_b = int_durations[b]
        if dur_b.sum() == 0:
            dur_b = torch.ones_like(dur_b)
        expanded_b = torch.repeat_interleave(features[b], dur_b, dim=-1)
        expanded_list.append(expanded_b)

    if max_len is None:
        max_len = max(x.size(-1) for x in expanded_list)
    max_len = max(max_len, 1)

    result = torch.zeros((B, D, max_len), dtype=features.dtype, device=device)
    for b in range(B):
        l = min(expanded_list[b].size(-1), max_len)
        result[b, :, :l] = expanded_list[b][:, :l]

    return result


class DurationPredictor(torch.nn.Module):
    """Lightweight duration predictor head (A2).
    
    Predicts log(1 + duration) from text encoder representations H.
    Architecture:
        D_in (text_hidden_dim)
        -> Conv1D (filter_channels) -> LayerNorm + ReLU + Dropout
        -> Conv1D (filter_channels) -> LayerNorm + ReLU + Dropout
        -> Linear(filter_channels, 1) -> log(1 + duration)
    """

    def __init__(
        self,
        in_channels: int = 320,
        filter_channels: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm1 = torch.nn.LayerNorm(filter_channels)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv1d(
            filter_channels,
            filter_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm2 = torch.nn.LayerNorm(filter_channels)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout)

        self.proj = torch.nn.Linear(filter_channels, 1)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting log(1 + duration).
        
        Args:
            h: Text representations [B, in_channels, T_text]
            mask: Optional text mask [B, 1, T_text] or [B, T_text]

        Returns:
            log_durations: Predicted log(1 + duration) [B, T_text]
        """
        # Conv1 + LayerNorm + ReLU + Dropout
        x = self.conv1(h)  # [B, filter_channels, T]
        x = x.transpose(1, 2)  # [B, T, filter_channels]
        x = self.drop1(self.relu1(self.norm1(x)))
        x = x.transpose(1, 2)  # [B, filter_channels, T]

        # Conv2 + LayerNorm + ReLU + Dropout
        x = self.conv2(x)  # [B, filter_channels, T]
        x = x.transpose(1, 2)  # [B, T, filter_channels]
        x = self.drop2(self.relu2(self.norm2(x)))  # [B, T, filter_channels]

        # Linear projection to scalar log(1 + d)
        out = self.proj(x).squeeze(-1)  # [B, T_text]

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.squeeze(1)
            out = out * mask.float()

        return out

    @torch.no_grad()
    def predict_durations(
        self,
        h: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pace_factor: float = 1.0,
        pause_frames: int = 20,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        pause_id: int = 5,
    ) -> torch.Tensor:
        """Inference duration generation with special-token handling and pace scaling.
        
        Rules:
        - Ordinary text tokens: expm1(log_d) * pace_factor, rounded & clamped to min 1
        - <pause>: deterministic pause duration (default 20 frames ~232ms)
        - <bos> / <eos>: set to 0 frames (boundary exclusion)
        - <pad>: set to 0 frames
        """
        log_d = self.forward(h, mask)
        raw_durations = torch.expm1(log_d).round().clamp(min=1.0)

        # Apply pace scaling to phonation tokens
        durations = (raw_durations * pace_factor).round().clamp(min=1.0)

        if tokens is not None:
            # Special token rules
            durations[tokens == pad_id] = 0.0
            durations[tokens == bos_id] = 0.0
            durations[tokens == eos_id] = 0.0
            durations[tokens == pause_id] = float(pause_frames)

        return durations


def compute_duration_loss(
    pred_log_durations: torch.Tensor,
    target_durations: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    exclude_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute MSE loss on log(1 + duration).
    
    Args:
        pred_log_durations: Predicted log(1 + d) [B, T_text]
        target_durations: Target integer/float frame counts [B, T_text]
        mask: Optional text mask [B, T_text]
        exclude_mask: Optional boolean mask of tokens to exclude (BOS/EOS/PAD) [B, T_text]

    Returns:
        loss: Mean squared error over valid, non-excluded tokens.
    """
    target_log_d = torch.log1p(target_durations.float())
    loss = (pred_log_durations - target_log_d) ** 2

    valid = torch.ones_like(loss, dtype=torch.bool)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.squeeze(1)
        valid = valid & (mask > 0)
    if exclude_mask is not None:
        valid = valid & (~exclude_mask)

    valid_count = valid.sum().clamp(min=1.0)
    return (loss * valid.float()).sum() / valid_count

