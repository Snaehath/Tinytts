"""Monotonic Alignment Search (MAS) for unsupervised text-to-speech alignment.

Finds the optimal monotonic path between text representations and mel spectrogram frames
using dynamic programming (Viterbi style), following Glow-TTS, Matcha-TTS, and VITS.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn


def compute_log_likelihood(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    mel: torch.Tensor,
) -> torch.Tensor:
    """Compute Gaussian log-likelihood matrix S [B, T_text, T_mel] using memory-efficient matrix products.
    
    Expands (mel - mu)^2 / sigma^2 = (mel^2 / sigma^2) - 2 * (mel * mu / sigma^2) + (mu^2 / sigma^2)
    via batched matrix multiplication, avoiding massive 4D [B, D, T_text, T_mel] tensor allocation.
    """
    B, D, T_text = mu.shape
    T_mel = mel.size(-1)

    # Precision/inv_sigma_sq: [B, D, T_text]
    inv_sigma_sq = torch.exp(-2.0 * log_sigma)
    mu_weighted = mu * inv_sigma_sq  # [B, D, T_text]

    # Term 1: 2 * (mu / sigma^2)^T @ mel -> [B, T_text, T_mel]
    cross_term = 2.0 * torch.bmm(mu_weighted.transpose(1, 2), mel)

    # Term 2: (1 / sigma^2)^T @ (mel^2) -> [B, T_text, T_mel]
    mel_sq = mel ** 2
    mel_sq_term = torch.bmm(inv_sigma_sq.transpose(1, 2), mel_sq)

    # Term 3: sum_d (mu^2 / sigma^2) -> [B, T_text, 1]
    mu_sq_term = torch.sum(mu * mu_weighted, dim=1).unsqueeze(-1)

    # Term 4: sum_d (2 * log_sigma) -> [B, T_text, 1]
    log_det_term = torch.sum(2.0 * log_sigma, dim=1).unsqueeze(-1)

    const = float(D * math.log(2.0 * math.pi))

    # Total log-likelihood S: [B, T_text, T_mel]
    S = -0.5 * (mel_sq_term - cross_term + mu_sq_term + log_det_term + const)
    return S


@torch.no_grad()
def monotonic_alignment_search(
    log_likelihood: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> torch.Tensor:
    """Computes hard monotonic alignment matrix A via dynamic programming.
    
    Args:
        log_likelihood: [B, T_text, T_mel]
        text_lengths: [B]
        mel_lengths: [B]

    Returns:
        alignment: Binary matrix [B, T_text, T_mel] with exactly one 1 per mel column.
    """
    B, T_text, T_mel = log_likelihood.shape
    device = log_likelihood.device
    alignment = torch.zeros((B, T_text, T_mel), dtype=torch.float32, device=device)

    # Process each batch item
    for b in range(B):
        t_len = int(text_lengths[b].item())
        m_len = int(mel_lengths[b].item())

        if t_len == 0 or m_len == 0:
            continue

        # Sub-matrix for valid lengths
        S_b = log_likelihood[b, :t_len, :m_len]  # [t_len, m_len]

        # DP table Q: max cumulative log-probability
        Q = torch.full((t_len, m_len), fill_value=-1e9, dtype=torch.float32, device=device)
        Q[0, 0] = S_b[0, 0]

        # Forward DP pass
        for j in range(1, m_len):
            # For each mel frame j, text token i can only come from i (stay) or i-1 (advance)
            # Token index i cannot exceed j (can't have more tokens than frames)
            max_i = min(j + 1, t_len)
            
            # i = 0: can only come from Q[0, j-1]
            Q[0, j] = Q[0, j - 1] + S_b[0, j]

            # i > 0
            if max_i > 1:
                stay = Q[1:max_i, j - 1]
                advance = Q[:max_i - 1, j - 1]
                best_prev = torch.maximum(stay, advance)
                Q[1:max_i, j] = best_prev + S_b[1:max_i, j]

        # Backward backtracking pass
        curr_i = t_len - 1
        alignment[b, curr_i, m_len - 1] = 1.0

        for j in range(m_len - 1, 0, -1):
            if curr_i == 0:
                # Must stay at 0
                pass
            else:
                prev_stay = Q[curr_i, j - 1].item()
                prev_advance = Q[curr_i - 1, j - 1].item()
                if prev_advance >= prev_stay:
                    curr_i -= 1

            alignment[b, curr_i, j - 1] = 1.0

    return alignment


class MonotonicAlignment(nn.Module):
    """Module wrapping log-likelihood calculation and Monotonic Alignment Search."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        mel: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment matrix and alignment loss.
        
        Args:
            mu: [B, D, T_text]
            log_sigma: [B, D, T_text]
            mel: [B, D, T_mel]
            text_lengths: [B]
            mel_lengths: [B]

        Returns:
            alignment: Binary alignment matrix [B, T_text, T_mel]
            align_loss: Negative log-likelihood of target mel under the aligned prior.
        """
        S = compute_log_likelihood(mu, log_sigma, mel)  # [B, T_text, T_mel]
        alignment = monotonic_alignment_search(S, text_lengths, mel_lengths)

        # Alignment loss: Negative log-likelihood along the optimal path
        # Sum over text and mel dimensions, normalized by total mel frames
        align_log_prob = torch.sum(S * alignment, dim=(1, 2))
        align_loss = -torch.mean(align_log_prob / mel_lengths.float().clamp(min=1.0))

        return alignment, align_loss
