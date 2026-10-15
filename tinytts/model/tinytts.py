"""TinyTTS-A0: Non-Autoregressive Text-to-Speech Architecture.

Integrates:
1. Tiny Transformer Text Encoder (d_model=192)
2. Monotonic Alignment Search (MAS) for unsupervised text-to-mel alignment
3. MAS-derived frame durations
4. Dilated Convolutional Conditional Flow Matching (Conv-CFM)
5. Pluggable Vocoder interface
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinytts.config import AudioConfig, ModelConfig, TinyTTSConfig
from tinytts.model.adapter import ConditioningAdapter
from tinytts.model.alignment import MonotonicAlignment
from tinytts.model.duration import (
    DurationPredictor,
    compute_duration_loss,
    durations_from_alignment,
    expand_by_alignment,
    expand_by_duration,
)
from tinytts.model.encoder import TextEncoder
from tinytts.model.flow import AcousticCTCHead, CTCAuxiliaryHead, ConditionalFlowMatching
from tinytts.model.vocoder import BaseVocoder, get_vocoder


@dataclass
class TinyTTSLosses:
    total: torch.Tensor
    flow_mel: torch.Tensor
    align: torch.Tensor
    prior: torch.Tensor
    duration: Optional[torch.Tensor] = None
    ctc: Optional[torch.Tensor] = None


class TinyTTSA0(nn.Module):
    """TinyTTS end-to-end model (10-30M parameter budget)."""

    def __init__(
        self,
        config: Optional[TinyTTSConfig] = None,
        vocoder: Optional[BaseVocoder] = None,
    ):
        super().__init__()
        self.config = config or TinyTTSConfig()
        m_cfg = self.config.model

        # 1. Text Encoder
        self.encoder = TextEncoder(
            vocab_size=m_cfg.vocab_size,
            d_model=m_cfg.text_hidden_dim,
            num_layers=m_cfg.text_num_layers,
            num_heads=m_cfg.text_num_heads,
            ffn_dim=m_cfg.text_ffn_dim,
            mel_dim=m_cfg.mel_channels,
            dropout=m_cfg.dropout,
        )

        # 2. Monotonic Alignment Search
        self.alignment = MonotonicAlignment()

        # 3. Duration Predictor (A2)
        self.duration_predictor = DurationPredictor(
            in_channels=m_cfg.text_hidden_dim,
            filter_channels=m_cfg.duration_filter_channels,
            kernel_size=m_cfg.duration_kernel_size,
            dropout=m_cfg.dropout,
        )

        # 4. Conditional Flow Matching
        self.flow = ConditionalFlowMatching(
            mel_channels=m_cfg.mel_channels,
            cond_channels=m_cfg.text_hidden_dim,
            hidden_channels=m_cfg.flow_hidden_dim,
            num_layers=m_cfg.flow_num_layers,
            use_adaln=getattr(m_cfg, "use_flow_adaln", False),
        )

        # 4b. Conditioning Adapter (Stage A2 & A2.1: Contextual smoothing of expanded phonemes)
        if getattr(m_cfg, "use_conditioning_adapter", False):
            self.conditioning_adapter = ConditioningAdapter(
                channels=m_cfg.text_hidden_dim,
                adapter_type=getattr(m_cfg, "adapter_type", "convnext"),
                kernel_size=getattr(m_cfg, "adapter_kernel_size", 7),
                kernel_sizes=getattr(m_cfg, "adapter_kernel_sizes", (3, 7, 15)),
                expansion_factor=getattr(m_cfg, "adapter_expansion", 2),
                num_layers=getattr(m_cfg, "adapter_num_layers", 1),
                dropout=getattr(m_cfg, "adapter_dropout", 0.1),
            )
        else:
            self.conditioning_adapter = None

        # 5. Training-only Acoustic CTC Head (A3-C v2: Operates on 80-channel generated Mel)
        self.ctc_head = AcousticCTCHead(
            in_channels=m_cfg.mel_channels,
            hidden_dim=128,
            vocab_size=m_cfg.vocab_size,
        )

        # 6. Vocoder
        self.vocoder = vocoder or get_vocoder("vocos", self.config.audio)

        # Loss weights
        self.lambda_align = 0.5
        self.lambda_prior = 1.0
        self.lambda_duration = 1.0
        self.lambda_ctc = 0.01  # Conservative initial weight (0.01)

    @property
    def total_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        train_duration: bool = True,
        freeze_encoder_for_duration: bool = False,
        use_ctc: bool = False,
        ctc_targets: Optional[torch.Tensor] = None,
        ctc_target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[TinyTTSLosses, Dict[str, Any]]:
        """Forward pass for training.
        
        Args:
            text: Token sequence [B, T_text]
            text_lengths: Lengths of text sequences [B]
            mel: Target mel spectrogram [B, mel_channels, T_mel]
            mel_lengths: Lengths of mel spectrograms [B]
            train_duration: Whether to compute duration loss
            freeze_encoder_for_duration: If True, detach h_text when feeding duration predictor (Stage A2.0)
            use_ctc: Whether to compute intermediate CTC loss (Stage A3)
            ctc_targets: Optional target token sequence for CTC
            ctc_target_lengths: Optional lengths of CTC targets

        Returns:
            losses: Named tuple containing total, flow, align, prior, duration, and ctc losses.
            outputs: Dictionary with intermediate representations and alignment.
        """
        B, C, T_mel = mel.shape
        T_text = text.size(1)

        # Step 1: Text encoding
        h_text, mu_text, log_sigma_text = self.encoder(text, text_lengths)

        # Step 2: Monotonic Alignment Search
        alignment_matrix, align_loss = self.alignment(
            mu=mu_text,
            log_sigma=log_sigma_text,
            mel=mel,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
        )

        # Step 3: Derive durations and expand text conditioning to mel length
        durations = durations_from_alignment(alignment_matrix)  # [B, T_text]
        frame_cond = expand_by_alignment(h_text, alignment_matrix)  # [B, d_model, T_mel]
        aligned_mu = expand_by_alignment(mu_text, alignment_matrix)  # [B, mel_channels, T_mel]

        # Text and Frame masks
        text_idx = torch.arange(T_text, device=text.device).unsqueeze(0).expand(B, -1)
        text_mask = (text_idx < text_lengths.unsqueeze(1)).float()  # [B, T_text]

        frame_idx = torch.arange(T_mel, device=mel.device).unsqueeze(0).expand(B, -1)
        mel_mask = (frame_idx < mel_lengths.unsqueeze(1)).unsqueeze(1).float()  # [B, 1, T_mel]

        # Step 3b: Contextual Conditioning Adapter (TinyTTS-20-A2)
        if self.conditioning_adapter is not None:
            frame_cond = self.conditioning_adapter(frame_cond, mel_mask)

        # Step 4: Conditional Flow Matching Loss & Acoustic Mel CTC (Stage A3-C v2)
        if use_ctc:
            flow_loss, _, mel_pred = self.flow.compute_loss(
                x_1=mel,
                cond=frame_cond,
                mask=mel_mask,
                return_mel_pred=True,
            )
            # Acoustic CTC Head operates strictly on 80-channel Mel (zero text shortcut!)
            ctc_logits = self.ctc_head(mel_pred)  # [T_mel, B, vocab_size]
            log_probs = F.log_softmax(ctc_logits.float(), dim=-1)

            if ctc_targets is None:
                # Extract clean 1D targets without PAD(0), BOS(1), EOS(2), BLANK(4)
                # PyTorch CTC requires targets to contain NO blank symbols!
                clean_targets_list = []
                clean_lengths_list = []
                for b in range(B):
                    seq = text[b, :text_lengths[b]]
                    clean = seq[(seq != 0) & (seq != 1) & (seq != 2) & (seq != 4)]
                    clean_targets_list.append(clean)
                    clean_lengths_list.append(len(clean))
                ctc_tgt = torch.cat(clean_targets_list)
                ctc_tgt_lens = torch.tensor(clean_lengths_list, dtype=torch.long, device=text.device)
            else:
                ctc_tgt = ctc_targets
                ctc_tgt_lens = ctc_target_lengths

            ctc_loss = F.ctc_loss(
                log_probs=log_probs,
                targets=ctc_tgt,
                input_lengths=mel_lengths,
                target_lengths=ctc_tgt_lens,
                blank=4,  # BLANK token ID in symbols.py
                zero_infinity=True,
            )

            # Hard sanity checks: abort immediately on any negative or non-finite loss
            if not torch.isfinite(ctc_loss):
                raise RuntimeError(f"Non-finite CTC loss detected: {ctc_loss.item()}")
            if ctc_loss.item() < -1e-5:
                raise RuntimeError(f"Invalid negative CTC loss detected: {ctc_loss.item()}")
        else:
            flow_loss, _ = self.flow.compute_loss(x_1=mel, cond=frame_cond, mask=mel_mask, return_mel_pred=False)
            ctc_loss = None
            ctc_logits = None

        # Step 5: Prior Loss (aligning text mean with target mel)
        prior_loss = torch.sum(((mel - aligned_mu) ** 2) * mel_mask) / (torch.sum(mel_mask) * C + 1e-6)

        # Step 6: Duration Predictor Loss (A2)
        dur_loss = None
        pred_log_dur = None
        if train_duration:
            h_dur = h_text.detach() if freeze_encoder_for_duration else h_text
            pred_log_dur = self.duration_predictor(h_dur, mask=text_mask)
            # Exclude special tokens BOS (1), EOS (2), PAD (0) from regression
            exclude_mask = (text == 0) | (text == 1) | (text == 2)
            dur_loss = compute_duration_loss(pred_log_dur, durations, mask=text_mask, exclude_mask=exclude_mask)

        # Total Loss
        total_loss = flow_loss + self.lambda_align * align_loss + self.lambda_prior * prior_loss
        if dur_loss is not None:
            total_loss = total_loss + self.lambda_duration * dur_loss
        if ctc_loss is not None:
            total_loss = total_loss + self.lambda_ctc * ctc_loss

        losses = TinyTTSLosses(
            total=total_loss,
            flow_mel=flow_loss,
            align=align_loss,
            prior=prior_loss,
            duration=dur_loss,
            ctc=ctc_loss,
        )

        outputs = {
            "alignment": alignment_matrix,
            "durations": durations,
            "pred_log_durations": pred_log_dur,
            "frame_cond": frame_cond,
            "aligned_mu": aligned_mu,
            "ctc_logits": ctc_logits,
        }

        return losses, outputs

    @torch.no_grad()
    def synthesize(
        self,
        text_tokens: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        use_duration_predictor: bool = False,
        frames_per_char: float = 3.0,
        num_flow_steps: Optional[int] = None,
        temperature: float = 0.9,
        pace_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize Mel spectrogram and audio waveform.
        
        Args:
            text_tokens: 1D [T_text] or 2D [1, T_text]
            durations: Optional [1, T_text] durations. If None and use_duration_predictor=False, uses frames_per_char.
            use_duration_predictor: If True, uses the learned duration predictor head.
            frames_per_char: Fallback average frames per token (default: 3.0 frames/token).
            num_flow_steps: Number of Euler ODE integration steps (default: config.model.flow_steps, e.g. 8).
            temperature: Sampling temperature for prior noise.
            pace_factor: Scaling factor for duration (1.0 = normal, 0.8 = faster, 1.2 = slower).

        Returns:
            mel: Generated log-Mel spectrogram [1, 80, T_mel]
            waveform: Generated audio waveform [1, 1, samples]
        """
        if text_tokens.ndim == 1:
            text_tokens = text_tokens.unsqueeze(0)

        B, T_text = text_tokens.shape
        device = text_tokens.device
        text_lengths = torch.tensor([T_text], device=device, dtype=torch.long)

        # Default flow steps from config (8 = quality default, 6 = low-latency, 16 = reference)
        if num_flow_steps is None:
            num_flow_steps = self.config.model.flow_steps

        # 1. Text encoding
        h_text, _, _ = self.encoder(text_tokens, text_lengths)

        # 2. Duration conditioning
        if durations is None:
            if use_duration_predictor:
                durations = self.duration_predictor.predict_durations(
                    h_text,
                    tokens=text_tokens,
                    pace_factor=pace_factor,
                )
            else:
                durations = torch.full((B, T_text), fill_value=frames_per_char, device=device)
        else:
            durations = durations[:, :T_text]

        frame_cond = expand_by_duration(h_text, durations)
        if self.conditioning_adapter is not None:
            frame_cond = self.conditioning_adapter(frame_cond)

        # 3. Flow Matching Euler ODE integration
        mel = self.flow.solve_euler(
            cond=frame_cond,
            num_steps=num_flow_steps,
            temperature=temperature,
        )

        # 4. Neural Vocoder decoding
        waveform = self.vocoder.decode(mel)

        return mel, waveform
