"""Modular Vocoder interface for TinyTTS.

Provides a unified interface `vocoder.decode(mel)` that allows seamlessly swapping
between Vocos, HiFi-GAN, and lightweight Griffin-Lim fallback without altering the acoustic model.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import torchaudio.transforms as T

from tinytts.config import AudioConfig


class BaseVocoder(ABC, nn.Module):
    """Abstract base class for all neural and signal-processing vocoders."""

    @abstractmethod
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from log-mel spectrogram.
        
        Args:
            mel: [B, 80, T] or [80, T] log-Mel spectrogram.

        Returns:
            waveform: [B, 1, samples] or [1, samples] audio waveform in [-1, 1].
        """
        pass


class GriffinLimVocoder(BaseVocoder):
    """Lightweight deterministic vocoder using Torchaudio Griffin-Lim.
    
    Serves as a robust zero-dependency offline fallback.
    """

    def __init__(self, config: AudioConfig = AudioConfig(), n_iter: int = 32):
        super().__init__()
        self.config = config
        self.n_iter = n_iter

        # Invert Mel basis: [n_fft // 2 + 1, n_mels]
        mel_fb = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            mel_scale=config.mel_norm,
        ).mel_scale.fb  # [n_fft // 2 + 1, n_mels] = [513, 80]
        # Invert from [80, 513] projection matrix: pinverse(fb.T) is [513, 80]
        mel_inv = torch.pinverse(mel_fb.t())
        self.register_buffer("mel_inv", mel_inv)

        self.griffin_lim = T.GriffinLim(
            n_fft=config.n_fft,
            n_iter=n_iter,
            win_length=config.win_length,
            hop_length=config.hop_length,
            power=config.power,
        )

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)

        # Invert log compression: mel = exp(log_mel)
        linear_mel = torch.exp(mel)  # [B, n_mels, T]

        # Map Mel back to linear magnitude spectrogram: [513, 80] @ [B, 80, T] = [B, 513, T]
        mel_inv = self.mel_inv.to(device=mel.device, dtype=mel.dtype)
        mag_spec = torch.matmul(mel_inv, linear_mel)
        mag_spec = torch.clamp(mag_spec, min=0.0)

        window_device = self.griffin_lim.window.device
        waveform = self.griffin_lim(mag_spec.to(window_device)).to(mel.device)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        return waveform


class VocosVocoder(BaseVocoder):
    """Wrapper around the lightweight Fourier-domain Vocos neural vocoder (80-mel 22.05kHz)."""

    def __init__(self, repo_id: str = "BSC-LT/vocos-mel-22khz", device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_id = repo_id
        try:
            from huggingface_hub import hf_hub_download
            from vocos.heads import ISTFTHead
            from vocos.models import VocosBackbone

            # 80-mel channels, 22050 Hz matching LJSpeech and TinyTTS-A0
            self.backbone = VocosBackbone(
                input_channels=80,
                dim=512,
                intermediate_dim=1536,
                num_layers=8,
            )
            self.head = ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="same")

            weights_path = hf_hub_download(self.repo_id, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

            bb_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            hd_state = {k.replace("head.", ""): v for k, v in state_dict.items() if k.startswith("head.")}

            self.backbone.load_state_dict(bb_state)
            self.head.load_state_dict(hd_state)

            self.backbone.to(self.device).eval()
            self.head.to(self.device).eval()
            self.available = True
            print(f"[Vocoder] Loaded 80-channel Vocos neural vocoder ({repo_id}) on {self.device}")
        except Exception as e:
            self.backbone = None
            self.head = None
            self.available = False
            self.error_msg = str(e)

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.available:
            raise RuntimeError(
                f"Vocos vocoder is not available ({self.error_msg}). "
                "Install with 'pip install vocos' or use GriffinLimVocoder fallback."
            )

        if mel.ndim == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.device)
        with torch.no_grad():
            features = self.backbone(mel)
            waveform = self.head(features)

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        return waveform


def get_vocoder(name: str = "vocos", config: AudioConfig = AudioConfig()) -> BaseVocoder:
    """Factory function to instantiate vocoders."""
    if name.lower() == "vocos":
        vocos = VocosVocoder()
        if vocos.available:
            return vocos
        print("[Vocoder] Vocos not installed or weight loading failed. Using GriffinLim fallback.")
        return GriffinLimVocoder(config)
    elif name.lower() == "griffin_lim":
        return GriffinLimVocoder(config)
    else:
        raise ValueError(f"Unknown vocoder: {name}")
