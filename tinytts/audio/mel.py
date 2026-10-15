"""Log-Mel Spectrogram extractor compatible with standard neural vocoders."""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from tinytts.config import AudioConfig


class MelSpectrogramExtractor(nn.Module):
    """Computes log-Mel spectrograms from raw waveforms on CPU or GPU."""

    def __init__(self, config: AudioConfig = AudioConfig()):
        super().__init__()
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            normalized=False,
            mel_scale=config.mel_norm,
            center=True,
            pad_mode="reflect",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram.
        
        Args:
            waveform: Tensor of shape [B, T] or [1, T] in range [-1.0, 1.0].
            
        Returns:
            mel: Tensor of shape [B, n_mels, n_frames].
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)

        if self.config.dynamic_range_compression:
            mel = torch.log(torch.clamp(mel, min=self.config.clip_val))

        return mel
