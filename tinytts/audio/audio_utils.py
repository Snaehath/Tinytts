"""Audio I/O and utility functions for TinyTTS."""

from pathlib import Path
from typing import Union
import soundfile as sf
import torch
import torchaudio


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 22050,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an audio file as a mono float32 tensor scaled to [-1.0, 1.0].
    
    Args:
        file_path: Path to the audio file (.wav, .flac, etc.)
        target_sr: Target sample rate for automatic resampling.
        normalize: Whether to peak-normalize the waveform to ~0.95.

    Returns:
        waveform: Float32 Tensor of shape [1, T].
    """
    path_str = str(file_path)
    # Using soundfile for robust Windows I/O
    audio_data, sr = sf.read(path_str, dtype="float32")
    waveform = torch.from_numpy(audio_data)

    # Convert to shape [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    # Mixdown to mono if multichannel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if sample rate doesn't match
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Peak normalize
    if normalize:
        peak = torch.max(torch.abs(waveform))
        if peak > 1e-4:
            waveform = (waveform / peak) * 0.95

    return waveform


def save_audio(
    waveform: torch.Tensor,
    file_path: Union[str, Path],
    sample_rate: int = 22050,
) -> None:
    """Save audio waveform tensor to disk as 16-bit PCM WAV.
    
    Args:
        waveform: Tensor of shape [1, 1, T], [1, T], or [T].
        file_path: Output filepath.
        sample_rate: Audio sampling rate.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu()
    
    # Squeeze leading batch and channel dimensions to get [T] for mono
    while waveform.ndim > 1 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    
    data = waveform.numpy()
    if data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T
        
    sf.write(str(file_path), data, sample_rate, format="WAV", subtype="PCM_16")
