"""Configuration module for TinyTTS.

Centralizes audio preprocessing, model hyperparameters, training configurations,
and directory paths for 4GB RTX 3050 GPU constraints.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    mel_norm: str = "slaney"
    power: float = 1.0  # magnitude mel spectrogram
    dynamic_range_compression: bool = True
    clip_val: float = 1e-5


@dataclass(frozen=True)
class TextConfig:
    add_blank_token: bool = True
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    pause_token: str = "<pause>"  # Explicit acoustic pause event for Class A deterministic tags


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters optimized for RTX 3050 4GB VRAM budget."""
    vocab_size: int = 128  # updated dynamically from tokenizer
    text_hidden_dim: int = 320
    text_num_layers: int = 6
    text_num_heads: int = 4
    text_ffn_dim: int = 1280
    
    # Control Vector Conditioning (Architecture ready from Day 1)
    control_dim: int = 64
    num_emotions: int = 8      # neutral, excited, happy, sad, angry, calm, worried, sarcastic
    num_deliveries: int = 6    # normal, whisper, shout, muttering, softly, dramatic
    
    # Acoustic Generator (Flow matching backbone)
    mel_channels: int = 80
    flow_hidden_dim: int = 320
    flow_num_layers: int = 8
    
    # Flow Matching ODE solver steps: 8 = quality default, 6 = low-latency mode, 16 = reference/debug mode
    flow_steps: int = 8
    
    # Duration Predictor
    duration_filter_channels: int = 256
    duration_kernel_size: int = 3
    dropout: float = 0.1

    # TinyTTS-20-A2 & A2.1 Conditioning Adapter
    use_conditioning_adapter: bool = False
    adapter_type: str = "convnext"  # "convnext" (A2, single k=7) or "multiscale" (A2.1, parallel k=(3, 7, 15))
    adapter_kernel_size: int = 7
    adapter_kernel_sizes: Tuple[int, ...] = (3, 7, 15)
    adapter_expansion: int = 2
    adapter_num_layers: int = 1
    adapter_dropout: float = 0.1

    # TinyTTS-20-A2.4 AdaLN / FiLM Conditioning in Flow Backbone
    use_flow_adaln: bool = False
    flow_adaln_cond_dim: int = 320


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16  # Fits comfortably in 4GB with AMP (fp16)
    grad_accumulation_steps: int = 2  # Effective batch size = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    max_epochs: int = 100
    precision: str = "fp16"  # "fp16" or "fp32"
    num_workers: int = 2     # Windows friendly
    pin_memory: bool = True
    save_interval: int = 2000
    eval_interval: int = 500
    log_interval: int = 50
    seed: int = 42


@dataclass
class TinyTTSConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    flow_steps: Optional[int] = None  # Quick shortcut: 8=quality default, 6=low-latency, 16=reference
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    ljspeech_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "LJSpeech-1.1")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "checkpoints")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "logs")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.flow_steps is not None:
            import dataclasses
            self.model = dataclasses.replace(self.model, flow_steps=self.flow_steps)
