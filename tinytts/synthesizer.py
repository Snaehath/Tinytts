"""High-level synthesis API for TinyTTS."""

from pathlib import Path
from typing import Optional, Tuple, Union
import soundfile as sf
import torch

from tinytts.config import AudioConfig, ModelConfig, TinyTTSConfig
from tinytts.model.tinytts import TinyTTSA0
from tinytts.text.phoneme_tokenizer import PhonemeTokenizer


RELEASE_CHECKPOINT_URL = "https://github.com/Snaehath/Tinytts/releases/download/v0.2.1/tinytts_20_a2_1_best.pt"


def load_model(
    checkpoint_path: Union[str, Path] = "checkpoints/tinytts_20_a2_1_best.pt",
    device: Optional[Union[str, torch.device]] = None,
) -> TinyTTSA0:
    """Load TinyTTS-20-A2.1 model from checkpoint (auto-downloads from GitHub Release if missing)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Checkpoint not found at {ckpt_path}. Downloading from GitHub Releases...")
        import urllib.request
        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=ckpt_path.name) as t:
            urllib.request.urlretrieve(RELEASE_CHECKPOINT_URL, filename=str(ckpt_path), reporthook=t.update_to)
        print(f"[✓] Checkpoint downloaded successfully to {ckpt_path}")

    # TinyTTS-20-A2.1 architecture configuration
    config = TinyTTSConfig(
        model=ModelConfig(
            vocab_size=128,
            text_hidden_dim=320,
            text_num_layers=6,
            text_ffn_dim=1280,
            flow_hidden_dim=448,
            flow_num_layers=10,
            duration_filter_channels=256,
            use_conditioning_adapter=True,
            adapter_type="multiscale",
            adapter_kernel_sizes=(3, 7, 15),
            adapter_expansion=2,
            adapter_num_layers=1,
            flow_steps=8,
        ),
        flow_steps=8,
    )

    model = TinyTTSA0(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def synthesize(
    text: str,
    output_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Union[str, Path] = "checkpoints/tinytts_20_a2_1_best.pt",
    model: Optional[TinyTTSA0] = None,
    device: Optional[Union[str, torch.device]] = None,
    flow_steps: int = 8,
    temperature: float = 0.8,
    pace_factor: float = 1.0,
) -> Tuple[torch.Tensor, int]:
    """Synthesize speech waveform from text.

    Args:
        text: Input text string (supports plain text and expressive tags like '[whisper]').
        output_path: Optional destination .wav path.
        checkpoint_path: Path to .pt checkpoint if model is not pre-loaded.
        model: Optional pre-loaded TinyTTSA0 model instance.
        device: 'cuda' or 'cpu'.
        flow_steps: Euler ODE steps (default: 8).
        temperature: Sampling temperature for prior noise.
        pace_factor: Speed multiplier (1.0 = normal, 0.8 = fast, 1.2 = slow).

    Returns:
        (waveform_tensor, sample_rate)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if model is None:
        model = load_model(checkpoint_path, device=device)

    tokenizer = PhonemeTokenizer(add_bos_eos=True, intersperse_blanks=True)
    tokens, _ = tokenizer.tokenize_with_tags(text)
    tokens_tensor = tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        _, wav = model.synthesize(
            text_tokens=tokens_tensor,
            use_duration_predictor=True,
            num_flow_steps=flow_steps,
            temperature=temperature,
            pace_factor=pace_factor,
        )

    wav = wav.squeeze().cpu()
    sr = model.config.audio.sample_rate

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), wav.numpy(), sr)

    return wav, sr
