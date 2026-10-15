"""TinyTTS: Expressive, Controllable, Lightweight Speech Synthesis for Consumer GPUs."""

__version__ = "0.2.1"

from tinytts.synthesizer import load_model, synthesize

__all__ = ["load_model", "synthesize"]
