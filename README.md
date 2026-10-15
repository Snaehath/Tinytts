# TinyTTS: Lightweight Flow-Matching Text-to-Speech

**TinyTTS** is an ultra-fast, lightweight (~21M parameter) non-autoregressive Text-to-Speech system built with PyTorch and designed for real-time speech generation on consumer GPUs.

It couples a compact Transformer Text Encoder with a Multi-Scale Temporal Conditioning Adapter, Monotonic Alignment Search (MAS), Conditional Flow Matching (CFM) with sinusoidal time embeddings, and a pre-trained Vocos neural vocoder.

---

## Sample Audio

A sample generated audio file is included directly in the root of the repository:
- **`sample.wav`**: Generated from the text *"The quick brown fox jumps over the lazy dog."* using the **TinyTTS-20-A2.1** checkpoint at 8 Flow Matching integration steps (22,050 Hz).

---

## Performance & Benchmarks (TinyTTS-20-A2.1)

Evaluated on **100 held-out sentences** from the LJSpeech evaluation set on a consumer **NVIDIA RTX 3050 (4GB VRAM)**:

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Total Parameters** | **21.08 M** | Compact non-autoregressive flow model |
| **Real-Time Factor (RTF)** | **0.043** | Synthesizes 1s of audio in ~43ms (23x faster than real-time) |
| **Inference Latency (P50)** | **94.1 ms** | High-speed response time for interactive use |
| **Flow Matching Steps** | **8 steps** | Fast Euler ODE integration |
| **Pitch Correlation ($r_{\text{F0}}$)** | **0.631** | Pearson correlation with ground-truth pitch contour |
| **Word Error Rate (WER)** | **50.2%** | Measured via pretrained Wav2Vec2-Large-960h ASR |
| **Character Error Rate (CER)** | **24.1%** | Character-level phoneme transcription accuracy |
| **VRAM Consumption** | **< 1.2 GB** | Fits comfortably on 4GB consumer GPUs |

---

## Dataset & License

### Dataset Used
- **Dataset**: [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/)
- **Speaker**: Single female speaker (Linda Johnson)
- **Audio Duration**: ~24 hours of 22,050 Hz 16-bit PCM WAV audio across 13,100 passages read from 7 non-fiction books.
- **Phoneme Representation**: ARPAbet phonemes with blank interspersing for monotonic alignment.

### Dataset License
- **License**: **Public Domain (CC0 1.0 Universal)**
- The LJSpeech dataset is in the public domain in the USA. It is free for both academic and commercial use without attribution requirements.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA recommended (also supports CPU inference)

### 2. Clone the Repository
```bash
git clone https://github.com/Snaehath/Tinytts.git
cd Tinytts
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

> **Model Checkpoint**: No Git LFS needed. The checkpoint (`tinytts_20_a2_1_best.pt`) is hosted on [GitHub Releases](https://github.com/Snaehath/Tinytts/releases/tag/v0.2.1). When you run `tinytts.synthesize()`, it will automatically download the checkpoint on first run. You can also manually download it into the `checkpoints/` directory if preferred.

---

## How to Use

### Quickstart (Python API)
Synthesize speech with two lines of code:

```python
import tinytts

# Generate speech and save to WAV
tinytts.synthesize(
    text="The quick brown fox jumps over the lazy dog.",
    output_path="output.wav"
)
```

### Advanced Usage & Expressive Controls

```python
from tinytts import load_model, synthesize

# Pre-load model once for low-latency batch generation
model = load_model("checkpoints/tinytts_20_a2_1_best.pt", device="cuda")

# Expressive text synthesis with tag parsing and speed adjustment
wav, sr = synthesize(
    text="[whisper] Artificial intelligence is changing the way we work.",
    output_path="outputs/ai_whisper.wav",
    model=model,
    flow_steps=8,       # 8 Euler steps for quality; 6 for ultra-low latency
    temperature=0.8,    # Sampling temperature
    pace_factor=1.0,    # 1.0 = normal, 0.8 = fast, 1.2 = slow
)
```

---

## Model Checkpoint & Release Download
- **Release Version**: `v0.2.1`
- **File**: `tinytts_20_a2_1_best.pt` (~137 MB)
- **Download**: Directly attach or download from [GitHub Releases](https://github.com/Snaehath/Tinytts/releases/tag/v0.2.1).
- **Auto-Download**: If the checkpoint is not found locally at `checkpoints/tinytts_20_a2_1_best.pt`, `tinytts.load_model()` / `tinytts.synthesize()` will automatically download it on first use.

---

## Modelfile

A standardized `Modelfile` is provided at the repository root detailing model hyperparameters, vocoder specifications, tag schemas, and benchmark statistics.

---

## License

- **Code**: [Apache License 2.0](LICENSE) / MIT
- **Dataset**: [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) (Public Domain / CC0 1.0)
