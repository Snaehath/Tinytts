# TinyTTS: Lightweight Flow-Matching Text-to-Speech

**TinyTTS** is an ultra-fast, lightweight (~21M parameter) non-autoregressive Text-to-Speech system built with PyTorch and designed for real-time speech generation on consumer GPUs.

It couples a compact Transformer Text Encoder with a Multi-Scale Temporal Conditioning Adapter, Monotonic Alignment Search (MAS), Conditional Flow Matching (CFM) with sinusoidal time embeddings, and a pre-trained Vocos neural vocoder.

---

## Sample Audio

A sample generated audio file is included directly in the root of the repository:
- **`sample.wav`**: Generated from the text *"The quick brown fox jumps over the lazy dog."* using the **TinyTTS-20-A2.1** checkpoint at 8 Flow Matching integration steps (22,050 Hz).

---

## Benchmark Results

Evaluated across **100 held-out sentences** from the LJSpeech evaluation manifest using automatic speech recognition (Wav2Vec2-Large-960h) and pitch correlation analysis:

| Model Architecture | Params | WER (%) | CER (%) | Substitution Ratio (%) | $r_{\text{F0}}$ (Pitch Corr) | Latency (P50) | RTF (RTX 3050) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline TinyTTS-20** | 20.77 M | 54.28% | 26.68% | 80.8% | 0.6196 | 83.1 ms | 0.038 |
| **TinyTTS-20-A2** (ConvNeXt $k=7$) | 21.18 M | **48.31%** | **23.10%** | **80.4%** | 0.6187 | 94.3 ms | 0.043 |
| **TinyTTS-20-A2.1** (Multi-Scale $k \in \{3,7,15\}$) | 21.08 M | 50.22% | 24.13% | 80.9% | **0.6307** | 94.1 ms | 0.043 |
| **TinyTTS-20-A2.4** (AdaLN FiLM Conditioning) | 23.96 M | *Architecture Ablation* | — | — | — | — | — |

### Key Benchmark Insights
- **Temporal Context**: Adding a small depthwise convolutional adapter (+0.31M params) produces a **-5.97% absolute reduction in WER**.
- **Multi-Scale Conditioning (A2.1)**: Parallel multi-resolution kernel receptive fields ($k \in \{3, 7, 15\}$) achieve the highest pitch correlation ($r_{\text{F0}} = 0.6307$), improving prosodic contour fidelity while keeping latency under 95 ms.
- **Inference Speed**: Using an 8-step Euler ODE solver, TinyTTS achieves an RTF of **0.043** (generating 1 second of speech in ~43 milliseconds on a budget RTX 3050 GPU).

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
- [Git Large File Storage (Git LFS)](https://git-lfs.com) (required to clone the 312MB `.pt` checkpoint)
- NVIDIA GPU with CUDA recommended (runs on CPU as well)

### 2. Clone the Repository
```bash
# Ensure Git LFS is installed and initialized
git lfs install

# Clone the repository
git clone https://github.com/<your-username>/tts.git
cd tts

# Fetch the large checkpoint file via LFS
git lfs pull
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

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
- **File**: `tinytts_20_a2_1_best.pt` (312 MB)
- **Download**: Directly attach or download from [GitHub Releases](https://github.com/Snaehath/Tinytts/releases/tag/v0.2.1).
- **Auto-Download**: If the checkpoint is not found locally at `checkpoints/tinytts_20_a2_1_best.pt`, `tinytts.load_model()` / `tinytts.synthesize()` will automatically download it on first use.

---

## Modelfile

A standardized `Modelfile` is provided at the repository root detailing model hyperparameters, vocoder specifications, tag schemas, and benchmark statistics.

---

## License

- **Code**: [Apache License 2.0](LICENSE) / MIT
- **Dataset**: [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) (Public Domain / CC0 1.0)
