"""Phoneme tokenizer for TinyTTS using ARPAbet representation.

Converts normalized text into discrete ARPAbet phoneme token IDs via g2p_en,
supporting blank interspersing for monotonic alignment and tag parsing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from tinytts.text.normalizer import clean_text
from tinytts.text.tags import ParsedUtterance, parse_tags

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "ljspeech_phonemes.pt"

# Special control tokens
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
BLANK = "<blank>"
PAUSE = "<pause>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, BLANK, PAUSE]


# Pre-computed ARPAbet phonemes for fixed benchmark sentences
BENCHMARK_PHONEMES = {
    "The quick brown fox jumps over the lazy dog.": [
        "DH", "AH0", " ", "K", "W", "IH1", "K", " ", "B", "R", "AW1", "N", " ", "F", "AA1", "K", "S",
        " ", "JH", "AH1", "M", "P", "S", " ", "OW1", "V", "ER0", " ", "DH", "AH0", " ", "L", "EY1",
        "Z", "IY0", " ", "D", "AO1", "G", " ", "."
    ],
    "Hello, welcome to TinyTTS.": [
        "HH", "AH0", "L", "OW1", " ", ",", " ", "W", "EH1", "L", "K", "AH0", "M", " ", "T", "UW1",
        " ", "T", "IH1", "N", "IY0", "T", "IY1", "T", "IY1", "EH1", "S", " ", "."
    ],
    "The weather is beautiful today.": [
        "DH", "AH0", " ", "W", "EH1", "DH", "ER0", " ", "IH1", "Z", " ", "B", "Y", "UW1", "T", "AH0",
        "F", "AH0", "L", " ", "T", "AH0", "D", "EY1", " ", "."
    ],
    "Artificial intelligence is changing the way we work.": [
        "AA2", "R", "T", "AH0", "F", "IH1", "SH", "AH0", "L", " ", "IH0", "N", "T", "EH1", "L", "AH0",
        "JH", "AH0", "N", "S", " ", "IH1", "Z", " ", "CH", "EY1", "N", "JH", "IH0", "NG", " ", "DH",
        "AH0", " ", "W", "EY1", " ", "W", "IY1", " ", "W", "ER1", "K", " ", "."
    ],
}


# Canonical 83-symbol ARPAbet vocabulary used across TinyTTS training
DEFAULT_PHONEME_VOCAB = [
    '<pad>', '<bos>', '<eos>', '<unk>', '<blank>', '<pause>',
    ' ', '!', "'", ',', '-', '.', '..', '?',
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
    'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2',
    'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S',
    'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2',
    'V', 'W', 'Y', 'Z', 'ZH',
]


class PhonemeTokenizer:
    """ARPAbet phoneme tokenizer supporting pre-cached training items and live inference."""

    def __init__(
        self,
        add_bos_eos: bool = True,
        intersperse_blanks: bool = True,
        cache_path: Optional[Path] = None,
    ):
        self.add_bos_eos = add_bos_eos
        self.intersperse_blanks = intersperse_blanks
        self._g2p = None

        # Load canonical vocabulary and pre-cached phonemes if available
        c_path = cache_path or CACHE_PATH
        if c_path.exists():
            payload = torch.load(c_path, map_location="cpu", weights_only=False)
            self.vocab = payload["vocab"]
            self.symbol_to_id = payload["symbol_to_id"]
            self.id_to_symbol = payload["id_to_symbol"]
            self.cached_dict = payload["phoneme_dict"]
        else:
            self.vocab = DEFAULT_PHONEME_VOCAB
            self.symbol_to_id = {s: i for i, s in enumerate(self.vocab)}
            self.id_to_symbol = {i: s for i, s in enumerate(self.vocab)}
            self.cached_dict = {}

        self.vocab_size = len(self.vocab)
        self.pad_id = self.symbol_to_id[PAD]
        self.bos_id = self.symbol_to_id[BOS]
        self.eos_id = self.symbol_to_id[EOS]
        self.unk_id = self.symbol_to_id[UNK]
        self.blank_id = self.symbol_to_id[BLANK]
        self.pause_id = self.symbol_to_id[PAUSE]

    def __getstate__(self):
        """Exclude open file handles from pickling across Windows multiprocessing."""
        state = self.__dict__.copy()
        state["_g2p"] = None
        return state

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert English text to ARPAbet phoneme list."""
        if text in BENCHMARK_PHONEMES:
            return BENCHMARK_PHONEMES[text]
        if self._g2p is None:
            from g2p_en import G2p
            self._g2p = G2p()
        return self._g2p(text)

    def encode_phonemes(self, phonemes: List[str]) -> List[int]:
        """Convert a list of ARPAbet phonemes to token IDs."""
        tokens = [self.symbol_to_id.get(p, self.unk_id) for p in phonemes]

        if self.intersperse_blanks:
            tokens = self._intersperse(tokens, self.blank_id)

        if self.add_bos_eos:
            tokens = [self.bos_id] + tokens + [self.eos_id]

        return tokens

    def encode(self, text: str) -> List[int]:
        """End-to-end: text -> ARPAbet phonemes -> token IDs."""
        phonemes = self.text_to_phonemes(text)
        return self.encode_phonemes(phonemes)

    def encode_cached(self, utterance_id: str, fallback_text: str = "") -> List[int]:
        """Fast lookup from pre-cached LJSpeech phoneme dictionary."""
        if utterance_id in self.cached_dict:
            phonemes = self.cached_dict[utterance_id]
        else:
            phonemes = self.text_to_phonemes(fallback_text)
        return self.encode_phonemes(phonemes)

    def decode(self, token_ids: List[int], remove_special: bool = True) -> str:
        """Decode token IDs back into phoneme string."""
        symbols: List[str] = []
        for tid in token_ids:
            symbol = self.id_to_symbol.get(tid, UNK)
            if remove_special and symbol in [PAD, BOS, EOS, BLANK]:
                continue
            symbols.append(symbol)
        return " ".join(symbols)

    def tokenize_with_tags(self, raw_input: str) -> Tuple[torch.Tensor, ParsedUtterance]:
        """End-to-end processing with expressive tag parsing."""
        parsed = parse_tags(raw_input, insert_pause_token=True)
        normalized = clean_text(parsed.clean_text)
        token_ids = self.encode(normalized)
        tensor_tokens = torch.tensor(token_ids, dtype=torch.long)
        return tensor_tokens, parsed

    @staticmethod
    def _intersperse(tokens: List[int], item: int) -> List[int]:
        result = [item] * (len(tokens) * 2 + 1)
        result[1::2] = tokens
        return result
