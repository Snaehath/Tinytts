"""Character tokenizer for TinyTTS.

Converts normalized text and special markup events into discrete token IDs,
supporting blank interspersing for monotonic alignment and control vector extraction.
"""

from typing import List, Optional, Tuple
import torch

from tinytts.text.normalizer import clean_text
from tinytts.text.symbols import (
    BLANK,
    BOS,
    EOS,
    ID_TO_SYMBOL,
    PAD,
    PAUSE,
    SPECIAL_TOKENS,
    SYMBOL_TO_ID,
    UNK,
    VOCAB_SIZE,
)
from tinytts.text.tags import ParsedUtterance, parse_tags


class CharacterTokenizer:
    """Character-level tokenizer supporting special acoustic tokens and expressive tag parsing."""

    def __init__(self, add_bos_eos: bool = True, intersperse_blanks: bool = False):
        self.add_bos_eos = add_bos_eos
        self.intersperse_blanks = intersperse_blanks
        self.vocab_size = VOCAB_SIZE
        self.pad_id = SYMBOL_TO_ID[PAD]
        self.bos_id = SYMBOL_TO_ID[BOS]
        self.eos_id = SYMBOL_TO_ID[EOS]
        self.unk_id = SYMBOL_TO_ID[UNK]
        self.blank_id = SYMBOL_TO_ID[BLANK]
        self.pause_id = SYMBOL_TO_ID[PAUSE]

    def encode(self, text: str) -> List[int]:
        """Convert a clean string into a list of token IDs."""
        tokens: List[int] = []

        # Tokenize by checking special tokens first (e.g. <pause>), then individual characters
        idx = 0
        n = len(text)
        while idx < n:
            matched_special = False
            for special in SPECIAL_TOKENS:
                if text.startswith(special, idx):
                    tokens.append(SYMBOL_TO_ID[special])
                    idx += len(special)
                    matched_special = True
                    break

            if not matched_special:
                char = text[idx]
                tokens.append(SYMBOL_TO_ID.get(char, self.unk_id))
                idx += 1

        if self.intersperse_blanks:
            tokens = self._intersperse(tokens, self.blank_id)

        if self.add_bos_eos:
            tokens = [self.bos_id] + tokens + [self.eos_id]

        return tokens

    def decode(self, token_ids: List[int], remove_special: bool = True) -> str:
        """Decode token IDs back into string."""
        chars: List[str] = []
        for tid in token_ids:
            symbol = ID_TO_SYMBOL.get(tid, UNK)
            if remove_special and symbol in [PAD, BOS, EOS, BLANK]:
                continue
            chars.append(symbol)
        return "".join(chars)

    def tokenize_with_tags(self, raw_input: str) -> Tuple[torch.Tensor, ParsedUtterance]:
        """End-to-end processing: tag parsing -> text normalization -> character tokenization.
        
        Args:
            raw_input: Raw text with or without tags, e.g. "[whisper] Don't tell anyone."

        Returns:
            (token_tensor, parsed_utterance)
        """
        # Step 1: Parse expressive tags and remove them from phonation text
        parsed = parse_tags(raw_input, insert_pause_token=True)

        # Step 2: Normalize phonation text (expand numbers, clean whitespace)
        normalized = clean_text(parsed.clean_text)

        # Step 3: Encode to IDs
        token_ids = self.encode(normalized)
        tensor_tokens = torch.tensor(token_ids, dtype=torch.long)

        return tensor_tokens, parsed

    @staticmethod
    def _intersperse(tokens: List[int], blank_id: int) -> List[int]:
        """Intersperses blank tokens between elements: [1, 2] -> [blank, 1, blank, 2, blank]."""
        result = [blank_id] * (len(tokens) * 2 + 1)
        result[1::2] = tokens
        return result
