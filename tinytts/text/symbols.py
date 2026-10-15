"""Character set and special tokens for TinyTTS character tokenizer."""

# Special control tokens
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
BLANK = "<blank>"  # Blank token useful for monotonic alignment / CTC
PAUSE = "<pause>"  # Deterministic acoustic pause marker (Class A)

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, BLANK, PAUSE]

# Standard English character vocabulary
CHARACTERS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?',.:;-\"()/")

# Full vocabulary: specials + characters
VOCAB = SPECIAL_TOKENS + [c for c in CHARACTERS if c not in SPECIAL_TOKENS]

# Lookup tables
SYMBOL_TO_ID = {s: i for i, s in enumerate(VOCAB)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
