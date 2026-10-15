"""Text normalizer for English speech synthesis.

Expands numbers, abbreviations, symbols, and dates into spoken words
while preserving special tokens like <pause>, <bos>, <eos>.
"""

import re
from typing import Dict, List, Tuple


# Common English abbreviations
ABBREVIATIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bMrs\.", re.IGNORECASE), "Missus"),
    (re.compile(r"\bMr\.", re.IGNORECASE), "Mister"),
    (re.compile(r"\bDr\.", re.IGNORECASE), "Doctor"),
    (re.compile(r"\bSt\.", re.IGNORECASE), "Saint"),
    (re.compile(r"\bCo\.", re.IGNORECASE), "Company"),
    (re.compile(r"\bCorp\.", re.IGNORECASE), "Corporation"),
    (re.compile(r"\bJr\.", re.IGNORECASE), "Junior"),
    (re.compile(r"\bSr\.", re.IGNORECASE), "Senior"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.", re.IGNORECASE), "etcetera"),
]

# Number names
ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
THOUSANDS = ["", "thousand", "million", "billion"]

ORDINALS: Dict[str, str] = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth"
}


def _number_to_words_below_thousand(n: int) -> str:
    if n == 0:
        return ""
    elif n < 20:
        return ONES[n]
    elif n < 100:
        remainder = n % 10
        return TENS[n // 10] + ("-" + ONES[remainder] if remainder else "")
    else:
        remainder = n % 100
        prefix = ONES[n // 100] + " hundred"
        return prefix + (" and " + _number_to_words_below_thousand(remainder) if remainder else "")


def number_to_words(num: int) -> str:
    """Convert integer to English words."""
    if num == 0:
        return "zero"
    if num < 0:
        return "minus " + number_to_words(abs(num))
    
    # Handle 4-digit years like 1999 or 2026 naturally
    if 1000 <= num <= 2099 and num % 100 != 0 and num % 1000 >= 100:
        century = num // 100
        decade = num % 100
        if decade < 10:
            return f"{_number_to_words_below_thousand(century)} oh {_number_to_words_below_thousand(decade)}"
        return f"{_number_to_words_below_thousand(century)} {_number_to_words_below_thousand(decade)}"

    parts = []
    chunk_idx = 0
    while num > 0:
        chunk = num % 1000
        if chunk != 0:
            chunk_str = _number_to_words_below_thousand(chunk)
            scale = THOUSANDS[chunk_idx]
            if scale:
                chunk_str += " " + scale
            parts.append(chunk_str)
        num //= 1000
        chunk_idx += 1

    return " ".join(reversed(parts))


def expand_numbers(text: str) -> str:
    """Expand numbers, ordinals, and currencies."""
    # Currencies
    text = re.sub(r"\$(\d+)", lambda m: f"{number_to_words(int(m.group(1)))} dollars", text)
    # Ordinals
    for ord_str, word in ORDINALS.items():
        text = re.sub(rf"\b{ord_str}\b", word, text, flags=re.IGNORECASE)
    # Generic integers
    text = re.sub(r"\b\d+\b", lambda m: number_to_words(int(m.group(0))), text)
    return text


def expand_abbreviations(text: str) -> str:
    """Expand standard English abbreviations."""
    for pattern, replacement in ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(text: str) -> str:
    """Applies normalization pipeline to input text.
    
    Preserves special tokens like <pause> while expanding numbers and abbreviations.
    """
    # 1. Protect special tokens
    special_placeholder_map = {
        "<pause>": "__SPECIAL_PAUSE__",
        "<bos>": "__SPECIAL_BOS__",
        "<eos>": "__SPECIAL_EOS__",
        "<pad>": "__SPECIAL_PAD__",
    }
    for token, placeholder in special_placeholder_map.items():
        text = text.replace(token, placeholder)

    # 2. Expand abbreviations
    text = expand_abbreviations(text)

    # 3. Expand numbers
    text = expand_numbers(text)

    # 4. Standardize dashes and quotes
    text = text.replace("—", " - ").replace("–", " - ")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 5. Restore special tokens
    for token, placeholder in special_placeholder_map.items():
        text = text.replace(placeholder, f" {token} ")

    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
