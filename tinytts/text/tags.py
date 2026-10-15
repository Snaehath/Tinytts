"""Tag parser and schema for controllable expressive speech synthesis.

Handles formal tag syntax [category:value key=val] and intuitive aliases:
- Class A (Deterministic Timing): [pause], [short pause], [long pause], [slow], [fast]
- Class B (Learned Delivery): [whisper], [shout], [softly], [muttering]
- Class C (Learned Emotion): [excited], [happy], [sad], [angry], [calm], [worried], [sarcastic]

Separates tags from phonation text so the acoustic model never pronounces tag names.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TagClass(str, Enum):
    CLASS_A = "deterministic_timing"  # Pause, pace
    CLASS_B = "learned_delivery"      # Whisper, shout
    CLASS_C = "learned_emotion"       # Excited, happy, angry
    NONVERBAL = "nonverbal"           # Laugh, sigh


# Canonical lists of known classes
EMOTIONS = ["neutral", "excited", "happy", "sad", "angry", "calm", "worried", "sarcastic"]
DELIVERIES = ["normal", "whisper", "shout", "softly", "muttering", "dramatic"]

EMOTION_TO_ID = {name: idx for idx, name in enumerate(EMOTIONS)}
DELIVERY_TO_ID = {name: idx for idx, name in enumerate(DELIVERIES)}


@dataclass
class TagEvent:
    category: str
    value: str
    tag_class: TagClass
    char_offset: int
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedUtterance:
    raw_text: str
    clean_text: str
    events: List[TagEvent]
    emotion: str = "neutral"
    delivery: str = "normal"
    pace: float = 1.0
    pitch: float = 1.0
    energy: float = 1.0

    @property
    def emotion_id(self) -> int:
        return EMOTION_TO_ID.get(self.emotion, 0)

    @property
    def delivery_id(self) -> int:
        return DELIVERY_TO_ID.get(self.delivery, 0)


# Friendly alias mapping
ALIAS_MAP: Dict[str, Tuple[str, str, TagClass, Dict[str, Any]]] = {
    # Class A - Timing & Pauses
    "pause": ("timing", "pause", TagClass.CLASS_A, {"duration_sec": 0.3}),
    "short pause": ("timing", "pause", TagClass.CLASS_A, {"duration_sec": 0.15}),
    "long pause": ("timing", "pause", TagClass.CLASS_A, {"duration_sec": 0.6}),
    "slow": ("timing", "pace", TagClass.CLASS_A, {"rate": 0.8}),
    "fast": ("timing", "pace", TagClass.CLASS_A, {"rate": 1.25}),
    "loud": ("timing", "energy", TagClass.CLASS_A, {"energy": 1.3}),
    "soft": ("timing", "energy", TagClass.CLASS_A, {"energy": 0.7}),

    # Class B - Learned Delivery
    "whisper": ("delivery", "whisper", TagClass.CLASS_B, {"intensity": 1.0}),
    "shout": ("delivery", "shout", TagClass.CLASS_B, {"intensity": 1.0}),
    "softly": ("delivery", "softly", TagClass.CLASS_B, {"intensity": 1.0}),
    "muttering": ("delivery", "muttering", TagClass.CLASS_B, {"intensity": 1.0}),
    "dramatic": ("delivery", "dramatic", TagClass.CLASS_B, {"intensity": 1.0}),

    # Class C - Learned Emotion
    "excited": ("emotion", "excited", TagClass.CLASS_C, {"intensity": 1.0}),
    "happy": ("emotion", "happy", TagClass.CLASS_C, {"intensity": 1.0}),
    "sad": ("emotion", "sad", TagClass.CLASS_C, {"intensity": 1.0}),
    "angry": ("emotion", "angry", TagClass.CLASS_C, {"intensity": 1.0}),
    "calm": ("emotion", "calm", TagClass.CLASS_C, {"intensity": 1.0}),
    "worried": ("emotion", "worried", TagClass.CLASS_C, {"intensity": 1.0}),
    "sarcastic": ("emotion", "sarcastic", TagClass.CLASS_C, {"intensity": 1.0}),

    # Non-verbal reactions
    "laughs": ("nonverbal", "laugh", TagClass.NONVERBAL, {}),
    "sighs": ("nonverbal", "sigh", TagClass.NONVERBAL, {}),
    "gasps": ("nonverbal", "gasp", TagClass.NONVERBAL, {}),
    "clears throat": ("nonverbal", "clear_throat", TagClass.NONVERBAL, {}),
}

# Regex to find tags enclosed in brackets: e.g. [whisper], [emotion:excited intensity=0.8]
TAG_PATTERN = re.compile(r"\[([a-zA-Z0-9_\s:=.\-]+)\]")
PARAM_PATTERN = re.compile(r"([a-zA-Z_]+)=([0-9.]+)")


def parse_tag_content(tag_str: str) -> Optional[Tuple[str, str, TagClass, Dict[str, Any]]]:
    """Parse tag content inside brackets into (category, value, tag_class, attributes)."""
    tag_str = tag_str.strip().lower()

    # 1. Check alias lookup
    if tag_str in ALIAS_MAP:
        cat, val, t_class, default_attrs = ALIAS_MAP[tag_str]
        return cat, val, t_class, dict(default_attrs)

    # 2. Check canonical syntax: category:value or category=value
    # e.g. [emotion:excited intensity=0.8] or [pace=0.8]
    parts = tag_str.split()
    if not parts:
        return None

    head = parts[0]
    attributes: Dict[str, Any] = {}

    for param_token in parts[1:]:
        match = PARAM_PATTERN.match(param_token)
        if match:
            k, v = match.groups()
            try:
                attributes[k] = float(v)
            except ValueError:
                attributes[k] = v

    if ":" in head:
        cat, val = head.split(":", 1)
    elif "=" in head:
        cat, val = head.split("=", 1)
        try:
            val_num = float(val)
            attributes[cat] = val_num
        except ValueError:
            pass
    else:
        # Unknown single tag, treat as custom delivery or ignore gracefully
        return None

    if cat == "emotion":
        t_class = TagClass.CLASS_C
    elif cat in ("delivery", "style"):
        t_class = TagClass.CLASS_B
    elif cat in ("timing", "pause", "pace"):
        t_class = TagClass.CLASS_A
    else:
        t_class = TagClass.CLASS_B

    return cat, val, t_class, attributes


def parse_tags(text: str, insert_pause_token: bool = True) -> ParsedUtterance:
    """Parses text containing expressive markup tags.
    
    Args:
        text: Input string, e.g. "[whisper] Don't tell anyone... [pause] We won!"
        insert_pause_token: If True, replaces deterministic pauses with '<pause>'
                            token so acoustic duration model can synthesize the silence.

    Returns:
        ParsedUtterance containing clean text, events, and resolved control states.
    """
    events: List[TagEvent] = []
    clean_parts: List[str] = []
    last_idx = 0
    current_char_offset = 0

    emotion = "neutral"
    delivery = "normal"
    pace = 1.0
    pitch = 1.0
    energy = 1.0

    for match in TAG_PATTERN.finditer(text):
        start, end = match.span()
        # Text preceding this tag
        preceding = text[last_idx:start]
        if preceding:
            clean_parts.append(preceding)
            current_char_offset += len(preceding)

        tag_content = match.group(1)
        parsed = parse_tag_content(tag_content)

        if parsed is not None:
            cat, val, t_class, attrs = parsed
            event = TagEvent(
                category=cat,
                value=val,
                tag_class=t_class,
                char_offset=current_char_offset,
                attributes=attrs,
            )
            events.append(event)

            # Update utterance-level dominant controls
            if cat == "emotion" and val in EMOTION_TO_ID:
                emotion = val
            elif cat == "delivery" and val in DELIVERY_TO_ID:
                delivery = val
            elif cat in ("timing", "pace") or "pace" in attrs or "rate" in attrs:
                if "pace" in attrs:
                    pace = float(attrs["pace"])
                elif "rate" in attrs:
                    pace = float(attrs["rate"])
            elif "energy" in attrs:
                energy = float(attrs["energy"])
            elif "pitch" in attrs:
                pitch = float(attrs["pitch"])

            # Class A pause replacement: insert acoustic pause marker if requested
            if cat == "timing" and val == "pause" and insert_pause_token:
                pause_str = " <pause> "
                clean_parts.append(pause_str)
                current_char_offset += len(pause_str)

        last_idx = end

    # Remaining text after last tag
    trailing = text[last_idx:]
    if trailing:
        clean_parts.append(trailing)

    clean_text = "".join(clean_parts)
    # Normalize multiple whitespace created by stripped tags
    clean_text = re.sub(r"[ \t]+", " ", clean_text).strip()

    return ParsedUtterance(
        raw_text=text,
        clean_text=clean_text,
        events=events,
        emotion=emotion,
        delivery=delivery,
        pace=pace,
        pitch=pitch,
        energy=energy,
    )
