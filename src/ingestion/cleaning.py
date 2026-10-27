"""Text normalisation applied to every extracted unit before chunking."""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise unicode and whitespace while keeping paragraph breaks.

    - NFKC normalisation (ligatures such as "ﬁ" become "fi")
    - control characters removed
    - words hyphenated across line breaks re-joined ("infor-\\nmation")
    - runs of spaces collapsed, single line breaks inside paragraphs kept
    - 3+ consecutive newlines collapsed to one blank line
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
