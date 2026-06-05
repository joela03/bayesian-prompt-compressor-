"""
Shared persona-detection patterns.

Lives in its own module so both `parser.py` and `validators.py` can import
from it without introducing a circular dependency between them.
"""

import re

PERSONA_PATTERNS = (
    'you are',
    'act as',
    'i want you to act',
    'pretend to be',
    'pretend you are',
    'imagine you are',
    "let's role[- ]?play as",
    'you will (be|act as|play)',
    'assume the role',
)


def persona_present(text: str) -> bool:
    """True if text opens with a recognised persona pattern."""
    first_line = text.lstrip().split('\n', 1)[0].strip()
    first_sentence = re.split(r'[.!?]', first_line, maxsplit=1)[0].strip().lower()
    return any(re.match(rf'^{pat}\b', first_sentence) for pat in PERSONA_PATTERNS)
