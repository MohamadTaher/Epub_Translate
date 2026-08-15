"""
What a piece of text costs to send.

Gemini's own tokenizer isn't available offline, so this counts with OpenAI's
cl100k_base instead. The two disagree, but everything derived from this number —
how chapters are packed into requests, how a run is paced, the estimate a reader
watches — only needs to be consistently approximate, and it is the same counter
on both sides of every comparison it takes part in.
"""

import threading
from typing import Optional

import tiktoken

from .console import logger

# Rough cost per character, by script, measured against cl100k_base on the
# languages this tool translates from. CJK runs about 1.25 tokens to the
# character; Cyrillic and Greek about 0.65; ASCII about 0.25. One ratio cannot
# serve all three, and assuming the ASCII one made the estimate five times too
# low for a Chinese book — in the direction that overruns the token budget.
_CJK_START = 0x2E80
_NON_LATIN_START = 0x250

_TOKENS_PER_CJK_CHAR = 1.25
_TOKENS_PER_NON_LATIN_CHAR = 0.65
_TOKENS_PER_ASCII_CHAR = 0.25

_lock = threading.Lock()
_encoding = None
_unavailable = False


def count_tokens(text: str) -> int:
    """
    How many tokens `text` is worth.

    Falls back to `estimate_tokens` if the tokenizer can't be loaded, which on
    a normal install means the image was built without its cached vocabulary.
    """
    encoding = _load_encoding()
    if encoding is None:
        return estimate_tokens(text)
    return len(encoding.encode(text))


def estimate_tokens(text: str) -> int:
    """
    A token count from the characters alone, for when the tokenizer is missing.

    Deliberately generous on non-Latin scripts. This number is what the rate
    limiter reserves against, so guessing high costs a little throughput and
    guessing low overruns the per-minute quota the limiter exists to respect.
    """
    cjk = 0
    non_latin = 0
    for char in text:
        code_point = ord(char)
        if code_point >= _CJK_START:
            cjk += 1
        elif code_point >= _NON_LATIN_START:
            non_latin += 1

    ascii_ish = len(text) - cjk - non_latin
    return round(cjk * _TOKENS_PER_CJK_CHAR
                 + non_latin * _TOKENS_PER_NON_LATIN_CHAR
                 + ascii_ish * _TOKENS_PER_ASCII_CHAR)


def _load_encoding() -> Optional["tiktoken.Encoding"]:
    """
    The tokenizer, loaded once.

    Checked without the lock first because this is called for every chapter and
    every request, from all the worker threads at once. A failure is remembered
    rather than retried: loading it can reach for the network, and a book of
    four hundred chapters would otherwise try four hundred times.
    """
    global _encoding, _unavailable

    if _encoding is not None or _unavailable:
        return _encoding

    with _lock:
        if _encoding is None and not _unavailable:
            try:
                _encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                _unavailable = True
                logger.warning(f"Token counting is falling back to an estimate: {e}")

    return _encoding
