"""
The glossary: terms this book translates the same way every time.

It is built during the run, not only before it. Every request comes back with
the terms the model met in those chapters, they are merged in, and they travel
back out with the later requests that mention them — which is what keeps a
character's name from drifting between chapter 1 and chapter 40, when the model
only ever sees one batch at a time.

    terms.py      the Glossary itself: what is known, and how it grows
    protocol.py   what the model is asked for, and the parser for what it sends
    matching.py   which terms are worth sending with a given batch
    storage.py    the file on disk, and the one definition of its format
"""

from .protocol import split_translation
from .terms import Glossary, MergeResult

__all__ = ["Glossary", "MergeResult", "split_translation"]
