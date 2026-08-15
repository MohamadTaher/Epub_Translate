"""
The EPUB itself: reading one, the chapters it is made of, and writing the
translated copy back out.

Everything here needs an archive to work on. The judgement of what language a
book is written in does not — it reads plain text — so it lives a level up in
`language.py`, beside `tokens.py`, and is imported by the reader and the server
alike.
"""

from .chapter import Chapter
from .reader import BookInfo, SourceBook, extract_chapter_title
from .writer import EpubWriter

__all__ = ["BookInfo", "Chapter", "EpubWriter", "SourceBook", "extract_chapter_title"]
