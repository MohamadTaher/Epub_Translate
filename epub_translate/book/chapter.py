"""One document out of the EPUB, as it travels through a run."""

from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup

from ..tokens import count_tokens


@dataclass(eq=False)
class Chapter:
    """
    A chapter of the source book, mutated in place as it is translated: the
    worker replaces `soup` with the model's answer, and the writer folds
    whatever is on the object into the EPUB at the next save.

    `eq=False` keeps identity equality and hashing. A chapter is a position in
    one book rather than a value worth comparing to another, and the server maps
    a chapter back to the patch carrying it by `id()`.
    """

    id: str
    title: str
    file_name: str
    # Out of the repr, both of them: a chapter is a whole document, and one in a
    # traceback or a log line would bury it.
    soup: BeautifulSoup = field(repr=False)

    # Set while reading the book, from how much of the text is still in the
    # source script.
    already_translated: bool = False
    skip_reason: Optional[str] = None

    # Filled in by `measure`, and kept, because a plan is re-costed every time a
    # reader ticks a chapter on or off.
    source_html: Optional[str] = field(default=None, repr=False)
    source_tokens: int = 0

    # Set by the worker once the model has answered.
    translated_title: Optional[str] = None

    def measure(self):
        """
        Cache the pre-translation HTML and what it costs to send.

        Idempotent, and it has to be: the cached HTML is the *source*, while
        `soup` is replaced by the translation partway through a run.
        """
        if self.source_html is None:
            self.source_html = str(self.soup)
            self.source_tokens = count_tokens(self.source_html)

    @property
    def display_title(self) -> str:
        """The translated title once there is one, the original until then."""
        return self.translated_title or self.title
