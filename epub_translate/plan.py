"""
What a translation would do, worked out before any of it is paid for.

Kept apart from the run itself because the server shows a reader this and then
waits: nothing here calls the API or changes a chapter, so a plan can be built,
discarded and built again as a reader changes their mind about which chapters to
translate.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .book import Chapter, SourceBook
from .packing import create_patches


@dataclass
class TranslationPlan:
    """
    What a translation run would do, worked out before any API call is made, so
    the cost can be reviewed and confirmed first.
    """
    chapters: List[Chapter]
    patches: List[List[Chapter]] = field(default_factory=list)
    total_tokens: int = 0
    source_language: str = "auto"


def build_plan(source: SourceBook, source_language: str, max_tokens_per_patch: int,
               only_chapter_ids: Optional[Set[str]] = None) -> TranslationPlan:
    """
    Work out what translating this book would involve, without calling the API.

    By default, chapters detected as already translated are left out. Pass
    `only_chapter_ids` to translate exactly the chapters named instead.

    Safe to call more than once on the same book: `source.chapters()` hands back
    fresh objects every time, so re-planning never inherits a previous plan's
    state.

    The returned `source_language` is what the book turned out to be written in.
    Detection happens inside `source.chapters` and nowhere else, so this is the
    one moment "auto" becomes a language name, and the caller is expected to keep
    it — a translator that goes on holding "auto" asks the model to translate
    "from auto". A book in a Latin script stays "auto"; the prompt words that
    case for itself.
    """
    chapters, resolved_language = source.chapters(source_language)

    if not chapters:
        return TranslationPlan(chapters=[], source_language=resolved_language)

    # Every chapter is measured, not just the ones about to be translated, so a
    # caller can re-cost a different selection without reading the book again.
    for chapter in chapters:
        chapter.measure()

    if only_chapter_ids is None:
        patches = create_patches(chapters, max_tokens_per_patch)
    else:
        selected = [chapter for chapter in chapters if chapter.id in only_chapter_ids]
        patches = create_patches(selected, max_tokens_per_patch, include_translated=True)

    return TranslationPlan(
        chapters=chapters,
        patches=patches,
        total_tokens=sum(ch.source_tokens for patch in patches for ch in patch),
        source_language=resolved_language,
    )
