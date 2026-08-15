"""
Grouping chapters into patches — the batches of chapters that become one API
request each.

Kept apart from the EPUB itself because the server packs a plan it has already
read: re-costing a reader's chapter selection goes through `pack_by_tokens`
without touching a book on disk.
"""

from typing import List, Sequence, Tuple, TypeVar

from .book import Chapter

T = TypeVar("T")

PATCH_OVERHEAD_TOKENS = 500      # room for the prompt wrapped around every patch
PATCH_SEPARATOR_TOKENS = 50      # the marker joining two chapters within a patch


def pack_by_tokens(items: Sequence[Tuple[T, int]], max_tokens_per_patch: int) -> List[List[T]]:
    """
    Group (item, token count) pairs into patches that stay under the token limit.

    Generic in the item, so the server can work out how many API requests a
    selection would take without re-reading the book. Both paths must pack
    identically, or the count shown to a reader would not be the count they are
    charged for — so this rule lives here and nowhere else.

    An item larger than the limit still gets a patch of its own; splitting a
    chapter mid-way would break the translation, so an oversized request is the
    lesser problem.
    """
    patches: List[List[T]] = []
    current_patch: List[T] = []
    current_tokens = PATCH_OVERHEAD_TOKENS

    for item, item_tokens in items:
        separator_tokens = PATCH_SEPARATOR_TOKENS if current_patch else 0

        if current_patch and current_tokens + separator_tokens + item_tokens > max_tokens_per_patch:
            patches.append(current_patch)
            current_patch = [item]
            current_tokens = PATCH_OVERHEAD_TOKENS + item_tokens
        else:
            current_patch.append(item)
            current_tokens += separator_tokens + item_tokens

    if current_patch:
        patches.append(current_patch)

    return patches


def create_patches(chapters: List[Chapter], max_tokens_per_patch: int,
                   include_translated: bool = False) -> List[List[Chapter]]:
    """
    Group chapters into patches, leaving out the ones that already look translated.

    `include_translated` overrides that exclusion, for when the caller would
    rather re-translate everything than trust the detection.
    """
    if include_translated:
        selected = list(chapters)
    else:
        selected = [chapter for chapter in chapters if not chapter.already_translated]

    # `build_plan` has already measured these, and `measure` is idempotent, so
    # this does no work on the path that exists today. It stays as the guard on
    # the precondition: unmeasured chapters all report zero tokens, which packs
    # the whole book into one patch instead of failing.
    for chapter in selected:
        chapter.measure()

    return pack_by_tokens([(chapter, chapter.source_tokens) for chapter in selected],
                          max_tokens_per_patch)
