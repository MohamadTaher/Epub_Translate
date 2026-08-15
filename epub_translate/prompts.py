"""
The prompt sent for one batch of chapters.

CHAPTER_SEPARATOR lives here rather than beside the code that splits on it, for
the same reason the glossary marker lives next to its parser: the instruction
telling the model to preserve the marker and the string being preserved are two
halves of one contract. Spelled out in two files, a change to one alone leaves
the model dutifully keeping a marker nothing looks for any more, and every batch
silently collapses into its first chapter.

The sections are ordered so that everything shaping the reply sits as close to
the reply as possible: what the book is, then how to translate it, then the
glossary, then the exact shape of the answer, then the input. The glossary
section states its own two rules — use these translations, report the new ones —
so the instruction list does not restate them.
"""

from .language import normalize

CHAPTER_SEPARATOR = "<!-- CHAPTER_SEPARATOR -->"


def build_translation_prompt(source_lang: str, target_lang: str, glossary_section: str,
                             combined_html: str) -> str:
    source = _source_phrase(source_lang)

    return f"""You are an expert literary translator, specializing in fiction. You have a deep understanding of literary devices, tone, cultural nuances, and character voice. You are not a literal, word-for-word machine translator; you are a creative partner tasked with preserving the original author's intent and spirit.

# GOAL
Your goal is to translate the book from {source} to {target_lang}. The final translation must read as if it were originally written in {target_lang}, while remaining completely faithful to the source's style, tone, and meaning.

# HOW TO TRANSLATE
- **Idioms & Culturalisms:** Do not translate idioms literally. Find the closest equivalent cultural idiom in {target_lang}. If no direct equivalent exists, convey the original meaning in a natural-sounding way.
- **Duplicated text:** If you see duplicate text (like "Chapter 1: Title Chapter 1: Title"), translate it only once.

# THE HTML
- Reproduce every tag and attribute exactly as it stands. Do not add, remove or alter any of them.
- Translate only the text between the tags.
{glossary_section}
# YOUR REPLY
{_reply_shape(combined_html.count(CHAPTER_SEPARATOR) + 1)}

INPUT HTML:
{combined_html}

TRANSLATED HTML:"""


def _reply_shape(chapter_count: int) -> str:
    """
    The one part of the prompt the caller can check the answer against.

    A dropped separator is the worst thing a reply can do quietly: the parts no
    longer line up with the chapters they belong to, so the text of two chapters
    lands in the first and the last chapters keep their original language. The
    count is stated here because a rule with a number in it is one the model can
    verify before answering, and the caller can verify afterwards.
    """
    if chapter_count == 1:
        return ("There is one chapter below. Return it translated, then the glossary block, "
                "and nothing else.")

    separators = chapter_count - 1
    if separators == 1:
        divided_by = f"a single {CHAPTER_SEPARATOR} marker"
        standing = "with that marker still standing between them"
        tally = "One marker goes in, one comes back."
    else:
        divided_by = f"{separators} {CHAPTER_SEPARATOR} markers"
        standing = f"with all {separators} markers still standing between them"
        tally = (f"The marker count is not negotiable: {separators} go in, "
                 f"{separators} come back.")

    return (f"There are {chapter_count} chapters below, divided by {divided_by}. Return all "
            f"{chapter_count} of them translated, in the same order, {standing} — then the "
            f"glossary block, and nothing else. {tally}")


def _source_phrase(source_lang: str) -> str:
    """
    How the source language is named in the prompt.

    "auto" still reaches here for any book whose language could not be named
    from its characters, which is every Latin-script one — nothing distinguishes
    Spanish from French that way. Printing it verbatim would ask the model to
    translate "from auto"; naming no language leaves it to read the text, which
    is what it would have to do regardless.
    """
    return "its original language" if normalize(source_lang) == "auto" else source_lang
