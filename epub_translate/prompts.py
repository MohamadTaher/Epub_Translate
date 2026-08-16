"""
The prompt sent for one patch of chapters.

CHAPTER_SEPARATOR, the instruction to preserve it, and the split that reads it
back all live here together, for the same reason the glossary marker lives next
to its parser: they are parts of one contract. Spelled out in two files, a
change to one alone leaves the model dutifully keeping a marker nothing looks
for any more, and every patch silently collapses into its first chapter.

The sections are ordered so that everything shaping the reply sits as close to
the reply as possible: what the book is, then how to translate it, then the
glossary, then the exact shape of the answer, then the input. The glossary
section states its own two rules — use these translations, report the new ones —
so the instruction list does not restate them.
"""

import re
from typing import List, Optional

from .language import normalize

CHAPTER_SEPARATOR = "<!-- CHAPTER_SEPARATOR -->"

# What counts as that marker coming back. A model tidying the markup around it
# can close the spaces up, change the case, or write the underscore as a space or
# a hyphen without meaning anything by it, and demanding the string exactly turns
# that into a patch that fails ten times over punctuation. The two English words
# are the part that has to survive — a marker that comes back *translated* is
# past saving here, which is why the prompt spends a sentence forbidding it.
_SEPARATOR_BACK = re.compile(r"<!--\s*CHAPTER[\s_-]*SEPARATOR\s*-->", re.IGNORECASE)


def build_translation_prompt(source_lang: str, target_lang: str, glossary_section: str,
                             combined_html: str, correction: Optional[str] = None) -> str:
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
- Leave HTML comments (`<!-- ... -->`) exactly as they are, in the original characters. They are notes to the program reading your reply, not words of the book, so they are never translated and never reworded.
{glossary_section}
# YOUR REPLY
{_reply_shape(combined_html.count(CHAPTER_SEPARATOR) + 1)}
{_correction_section(correction)}
INPUT HTML:
{combined_html}

TRANSLATED HTML:"""


def _correction_section(correction: Optional[str]) -> str:
    """
    What the last attempt did wrong, on the attempts that follow it.

    Last of the instructions and directly above the input, for the reason the
    module docstring gives: what shapes the reply sits closest to the reply. It
    stops short of the input itself, though — text after the chapters is text
    the model may reasonably take for more book to translate.

    Empty for a first attempt, and empty for a failure the model did not cause,
    which leaves the prompt byte-for-byte what it was.
    """
    if not correction:
        return ""

    return f"""
# YOUR LAST ATTEMPT WAS REJECTED
{correction} Nothing about the input below has changed, and the rules above still stand — send the whole patch again, correctly this time.
"""


def split_chapters(translated_html: str) -> List[str]:
    """
    A reply broken back into the chapters that went into it.

    Deliberately forgiving, because the caller checks the count against what it
    sent and that is the real guard: every spelling accepted here is one the
    model plainly meant as the marker, and every one turned away costs a whole
    patch its ten attempts.
    """
    return _SEPARATOR_BACK.split(translated_html)


def _reply_shape(chapter_count: int) -> str:
    """
    The one part of the prompt the caller can check the answer against.

    A dropped separator is the worst thing a reply can do quietly: the parts no
    longer line up with the chapters they belong to, so the text of two chapters
    lands in the first and the last chapters keep their original language. The
    count is stated here because a rule with a number in it is one the model can
    verify before answering, and the caller can verify afterwards.

    The marker is also called out as a literal to copy, because the rest of the
    prompt reads as an instruction to translate it: an HTML comment is not a tag
    or an attribute, so "reproduce every tag exactly" does not reach it, while
    its contents are text and CHAPTER_SEPARATOR is a legible English word. A
    marker that comes back translated is not a near miss — the split finds
    nothing, and the patch spends all ten attempts being refused the same way.
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
            f"glossary block, and nothing else. Copy each marker across character for "
            f"character, in the Latin letters shown: it is a label this program searches for, "
            f"not a line of the book, so it is never translated, spaced out or reworded. "
            f"{tally}")


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
