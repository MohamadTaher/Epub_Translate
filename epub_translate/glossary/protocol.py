"""
The contract with the model: how known terms are handed over, and how new ones
come back.

The instruction that asks for the block and the parser that reads it live in one
file on purpose. They are two halves of one format, and a marker changed in only
one of them would cost a whole run's worth of learned terms without failing
anything.
"""

import json
from typing import Dict, Tuple

# Deliberately shaped like CHAPTER_SEPARATOR: an HTML comment survives a model
# that decides to tidy up the markup around it.
MARKER = "<!-- GLOSSARY -->"


def prompt_section(known_terms: Dict[str, str]) -> str:
    """
    The glossary's half of the prompt: what has been settled, and the request
    for whatever is new.

    Never empty, even for a book with no glossary yet — the request for new
    terms is how the glossary gets its first entries.
    """
    return _known_section(known_terms) + _new_terms_section(bool(known_terms))


def split_translation(response_text: str) -> Tuple[str, Dict[str, str]]:
    """
    Separate the translated HTML from the glossary block appended to it.

    A response with no block is not a failure — the chapters are the point and
    the terms are a bonus — so the whole text comes back as the translation and
    the glossary is left as it was. The last marker wins: the block belongs at
    the end, so an earlier one is the model quoting its instructions.
    """
    body, marker, tail = response_text.rpartition(MARKER)
    if not marker:
        return response_text, {}

    # A model that fences the HTML now has something to close the fence before,
    # so the closing ``` ends up here rather than at the end of the response
    # where the caller already strips it. Left alone it becomes a line of text
    # at the bottom of the last chapter.
    body = body.rstrip().removesuffix("```").rstrip()

    return body, _parse_terms(tail)


def _new_terms_section(has_known_terms: bool) -> str:
    """
    The request for the terms this batch introduced.

    Being new is the request itself, not a condition tacked on after it: asked
    for every term in the chapters and then told to leave out the settled ones,
    a model returns the settled ones anyway. They cost output tokens on every
    request of the run and are thrown away on arrival, since a term's first
    translation is the one that stands.

    The exclusion is dropped entirely when nothing has been settled yet, because
    there is no list above for it to point at.

    One gap this cannot close: only the terms this batch mentions are listed
    above (see `Glossary.prompt_section`), so a name settled in chapter 2 and
    absent from these chapters is not something the model has been shown, and
    coming back with it is not disobedience. `merge` drops it either way.
    """
    only_new = "is new to this book and " if has_known_terms else ""

    rules = ["Use the source spelling exactly as it appears in the text as the key."]
    if has_known_terms:
        rules.append("Nothing from the list above belongs here, in any spelling — those are settled.")
    rules += [
        "Write {} if these chapters introduced nothing new.",
        "Nothing may follow the JSON object.",
    ]

    # One long line rather than a wrapped paragraph: the insert above changes
    # its length, so any hand-wrapping is ragged for one of the two cases.
    ask = (f"After the last chapter, on a line of its own, write the marker {MARKER} and then a "
           f"single JSON object: every name or term in these chapters that {only_new}will come up "
           f"again — characters, places, titles, invented vocabulary — mapped to the translation "
           f"you just used for it.")

    return f"""
# NEW TERMS
{ask}

""" + "\n".join(f"- {rule}" for rule in rules) + "\n"


def _known_section(known_terms: Dict[str, str]) -> str:
    if not known_terms:
        return ""

    lines = "\n".join(f'"{term}": "{translation}"' for term, translation in known_terms.items())
    return f"""
# GLOSSARY TERMS
These translations are already established in this book. Use them exactly:
{lines}
"""


def _parse_terms(tail: str) -> Dict[str, str]:
    """
    The JSON object out of whatever the model put after the marker.

    Read between the outermost braces rather than parsing the tail whole, which
    takes a code fence, a stray sentence or a trailing newline in its stride.
    Anything unreadable is no terms: a garbled glossary block must not cost the
    chapters it was attached to.
    """
    start, end = tail.find('{'), tail.rfind('}')
    if start == -1 or end < start:
        return {}

    try:
        data = json.loads(tail[start:end + 1])
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        term.strip(): translation.strip()
        for term, translation in data.items()
        if isinstance(term, str) and isinstance(translation, str)
        and term.strip() and translation.strip()
    }
