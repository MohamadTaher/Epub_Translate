"""Which glossary terms are worth sending with a given patch."""

from difflib import SequenceMatcher
from typing import Dict

# How alike a word and a term must be to count as the same name. Below this,
# unrelated words start matching; above it, an inflected name stops matching.
SIMILARITY_THRESHOLD = 0.7

# Shorter words are skipped: at two characters almost everything is 70% similar
# to almost everything.
MIN_WORD_LENGTH = 3


def relevant_terms(terms: Dict[str, str], text: str) -> Dict[str, str]:
    """
    The terms `text` mentions, matched exactly or closely enough.

    Close enough matters for inflected languages: a name in the text carries a
    case ending the glossary key doesn't have, and the model still needs to be
    told what that name is called.

    Words are deduplicated first. A chapter says its characters' names hundreds
    of times, and every repeat would otherwise be scored against every term.
    """
    text_lower = text.lower()
    words = {word.lower() for word in text.split() if len(word) > MIN_WORD_LENGTH - 1}

    relevant = {}
    matcher = SequenceMatcher()

    for term, translation in terms.items():
        term_lower = term.lower()
        if term_lower in text_lower:
            relevant[term] = translation
            continue

        # seq2 is the expensive side to change, so set the term once and vary the word.
        matcher.set_seq2(term_lower)
        for word in words:
            matcher.set_seq1(word)
            if (matcher.real_quick_ratio() > SIMILARITY_THRESHOLD
                    and matcher.quick_ratio() > SIMILARITY_THRESHOLD
                    and matcher.ratio() > SIMILARITY_THRESHOLD):
                relevant[term] = translation
                break

    return relevant
