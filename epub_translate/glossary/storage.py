"""The glossary file, and the one definition of what may be in it."""

import json
import os
import tempfile
from typing import Dict


def read_terms(path: str) -> Dict[str, str]:
    """
    The terms in a glossary file.

    Raises for a missing, unreadable or malformed file rather than quietly
    returning nothing, so a mistyped `--glossary` is reported at load instead of
    surfacing as a book translated without one.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        return _validated(json.load(handle))


def write_terms(path: str, terms: Dict[str, str]):
    """
    Write the glossary out.

    Through a temporary file and a rename, because a run merges terms into this
    file while a reader may be fetching it — and half a JSON file reads as no
    glossary at all.
    """
    directory = os.path.dirname(os.path.abspath(path))
    handle, temp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            json.dump(terms, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except Exception:
        os.unlink(temp_path)
        raise


def _validated(data) -> Dict[str, str]:
    """
    A glossary is a flat object of "term": "translation".

    Anything else is refused here, where the file is named and the problem can
    be explained. Left to be discovered later it would raise inside a worker,
    which retries ten times before giving up on the patch.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f'a glossary must be a JSON object of "term": "translation" pairs, '
            f'not {type(data).__name__}'
        )

    terms = {}
    for term, translation in data.items():
        if not isinstance(translation, str):
            raise ValueError(f'the translation for "{term}" must be text, not {type(translation).__name__}')
        if term.strip():
            terms[term.strip()] = translation.strip()

    return terms
