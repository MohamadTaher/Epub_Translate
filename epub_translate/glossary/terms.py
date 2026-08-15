"""The glossary itself: what this book has settled on, and how it grows."""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..console import logger
from . import matching, protocol, storage


@dataclass
class MergeResult:
    """What one response added to the glossary."""
    added: Dict[str, str] = field(default_factory=dict)
    # Terms the model translated differently the second time round, kept as
    # they were. Worth counting: a book full of them means the model is
    # ignoring the glossary it was given.
    conflicts: int = 0


class Glossary:
    """
    The names and terms this book has settled on, shared by every worker.

    It starts as whatever the reader typed and grows as the book is translated:
    each response reports the terms it met, `merge` keeps the ones that are new,
    and `prompt_section` sends them back out with the later batches that mention
    them. That loop is the point — chapter 40 has to call a character what
    chapter 1 called them, and the model sees only one batch at a time.

    First translation wins. A term already settled keeps its translation however
    the model renders it later, because an inconsistent name is the one thing a
    glossary exists to prevent.

    Every worker thread reaches this object at once, so all of it is under one
    lock and callers only ever receive snapshots.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._terms: Dict[str, str] = {}
        self._lock = threading.Lock()

        if path:
            self._load(path)

    def __len__(self) -> int:
        with self._lock:
            return len(self._terms)

    @property
    def terms(self) -> Dict[str, str]:
        """A snapshot of everything known, blank translations included."""
        with self._lock:
            return dict(self._terms)

    def replace(self, terms: Dict[str, str]):
        """
        Swap the whole glossary for `terms`, as the reader's editor does.

        Memory only: whoever calls this owns the file, and writes it in the same
        breath. Takes effect on batches not yet sent.
        """
        with self._lock:
            self._terms = dict(terms)

    def merge(self, new_terms: Dict[str, str]) -> MergeResult:
        """
        Take in the terms one response reported and keep what is new.

        A term whose translation is blank counts as unsettled — the reader named
        it but left the translation to us — so the model's answer fills it in.
        """
        result = MergeResult()

        with self._lock:
            # Compared case-insensitively so "Ivan" and "IVAN" cannot end up in
            # the prompt as two terms giving conflicting instructions.
            settled = {term.lower(): translation
                       for term, translation in self._terms.items() if translation}

            for term, translation in new_terms.items():
                current = settled.get(term.lower())
                if current:
                    if current != translation:
                        result.conflicts += 1
                    continue

                self._terms[term] = translation
                settled[term.lower()] = translation
                result.added[term] = translation

        if result.added and self.path:
            self.save()

        return result

    def prompt_section(self, text: str) -> str:
        """
        The glossary's half of the prompt for this batch.

        Only the terms the batch actually mentions travel with it: the glossary
        grows all run, and sending all of it every time would spend the token
        budget on names from chapters that aren't in this request.
        """
        return protocol.prompt_section(self._settled_terms(text))

    def save(self):
        """
        Write the glossary to its file.

        A failed save costs the terms learned so far, not the run, so it is
        reported and stepped over — this is called from a worker between two
        API requests.
        """
        if not self.path:
            raise ValueError("This glossary has no file to save to")

        try:
            storage.write_terms(self.path, self.terms)
        except OSError as e:
            logger.warning(f"Could not save the glossary: {e}")

    def _settled_terms(self, text: str) -> Dict[str, str]:
        with self._lock:
            settled = {term: translation
                       for term, translation in self._terms.items() if translation}

        return matching.relevant_terms(settled, text)

    def _load(self, path: str):
        try:
            self._terms = storage.read_terms(path)
        except FileNotFoundError:
            logger.warning(f"Glossary file {path} not found. Starting with an empty glossary.")
        except (OSError, ValueError) as e:
            # ValueError covers a malformed JSON file as well as a well-formed
            # one holding something that is not a glossary.
            logger.error(f"Could not read {path}: {e}. Starting with an empty glossary.")
