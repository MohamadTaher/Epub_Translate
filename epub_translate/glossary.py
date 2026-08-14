import json
import os
import tempfile
import threading
from difflib import SequenceMatcher
from typing import Dict

from .logging_utils import logger


class GlossaryManager:
    """
    Manages a glossary of term translations to ensure consistency across the book.
    Terms are saved/loaded from JSON and applied during translation.
    """

    def __init__(self, glossary_file_path: str = None):
        self.master_glossary: Dict[str, str] = {}
        self.glossary_file_path = glossary_file_path
        self.lock = threading.Lock()

        if glossary_file_path:
            self.load_glossary(glossary_file_path)

    def load_glossary(self, file_path: str):
        """Load glossary from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.master_glossary = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Glossary file {file_path} not found. Starting with empty glossary.")
            self.master_glossary = {}
        except Exception as e:
            logger.error(f"Error loading glossary: {e}")
            self.master_glossary = {}

    def replace_terms(self, terms: Dict[str, str]):
        """Replace the whole glossary with `terms`."""
        with self.lock:
            self.master_glossary = dict(terms)

    def save_glossary(self, file_path: str = None):
        """
        Write the glossary to JSON. Writes to a temporary file and renames it, so
        a translation thread reading the file never sees a half-written one.
        """
        target = file_path or self.glossary_file_path
        if not target:
            raise ValueError("No glossary file path to save to")

        with self.lock:
            snapshot = dict(self.master_glossary)

        directory = os.path.dirname(os.path.abspath(target))
        handle, temp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, target)
        except Exception:
            os.unlink(temp_path)
            raise

    def extract_relevant_terms(self, text: str, similarity_threshold: float = 0.7) -> Dict[str, str]:
        """
        Extract terms from master glossary that appear in the given text.
        Uses both exact matching and fuzzy matching for flexibility.
        """
        relevant_terms = {}
        text_lower = text.lower()
        words = [word.lower() for word in text.split() if len(word) > 2]  # Skip very short words

        with self.lock:
            glossary_snapshot = dict(self.master_glossary)

        matcher = SequenceMatcher()
        for original_term, translated_term in glossary_snapshot.items():
            term_lower = original_term.lower()
            if term_lower in text_lower:
                relevant_terms[original_term] = translated_term
                continue

            # seq2 is the expensive side to change, so set the term once and vary the word.
            matcher.set_seq2(term_lower)
            for word in words:
                matcher.set_seq1(word)
                if (matcher.real_quick_ratio() > similarity_threshold
                        and matcher.quick_ratio() > similarity_threshold
                        and matcher.ratio() > similarity_threshold):
                    relevant_terms[original_term] = translated_term
                    break

        return relevant_terms

    def get_glossary_size(self) -> int:
        """Get the current size of the master glossary."""
        with self.lock:
            return len(self.master_glossary)

    def create_glossary_prompt_section(self, relevant_terms: Dict[str, str]) -> str:
        """Create the glossary section for the translation prompt."""
        if not relevant_terms:
            return ""

        terms_text = "\n".join([f'"{original}": "{translated}"' for original, translated in relevant_terms.items()])

        return f"""
# GLOSSARY TERMS
Use these established translations for consistency. These terms have been used in previous chapters:
{terms_text}

IMPORTANT: When you encounter these terms, use the exact translations provided above. To repeat, please make sure to use the exact translations provided above for these specific terms.
"""
