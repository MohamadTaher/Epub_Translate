import ebooklib
from ebooklib import epub
import google.generativeai as genai
import time
import socket
import threading
import sys
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set, Tuple
from collections import deque
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from difflib import SequenceMatcher
import signal
from difflib import SequenceMatcher

class ColoredLogger:
    """Custom logger with colored output for different message types."""
    
    @staticmethod
    def system(message: str):
        print(f"\033[96m[SYSTEM]\033[0m \033[97m{message}\033[0m")
    
    @staticmethod
    def api(message: str):
        print(f"\033[94m[API]\033[0m \033[97m{message}\033[0m")
    
    @staticmethod
    def progress(message: str):
        print(f"\033[92m[PROGRESS]\033[0m \033[97m{message}\033[0m")
    
    @staticmethod
    def concurrency(message: str):
        print(f"\033[95m[CONCURRENCY]\033[0m \033[97m{message}\033[0m")
    
    @staticmethod
    def rate_limit(message: str):
        print(f"\033[93m[RATE_LIMIT]\033[0m \033[33m{message}\033[0m")
    
    @staticmethod
    def interrupt(message: str):
        print(f"\033[41m\033[97m[INTERRUPT]\033[0m \033[91m{message}\033[0m")
    
    @staticmethod
    def error(message: str):
        print(f"\033[91m[ERROR]\033[0m \033[31m{message}\033[0m")
    
    @staticmethod
    def warning(message: str):
        print(f"\033[93m[WARNING]\033[0m \033[33m{message}\033[0m")
    
    @staticmethod
    def success(message: str):
        print(f"\033[92m[SUCCESS]\033[0m \033[32m{message}\033[0m")
    
    @staticmethod
    def info(message: str):
        print(f"\033[96m[INFO]\033[0m \033[97m{message}\033[0m")

logger = ColoredLogger()

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate of 4 characters per token
        return len(text) // 4

@dataclass
class PatchResult:
    """Container for translation results of a single patch (group of chapters)"""
    patch_index: int
    success: bool
    chapters: List[Dict]
    attempts: int
    error_message: Optional[str] = None
    processing_time: float = 0.0

class RateLimiter:
    """
    Manages API rate limiting using a rolling window approach.
    Tracks both request count and token usage over time.
    """

    def __init__(self, max_requests_per_minute: int = 4, max_tokens_per_minute: int = 250000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_timestamps_minute = deque()  # Rolling window of request timestamps
        self.token_usage_minute = deque()  # Rolling window of (timestamp, tokens) pairs
        self.lock = threading.Lock()

    def _cleanup_windows(self):
        """Remove entries older than 1 minute from the rolling windows."""
        current_time = time.time()
        cutoff_time = current_time - 60  # 60 seconds ago
        
        # Clean up request timestamps
        while self.request_timestamps_minute and self.request_timestamps_minute[0] < cutoff_time:
            self.request_timestamps_minute.popleft()
        
        # Clean up token usage
        while self.token_usage_minute and self.token_usage_minute[0][0] < cutoff_time:
            self.token_usage_minute.popleft()

    def can_make_request(self, tokens_to_add: int) -> tuple[bool, float, str]:
        """Check if we can make a request and return wait time and reason if not."""
        with self.lock:
            self._cleanup_windows()
            
            current_requests_minute = len(self.request_timestamps_minute)
            current_tokens_minute = sum(tokens for _, tokens in self.token_usage_minute)

            # Check if we're within both limits
            if (current_requests_minute < self.max_requests_per_minute and 
                current_tokens_minute + tokens_to_add <= self.max_tokens_per_minute):
                return True, 0.0, ""

            # Calculate how long to wait if we're over limits
            wait_time = 0.0
            reason = ""
            if current_requests_minute >= self.max_requests_per_minute:
                wait_time = max(wait_time, 60 - (time.time() - self.request_timestamps_minute[0]) + 1)
                reason = "per-minute request limit"
            if current_tokens_minute + tokens_to_add > self.max_tokens_per_minute:
                wait_time = max(wait_time, 60 - (time.time() - self.token_usage_minute[0][0]) + 1)
                reason = "per-minute token limit"

            return False, wait_time, reason

    def wait_for_availability(self, tokens_to_add: int):
        """Wait until we can make a request within rate limits."""
        while True:
            can_proceed, wait_time, reason = self.can_make_request(tokens_to_add)
            if can_proceed:
                return
            
            logger.rate_limit(f"Rolling window limit reached ({reason}), waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    def record_request(self, tokens_used: int):
        """Record a successful request."""
        with self.lock:
            current_time = time.time()
            self.request_timestamps_minute.append(current_time)
            self.token_usage_minute.append((current_time, tokens_used))


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
    
    def save_glossary(self, file_path: str = None):
        """Save glossary to JSON file."""
        save_path = file_path or self.glossary_file_path
        if not save_path:
            logger.warning("No glossary file path specified for saving.")
            return
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.master_glossary, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.master_glossary)} terms to {save_path}")
        except Exception as e:
            logger.error(f"Error saving glossary: {e}")

    def extract_relevant_terms(self, text: str, similarity_threshold: float = 0.7) -> Dict[str, str]:
        """
        Extract terms from master glossary that appear in the given text.
        Uses both exact matching and fuzzy matching for flexibility.
        """
        relevant_terms = {}
        text_lower = text.lower()
        
        # Create a snapshot of the glossary under lock protection
        with self.lock:
            glossary_snapshot = dict(self.master_glossary)
        
        # Now iterate over the snapshot safely
        for original_term, translated_term in glossary_snapshot.items():
            # Check for exact matches (case insensitive)
            if original_term.lower() in text_lower:
                relevant_terms[original_term] = translated_term
                continue
            
            # Check for partial matches using sequence matching
            for word in text.split():
                if len(word) > 2:  # Skip very short words
                    similarity = SequenceMatcher(None, original_term.lower(), word.lower()).ratio()
                    if similarity > similarity_threshold:
                        relevant_terms[original_term] = translated_term
                        break
        
        return relevant_terms
 
    def add_new_terms(self, new_terms: Dict[str, str]):
        """Add new terms to the master glossary."""
        if not new_terms:
            return
        
        with self.lock:
            added_count = 0
            for original, translated in new_terms.items():
                if original not in self.master_glossary:
                    self.master_glossary[original] = translated
                    added_count += 1
            
            if added_count > 0:
                logger.info(f"Added {added_count} new terms to glossary")
                if self.glossary_file_path:
                    self.save_glossary()
                    
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


class EPUBTranslator:
    """
    Main translator class that handles EPUB book translation using Google's Gemini API.
    Features concurrent translation, rate limiting, auto-save, and glossary management.
    """

    def __init__(self, api_key: str, source_language: str = "auto", target_language: str = "English", 
                 max_concurrent: int = 5, glossary_file_path: str = None, max_requests_per_minute: int = 4,
                 max_tokens_per_minute: int = 250000, model_name: str = "gemini-2.5-pro"):
        if not api_key:
            raise ValueError("API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)                              
        self.source_lang = source_language
        self.target_lang = target_language
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute
        )
        self.should_stop = False
        self.translation_start_time = None
        self.total_api_requests = 0
        self.total_tokens_processed = 0
        self.completed_patches_count = 0
        self.consecutive_failures = 0
        self.lock = threading.Lock()
        
        # Store references for auto-saving progress
        self.current_epub_path = None
        self.current_output_path = None
        self.current_total_patches = 0
        self.current_chapters = []
        
        # Initialize glossary manager
        self.glossary_manager = GlossaryManager(glossary_file_path)
        
        logger.system(f"Concurrency limit: {max_concurrent}")
        logger.system(f"Rate limits: {max_requests_per_minute} req/min, {max_tokens_per_minute:,} tokens/min")
        if glossary_file_path:
            logger.system(f"Glossary loaded: {self.glossary_manager.get_glossary_size()} terms")

    def translate_book(self, epub_path: str, output_path: str, max_tokens_per_patch: int = 15000):
        """
        Main entry point for translating an EPUB book.
        Extracts chapters, groups them into patches, and manages concurrent translation.
        """
        chapters = self._extract_chapters(epub_path)
        
        if not chapters:
            logger.error("No chapters found in the EPUB.")
            return

        # Group chapters into patches for efficient API usage
        patches = self._create_patches(chapters, max_tokens_per_patch)
        untranslated_chapters = [ch for ch in chapters if not ch.get('already_translated', False)]
        
        if not patches:
            logger.system("All chapters are already translated. Creating final EPUB...")
            self.create_translated_epub(chapters, epub_path, output_path)
            return
        
        total_tokens = sum(count_tokens(str(ch['soup'])) for ch in untranslated_chapters)
        
        # Store current state for auto-saves
        self.current_epub_path = epub_path
        self.current_output_path = output_path
        self.current_total_patches = len(patches)
        self.current_chapters = chapters
        
        logger.system(f"Total tokens to translate: {total_tokens:,}")
        logger.system(f"Max tokens per patch: {max_tokens_per_patch:,}")
        logger.system(f"Patches created: {len(patches)}")
       
        proceed = input(f"\n\033[93mProceed with translation? (yes/no): \033[0m")
        if proceed.lower() != 'yes':
            logger.system("Translation cancelled.")
            return

        print(f"\n\033[1m\033[96m🚀 Starting concurrent translation with auto-save enabled...\033[0m")
        logger.system("Progress will be saved after each successful patch")
        
        self.translation_start_time = time.time()
        self.completed_patches_count = 0
        self.consecutive_failures = 0
        results = []
        
        # Use ThreadPoolExecutor for concurrent translation
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all patches for processing
            future_to_patch = {
                executor.submit(self._translate_patch_worker, patch, i+1, len(patches)): i
                for i, patch in enumerate(patches)
            }
            
            logger.concurrency(f"Starting batch of {min(self.max_concurrent, len(patches))} patches. {len(patches)} total in queue")
            
            completed_count = 0
            for future in as_completed(future_to_patch):
                if self.should_stop:
                    logger.interrupt(f"Stop signal received. Cancelling remaining patches...")
                    # Cancel all remaining futures
                    for f in future_to_patch:
                        if not f.done():
                            f.cancel()
                    
                    logger.interrupt(f"Final save with {len(results)} completed patches...")
                    break
                
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if result.success:
                        # Apply this successful result immediately to chapters
                        for translated_chapter in result.chapters:
                            for main_chapter in chapters:
                                if main_chapter['id'] == translated_chapter['id']:
                                    main_chapter['soup'] = translated_chapter['soup']
                                    if 'translated_title' in translated_chapter:
                                        main_chapter['translated_title'] = translated_chapter['translated_title']
                                    break
                        
                        # Auto-save after each successful patch
                        self._auto_save_progress(chapters)
                        
                    else:
                        logger.error(f"Patch {result.patch_index}/{len(patches)} failed after {result.attempts} attempts")
                        
                        # Check if we should stop due to consecutive failures
                        if self.should_stop:
                            logger.error("Stopping due to consecutive API failures")
                            break
                            
                except Exception as e:
                    logger.error(f"Unexpected error in patch processing: {e}")
        
        # Sort results by patch index
        results.sort(key=lambda x: x.patch_index)
        successful_results = [r for r in results if r.success]
        
        logger.info(f"Translation completed: {len(successful_results)}/{len(patches)} patches successful")
        logger.info(f"Final glossary size: {self.glossary_manager.get_glossary_size()} terms")
        logger.info(f"Total API requests made: {self.total_api_requests}")
        
        if successful_results:
            if len(successful_results) == len(patches):
                logger.info("All patches completed successfully. Final save...")
                self.create_translated_epub(chapters, epub_path, output_path)
            else:
                logger.info(f"Translation incomplete: {len(successful_results)}/{len(patches)} patches completed")
                logger.info(f"Partial translation saved to: {output_path}")
        else:
            logger.info("No patches were successfully translated.")

    def _auto_save_progress(self, chapters: List[Dict]):
        """Auto-save the current translation progress after each successful patch."""
        try:
            with self.lock:
                logger.progress(f"Auto-saving progress: {self.completed_patches_count}/{self.current_total_patches} patches completed")
                self.create_translated_epub(chapters, self.current_epub_path, self.current_output_path)
                logger.success(f"Progress saved to: {self.current_output_path}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    def _is_chapter_already_translated(self, soup: BeautifulSoup) -> bool:
        """
        Check if a chapter is already translated by looking for English markers.
        Simple heuristic: presence of "chapter" in title suggests already translated.
        """
        # Get the chapter title
        chapter_title = self._extract_chapter_title(soup)
        
        # Check if title contains "chapter" (case insensitive) - simple and direct approach
        if "chapter" in chapter_title.lower():
            return True
        if "(Translated)" in chapter_title:
            return True
        if "Original_full_title" in chapter_title:
            return True
        if "more" in chapter_title:
            return True
                    
        return False

    def _translate_patch_worker(self, patch: List[Dict], patch_index: int, total_patches: int, max_retries: int = 10) -> PatchResult:
        """
        Worker function to translate a single patch with glossary support.
        Handles retries, rate limiting, and glossary term extraction.
        """
        start_time = time.time()
        
        # Check for stop signal at the start
        if self.should_stop:
            return PatchResult(
                patch_index=patch_index,
                success=False,
                chapters=patch,
                attempts=0,
                error_message="Stopped by user",
                processing_time=time.time() - start_time
            )
        
        # Combine all chapters in the patch into one HTML string
        html_parts = [str(ch['soup']) for ch in patch]
        combined_html = "\n<!-- CHAPTER_SEPARATOR -->\n".join(html_parts)
        
        # Extract relevant glossary terms for this patch
        combined_text = BeautifulSoup(combined_html, 'html.parser').get_text()
        try:
            relevant_terms = self.glossary_manager.extract_relevant_terms(combined_text)
            glossary_section = self.glossary_manager.create_glossary_prompt_section(relevant_terms)
        except RuntimeError as e:
            # Handle dictionary changed size during iteration
            logger.warning(f"Glossary iteration error for patch {patch_index}, using empty glossary: {e}")
            relevant_terms = {}
            glossary_section = ""
        
        prompt_tokens = count_tokens(combined_html) + count_tokens(glossary_section) + 1000
        estimated_total_tokens = int(prompt_tokens * 1.4)
        
        # Build the translation prompt with glossary terms
        prompt = f"""You are an expert literary translator, specializing in fiction. You have a deep understanding of literary devices, tone, cultural nuances, and character voice. You are not a literal, word-for-word machine translator; you are a creative partner tasked with preserving the original author's intent and spirit.

# GOAL
Your goal is to translate the book from {self.source_lang} to {self.target_lang}. The final translation must read as if it were originally written in the {self.target_lang}, while remaining completely faithful to the source's style, tone, and meaning.

{glossary_section}

CRITICAL INSTRUCTIONS:
1. Maintain the exact HTML structure and tags
2. Only translate TEXT CONTENT inside HTML tags
3. Do NOT modify, add, or remove HTML tags or attributes
4. Keep the <!-- CHAPTER_SEPARATOR --> markers exactly as they are
5. **Idioms & Culturalisms:** Do not translate idioms literally. Find the closest equivalent cultural idiom in the {self.target_lang}. If no direct equivalent exists, convey the original meaning in a natural-sounding way.
6. If you see duplicate text (like "Chapter 1: Title Chapter 1: Title"), translate it only once
7. **GLOSSARY CONSISTENCY:** Use the exact translations provided in the glossary section above for those specific terms.

INPUT HTML:
{combined_html}

TRANSLATED HTML:"""

        last_error = None
        
        # Retry loop for handling transient API failures
        for attempt in range(max_retries):
            # Check for stop signal before each attempt
            if self.should_stop:
                return PatchResult(
                    patch_index=patch_index,
                    success=False,
                    chapters=patch,
                    attempts=attempt,
                    error_message="Stopped by user",
                    processing_time=time.time() - start_time
                )
                
            try:
                # Wait for rate limit availability
                self.rate_limiter.wait_for_availability(estimated_total_tokens)
                
                logger.api(f"Sending patch {patch_index} (attempt {attempt + 1}/{max_retries})")
                
                response = self.model.generate_content(prompt)
                if not response or not response.text:
                    raise Exception("Empty response from API")
                
                response_text = response.text.strip()
                
                # Clean up markdown formatting
                for marker in ["```html", "```"]:
                    if response_text.startswith(marker):
                        response_text = response_text[len(marker):]
                    if response_text.endswith(marker):
                        response_text = response_text[:-len(marker)]
                response_text = response_text.strip()
                
                # Split the response back into individual chapters
                translated_parts = response_text.split("<!-- CHAPTER_SEPARATOR -->")
                
                for i, (chapter, translated_part) in enumerate(zip(patch, translated_parts)):
                    if translated_part.strip():
                        translated_soup = BeautifulSoup(translated_part.strip(), 'html.parser')
                        chapter['soup'] = translated_soup
                        chapter['translated_title'] = self._extract_chapter_title(translated_soup)
                
                # NOTE: Glossary term extraction is commented out to improve performance
                # Uncomment if you want to automatically build glossary from translations
                # try:
                #     new_terms = self._extract_new_terms_from_translation(combined_html, response_text)
                #     if new_terms:
                #         self.glossary_manager.add_new_terms(new_terms)
                #         logger.info(f"Patch {patch_index}: Found {len(new_terms)} new glossary terms")
                # except Exception as e:
                #     logger.warning(f"Patch {patch_index}: Failed to extract new terms - {str(e)[:50]}")
                
                # Record API usage for rate limiting
                actual_total_tokens = prompt_tokens + count_tokens(response_text)
                self.rate_limiter.record_request(actual_total_tokens)
                self.total_api_requests += 1
                self.total_tokens_processed += actual_total_tokens
                
                with self.lock:
                    self.completed_patches_count += 1
                    self.consecutive_failures = 0  # Reset failure counter on success
                    completion_pct = (self.completed_patches_count / total_patches) * 100
                
                processing_time = time.time() - start_time
                logger.success(f"Patch {patch_index} completed in {processing_time:.1f}s [{self.completed_patches_count}/{total_patches}]")
                
                return PatchResult(
                    patch_index=patch_index,
                    success=True,
                    chapters=patch,
                    attempts=attempt + 1,
                    processing_time=processing_time
                )
                
            except Exception as e:
                last_error = str(e).split(' ')[0]
                logger.warning(f"Patch {patch_index}/{total_patches} attempt {attempt + 1}/{max_retries} failed: {last_error}")
        
        # If we reach here, all attempts failed
        with self.lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 6:
                logger.error(f"6 consecutive patch failures detected. API may be down. Stopping translation.")
                self.should_stop = True
        
        processing_time = time.time() - start_time
        logger.error(f"Patch {patch_index}/{total_patches} failed after {max_retries} attempts")
        
        return PatchResult(
            patch_index=patch_index,
            success=False,
            chapters=patch,
            attempts=max_retries,
            error_message=last_error,
            processing_time=processing_time
        )

    def _extract_chapters(self, epub_path: str) -> List[Dict]:
        """Extract all chapters from the EPUB, including already translated ones."""
        chapters = []
        book = epub.read_epub(epub_path)
        chapter_counter = 1
        already_translated_count = 0
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            
            if soup.body and soup.body.get_text(strip=True):
                chapter_title = self._extract_chapter_title(soup)
                if not chapter_title or chapter_title == "Untitled Chapter":
                    chapter_title = f"Chapter {chapter_counter}"
                
                # Check if chapter is already translated
                is_translated = self._is_chapter_already_translated(soup)
                if is_translated:
                    already_translated_count += 1
                
                chapters.append({
                    'id': item.get_id() or f"chapter_{chapter_counter}",
                    'title': chapter_title,
                    'soup': soup,
                    'file_name': item.get_name(),
                    'already_translated': is_translated
                })
                chapter_counter += 1
        
        logger.system(f"Extracted {len(chapters)} total chapters")
        if already_translated_count > 0:
            logger.system(f"Found {already_translated_count} already translated chapters (will be preserved)")
        
        return chapters

    def _create_patches(self, chapters: List[Dict], max_tokens_per_patch: int = 15000) -> List[List[Dict]]:
        """
        Group chapters into patches based on token limits, excluding already translated chapters.
        This batching improves API efficiency while staying under token limits.
        """
        patches = []
        current_patch = []
        current_tokens = 500  # Reserve tokens for prompt overhead
        
        # Only include chapters that need translation
        untranslated_chapters = [ch for ch in chapters if not ch.get('already_translated', False)]
        
        for chapter in untranslated_chapters:
            chapter_tokens = count_tokens(str(chapter['soup']))
            
            # Start new patch if adding this chapter would exceed limit
            if current_patch and current_tokens + chapter_tokens + 50 > max_tokens_per_patch:
                patches.append(current_patch)
                current_patch = [chapter]
                current_tokens = 500 + chapter_tokens
            else:
                current_patch.append(chapter)
                current_tokens += chapter_tokens + (50 if current_patch else 0)
        
        if current_patch:
            patches.append(current_patch)
        
        return patches

    def _clean_and_validate_glossary_response(self, response_text: str) -> Dict[str, str]:
        """Clean and validate the AI response for new glossary terms."""
        # Remove markdown formatting if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            # Parse JSON
            data = json.loads(response_text)
            
            # Handle different response formats
            if isinstance(data, dict):
                # Check if it's wrapped in a structure like {"new_terms": {...}}
                if "new_terms" in data:
                    data = data["new_terms"]
                elif "proper_noun_mappings" in data:
                    if isinstance(data["proper_noun_mappings"], list):
                        # Convert list format to dict format
                        glossary_data = {}
                        for item in data["proper_noun_mappings"]:
                            if isinstance(item, dict) and "original" in item and "translated" in item:
                                glossary_data[item["original"]] = item["translated"]
                        data = glossary_data
                    else:
                        data = data["proper_noun_mappings"]
                
                # Remove any other non-glossary keys
                clean_data = {}
                for key, value in data.items():
                    if isinstance(key, str) and isinstance(value, str):
                        clean_data[key] = value
                data = clean_data
            
            if not isinstance(data, dict):
                return {}
            
            # Filter out invalid entries
            cleaned_glossary = {}
            for original, translated in data.items():
                # Skip if either term is empty
                if not original.strip() or not translated.strip():
                    continue
                
                # Skip if translated term is all lowercase (likely not a proper noun)
                if translated.islower():
                    continue
                
                # Skip single word verbs
                if len(translated.split()) == 1:
                    verb_endings = ['ed', 'ing', 'es', 's']
                    if any(translated.lower().endswith(ending) for ending in verb_endings):
                        continue
                
                cleaned_glossary[original] = translated
            
            return cleaned_glossary
            
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _extract_new_terms_from_translation(self, original_html: str, translated_html: str) -> Dict[str, str]:
        """
        Extract new glossary terms from the translation pair.
        Uses AI to identify proper nouns and their translations.
        """
        # Create a prompt to extract new terms
        extract_prompt = f"""Your job is to ONLY identify new proper nouns from the original text and find their translations in the translated text.

**CRITICAL RULES:**
1. ONLY extract proper nouns (names, places, techniques, organizations, titles)
2. DO NOT translate single-word verbs or common words
3. The translated term must have at least one uppercase letter (proper nouns are capitalized)
4. Return ONLY a flat JSON dictionary format: {{"original_term": "Translated_Term"}}
5. Only extract terms that appear in BOTH texts
6. Make sure no that english is not on both sides of the mapping of "original_term": "Translated_Term"

**ORIGINAL TEXT:**
{BeautifulSoup(original_html, 'html.parser').get_text()}

**TRANSLATED TEXT:**
{BeautifulSoup(translated_html, 'html.parser').get_text()}

**OUTPUT FORMAT:**
Return ONLY a valid JSON dictionary. No explanations, no markdown, no other text. Make sure one-to-one mappings is original language to translated language.You should not have both as english.
Example: {{
    "道玄": "Dao Xuan",
    "灵宝天玄镜": "Tianxuan Mirror"}}

**JSON RESPONSE:**"""

        try:
            # Estimate tokens for the extraction request
            extract_tokens = count_tokens(extract_prompt) + 500
            
            # Wait for rate limit availability
            self.rate_limiter.wait_for_availability(extract_tokens)
            
            response = self.model.generate_content(extract_prompt)
            if response and response.text:
                new_terms = self._clean_and_validate_glossary_response(response.text.strip())
                self.rate_limiter.record_request(extract_tokens)
                return new_terms
        except Exception as e:
            logger.warning(f"Failed to extract new terms: {str(e)[:100]}")
        
        return {}

    def _extract_chapter_title(self, soup: BeautifulSoup) -> str:
        """Extract a meaningful title from chapter content."""
        # Try to find title in the HTML title tag
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(strip=True)
        
        # Look for heading tags (h1, h2, h3)
        for tag_name in ['h1', 'h2', 'h3']:
            heading = soup.find(tag_name)
            if heading and heading.get_text(strip=True):
                return heading.get_text(strip=True)
        
        # Fall back to first paragraph if it's short enough to be a title
        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text(strip=True)
            if text and len(text) < 100:
                return text
        
        return "Untitled Chapter"

    def _update_table_of_contents(self, book, chapters: List[Dict]):
        """Update the table of contents with translated chapter titles."""
        toc_items = []
        
        for chapter in chapters:
            translated_title = chapter.get('translated_title', chapter['title'])
            file_name = chapter.get('file_name', f"{chapter['id']}.html")
            
            toc_item = epub.Link(file_name, translated_title, chapter['id'])
            toc_items.append(toc_item)
        
        book.toc = toc_items
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

    def create_translated_epub(self, chapters: List[Dict], original_book_path: str, output_path: str):
        """
        Creates translated EPUB by modifying the original book.
        Preserves structure and metadata while updating content.
        """
        book = epub.read_epub(original_book_path)
        
        # Create a mapping of chapter IDs to their translated content
        translated_chapters = {ch['id']: ch['soup'] for ch in chapters}
        
        # Update each document item with translated content
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_id = item.get_id()
            if item_id in translated_chapters:
                translated_soup = translated_chapters[item_id]
                
                # Ensure proper HTML structure
                if not translated_soup.html:
                    html_content = f"""<!DOCTYPE html>
                    <html xmlns="http://www.w3.org/1999/xhtml">
                    <head>
                        <title>Chapter</title>
                        <meta charset="utf-8"/>
                    </head>
                    <body>
                    {str(translated_soup)}
                    </body>
                    </html>"""
                    translated_soup = BeautifulSoup(html_content, 'html.parser')
                
                item.content = str(translated_soup).encode('utf-8')
        
        # Update table of contents with translated titles
        self._update_table_of_contents(book, chapters)
        
        # Update book metadata
        original_title = book.get_metadata('DC', 'title')
        if original_title:
            book.set_title(f"{original_title[0][0]} (Translated)")
        else:
            book.set_title("Translated Book")
        
        book.set_language('en')
        
        # Write the final EPUB file
        epub.write_epub(output_path, book, {})
        logger.success(f'Successfully created translated EPUB: {output_path}')

def main():
    """
    Main entry point for the EPUB translator.
    Configure your settings here before running.
    """
    
    # API Configuration
    API_KEY = "xxxxxxx"  # Replace with your actual Gemini API key
    
    # File paths
    INPUT_EPUB = "The More You Believe Me, the Truer It Becomes.epub"
    OUTPUT_EPUB = "The More You Believe Me, the Truer It Becomes.epub"
    GLOSSARY_FILE = "Believe_Me.json"  # Path to your glossary JSON file

    # Performance and rate limiting parameters
    MAX_CONCURRENT = 2  # Number of patches to translate simultaneously
    MAX_REQUESTS_PER_MINUTE = 2  # API request limit
    MAX_TOKENS_PER_MINUTE = 250000  # Token usage limit
    MAX_TOKENS_PER_PATCH = 26000  # How many tokens per batch (affects grouping)
    MODEL_NAME = "gemini-2.5-pro"  # Which Gemini model to use

    # Initialize the translator with configured parameters
    translator = EPUBTranslator(
        api_key=API_KEY, 
        max_concurrent=MAX_CONCURRENT,
        glossary_file_path=GLOSSARY_FILE,
        max_requests_per_minute=MAX_REQUESTS_PER_MINUTE,
        max_tokens_per_minute=MAX_TOKENS_PER_MINUTE,
        model_name=MODEL_NAME
    )
    
    # Start the translation process
    translator.translate_book(INPUT_EPUB, OUTPUT_EPUB, MAX_TOKENS_PER_PATCH)


if __name__ == "__main__":
    main()
