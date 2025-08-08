import ebooklib
from ebooklib import epub
import google.generativeai as genai
import time
import socket
import threading
import select
import sys
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from collections import deque
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from dataclasses import dataclass


socket.setdefaulttimeout(300)

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
        return len(text) // 4

@dataclass
class PatchResult:
    patch_index: int
    success: bool
    chapters: List[Dict]
    attempts: int
    error_message: Optional[str] = None
    processing_time: float = 0.0

class RateLimiter:
    """Handles API rate limiting with per-minute limits and concurrent request tracking."""
    def __init__(self, max_requests_per_minute: int = 4, max_tokens_per_minute: int = 250000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_timestamps_minute = deque()
        self.token_usage_minute = deque()
        self.lock = threading.Lock()

    def _cleanup_windows(self):
        """Remove entries older than one minute."""
        current_time = time.time()
        while self.request_timestamps_minute and self.request_timestamps_minute[0] < current_time - 60:
            self.request_timestamps_minute.popleft()
        while self.token_usage_minute and self.token_usage_minute[0][0] < current_time - 60:
            self.token_usage_minute.popleft()

    def can_make_request(self, tokens_to_add: int) -> tuple[bool, float]:
        """Check if we can make a request and return wait time if not."""
        with self.lock:
            self._cleanup_windows()
            
            current_requests_minute = len(self.request_timestamps_minute)
            current_tokens_minute = sum(tokens for _, tokens in self.token_usage_minute)

            if (current_requests_minute < self.max_requests_per_minute and 
                current_tokens_minute + tokens_to_add <= self.max_tokens_per_minute):
                return True, 0.0

            wait_time = 0.0
            if current_requests_minute >= self.max_requests_per_minute:
                wait_time = max(wait_time, 60 - (time.time() - self.request_timestamps_minute[0]) + 1)
            if current_tokens_minute + tokens_to_add > self.max_tokens_per_minute:
                wait_time = max(wait_time, 60 - (time.time() - self.token_usage_minute[0][0]) + 1)

            return False, wait_time

    def wait_for_availability(self, tokens_to_add: int):
        """Wait until we can make a request within rate limits."""
        while True:
            can_proceed, wait_time = self.can_make_request(tokens_to_add)
            if can_proceed:
                return
            
            logger.rate_limit(f"Rolling window limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    def record_request(self, tokens_used: int):
        """Record a successful request."""
        with self.lock:
            current_time = time.time()
            self.request_timestamps_minute.append(current_time)
            self.token_usage_minute.append((current_time, tokens_used))

class EPUBTranslator:
    def __init__(self, api_key: str, source_language: str = "auto", target_language: str = "English", max_concurrent: int = 4):
        if not api_key:
            raise ValueError("API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')                              
        self.source_lang = source_language
        self.target_lang = target_language
        # Sync max_concurrent with rate limiter max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(max_requests_per_minute=max_concurrent)
        self.should_stop = False
        self.translation_start_time = None
        self.total_api_requests = 0
        self.total_tokens_processed = 0
        self.completed_patches_count = 0  # Track actual completion count
        self.lock = threading.Lock()  # For thread-safe completion counting
        self._setup_interrupt_handler()
        
        logger.system(f"EPUB Translator initialized: {source_language} → {target_language}")
        logger.system(f"Concurrency limit: {max_concurrent} (synced with rate limit)")

    def _setup_interrupt_handler(self):
        """Setup background thread to listen for 's' key press."""
        def listen_for_stop():
            while not self.should_stop:
                try:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        if key == 's':
                            logger.interrupt("'S' key pressed - saving progress and stopping immediately...")
                            self.should_stop = True
                            break
                except:
                    time.sleep(0.5)
        
        if hasattr(select, 'select'):
            listener_thread = threading.Thread(target=listen_for_stop, daemon=True)
            listener_thread.start()
        else:
            logger.system("Interrupt handler not available on this system")

    def _extract_chapter_title(self, soup: BeautifulSoup) -> str:
        """Extract a meaningful title from chapter content."""
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(strip=True)
        
        for tag_name in ['h1', 'h2', 'h3']:
            heading = soup.find(tag_name)
            if heading and heading.get_text(strip=True):
                return heading.get_text(strip=True)
        
        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text(strip=True)
            if text and len(text) < 100:
                return text
        
        return "Untitled Chapter"

    def _extract_chapters(self, epub_path: str) -> List[Dict]:
        """Extract all chapters from the EPUB."""
        chapters = []
        book = epub.read_epub(epub_path)
        chapter_counter = 1
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            
            if soup.body and soup.body.get_text(strip=True):
                chapter_title = self._extract_chapter_title(soup)
                if not chapter_title or chapter_title == "Untitled Chapter":
                    chapter_title = f"Chapter {chapter_counter}"
                
                chapters.append({
                    'id': item.get_id() or f"chapter_{chapter_counter}",
                    'title': chapter_title,
                    'soup': soup,
                    'file_name': item.get_name()
                })
                chapter_counter += 1
        
        logger.system(f"Extracted {len(chapters)} chapters")
        return chapters

    def _create_patches(self, chapters: List[Dict], max_tokens_per_patch: int = 60000) -> List[List[Dict]]:
        """Group chapters into patches based on token limits."""
        patches = []
        current_patch = []
        current_tokens = 500
        
        for chapter in chapters:
            chapter_tokens = count_tokens(str(chapter['soup']))
            
            if current_patch and current_tokens + chapter_tokens + 50 > max_tokens_per_patch:
                patches.append(current_patch)
                current_patch = [chapter]
                current_tokens = 500 + chapter_tokens
            else:
                current_patch.append(chapter)
                current_tokens += chapter_tokens + (50 if current_patch else 0)
        
        if current_patch:
            patches.append(current_patch)
        
        logger.system(f"Created {len(patches)} patches")
        return patches

    def _translate_patch_worker(self, patch: List[Dict], patch_index: int, total_patches: int, max_retries: int = 10) -> PatchResult:
        """Worker function to translate a single patch with retry logic."""
        start_time = time.time()
        
        html_parts = [str(ch['soup']) for ch in patch]
        combined_html = "\n<!-- CHAPTER_SEPARATOR -->\n".join(html_parts)
        
        prompt_tokens = count_tokens(combined_html) + 500
        estimated_total_tokens = int(prompt_tokens * 1.4)
        
        prompt = f"""You are an expert literary translator, specializing in fiction. You have a deep understanding of literary devices, tone, cultural nuances, and character voice. You are not a literal, word-for-word machine translator; you are a creative partner tasked with preserving the original author's intent and spirit.

# GOAL
Your goal is to translate the book from {self.source_lang} to {self.target_lang}. The final translation must read as if it were originally written in the {self.target_lang}, while remaining completely faithful to the source's style, tone, and meaning.

CRITICAL INSTRUCTIONS:
1. Maintain the exact HTML structure and tags
2. Only translate TEXT CONTENT inside HTML tags
3. Do NOT modify, add, or remove HTML tags or attributes
4. Keep the <!-- CHAPTER_SEPARATOR --> markers exactly as they are
5. Return ONLY the translated HTML - no explanations
6. **Idioms & Culturalisms:** Do not translate idioms literally. Find the closest equivalent cultural idiom in the {self.target_lang}. If no direct equivalent exists, convey the original meaning in a natural-sounding way.
7. If you see duplicate text (like "Chapter 1: Title Chapter 1: Title"), translate it only once

INPUT HTML:
{combined_html}

TRANSLATED HTML:"""

        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Wait for rate limit availability
                self.rate_limiter.wait_for_availability(estimated_total_tokens)
                
                logger.api(f"Sending patch {patch_index} (attempt {attempt + 1}/{max_retries})")
                
                response = self.model.generate_content(prompt)
                if not response or not response.text:
                    raise Exception("Empty response from API")
                
                translated_html = response.text.strip()
                
                # Clean up markdown formatting
                for marker in ["```html", "```"]:
                    if translated_html.startswith(marker):
                        translated_html = translated_html[len(marker):]
                    if translated_html.endswith(marker):
                        translated_html = translated_html[:-len(marker)]
                translated_html = translated_html.strip()
                
                # Update chapters with translated content
                translated_parts = translated_html.split("<!-- CHAPTER_SEPARATOR -->")
                
                for i, (chapter, translated_part) in enumerate(zip(patch, translated_parts)):
                    if translated_part.strip():
                        translated_soup = BeautifulSoup(translated_part.strip(), 'html.parser')
                        chapter['soup'] = translated_soup
                        chapter['translated_title'] = self._extract_chapter_title(translated_soup)
                
                actual_total_tokens = prompt_tokens + count_tokens(translated_html)
                self.rate_limiter.record_request(actual_total_tokens)
                self.total_api_requests += 1
                self.total_tokens_processed += actual_total_tokens
                
                
                with self.lock:
                    self.completed_patches_count += 1
                    completion_pct = (self.completed_patches_count / total_patches) * 100
                
                processing_time = time.time() - start_time
                logger.success(f"Patch {patch_index} completed in {processing_time:.1f}s ({completion_pct:.0f}%)")
                
                return PatchResult(
                    patch_index=patch_index,
                    success=True,
                    chapters=patch,
                    attempts=attempt + 1,
                    processing_time=processing_time
                )
                
            except Exception as e:
                # Trim the error message to only show the first part
                last_error = str(e).split(' ')[0]
                logger.warning(f"Patch {patch_index}/{total_patches} attempt {attempt + 1}/{max_retries} failed: {last_error}")
                # No delay between retries as requested
        
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

    def save_partial_translation(self, results: List[PatchResult], original_book_path: str, output_path: str, total_patches: int, chapters: List[Dict]):
        """Save partially translated EPUB."""
        successful_results = [r for r in results if r.success]
        completed_patches = len(successful_results)
        
        partial_output = output_path.replace('.epub', f'_partial_{completed_patches}of{total_patches}.epub')
        
        logger.progress(f"Saving partial translation to: {partial_output}")
        self.create_translated_epub(chapters, original_book_path, partial_output)
        
        return partial_output

    def create_translated_epub(self, chapters: List[Dict], original_book_path: str, output_path: str):
        """Creates translated EPUB by modifying the original book."""
        book = epub.read_epub(original_book_path)
        
        translated_chapters = {ch['id']: ch['soup'] for ch in chapters}
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_id = item.get_id()
            if item_id in translated_chapters:
                translated_soup = translated_chapters[item_id]
                
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
        
        self._update_table_of_contents(book, chapters)
        
        original_title = book.get_metadata('DC', 'title')
        if original_title:
            book.set_title(f"{original_title[0][0]} (Translated)")
        else:
            book.set_title("Translated Book")
        
        book.set_language('en')
        
        epub.write_epub(output_path, book, {})
        logger.success(f'Successfully created translated EPUB: {output_path}')

    def translate_book(self, epub_path: str, output_path: str):
        """Main translation function with concurrent processing."""
        logger.system("Analyzing book content...")
        chapters = self._extract_chapters(epub_path)
        
        if not chapters:
            logger.error("No chapters found in the EPUB.")
            return

        patches = self._create_patches(chapters)
        total_tokens = sum(count_tokens(str(ch['soup'])) for ch in chapters)
        
        logger.system(f"Total tokens: {total_tokens:,}")
        
        for i, patch in enumerate(patches):
            patch_tokens = sum(count_tokens(str(ch['soup'])) for ch in patch)
            logger.info(f"Patch {i+1}: {len(patch)} chapters, {patch_tokens:,} tokens")
        
        proceed = input(f"\n\033[93mProceed with translation? (yes/no): \033[0m")
        if proceed.lower() != 'yes':
            logger.system("Translation cancelled.")
            return

        print(f"\n\033[1m\033[96m🚀 Starting concurrent translation...\033[0m")
        if hasattr(select, 'select'):
            logger.system("Press 's' and Enter at any time to save progress and stop immediately")
        
        self.translation_start_time = time.time()
        self.completed_patches_count = 0  # Reset counter at start
        results = []
        
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
                    logger.interrupt(f"Stopping immediately. Cancelling remaining patches...")
                    # Cancel all remaining futures
                    for f in future_to_patch:
                        if not f.done():
                            f.cancel()
                    
                    # Save progress with what we have so far
                    logger.interrupt(f"Saving partial translation with {len(results)} completed patches...")
                    if results:
                        self.save_partial_translation(results, epub_path, output_path, len(patches), chapters)
                    else:
                        logger.interrupt("No patches completed yet, nothing to save.")
                    return
                
                result = future.result()
                results.append(result)
                completed_count += 1
                
                if not result.success:
                    logger.error(f"Patch {result.patch_index}/{len(patches)} failed after {result.attempts} attempts")
        
        # Sort results by patch index
        results.sort(key=lambda x: x.patch_index)
        successful_results = [r for r in results if r.success]
        
        logger.info(f"[SYSTEM] Translation completed: {len(successful_results)}/{len(patches)} patches successful")
        
        if len(successful_results) == len(patches):
            logger.info("[SYSTEM] Creating complete translated EPUB...")
            self.create_translated_epub(chapters, epub_path, output_path)
        elif successful_results:
            logger.info("[SYSTEM] Some patches failed. Saving partial translation...")
            self.save_partial_translation(results, epub_path, output_path, len(patches), chapters)
        else:
            logger.info("[SYSTEM] No patches were successfully translated.")

def main():
    API_KEY = "xxxxxxxxxxxxxxx"  # Replace with your key
    INPUT_EPUB = "file_name.epub"
    OUTPUT_EPUB = "output_name.epub"

    translator = EPUBTranslator(api_key=API_KEY, max_concurrent=4)
    translator.translate_book(INPUT_EPUB, OUTPUT_EPUB)

if __name__ == "__main__":
    main()
