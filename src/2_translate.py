import json
import asyncio
import aiohttp
from pathlib import Path
import time
from datetime import datetime
from typing import List, Dict, Any
import re
import yaml
import logging
from logging.handlers import RotatingFileHandler

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration"""
    log_file = config.get('log_file', 'translator.log')
    
    # Ensure the log directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('translator')
    logger.setLevel(logging.INFO)
    
    # File handler with rotation and UTF-8 encoding
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class AsyncTranslator:
    def __init__(self, config: dict):
        self.logger = setup_logging(config)
        self.cache_file = Path(config['cache_file'])
        self.backup_interval = config.get('cache_save_interval', 50)
        self.base_url = "https://translate.googleapis.com/translate_a/single"
        self.batch_size = config['batch_size']
        self.target_language = config['target_language']
        self.min_text_length = config['min_text_length']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay']
        self.translate_fields = set(config['translate_fields'])
        self.cache = self._load_cache()
        self.cache_modified = False
        self.cache_save_interval = config.get('cache_save_interval', 50)  # Save every 50 translations
        self.translations_since_save = 0
        self.source_languages = config.get('source_languages', ['auto'])
        
    def _is_translation_needed(self, text: str) -> bool:
        """
        Check if text needs translation based on language detection.
        Returns False if text is already in target language.
        """
        try:
            # Common words in various languages to skip translation if target is English
            english_indicators = {'the', 'and', 'in', 'to', 'of', 'for'}
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            
            # If text contains several English words, assume it's English
            if self.target_language == 'en' and len(text_words.intersection(english_indicators)) >= 2:
                return False
                
            return True
        except Exception as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return True
            
    def _is_english(self, text: str) -> bool:
        """
        Check if text is likely English using basic heuristics.
        Returns True if the text appears to be English.
        """
        # Common English words that rarely appear in other languages
        english_words = {'the', 'and', 'for', 'in', 'to', 'of', 'at', 'on', 'with', 'by'}
        
        # Convert to lowercase and split into words
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # If text contains several common English words, it's likely English
        common_words_count = len(words.intersection(english_words))
        return common_words_count >= 2

    def _load_cache(self) -> Dict[str, str]:
        """Load the translation cache from disk with error handling"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                self.logger.info(f"Loaded {len(cache_data)} cached translations")
                return cache_data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.warning(f"Cache file corrupted, starting fresh: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
        return {}
    
    def _save_cache(self, force: bool = False) -> None:
        """Save the translation cache."""
        try:
            # Save current cache
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {str(e)}")

    def _get_cache_key(self, text: str, target_lang: str) -> str:
        """Generate a unique cache key for the text and target language"""
        return f"{text}|{target_lang}"
            
    async def translate_batch(self, session: aiohttp.ClientSession, texts: List[str], dest='en') -> List[str]:
        """Translate a batch of texts with caching"""
        if not texts:
            return []
            
        results = []
        texts_to_translate = []
        cache_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, dest)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                texts_to_translate.append(text)
                cache_indices.append(i)
                
        # Only translate what's not in cache
        if texts_to_translate:
            self.logger.info(f"Cache hit rate: {(len(texts) - len(texts_to_translate)) / len(texts):.2%}")
            
            for text in texts_to_translate:
                try:
                    translated = await self._translate_single_text(session, text, dest)
                    cache_key = self._get_cache_key(text, dest)
                    self.cache[cache_key] = translated
                    self.cache_modified = True
                    self.translations_since_save += 1
                    
                    # Periodic cache saving
                    if self.translations_since_save >= self.cache_save_interval:
                        self._save_cache()
                        
                    results.append(translated)
                except Exception as e:
                    self.logger.error(f"Translation failed for text: {text[:100]}... Error: {str(e)}")
                    results.append(text)  # Fallback to original text
                    
        # Save cache at the end if modified
        if self.cache_modified:
            self._save_cache(force=True)
            
        return results
                    
    async def _translate_single_text(self, session: aiohttp.ClientSession, text: str, dest: str) -> str:
        """Handle translation of a single text with proper error handling"""
        if not self._is_translation_needed(text):
            return text
            
        params = {
            'client': 'gtx',
            'sl': 'auto',  # Auto-detect source language
            'tl': dest,
            'dt': 't',
            'q': text
        }
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        wait_time = self.retry_delay * (attempt + 1)
                        self.logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    if response.status != 200:
                        raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                    data = await response.json()
                    if not data or not data[0] or not data[0][0] or not data[0][0][0]:
                        raise ValueError("Invalid response format")
                        
                    translated = data[0][0][0]
                    
                    # Basic quality checks
                    if len(translated) < self.min_text_length:
                        self.logger.warning(f"Translation too short: {text[:100]}...")
                        return text
                        
                    # Check for common translation artifacts
                    if '[' in translated and ']' in translated and '[' not in text:
                        self.logger.warning(f"Possible machine translation artifacts detected")
                        return text
                        
                    self.cache[text] = translated
                    return translated
                    
            except aiohttp.ClientError as e:
                self.logger.warning(f"Network error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

async def translate_json_file(input_file_path: str, output_file_path: str, config_path: str):
    start_time = time.time()
    print(f"Starting translation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config and initialize translator
    config = load_config(config_path)
    translator = AsyncTranslator(config)
    
    # Load JSON data
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Collect strings to translate
    total_strings = 0
    original_strings: List[tuple[str, Any]] = []
    
    def collect_strings(obj: Any, path: List[Any] = None) -> None:
        nonlocal total_strings
        if path is None:
            path = []
            
        if isinstance(obj, dict):
            # Only collect non-English title and summary fields
            if 'title' in obj and isinstance(obj['title'], str) and len(obj['title'].strip()) > 3:
                if not translator._is_english(obj['title']):
                    original_strings.append((obj['title'], ['title']))
                    total_strings += 1
            if 'summary' in obj and isinstance(obj['summary'], str) and len(obj['summary'].strip()) > 3:
                if not translator._is_english(obj['summary']):
                    original_strings.append((obj['summary'], ['summary']))
                    total_strings += 1
            
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    collect_strings(value, path + [key])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                collect_strings(item, path + [i])

    collect_strings(data)
    strings_to_translate = [s[0] for s in original_strings]
    print(f"Found {total_strings} strings to translate")

    # Translate strings in batches
    translated_strings = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(strings_to_translate), translator.batch_size):
            batch = strings_to_translate[i:i + translator.batch_size]
            tasks.append(translator.translate_batch(session, batch))
            
        batch_results = await asyncio.gather(*tasks)
        for result in batch_results:
            translated_strings.extend(result)

    # Create translation mapping
    translations = dict(zip(strings_to_translate, translated_strings))

    # Map translations back to original structure
    def map_translations(obj: Any) -> Any:
        if isinstance(obj, dict):
            result = obj.copy()
            if 'title' in result and result['title'] in translations:
                result['title'] = translations[result['title']]
            if 'summary' in result and result['summary'] in translations:
                result['summary'] = translations[result['summary']]
            return result
        elif isinstance(obj, list):
            return [map_translations(item) for item in obj]
        return obj

    translated_data = map_translations(data)

    # Save the translated JSON
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(translated_data, file, ensure_ascii=False, indent=2)
        print(f"Translation completed! Saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving translated file: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    strings_per_second = total_strings / total_time if total_time > 0 else 0
    
    print("\nTranslation Statistics:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Strings translated: {total_strings}")
    print(f"Average speed: {strings_per_second:.2f} strings/second")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Load configuration
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / "config" / "2_translator_config.yaml"
    config = load_config(str(config_path))
    
    # Create necessary directories
    Path(config['output_file']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['cache_file']).parent.mkdir(parents=True, exist_ok=True)
    
    # Run translation with config path instead of cache file path
    asyncio.run(translate_json_file(
        config['input_file'],
        config['output_file'],
        str(config_path)  # Pass config path instead of cache file path
    ))