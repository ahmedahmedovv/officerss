# app.py
import feedparser
import json
import os
import yaml
from datetime import datetime, timedelta
import logging
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from aiohttp import ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'fetcher.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class FetchError(Exception):
    """Custom exception for fetch errors with context"""
    url: str
    message: str
    original_error: Optional[Exception] = None

    def __str__(self) -> str:
        return f"Error fetching {self.url}: {self.message}"

class NewsFetcher:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.cache_file_path = os.path.join(os.path.dirname(__file__), '../cache/news_cache.json')
        self.timeout = ClientTimeout(total=30)  # 30 seconds timeout

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file with error handling"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            raise

    async def fetch_feed_async(self, url: str, cache_data: Dict, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Fetch feed asynchronously with enhanced error handling and retries"""
        try:
            if url in cache_data and self._is_cache_valid(cache_data[url]['fetched_at']):
                logger.info(f"Using cached data for {url}")
                return cache_data[url]['articles']
            
            logger.info(f"Fetching fresh data for {url}")
            async with session.get(url, timeout=self.timeout) as response:
                response.raise_for_status()  # Raise exception for bad status codes
                feed_content = await response.text()
                
            feed = feedparser.parse(feed_content)
            
            if hasattr(feed, 'bozo_exception'):
                raise FetchError(url, f"Feed parsing error: {feed.bozo_exception}")
            
            articles = self._process_feed_entries(feed, url)
            
            cache_data[url] = {
                'fetched_at': datetime.now().isoformat(),
                'articles': articles
            }
            return articles
            
        except aiohttp.ClientError as e:
            logger.error(f"Network error for {url}: {str(e)}")
            return self._handle_fetch_error(url, cache_data, e)
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}", exc_info=True)
            return self._handle_fetch_error(url, cache_data, e)

    def _process_feed_entries(self, feed: feedparser.FeedParserDict, url: str) -> List[Dict[str, Any]]:
        """Process feed entries with validation"""
        articles = []
        for entry in feed.entries:
            try:
                article = {
                    'title': getattr(entry, 'title', 'No title'),
                    'link': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'summary': getattr(entry, 'summary', ''),
                    'source_url': url
                }
                
                # Validate required fields
                if not article['link'] or not article['title']:
                    logger.warning(f"Skipping article from {url} - missing required fields")
                    continue
                    
                articles.append(article)
            except AttributeError as e:
                logger.error(f"Error processing entry from {url}: {str(e)}")
                continue
        return articles

    def _handle_fetch_error(self, url: str, cache_data: Dict, error: Exception) -> List[Dict[str, Any]]:
        """Handle fetch errors and return cached data if available"""
        if url in cache_data:
            logger.info(f"Using cached data due to error for {url}")
            return cache_data[url]['articles']
        return []

    def _is_cache_valid(self, cache_time: str, max_age_minutes: int = 15) -> bool:
        """Check if cache data is recent enough"""
        try:
            age = datetime.now() - datetime.fromisoformat(cache_time)
            return age < timedelta(minutes=max_age_minutes)
        except ValueError as e:
            logger.error(f"Invalid cache time format: {str(e)}")
            return False

    async def fetch_all_feeds(self, urls: List[str], cache_data: Dict) -> List[Dict[str, Any]]:
        """Fetch all feeds concurrently with connection pooling"""
        connector = aiohttp.TCPConnector(limit=10)  # Limit concurrent connections
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_feed_async(url, cache_data, session) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and flatten results
            articles = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {str(result)}")
                    continue
                articles.extend(result)
            return articles

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def save_cache(self, cache_path: str, cache_data: Dict) -> None:
        """Save cache with retry mechanism"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving cache file: {str(e)}")
            raise

def main():
    try:
        # Initialize fetcher
        config_path = os.path.join(os.path.dirname(__file__), '../config/1_fetcher_config.yaml')
        fetcher = NewsFetcher(config_path)
        
        # Load cache
        cache_data = {}
        if os.path.exists(fetcher.cache_file_path):
            try:
                with open(fetcher.cache_file_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Cache file corrupted: {str(e)}")
            except IOError as e:
                logger.error(f"Error reading cache file: {str(e)}")

        # Read URLs
        url_file_path = os.path.join(os.path.dirname(__file__), fetcher.config['url_file_path'])
        try:
            with open(url_file_path, 'r') as file:
                urls = [line.strip() for line in file if line.strip()]
        except IOError as e:
            logger.error(f"Error reading URL file: {str(e)}")
            raise

        # Fetch feeds
        all_articles = asyncio.run(fetcher.fetch_all_feeds(urls, cache_data))

        # Save results
        fetcher.save_cache(fetcher.cache_file_path, cache_data)
        
        json_folder_path = os.path.join(os.path.dirname(__file__), fetcher.config['json_folder_path'])
        os.makedirs(json_folder_path, exist_ok=True)
        with open(os.path.join(json_folder_path, 'news_articles.json'), 'w', encoding='utf-8') as json_file:
            json.dump(all_articles, json_file, indent=4, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()