from transformers import pipeline
import json
import os
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_cache_key(text):
    """Generate a unique cache key for the text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_cache(cache_path):
    """Load the cache from file"""
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache, cache_path):
    """Save the cache to file"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)  # Ensure cache directory exists
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.error(f"Error saving cache: {e}")

def classify_article(article, classifier, categories, cache):
    """Classify a single article"""
    text = f"{article['title']} {article['summary']}"
    cache_key = get_cache_key(text)
    
    if cache_key in cache:
        return {'article': article, 'category': cache[cache_key], 'cache_key': cache_key, 'cached': True}
    
    try:
        result = classifier(text, categories, multi_label=False)
        category = result['labels'][0]
        return {'article': article, 'category': category, 'cache_key': cache_key, 'cached': False}
    except Exception as e:
        logging.error(f"Error classifying article: {article['title']}. Error: {e}")
        return {'article': article, 'category': "Uncategorized", 'cache_key': cache_key, 'cached': False}

def categorize_news():
    # Load configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(current_dir), 'config', '3_category_config.yaml')
    config = load_config(config_path)
    
    # Setup logging
    logging.basicConfig(
        filename=config['log_file'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load cache
        cache = load_cache(config['cache_file'])
        cache_hits = 0
        cache_misses = 0
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() and config['device'] != 'cpu' else "cpu"
        
        # Load pre-trained zero-shot classification pipeline
        classifier = pipeline("zero-shot-classification",
                            model=config['model_name'],
                            device=device,
                            batch_size=config['batch_size'])
        
        # Load news articles
        try:
            with open(config['input_file'], 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: {config['input_file']}")
            return
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from file: {config['input_file']}")
            return
        
        # Process articles in parallel
        max_workers = min(config['max_workers'], len(articles))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {
                executor.submit(classify_article, article, classifier, config['categories'], cache): article 
                for article in articles
            }
            
            # Process results as they complete
            processed_count = 0
            for future in as_completed(future_to_article):
                result = future.result()
                article = result['article']
                article['category'] = result['category']
                
                if result['cached']:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    cache[result['cache_key']] = result['category']
                
                processed_count += 1
                if processed_count % config['cache_save_interval'] == 0:
                    save_cache(cache, config['cache_file'])
        
        # Final cache save
        if cache_misses > 0:
            save_cache(cache, config['cache_file'])
        
        # Log statistics
        logging.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        logging.info(f"Using device: {device}")
        
        # Save categorized articles
        try:
            with open(config['output_file'], 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logging.error(f"Error writing to file: {config['output_file']}. Error: {e}")
    
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    categorize_news()