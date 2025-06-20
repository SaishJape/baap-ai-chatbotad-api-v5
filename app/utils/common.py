import re
import time
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urljoin, urlparse
import warnings
import logging
from typing import Dict, List

from app.db.qdrant import is_cancellation_requested

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def scrape_url(url: str) -> str:
    """Scrape content from a single URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return ""

def clean_text(html: str) -> str:
    """Clean HTML and extract meaningful text."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from meaningful tags
        texts = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "span"])
        clean_texts = []
        
        for element in texts:
            text = element.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short texts
                clean_texts.append(text)
        
        return "\n".join(clean_texts)
    except Exception as e:
        logging.error(f"Failed to clean text: {e}")
        return ""

def crawl_website(start_url: str, max_pages: int = None, task_id: str = None) -> Dict[str, str]:
    """Crawl a website starting from the given URL. If max_pages is None, crawl all pages."""
    visited = set()
    data = {}
    urls_to_visit = [start_url]
   
    while urls_to_visit:
        # Check for cancellation before processing each URL
        if task_id and is_cancellation_requested(task_id):
            logging.info(f"Crawling cancelled for task {task_id} after {len(visited)} pages")
            break
            
        # If max_pages is set and we've reached the limit, stop
        if max_pages is not None and len(visited) >= max_pages:
            break
           
        current_url = urls_to_visit.pop(0)
       
        if current_url in visited:
            continue
           
        try:
            logging.info(f"Crawling: {current_url} (Total crawled: {len(visited)})")
            html = scrape_url(current_url)
           
            if not html:
                continue
               
            # Check if it's HTML content
            try:
                response_check = requests.head(current_url, timeout=5)
                content_type = response_check.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue
            except:
                # If head request fails, assume it's HTML and continue
                pass
               
            visited.add(current_url)
            data[current_url] = html
           
            # Extract links for further crawling
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(current_url, link["href"])
               
                # Only crawl links from the same domain
                if (urlparse(full_url).netloc == urlparse(start_url).netloc
                    and full_url not in visited
                    and full_url not in urls_to_visit):
                   
                    # Skip certain file types and fragments
                    if not should_skip_url(full_url):
                        urls_to_visit.append(full_url)
            
            # Add a small delay to make cancellation more responsive
            time.sleep(0.1)
                   
        except Exception as e:
            logging.error(f"Error crawling {current_url}: {e}")
            continue
   
    if task_id and is_cancellation_requested(task_id):
        logging.info(f"Crawling was cancelled for task {task_id}. Pages crawled: {len(data)}")
    else:
        logging.info(f"Crawling completed. Total pages crawled: {len(data)}")
    
    return data

def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped based on file extension or other criteria."""
    skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.rar', '.exe', '.dmg', '.mp4', '.mp3', '.avi']
    skip_patterns = ['#', 'mailto:', 'tel:', 'javascript:', 'ftp://']
    
    url_lower = url.lower()
    
    # Skip if it has a file extension we don't want
    for ext in skip_extensions:
        if url_lower.endswith(ext):
            return True
    
    # Skip if it matches certain patterns
    for pattern in skip_patterns:
        if pattern in url_lower:
            return True
    
    return False


def preprocess_text(text: str) -> str:
    """Preprocess text to ensure it's clean and properly formatted."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Create overlapping chunks from text with fixed size."""
    if not text:
        return []
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks