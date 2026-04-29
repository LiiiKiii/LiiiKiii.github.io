#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General external resource search module.
Retrieve resources through general search and scraping without relying on a specific API.
Supports text, video, image, and code search.
"""

import os
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote, urlencode, urlparse, parse_qs
from typing import List, Dict
import json
import time
import random
import unicodedata

VERBOSE_LOGS = os.environ.get("AI_PEDIA_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}


def debug_print(*args, force: bool = False, **kwargs) -> None:
    """Handle debug print."""
    if force or VERBOSE_LOGS:
        print(*args, **kwargs)

AI_RELEVANT_KEYWORDS = [
    'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 'dl',
    'neural network', 'neural networks', 'neural net', 'nn',
    
    'algorithm', 'model', 'training', 'inference', 'prediction', 'classification', 
    'regression', 'clustering', 'supervised learning', 'unsupervised learning', 
    'reinforcement learning', 'semi-supervised', 'transfer learning', 'meta learning',
    'few-shot learning', 'zero-shot learning',
    
    'convolutional neural network', 'cnn', 'recurrent neural network', 'rnn', 
    'lstm', 'gru', 'transformer', 'attention mechanism', 'self-attention',
    'generative adversarial network', 'gan', 'variational autoencoder', 'vae',
    'autoencoder', 'encoder-decoder', 'seq2seq', 'bert', 'gpt', 'large language model', 'llm',
    
    'computer vision', 'cv', 'image recognition', 'object detection', 'semantic segmentation',
    'image classification', 'face recognition', 'optical character recognition', 'ocr',
    'image generation', 'image processing',
    
    'natural language processing', 'nlp', 'text classification', 'sentiment analysis',
    'named entity recognition', 'ner', 'machine translation', 'text generation',
    'language model', 'word embedding', 'tokenization', 'text mining',
    
    'gradient descent', 'backpropagation', 'activation function', 'loss function',
    'optimization', 'regularization', 'dropout', 'batch normalization',
    'overfitting', 'underfitting', 'cross-validation', 'hyperparameter tuning',
    'feature engineering', 'feature extraction', 'feature selection',
    
    'dataset', 'training data', 'test data', 'validation data', 'data preprocessing',
    'data augmentation', 'feature scaling', 'normalization',
    
    'recommendation system', 'recommender system', 'speech recognition', 'speech synthesis',
    'reinforcement learning', 'rl', 'autonomous driving', 'robotics',
    'knowledge graph', 'knowledge representation', 'reasoning',
    
    'research paper', 'academic', 'arxiv', 'conference', 'journal', 'publication',
    'thesis', 'dissertation', 'tutorial', 'survey',
    
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
    'jupyter', 'notebook', 'python', 'tensor', 'gpu', 'cuda',
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",  # Prefer English content
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def _build_http_session() -> requests.Session:
    """Build an HTTP session with connection pooling and light retries to reduce repeated handshakes."""
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_SESSION = _build_http_session()


def http_get(url: str, headers=None, timeout: int = 15):
    """Shared GET entry point that reuses connections while preserving header and timeout behavior."""
    return HTTP_SESSION.get(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)


def is_english_content(text: str, min_english_ratio: float = 0.7) -> bool:
    """Return whether english content."""
    if not text or len(text.strip()) < 10:
        return False
    
    total_chars = 0
    english_chars = 0
    non_latin_chars = 0
    
    for char in text:
        if char.isspace():
            continue
        
        total_chars += 1
        char_code = ord(char)
        
        if char_code < 128:
            english_chars += 1
        elif (
            (0x0900 <= char_code <= 0x097F) or  # Devanagari (Hindi, Sanskrit, etc.)
            (0x0980 <= char_code <= 0x09FF) or  # Bengali
            (0x0A00 <= char_code <= 0x0A7F) or  # Gurmukhi (Punjabi)
            (0x0A80 <= char_code <= 0x0AFF) or  # Gujarati
            (0x0B00 <= char_code <= 0x0B7F) or  # Oriya
            (0x0B80 <= char_code <= 0x0BFF) or  # Tamil
            (0x0C00 <= char_code <= 0x0C7F) or  # Telugu
            (0x0C80 <= char_code <= 0x0CFF) or  # Kannada
            (0x0D00 <= char_code <= 0x0D7F) or  # Malayalam
            (0x4E00 <= char_code <= 0x9FFF) or  # CJK Unified Ideographs (Chinese, Japanese, Korean)
            (0x0600 <= char_code <= 0x06FF) or  # Arabic
            (0x0590 <= char_code <= 0x05FF) or  # Hebrew
            (0x0400 <= char_code <= 0x04FF)     # Cyrillic (Russian, etc.)
        ):
            non_latin_chars += 1
    
    if total_chars == 0:
        return False
    
    english_ratio = english_chars / total_chars if total_chars > 0 else 0
    non_latin_ratio = non_latin_chars / total_chars if total_chars > 0 else 0
    
    if non_latin_ratio > 0.3:
        return False
    
    return english_ratio >= min_english_ratio


def filter_english_content(resources: List[Dict], content_key: str = "content") -> List[Dict]:
    """Handle filter english content."""
    english_resources = []
    
    for res in resources:
        texts_to_check = []
        if "title" in res:
            texts_to_check.append(res["title"])
        if content_key in res:
            texts_to_check.append(res[content_key])
        if "description" in res:
            texts_to_check.append(res["description"])
        
        combined_text = " ".join(str(t) for t in texts_to_check if t)
        
        if is_english_content(combined_text, min_english_ratio=0.6):
            english_resources.append(res)
        else:
            debug_print(f"  Filtered non-English content: {res.get('title', 'Unknown')[:50]}")
    
    return english_resources


def search_text_resources(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    Search for academic text resources.
    Prioritize high-quality material such as academic articles, papers, and detailed explanations.
    Return: List[Dict], each with {title, content, source, url}.
    """
    results = []
    
    academic_queries = [
        f"{keyword} research paper",
        f"{keyword} academic article",
        f"{keyword} scholarly",
        f"{keyword}",
    ]
    
    try:
        wiki_result = fetch_wikipedia_article(keyword)
        if wiki_result:
            results.append(wiki_result)
            debug_print(f"  Found a Wikipedia article")
    except Exception as e:
        print(f"Wikipedia search error for {keyword}: {e}")
    
    try:
        scholar_results = fetch_google_scholar_results(keyword, max_results=5)
        results.extend(scholar_results)
        debug_print(f"  Google Scholar returned {len(scholar_results)} results")
    except Exception as e:
        print(f"Google Scholar search error for {keyword}: {e}")
    
    try:
        arxiv_results = fetch_arxiv_results(keyword, max_results=3)
        results.extend(arxiv_results)
        debug_print(f"  arXiv returned {len(arxiv_results)} results")
    except Exception as e:
        print(f"arXiv search error for {keyword}: {e}")
    
    
    seen_urls = set()
    unique_results = []
    for res in results:
        url = res.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)
    
    source_priority = {
        "Wikipedia": 1,
        "Google Scholar": 2,
        "arXiv": 3,
        "Academic Website": 4,
        "Web Article": 5,
        "Web Link": 6
    }
    unique_results.sort(key=lambda x: source_priority.get(x.get("source", ""), 99))
    
    english_results = filter_english_content(unique_results, content_key="content")
    if len(english_results) < max_results:
        debug_print(f"  English-content filtering: {len(unique_results)} -> {len(english_results)} results")
    
    return english_results[:max_results]


def search_youtube_videos(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    Search YouTube videos through HTML scraping without an API.
    Return: List[Dict], each with {title, url, description, video_id}.
    """
    results = []
    
    try:
        search_url = f"https://www.youtube.com/results?search_query={quote(keyword)}&sp=EgIoAQ%253D%253D"  # Add an English-content filter
        headers = DEFAULT_HEADERS.copy()
        
        resp = http_get(search_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            html = resp.text
            
            import re as regex_module
            pattern = r'var ytInitialData = ({.*?});'
            match = regex_module.search(pattern, html)
            
            if match:
                try:
                    data = json.loads(match.group(1))
                    videos = extract_youtube_videos_from_json(data, max_results)
                    results.extend(videos)
                except json.JSONDecodeError:
                    pass
            
            if not results:
                videos = extract_youtube_videos_from_html(html, max_results)
                results.extend(videos)
        
        if not results:
            results.append({
                "title": f"YouTube results: {keyword}",
                "url": search_url,
                "description": f"Open YouTube results for videos about '{keyword}'",
                "video_id": None,
                "thumbnail": "",
                "type": "video"
            })
    except Exception as e:
        print(f"YouTube search error for {keyword}: {e}")
        results.append({
            "title": f"YouTube results: {keyword}",
            "url": f"https://www.youtube.com/results?search_query={quote(keyword)}&sp=EgIoAQ%253D%253D",
            "description": f"Open YouTube results for videos about '{keyword}'",
            "video_id": None,
            "thumbnail": "",
            "type": "video"
        })
    
    english_results = filter_english_content(results, content_key="description")
    if len(english_results) < len(results):
        debug_print(f"  English-content filtering: {len(results)} -> {len(english_results)} results")
    
    return english_results[:max_results]


def search_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    Search images across multiple sources.
    Return: List[Dict], each with {title, url, thumbnail, source}.
    """
    results = []
    
    try:
        google_images = fetch_google_images(keyword, max_results=max_results // 3)
        results.extend(google_images)
        debug_print(f"  Google Images returned {len(google_images)} results")
    except Exception as e:
        print(f"Google Images search error for {keyword}: {e}")
    
    try:
        bing_images = fetch_bing_images(keyword, max_results=max_results // 3)
        results.extend(bing_images)
        debug_print(f"  Bing Images returned {len(bing_images)} results")
    except Exception as e:
        print(f"Bing Images search error for {keyword}: {e}")
    
    try:
        unsplash_images = fetch_unsplash_images(keyword, max_results=max_results // 4)
        results.extend(unsplash_images)
        debug_print(f"  Unsplash returned {len(unsplash_images)} results")
    except Exception as e:
        print(f"Unsplash search error for {keyword}: {e}")
    
    try:
        pexels_images = fetch_pexels_images(keyword, max_results=max_results // 4)
        results.extend(pexels_images)
        debug_print(f"  Pexels returned {len(pexels_images)} results")
    except Exception as e:
        print(f"Pexels search error for {keyword}: {e}")
    
    if len(results) < 3:
        try:
            generic_images = fetch_generic_image_links(keyword, max_results=3)
            results.extend(generic_images)
            debug_print(f"  Generic image search returned {len(generic_images)} results")
        except Exception as e:
            print(f"Generic image search error for {keyword}: {e}")
    
    seen_urls = set()
    unique_results = []
    for res in results:
        url = res.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)
    
    english_results = filter_english_content(unique_results, content_key="title")
    if len(english_results) < len(unique_results):
        debug_print(f"  English-content filtering: {len(unique_results)} -> {len(english_results)} results")
    
    return english_results[:max_results]


def search_code_resources(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    Search code resources from GitHub.
    Return: List[Dict], each with {title, url, description, source, type}.
    """
    results = []
    
    try:
        github_results = fetch_github_code(keyword, max_results=max_results)
        results.extend(github_results)
        debug_print(f"  GitHub returned {len(github_results)} code resources")
    except Exception as e:
        print(f"GitHub search error for {keyword}: {e}")
    
    seen_urls = set()
    unique_results = []
    for res in results:
        url = res.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)
    
    english_results = filter_english_content(unique_results, content_key="description")
    if len(english_results) < len(unique_results):
        debug_print(f"  English-content filtering: {len(unique_results)} -> {len(english_results)} results")
    
    return english_results[:max_results]



def fetch_github_code(keyword: str, max_results: int = 10) -> List[Dict]:
    """Search GitHub repositories and return direct repository links rather than the search page."""
    results = []
    
    try:
        search_query = f"{keyword} machine-learning OR deep-learning OR pytorch OR tensorflow"
        search_url = f"https://github.com/search?q={quote(search_query)}&type=repositories&s=stars&o=desc"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            try:
                json_pattern = r'application/json[^>]*>([^<]+)'
                json_matches = re.findall(json_pattern, html)
                for json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        repos = extract_repos_from_json(data, max_results)
                        for repo in repos:
                            if len(results) >= max_results:
                                break
                            if repo not in results:
                                results.append(repo)
                    except:
                        continue
            except:
                pass
            
            repo_link_pattern = r'href="/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"[^>]*>'
            all_matches = re.findall(repo_link_pattern, html)
            
            seen_urls = set()
            for repo_path in all_matches:
                if len(results) >= max_results:
                    break
                
                if '/' in repo_path and len(repo_path.split('/')) == 2:
                    excluded_paths = ['search', 'explore', 'trending', 'topics', 'collections', 
                                      'settings', 'login', 'signup', 'join', 'pricing', 'enterprise',
                                      'features', 'security', 'marketplace', 'sponsors', 'about']
                    if any(excluded in repo_path.lower() for excluded in excluded_paths):
                        continue
                    
                    repo_url = f"https://github.com/{repo_path}"
                    
                    if repo_url not in seen_urls:
                        seen_urls.add(repo_url)
                        
                        repo_name = repo_path.split('/')[-1]
                        
                        repo_text = f"{repo_name} {repo_path}".lower()
                        has_ai_keywords = any(kw in repo_text for kw in ['machine', 'learning', 'deep', 'neural', 'ai', 'ml', 'dl', 'pytorch', 'tensorflow', 'keras', 'scikit', 'transformer', 'cnn', 'rnn', 'lstm', 'gan', 'bert', 'gpt'])
                        
                        if has_ai_keywords or keyword.lower() in repo_text:
                            description = f"GitHub repository: {repo_name}"
                            
                            results.append({
                                "title": repo_name,
                                "url": repo_url,
                                "description": description,
                                "source": "GitHub",
                                "type": "code"
                            })
            
            search_result_pattern = r'<div[^>]*class="[^"]*repo-list[^"]*"[^>]*>.*?href="/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"[^>]*>([^<]+)</a>'
            search_matches = re.findall(search_result_pattern, html, re.DOTALL)
            
            for match in search_matches:
                if len(results) >= max_results:
                    break
                repo_path = match[0]
                repo_name = match[1] if len(match) > 1 else repo_path.split('/')[-1]
                
                if '/' in repo_path and len(repo_path.split('/')) == 2:
                    repo_url = f"https://github.com/{repo_path}"
                    if repo_url not in seen_urls:
                        seen_urls.add(repo_url)
                        repo_name = re.sub(r'<[^>]+>', '', repo_name).strip()
                        repo_text = f"{repo_name} {repo_path}".lower()
                        has_ai_keywords = any(kw in repo_text for kw in ['machine', 'learning', 'deep', 'neural', 'ai', 'ml', 'dl', 'pytorch', 'tensorflow'])
                        
                        if has_ai_keywords or keyword.lower() in repo_text:
                            results.append({
                                "title": repo_name,
                                "url": repo_url,
                                "description": f"GitHub repository: {repo_name}",
                                "source": "GitHub",
                                "type": "code"
                            })
            
            seen = set()
            unique_results = []
            for res in results:
                url = res.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    unique_results.append(res)
            results = unique_results[:max_results]
            
    except Exception as e:
        print(f"Error fetching GitHub code: {e}")
    
    return results[:max_results]


def extract_repos_from_json(data, max_results: int = 10) -> List[Dict]:
    """Recursively extract repository information from GitHub JSON data."""
    results = []
    
    def traverse(obj, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return
        if len(results) >= max_results:
            return
        
        if isinstance(obj, dict):
            if 'full_name' in obj and 'html_url' in obj:
                repo_name = obj.get('full_name', '').split('/')[-1]
                repo_url = obj.get('html_url', '')
                if repo_url and 'github.com' in repo_url and '/search' not in repo_url:
                    results.append({
                        "title": repo_name,
                        "url": repo_url,
                        "description": obj.get('description', f"GitHub repository: {repo_name}"),
                        "source": "GitHub",
                        "type": "code"
                    })
            
            for value in obj.values():
                traverse(value, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item, depth + 1)
    
    traverse(data)
    return results



def fetch_wikipedia_article(keyword: str) -> Dict:
    """Fetch wikipedia article."""
    try:
        wiki_url = f"https://en.wikipedia.org/wiki/{quote(keyword.replace(' ', '_'))}"
        resp = http_get(wiki_url, headers=DEFAULT_HEADERS, timeout=10)
        
        if resp.status_code == 200:
            html = resp.text
            title_match = re.search(r'<h1[^>]*id="firstHeading"[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
            title = keyword
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
            
            content = extract_article_content(wiki_url, max_length=5000)
            
            if content and len(content) >= 200:
                return {
                    "title": f"{title} - Wikipedia",
                    "content": content,
                    "source": "Wikipedia",
                    "url": wiki_url,
                    "type": "txt"
                }
            else:
                return {
                    "title": f"{title} - Wikipedia",
                    "content": f"Wikipedia article: {title}\n\nLink: {wiki_url}\n\nOpen the link to read the full entry.",
                    "source": "Wikipedia",
                    "url": wiki_url,
                    "type": "txt"
                }
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
    
    return None


def fetch_google_scholar_results(keyword: str, max_results: int = 5) -> List[Dict]:
    """Fetch google scholar results."""
    results = []
    
    try:
        search_url = f"https://scholar.google.com/scholar?q={quote(keyword)}"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            paper_pattern = r'<div class="gs_ri"[^>]*>(.*?)</div>\s*</div>'
            papers = re.findall(paper_pattern, html, re.DOTALL | re.IGNORECASE)
            
            for paper_html in papers[:max_results * 2]:
                if len(results) >= max_results:
                    break
                
                title_match = re.search(r'<h3[^>]*class="gs_rt"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', paper_html, re.DOTALL | re.IGNORECASE)
                if title_match:
                    url = title_match.group(1)
                    title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
                    
                    abstract_match = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', paper_html, re.DOTALL | re.IGNORECASE)
                    abstract = ""
                    if abstract_match:
                        abstract = re.sub(r'<[^>]+>', '', abstract_match.group(1)).strip()
                    
                    authors_match = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', paper_html, re.DOTALL | re.IGNORECASE)
                    authors = ""
                    if authors_match:
                        authors = re.sub(r'<[^>]+>', '', authors_match.group(1)).strip()
                    
                    if title and url:
                        content = f"Paper title: {title}\n\n"
                        if authors:
                            content += f"Authors/source: {authors}\n\n"
                        if abstract:
                            content += f"Abstract: {abstract}\n\n"
                        content += f"Paper link: {url}\n\nOpen the link to view the full paper."
                        
                        results.append({
                            "title": title,
                            "content": content,
                            "source": "Google Scholar",
                            "url": url,
                            "type": "txt"
                        })
            
            if not results:
                results.append({
                    "title": f"{keyword} - Google Scholar",
                    "content": f"Google Scholar results: {search_url}\n\nOpen the link to view related academic papers.",
                    "source": "Google Scholar",
                    "url": search_url,
                    "type": "txt"
                })
    except Exception as e:
        print(f"Error fetching Google Scholar results: {e}")
        try:
            search_url = f"https://scholar.google.com/scholar?q={quote(keyword)}"
            results.append({
                "title": f"{keyword} - Google Scholar",
                "content": f"Google Scholar search link: {search_url}\n\nOpen the link to view related academic papers.",
                "source": "Google Scholar",
                "url": search_url,
                "type": "txt"
            })
        except:
            pass
    
    return results


def fetch_arxiv_results(keyword: str, max_results: int = 3) -> List[Dict]:
    """Fetch arxiv results."""
    results = []
    
    try:
        search_url = f"http://export.arxiv.org/api/query?search_query=all:{quote(keyword)}&start=0&max_results={max_results}"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            xml_content = resp.text
            
            entries = re.findall(r'<entry>(.*?)</entry>', xml_content, re.DOTALL)
            
            for entry in entries:
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                title = ""
                if title_match:
                    title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                    title = title.replace('\n', ' ').strip()
                
                id_match = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                url = ""
                if id_match:
                    url = id_match.group(1).strip()
                
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                abstract = ""
                if summary_match:
                    abstract = re.sub(r'<[^>]+>', '', summary_match.group(1)).strip()
                    abstract = abstract.replace('\n', ' ').strip()
                
                authors = []
                author_matches = re.findall(r'<name>(.*?)</name>', entry, re.DOTALL)
                for author_match in author_matches:
                    author = re.sub(r'<[^>]+>', '', author_match).strip()
                    if author:
                        authors.append(author)
                
                if title and url:
                    content = f"Paper title: {title}\n\n"
                    if authors:
                        content += f"Authors: {', '.join(authors[:5])}\n\n"  # Show up to 5 authors
                    if abstract:
                        content += f"Abstract: {abstract[:1000]}{'...' if len(abstract) > 1000 else ''}\n\n"
                    content += f"arXiv link: {url}\n\nOpen the link to view the full paper."
                    
                    results.append({
                        "title": title,
                        "content": content,
                        "source": "arXiv",
                        "url": url,
                        "type": "txt"
                    })
    except Exception as e:
        print(f"Error fetching arXiv results: {e}")
        try:
            search_url = f"https://arxiv.org/search/?query={quote(keyword)}&searchtype=all"
            results.append({
                "title": f"{keyword} - arXiv",
                "content": f"arXiv search link: {search_url}\n\nOpen the link to view related preprint papers.",
                "source": "arXiv",
                "url": search_url,
                "type": "txt"
            })
        except:
            pass
    
    return results


def is_irrelevant_url(url: str, title: str = "") -> bool:
    """Return whether irrelevant url."""
    url_lower = url.lower()
    title_lower = title.lower() if title else ""
    combined = f"{url_lower} {title_lower}"
    
    irrelevant_url_patterns = [
        r'raac',  # RAAC report
        r'report[_-]?for[_-]?publishing',  # Report-style PDF
        r'/[^/]+report[^/]*\.pdf',  # PDF report
        r'school[_-]?(closure|building|infrastructure)',  # School building issue
        r'pandemic[_-]?impact',  # Pandemic impact
        r'ministerial[_-]?(visit|report)',  # Ministerial visit report
        r'covid[_-]?19[_-]?(impact|effect)',  # COVID-19 impact
    ]
    
    for pattern in irrelevant_url_patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return True
    
    if re.search(r'\b(report|policy|impact|disruption|building|infrastructure)\b', combined, re.IGNORECASE):
        has_ai_keywords = any(keyword in combined for keyword in AI_RELEVANT_KEYWORDS)
        if not has_ai_keywords:
            return True
    
    return False


def fetch_academic_web_results(keyword: str, max_results: int = 3) -> List[Dict]:
    """Fetch academic web results."""
    return []


def fetch_ddg_instant_answer(keyword: str) -> Dict:
    """Fetch ddg instant answer."""
    return None


def clean_title(title: str) -> str:
    """Clean title."""
    if not title:
        return title
    
    title = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', title)
    
    title = re.sub(r'https?://[^\s]+', '', title, flags=re.IGNORECASE)
    title = re.sub(r'www\.[^\s]+', '', title, flags=re.IGNORECASE)
    
    title = re.sub(r'\b[a-zA-Z0-9.-]+\.(ac\.uk|edu\.|edu\b)\b', '', title, flags=re.IGNORECASE)
    
    title = re.sub(r'^Department\s+of\s+[A-Za-z\s]+$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^Faculty\s+of\s+[A-Za-z\s]+$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^School\s+of\s+[A-Za-z\s]+$', '', title, flags=re.IGNORECASE)
    
    title = re.sub(r'^(Department|Faculty|School|Institute|College)\s+of\s+', '', title, flags=re.IGNORECASE)
    
    title = re.sub(r'\s+', ' ', title)  # Collapse multiple spaces into one
    title = re.sub(r'^\s*[,\-–—]\s*', '', title)  # Remove leading commas, dashes, and similar punctuation
    title = re.sub(r'\s*[,\-–—]\s*$', '', title)  # Remove trailing commas, dashes, and similar punctuation
    title = title.strip()
    
    if len(title) < 5:
        original = title if title else ""
        if not original:
            return "Untitled"
        parts = re.split(r'[—–\-:]', original)
        if parts:
            title = parts[0].strip()
        if len(title) < 5:
            return original[:50] if original else "Untitled"
    
    if len(title) > 100:
        title = title[:97] + "..."
    
    return title


def clean_extracted_content(content: str) -> str:
    """Clean extracted content."""
    if not content:
        return content
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if len(line) < 10:
            continue
        
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line):
            continue
        
        contact_patterns = [
            r'^Contact\s*:',
            r'^Email\s*:',
            r'^Phone\s*:',
            r'^Tel\s*:',
            r'^Fax\s*:',
            r'^Address\s*:',
        ]
        
        is_contact_header = any(re.match(pattern, line, re.IGNORECASE) for pattern in contact_patterns)
        if is_contact_header and len(line) < 150:
            continue
        
        if re.match(r'^\s*@\s*[a-zA-Z0-9.-]+\.(ac\.uk|edu|org|com)\s*$', line, re.IGNORECASE):
            continue
        
        if re.search(r'\b[a-zA-Z0-9.-]+\.(ac\.uk|edu\.|edu\b)', line):
            url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.(ac\.uk|edu\.|edu\b)'
            urls_in_line = re.findall(url_pattern, line)
            
            if urls_in_line and len(line) < 80:
                continue
            
            if urls_in_line and len(line) > 80:
                line_without_urls = re.sub(url_pattern, '', line).strip()
                if len(line_without_urls) > 30:
                    line = line_without_urls
                else:
                    continue
        
        if re.search(r'^(Department|Faculty|School|Institute|College)\s+of\s+[A-Za-z\s]{0,50}$', line, re.IGNORECASE):
            quick_ai_keywords = ['study', 'research', 'paper', 'article', 'method', 'theory', 
                               'algorithm', 'model', 'data', 'analysis', 'learning', 'system', 
                               'course', 'program', 'curriculum', 'ai', 'ml', 'neural', 'deep learning']
            if not any(keyword in line.lower() for keyword in quick_ai_keywords):
                continue
        
        navigation_keywords = [
            'home', 'about', 'contact', 'login', 'register', 'menu', 'search',
            'skip to', 'back to', 'next page', 'previous page', 'breadcrumb'
        ]
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in navigation_keywords) and len(line) < 100:
            continue
        
        if len(line) < 60 and re.search(r'(@|email|phone|tel|fax|address|contact)', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    cleaned_content = re.sub(r'[^\n]{0,100}@[^\n]{0,100}(ac\.uk|edu|org|com)[^\n]{0,100}\n?', '', cleaned_content, flags=re.IGNORECASE)
    
    cleaned_content = re.sub(r'^https?://[^\s]+\s*$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_content = re.sub(r'^www\.[^\s]+\s*$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    
    cleaned_content = re.sub(r'^Department of [A-Za-z\s]{0,50}$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_content = re.sub(r'^Faculty of [A-Za-z\s]{0,50}$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_content = re.sub(r'^School of [A-Za-z\s]{0,50}$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content


def extract_article_content(url: str, max_length: int = 3000) -> str:
    """Extract article content."""
    try:
        resp = http_get(url, headers=DEFAULT_HEADERS, timeout=12)
        if resp.status_code == 200:
            html = resp.text
            
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            html = re.sub(r'<div[^>]*class="[^"]*(?:nav|menu|sidebar|ad|advertisement|cookie|banner)[^"]*"[^>]*>.*?</div>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            content_patterns = [
                (r'<article[^>]*>(.*?)</article>', 'article'),
                (r'<main[^>]*>(.*?)</main>', 'main'),
                (r'<div[^>]*id="[^"]*content[^"]*"[^>]*>(.*?)</div>', 'content-id'),
                (r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>', 'content-class'),
                (r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>', 'article-class'),
                (r'<div[^>]*class="[^"]*post[^"]*"[^>]*>(.*?)</div>', 'post'),
                (r'<div[^>]*class="[^"]*entry[^"]*"[^>]*>(.*?)</div>', 'entry'),
                (r'<section[^>]*>(.*?)</section>', 'section'),
            ]
            
            extracted_text = ""
            best_text = ""
            
            for pattern, name in content_patterns:
                matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
                if matches:
                    text = " ".join(matches)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'&[a-z]+;', ' ', text)  # HTML entity
                    text = re.sub(r'&#\d+;', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 200 and not re.search(r'^(home|about|contact|login|register|menu|search)', text[:50], re.IGNORECASE):
                        if len(text) > len(extracted_text):
                            extracted_text = text
                            best_text = text
            
            if not extracted_text or len(extracted_text) < 200:
                p_matches = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
                if p_matches:
                    paragraphs = []
                    for p in p_matches:
                        text = re.sub(r'<[^>]+>', ' ', p)
                        text = re.sub(r'&[a-z]+;', ' ', text)
                        text = re.sub(r'&#\d+;', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if len(text) > 50:
                            paragraphs.append(text)
                    
                    if paragraphs:
                        extracted_text = " ".join(paragraphs)
            
            if not extracted_text or len(extracted_text) < 200:
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
                if body_match:
                    body_html = body_match.group(1)
                    body_html = re.sub(r'<nav[^>]*>.*?</nav>', '', body_html, flags=re.DOTALL | re.IGNORECASE)
                    body_html = re.sub(r'<header[^>]*>.*?</header>', '', body_html, flags=re.DOTALL | re.IGNORECASE)
                    body_html = re.sub(r'<footer[^>]*>.*?</footer>', '', body_html, flags=re.DOTALL | re.IGNORECASE)
                    
                    text = re.sub(r'<[^>]+>', ' ', body_html)
                    text = re.sub(r'&[a-z]+;', ' ', text)
                    text = re.sub(r'&#\d+;', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    extracted_text = text
            
            if extracted_text:
                noise_patterns = [
                    r'cookie\s+policy',
                    r'privacy\s+policy',
                    r'terms\s+of\s+service',
                    r'skip\s+to\s+content',
                    r'menu',
                    r'search',
                    r'login',
                    r'register',
                ]
                
                sentences = re.split(r'[.!?]\s+', extracted_text)
                filtered_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 20:
                        continue
                    is_noise = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in noise_patterns)
                    if not is_noise:
                        filtered_sentences.append(sentence)
                
                if filtered_sentences:
                    extracted_text = ". ".join(filtered_sentences)
                else:
                    pass
            
            if not extracted_text or len(extracted_text) < 100:
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'&[a-z]+;', ' ', text)
                text = re.sub(r'&#\d+;', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                extracted_text = text
            
            if len(extracted_text) > max_length:
                truncated = extracted_text[:max_length]
                last_period = truncated.rfind('.')
                if last_period > max_length * 0.8:  # If the last sentence starts after the 80% mark
                    extracted_text = truncated[:last_period + 1]
                else:
                    extracted_text = truncated + "..."
            
            extracted_text = extracted_text.strip()
            
            cleaned_text = clean_extracted_content(extracted_text)
            
            if cleaned_text and len(cleaned_text) >= 50:
                url_count = len(re.findall(r'https?://', cleaned_text))
                if url_count < len(cleaned_text) / 20:  # Keep link count below 5% of the text length
                    return cleaned_text
            
    except requests.exceptions.Timeout:
        print(f"Timeout extracting content from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Request error extracting content from {url}: {e}")
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    
    return ""


def fetch_ddg_web_results(keyword: str, max_results: int = 10, academic_only: bool = False) -> List[Dict]:
    """Fetch ddg web results."""
    return []


def fetch_google_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """Fetch google images."""
    results = []
    
    try:
        search_url = f"https://www.google.com/search?tbm=isch&q={quote(keyword)}&safe=images"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            json_pattern = r'AF_initDataCallback\([^)]+\)'
            json_matches = re.findall(json_pattern, html)
            
            img_patterns = [
                r'"ou":"([^"]+)"',  # Google Images raw URL field (most reliable)
                r'"ow":\d+,"oh":\d+,"ou":"([^"]+)"',  # URL with size metadata
                r'data-src="(https://[^"]+\.(jpg|jpeg|png|gif|webp))"',
                r'src="(https://encrypted-tbn[^"]+)"',  # Google thumbnail URL (usable as a fallback)
                r'https://[^"\s]+\.(jpg|jpeg|png|gif|webp|svg)',  # Direct image URL match
            ]
            
            seen_urls = set()
            for pattern in img_patterns:
                if len(results) >= max_results:
                    break
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if len(results) >= max_results:
                        break
                    img_url = match[0] if isinstance(match, tuple) else match
                    
                    if img_url and img_url.startswith('http') and img_url not in seen_urls:
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/search', '/webhp']):
                            continue
                        if any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', 'image', 'img', 'photo', 'picture']):
                            seen_urls.add(img_url)
                            results.append({
                                "title": f"Image: {keyword}",
                                "url": img_url,
                                "thumbnail": img_url,
                                "source": "Google Images",
                                "type": "image"
                            })
    except Exception as e:
        print(f"Error fetching Google Images: {e}")
    
    if not results:
        results.append({
            "title": f"Google image search: {keyword}",
            "url": f"https://www.google.com/search?tbm=isch&q={quote(keyword)}",
            "thumbnail": "",
            "source": "Google Images",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_bing_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """Fetch bing images."""
    results = []
    
    try:
        search_url = f"https://www.bing.com/images/search?q={quote(keyword)}&safe=strict"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            img_patterns = [
                r'"murl":"([^"]+)"',  # Bing Images media URL field (most reliable)
                r'"thumb":"([^"]+)"',  # Thumbnail URL
                r'data-src="(https://[^"]+\.(jpg|jpeg|png|gif|webp))"',
                r'src="(https://[^"]+th\.bing\.com[^"]+)"',  # Bing thumbnail server
                r'https://[^"\s]+\.(jpg|jpeg|png|gif|webp|svg)',  # Direct image URL match
            ]
            
            seen_urls = set()
            for pattern in img_patterns:
                if len(results) >= max_results:
                    break
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if len(results) >= max_results:
                        break
                    img_url = match[0] if isinstance(match, tuple) else match
                    
                    if img_url and img_url.startswith('http') and img_url not in seen_urls:
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/search']):
                            continue
                        if any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', 'image', 'img', 'photo', 'picture', 'th.bing.com']):
                            seen_urls.add(img_url)
                            results.append({
                                "title": f"Image: {keyword}",
                                "url": img_url,
                                "thumbnail": img_url,
                                "source": "Bing Images",
                                "type": "image"
                            })
    except Exception as e:
        print(f"Error fetching Bing Images: {e}")
    
    if not results:
        results.append({
            "title": f"Bing image search: {keyword}",
            "url": f"https://www.bing.com/images/search?q={quote(keyword)}",
            "thumbnail": "",
            "source": "Bing Images",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_unsplash_images(keyword: str, max_results: int = 5) -> List[Dict]:
    """Fetch unsplash images."""
    results = []
    
    try:
        search_url = f"https://unsplash.com/s/photos/{quote(keyword)}"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            img_patterns = [
                r'"raw":"([^"]+)"',  # Unsplash raw image URL (most reliable)
                r'"regular":"([^"]+)"',  # Unsplash regular size
                r'"small":"([^"]+)"',  # Small size
                r'src="(https://images\.unsplash\.com[^"]+)"',
                r'data-src="(https://images\.unsplash\.com[^"]+)"',
                r'url\(["\']?(https://images\.unsplash\.com[^"\']+)["\']?\)',
            ]
            
            seen_urls = set()
            for pattern in img_patterns:
                if len(results) >= max_results:
                    break
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if len(results) >= max_results:
                        break
                    img_url = match[0] if isinstance(match, tuple) else match
                    
                    if img_url and img_url.startswith('http') and img_url not in seen_urls:
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/s/photos']):
                            continue
                        seen_urls.add(img_url)
                        results.append({
                            "title": f"Unsplash image: {keyword}",
                            "url": img_url,
                            "thumbnail": img_url,
                            "source": "Unsplash",
                            "type": "image"
                        })
    except Exception as e:
        print(f"Error fetching Unsplash images: {e}")
    
    if not results:
        results.append({
            "title": f"Unsplash image search: {keyword}",
            "url": f"https://unsplash.com/s/photos/{quote(keyword)}",
            "thumbnail": "",
            "source": "Unsplash",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_pexels_images(keyword: str, max_results: int = 5) -> List[Dict]:
    """Fetch pexels images."""
    results = []
    
    try:
        search_url = f"https://www.pexels.com/search/{quote(keyword)}/"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            img_patterns = [
                r'data-bg="([^"]+)"',  # Pexels background image
                r'data-src="(https://images\.pexels\.com[^"]+)"',  # Lazy-loaded image
                r'src="(https://images\.pexels\.com[^"]+)"',  # Direct src
                r'url\(["\']?(https://images\.pexels\.com[^"\']+)["\']?\)',  # CSS background image
                r'"original":"([^"]+)"',  # Pexels raw image URL
                r'"large":"([^"]+)"',  # Large image
            ]
            
            seen_urls = set()
            for pattern in img_patterns:
                if len(results) >= max_results:
                    break
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if len(results) >= max_results:
                        break
                    img_url = match[0] if isinstance(match, tuple) else match
                    
                    if img_url and img_url.startswith('http') and img_url not in seen_urls:
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/search']):
                            continue
                        seen_urls.add(img_url)
                        results.append({
                            "title": f"Pexels image: {keyword}",
                            "url": img_url,
                            "thumbnail": img_url,
                            "source": "Pexels",
                            "type": "image"
                        })
    except Exception as e:
        print(f"Error fetching Pexels images: {e}")
    
    if not results:
        results.append({
            "title": f"Pexels image search: {keyword}",
            "url": f"https://www.pexels.com/search/{quote(keyword)}/",
            "thumbnail": "",
            "source": "Pexels",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_generic_image_links(keyword: str, max_results: int = 3) -> List[Dict]:
    """Fetch generic image links."""
    results = []
    
    image_search_sources = [
        {
            "title": f"Google image search: {keyword}",
            "url": f"https://www.google.com/search?tbm=isch&q={quote(keyword)}",
            "source": "Google Images"
        },
        {
            "title": f"Bing image search: {keyword}",
            "url": f"https://www.bing.com/images/search?q={quote(keyword)}",
            "source": "Bing Images"
        },
        {
            "title": f"Unsplash library: {keyword}",
            "url": f"https://unsplash.com/s/photos/{quote(keyword)}",
            "source": "Unsplash"
        },
        {
            "title": f"Pexels library: {keyword}",
            "url": f"https://www.pexels.com/search/{quote(keyword)}/",
            "source": "Pexels"
        },
    ]
    
    for source in image_search_sources[:max_results]:
        results.append({
            "title": source["title"],
            "url": source["url"],
            "thumbnail": "",
            "source": source["source"],
            "type": "image"
        })
    
    return results


def fetch_ddg_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """Fetch ddg images."""
    return []





def extract_youtube_videos_from_json(data: dict, max_results: int) -> List[Dict]:
    """Extract youtube videos from json."""
    videos = []
    
    try:
        def find_videos(obj, depth=0):
            if depth > 10:  # Prevent infinite recursion
                return
            
            if isinstance(obj, dict):
                if "videoRenderer" in obj:
                    renderer = obj["videoRenderer"]
                    video_id = renderer.get("videoId", "")
                    title_obj = renderer.get("title", {})
                    title = title_obj.get("runs", [{}])[0].get("text", "") if isinstance(title_obj.get("runs"), list) else ""
                    snippet = renderer.get("thumbnail", {})
                    thumbnails = snippet.get("thumbnails", [])
                    thumbnail = thumbnails[0].get("url", "") if thumbnails else ""
                    
                    if video_id and title:
                        videos.append({
                            "title": title,
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "description": f"YouTube video: {title}",
                            "video_id": video_id,
                            "thumbnail": thumbnail,
                            "type": "video"
                        })
                
                for value in obj.values():
                    find_videos(value, depth + 1)
            
            elif isinstance(obj, list):
                for item in obj:
                    find_videos(item, depth + 1)
        
        find_videos(data)
    except Exception as e:
        print(f"Error extracting YouTube videos from JSON: {e}")
    
    return videos[:max_results]


def extract_youtube_videos_from_html(html: str, max_results: int) -> List[Dict]:
    """Extract youtube videos from html."""
    videos = []
    
    try:
        pattern = r'/watch\?v=([a-zA-Z0-9_-]{11})'
        video_ids = list(set(re.findall(pattern, html)))[:max_results]
        
        for video_id in video_ids:
            videos.append({
                "title": f"YouTube video {video_id}",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": f"YouTube video ID: {video_id}",
                "video_id": video_id,
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/default.jpg",
                "type": "video"
            })
    except Exception as e:
        print(f"Error extracting YouTube videos from HTML: {e}")
    
    return videos



def search_all_resources(keywords: List[str], max_per_type: int = 10, progress_callback=None) -> Dict[str, List[Dict]]:
    """Search all resources."""
    all_txt = []
    all_video = []
    all_code = []
    
    for i, keyword in enumerate(keywords):
        debug_print(f"Searching keyword {i+1}/{len(keywords)}: {keyword}")
        
        if progress_callback:
            progress_callback({
                "type": "keyword_start",
                "keyword": keyword,
                "index": i + 1,
                "total": len(keywords)
            })
        
        try:
            txt_results = search_text_resources(keyword, max_per_type * 2)
            all_txt.extend(txt_results)
            debug_print(f"  Found {len(txt_results)} text resources")
            
            if progress_callback:
                progress_callback({
                    "type": "keyword_text_done",
                    "keyword": keyword,
                    "count": len(txt_results)
                })
        except Exception as e:
            print(f"  Text search error: {e}")
        
        try:
            video_results = search_youtube_videos(keyword, max_per_type)
            all_video.extend(video_results)
            debug_print(f"  Found {len(video_results)} video resources")
            
            if progress_callback:
                progress_callback({
                    "type": "keyword_video_done",
                    "keyword": keyword,
                    "count": len(video_results)
                })
        except Exception as e:
            print(f"  Video search error: {e}")
        
        try:
            code_results = search_code_resources(keyword, max_per_type)
            all_code.extend(code_results)
            debug_print(f"  Found {len(code_results)} code resources")
            
            if progress_callback:
                progress_callback({
                    "type": "keyword_code_done",
                    "keyword": keyword,
                    "count": len(code_results)
                })
        except Exception as e:
            print(f"  Code search error: {e}")
        
        if progress_callback:
            progress_callback({
                "type": "keyword_done",
                "keyword": keyword,
                "index": i + 1,
                "total": len(keywords)
            })
        
        if i < len(keywords) - 1:
            time.sleep(random.uniform(0.5, 1.5))
    
    seen_txt_urls = set()
    unique_txt = []
    for txt in all_txt:
        url = txt.get("url", "")
        if url and url not in seen_txt_urls:
            seen_txt_urls.add(url)
            unique_txt.append(txt)
    
    seen_video_ids = set()
    seen_video_urls = set()
    unique_video = []
    for video in all_video:
        video_id = video.get("video_id")
        url = video.get("url", "")
        
        if video_id and video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            seen_video_urls.add(url)
            unique_video.append(video)
        elif url and url not in seen_video_urls:
            seen_video_urls.add(url)
            unique_video.append(video)
    
    seen_code_urls = set()
    unique_code = []
    for code in all_code:
        url = code.get("url", "")
        if url and url not in seen_code_urls:
            seen_code_urls.add(url)
            unique_code.append(code)
    
    
    debug_print(f"After deduplication: text {len(all_txt)} -> {len(unique_txt)}, video {len(all_video)} -> {len(unique_video)}, code {len(all_code)} -> {len(unique_code)}")
    
    return {
        "txt": unique_txt,
        "video": unique_video,
        "code": unique_code
    }
