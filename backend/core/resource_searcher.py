#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用外部资源搜索模块
不依赖特定API，通过通用搜索和网页爬取获取资源
支持搜索：文本、视频、图片、代码
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

# AI领域的核心关键词列表（用于相关性判断和内容过滤）
AI_RELEVANT_KEYWORDS = [
    # 核心概念
    'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 'dl',
    'neural network', 'neural networks', 'neural net', 'nn',
    
    # 算法和方法
    'algorithm', 'model', 'training', 'inference', 'prediction', 'classification', 
    'regression', 'clustering', 'supervised learning', 'unsupervised learning', 
    'reinforcement learning', 'semi-supervised', 'transfer learning', 'meta learning',
    'few-shot learning', 'zero-shot learning',
    
    # 深度学习架构
    'convolutional neural network', 'cnn', 'recurrent neural network', 'rnn', 
    'lstm', 'gru', 'transformer', 'attention mechanism', 'self-attention',
    'generative adversarial network', 'gan', 'variational autoencoder', 'vae',
    'autoencoder', 'encoder-decoder', 'seq2seq', 'bert', 'gpt', 'large language model', 'llm',
    
    # 计算机视觉
    'computer vision', 'cv', 'image recognition', 'object detection', 'semantic segmentation',
    'image classification', 'face recognition', 'optical character recognition', 'ocr',
    'image generation', 'image processing',
    
    # 自然语言处理
    'natural language processing', 'nlp', 'text classification', 'sentiment analysis',
    'named entity recognition', 'ner', 'machine translation', 'text generation',
    'language model', 'word embedding', 'tokenization', 'text mining',
    
    # 技术术语
    'gradient descent', 'backpropagation', 'activation function', 'loss function',
    'optimization', 'regularization', 'dropout', 'batch normalization',
    'overfitting', 'underfitting', 'cross-validation', 'hyperparameter tuning',
    'feature engineering', 'feature extraction', 'feature selection',
    
    # 数据处理
    'dataset', 'training data', 'test data', 'validation data', 'data preprocessing',
    'data augmentation', 'feature scaling', 'normalization',
    
    # 应用领域
    'recommendation system', 'recommender system', 'speech recognition', 'speech synthesis',
    'reinforcement learning', 'rl', 'autonomous driving', 'robotics',
    'knowledge graph', 'knowledge representation', 'reasoning',
    
    # 学术相关
    'research paper', 'academic', 'arxiv', 'conference', 'journal', 'publication',
    'thesis', 'dissertation', 'tutorial', 'survey',
    
    # 缩写和常用术语
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
    'jupyter', 'notebook', 'python', 'tensor', 'gpu', 'cuda',
]

# 通用User-Agent
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",  # 优先请求英文内容
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def _build_http_session() -> requests.Session:
    """构建带连接池和轻量重试的HTTP会话，减少重复握手开销。"""
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
    """统一的GET入口，复用连接并保持原有headers/timeout行为。"""
    return HTTP_SESSION.get(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)


def is_english_content(text: str, min_english_ratio: float = 0.7) -> bool:
    """
    检测文本内容是否主要是英文
    使用启发式方法：检测非拉丁字符的比例
    
    Args:
        text: 要检测的文本
        min_english_ratio: 英文字符的最小比例（默认70%）
    
    Returns:
        True 如果主要是英文，False 否则
    """
    if not text or len(text.strip()) < 10:
        return False
    
    # 统计字符类型
    total_chars = 0
    english_chars = 0
    non_latin_chars = 0
    
    for char in text:
        if char.isspace():
            continue
        
        total_chars += 1
        char_code = ord(char)
        
        # 英文字母、数字、基本标点符号（ASCII范围）
        if char_code < 128:
            english_chars += 1
        # 常见的非拉丁字符范围（印度语、中文、阿拉伯语等）
        elif (
            (0x0900 <= char_code <= 0x097F) or  # 天城文（印地语、梵语等）
            (0x0980 <= char_code <= 0x09FF) or  # 孟加拉语
            (0x0A00 <= char_code <= 0x0A7F) or  # 古木基文（旁遮普语）
            (0x0A80 <= char_code <= 0x0AFF) or  # 古吉拉特语
            (0x0B00 <= char_code <= 0x0B7F) or  # 奥里亚语
            (0x0B80 <= char_code <= 0x0BFF) or  # 泰米尔语
            (0x0C00 <= char_code <= 0x0C7F) or  # 泰卢固语
            (0x0C80 <= char_code <= 0x0CFF) or  # 卡纳达语
            (0x0D00 <= char_code <= 0x0D7F) or  # 马拉雅拉姆语
            (0x4E00 <= char_code <= 0x9FFF) or  # 中日韩统一表意文字（中文、日文、韩文）
            (0x0600 <= char_code <= 0x06FF) or  # 阿拉伯语
            (0x0590 <= char_code <= 0x05FF) or  # 希伯来语
            (0x0400 <= char_code <= 0x04FF)     # 西里尔字母（俄语等）
        ):
            non_latin_chars += 1
    
    if total_chars == 0:
        return False
    
    # 计算英文比例
    english_ratio = english_chars / total_chars if total_chars > 0 else 0
    non_latin_ratio = non_latin_chars / total_chars if total_chars > 0 else 0
    
    # 如果非拉丁字符比例超过30%，判定为非英文
    if non_latin_ratio > 0.3:
        return False
    
    # 如果英文字符比例达到阈值，判定为英文
    return english_ratio >= min_english_ratio


def filter_english_content(resources: List[Dict], content_key: str = "content") -> List[Dict]:
    """
    过滤资源列表，只保留英文内容
    
    Args:
        resources: 资源列表
        content_key: 内容字段的键名（可能是 "content", "title", "description" 等）
    
    Returns:
        过滤后的资源列表（只包含英文内容）
    """
    english_resources = []
    
    for res in resources:
        # 检查多个字段（title, content, description）
        texts_to_check = []
        if "title" in res:
            texts_to_check.append(res["title"])
        if content_key in res:
            texts_to_check.append(res[content_key])
        if "description" in res:
            texts_to_check.append(res["description"])
        
        # 合并所有文本进行检查
        combined_text = " ".join(str(t) for t in texts_to_check if t)
        
        # 如果主要是英文，保留该资源
        if is_english_content(combined_text, min_english_ratio=0.6):
            english_resources.append(res)
        else:
            # 记录过滤掉的非英文资源（用于调试）
            print(f"  过滤非英文内容: {res.get('title', 'Unknown')[:50]}")
    
    return english_resources


def search_text_resources(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    学术文本资源搜索
    优先搜索学术文章、论文、详细介绍等高质量资源
    返回: List[Dict] 每个包含 {title, content, source, url}
    """
    results = []
    
    # 构建学术搜索查询（添加学术相关关键词）
    academic_queries = [
        f"{keyword} research paper",
        f"{keyword} academic article",
        f"{keyword} scholarly",
        f"{keyword}",
    ]
    
    # 1. 优先搜索Wikipedia（详细介绍）
    try:
        wiki_result = fetch_wikipedia_article(keyword)
        if wiki_result:
            results.append(wiki_result)
            print(f"  找到Wikipedia文章")
    except Exception as e:
        print(f"Wikipedia search error for {keyword}: {e}")
    
    # 2. 搜索Google Scholar（学术论文）
    try:
        scholar_results = fetch_google_scholar_results(keyword, max_results=5)
        results.extend(scholar_results)
        print(f"  Google Scholar找到 {len(scholar_results)} 个结果")
    except Exception as e:
        print(f"Google Scholar search error for {keyword}: {e}")
    
    # 3. 搜索arXiv（预印本论文）
    try:
        arxiv_results = fetch_arxiv_results(keyword, max_results=3)
        results.extend(arxiv_results)
        print(f"  arXiv找到 {len(arxiv_results)} 个结果")
    except Exception as e:
        print(f"arXiv search error for {keyword}: {e}")
    
    # 4. 学术网站搜索已移除（不再使用DuckDuckGo）
    
    # 去重（基于URL）
    seen_urls = set()
    unique_results = []
    for res in results:
        url = res.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)
    
    # 按来源优先级排序（Wikipedia > Scholar > arXiv > 其他学术网站 > 其他）
    source_priority = {
        "Wikipedia": 1,
        "Google Scholar": 2,
        "arXiv": 3,
        "Academic Website": 4,
        "Web Article": 5,
        "Web Link": 6
    }
    unique_results.sort(key=lambda x: source_priority.get(x.get("source", ""), 99))
    
    # 过滤非英文内容
    english_results = filter_english_content(unique_results, content_key="content")
    if len(english_results) < max_results:
        print(f"  英文内容过滤: {len(unique_results)} -> {len(english_results)} 个结果")
    
    return english_results[:max_results]


def search_youtube_videos(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    搜索YouTube视频（通过HTML爬取，不需要API）
    返回: List[Dict] 每个包含 {title, url, description, video_id}
    """
    results = []
    
    try:
        # 使用YouTube搜索页面（添加语言参数限制为英文）
        search_url = f"https://www.youtube.com/results?search_query={quote(keyword)}&sp=EgIoAQ%253D%253D"  # 添加英文内容过滤器
        headers = DEFAULT_HEADERS.copy()
        
        resp = http_get(search_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            # 从HTML中提取视频信息
            html = resp.text
            
            # YouTube在页面中嵌入了JSON数据
            # 查找 var ytInitialData
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
            
            # 如果JSON解析失败，尝试从HTML中直接提取
            if not results:
                videos = extract_youtube_videos_from_html(html, max_results)
                results.extend(videos)
        
        # 如果仍然没有结果，至少返回搜索链接
        if not results:
            results.append({
                "title": f"YouTube搜索结果: {keyword}",
                "url": search_url,
                "description": f"点击查看YouTube上关于'{keyword}'的视频",
                "video_id": None,
                "thumbnail": "",
                "type": "video"
            })
    except Exception as e:
        print(f"YouTube search error for {keyword}: {e}")
        # Fallback: 返回搜索链接
        results.append({
            "title": f"YouTube搜索结果: {keyword}",
            "url": f"https://www.youtube.com/results?search_query={quote(keyword)}&sp=EgIoAQ%253D%253D",
            "description": f"点击查看YouTube上关于'{keyword}'的视频",
            "video_id": None,
            "thumbnail": "",
            "type": "video"
        })
    
    # 过滤非英文内容（视频主要检查标题和描述）
    english_results = filter_english_content(results, content_key="description")
    if len(english_results) < len(results):
        print(f"  英文内容过滤: {len(results)} -> {len(english_results)} 个结果")
    
    return english_results[:max_results]


def search_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    多源图片搜索，从多个来源搜索相关图片
    返回: List[Dict] 每个包含 {title, url, thumbnail, source}
    """
    results = []
    
    # 1. Google Images搜索
    try:
        google_images = fetch_google_images(keyword, max_results=max_results // 3)
        results.extend(google_images)
        print(f"  Google Images找到 {len(google_images)} 个结果")
    except Exception as e:
        print(f"Google Images search error for {keyword}: {e}")
    
    # 2. Bing Images搜索
    try:
        bing_images = fetch_bing_images(keyword, max_results=max_results // 3)
        results.extend(bing_images)
        print(f"  Bing Images找到 {len(bing_images)} 个结果")
    except Exception as e:
        print(f"Bing Images search error for {keyword}: {e}")
    
    # 3. Unsplash免费图片库（高质量）
    try:
        unsplash_images = fetch_unsplash_images(keyword, max_results=max_results // 4)
        results.extend(unsplash_images)
        print(f"  Unsplash找到 {len(unsplash_images)} 个结果")
    except Exception as e:
        print(f"Unsplash search error for {keyword}: {e}")
    
    # 4. Pexels免费图片库（高质量）
    try:
        pexels_images = fetch_pexels_images(keyword, max_results=max_results // 4)
        results.extend(pexels_images)
        print(f"  Pexels找到 {len(pexels_images)} 个结果")
    except Exception as e:
        print(f"Pexels search error for {keyword}: {e}")
    
    # 5. 如果结果不足，添加通用图片搜索链接
    if len(results) < 3:
        try:
            generic_images = fetch_generic_image_links(keyword, max_results=3)
            results.extend(generic_images)
            print(f"  通用图片搜索找到 {len(generic_images)} 个结果")
        except Exception as e:
            print(f"Generic image search error for {keyword}: {e}")
    
    # 去重（基于URL）
    seen_urls = set()
    unique_results = []
    for res in results:
        url = res.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)
    
    # 过滤非英文内容（图片主要检查标题）
    english_results = filter_english_content(unique_results, content_key="title")
    if len(english_results) < len(unique_results):
        print(f"  英文内容过滤: {len(unique_results)} -> {len(english_results)} 个结果")
    
    return english_results[:max_results]


def search_code_resources(keyword: str, max_results: int = 10) -> List[Dict]:
    """
    代码搜索，从GitHub搜索相关代码资源
    返回: List[Dict] 每个包含 {title, url, description, source, type}
    """
    results = []
    
    # GitHub代码搜索
    try:
        github_results = fetch_github_code(keyword, max_results=max_results)
        results.extend(github_results)
        print(f"  GitHub找到 {len(github_results)} 个代码资源")
    except Exception as e:
        print(f"GitHub search error for {keyword}: {e}")
    
    # 去重（基于URL）
    seen_urls = set()
    unique_results = []
    for res in results:
        url = res.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)
    
    # 过滤非英文内容（代码主要检查标题和描述）
    english_results = filter_english_content(unique_results, content_key="description")
    if len(english_results) < len(unique_results):
        print(f"  英文内容过滤: {len(unique_results)} -> {len(english_results)} 个结果")
    
    return english_results[:max_results]


# ====== 代码搜索相关函数 ======

def fetch_github_code(keyword: str, max_results: int = 10) -> List[Dict]:
    """从GitHub搜索代码仓库，返回具体的仓库链接（不是搜索页面）"""
    results = []
    
    try:
        # GitHub搜索URL（搜索代码仓库，添加AI相关关键词以提高相关性）
        search_query = f"{keyword} machine-learning OR deep-learning OR pytorch OR tensorflow"
        search_url = f"https://github.com/search?q={quote(search_query)}&type=repositories&s=stars&o=desc"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            # 方法1: 尝试从嵌入的JSON数据中提取（GitHub在页面中嵌入JSON）
            try:
                # 查找包含仓库信息的JSON数据
                json_pattern = r'application/json[^>]*>([^<]+)'
                json_matches = re.findall(json_pattern, html)
                for json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        # 递归搜索JSON中的仓库链接
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
            
            # 方法2: 从HTML中直接提取仓库链接（使用更精确的模式）
            # GitHub搜索结果中的仓库链接格式：href="/username/repo-name"
            # 需要确保是真正的仓库链接，不是其他链接
            repo_link_pattern = r'href="/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"[^>]*>'
            all_matches = re.findall(repo_link_pattern, html)
            
            seen_urls = set()
            for repo_path in all_matches:
                if len(results) >= max_results:
                    break
                
                # 验证是有效的仓库路径（格式：username/repo-name，排除特殊路径）
                if '/' in repo_path and len(repo_path.split('/')) == 2:
                    # 排除非仓库路径
                    excluded_paths = ['search', 'explore', 'trending', 'topics', 'collections', 
                                      'settings', 'login', 'signup', 'join', 'pricing', 'enterprise',
                                      'features', 'security', 'marketplace', 'sponsors', 'about']
                    if any(excluded in repo_path.lower() for excluded in excluded_paths):
                        continue
                    
                    repo_url = f"https://github.com/{repo_path}"
                    
                    if repo_url not in seen_urls:
                        seen_urls.add(repo_url)
                        
                        # 提取仓库名称
                        repo_name = repo_path.split('/')[-1]
                        
                        # 检查是否包含AI相关关键词（提高相关性）
                        repo_text = f"{repo_name} {repo_path}".lower()
                        has_ai_keywords = any(kw in repo_text for kw in ['machine', 'learning', 'deep', 'neural', 'ai', 'ml', 'dl', 'pytorch', 'tensorflow', 'keras', 'scikit', 'transformer', 'cnn', 'rnn', 'lstm', 'gan', 'bert', 'gpt'])
                        
                        # 如果包含AI关键词或者是原始关键词匹配，添加
                        if has_ai_keywords or keyword.lower() in repo_text:
                            # 尝试获取仓库描述（从HTML中提取）
                            description = f"GitHub代码仓库: {repo_name}"
                            
                            results.append({
                                "title": repo_name,
                                "url": repo_url,
                                "description": description,
                                "source": "GitHub",
                                "type": "code"
                            })
            
            # 方法3: 从搜索结果区域的特定HTML结构中提取
            # GitHub搜索结果通常在 <div class="repo-list-item"> 或类似结构中
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
                                "description": f"GitHub代码仓库: {repo_name}",
                                "source": "GitHub",
                                "type": "code"
                            })
            
            # 去重
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
    
    # 重要：如果无法提取实际仓库，返回空列表，而不是搜索链接
    # 这样用户就知道没有找到具体的仓库，而不是得到一个无用的搜索链接
    return results[:max_results]


def extract_repos_from_json(data, max_results: int = 10) -> List[Dict]:
    """从GitHub的JSON数据中递归提取仓库信息"""
    results = []
    
    def traverse(obj, depth=0):
        if depth > 10:  # 防止无限递归
            return
        if len(results) >= max_results:
            return
        
        if isinstance(obj, dict):
            # 检查是否包含仓库信息
            if 'full_name' in obj and 'html_url' in obj:
                repo_name = obj.get('full_name', '').split('/')[-1]
                repo_url = obj.get('html_url', '')
                if repo_url and 'github.com' in repo_url and '/search' not in repo_url:
                    results.append({
                        "title": repo_name,
                        "url": repo_url,
                        "description": obj.get('description', f"GitHub代码仓库: {repo_name}"),
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


# ====== DuckDuckGo相关函数 ======

def fetch_wikipedia_article(keyword: str) -> Dict:
    """
    从Wikipedia获取文章内容
    返回详细的介绍性文章
    """
    try:
        # 尝试英文Wikipedia
        wiki_url = f"https://en.wikipedia.org/wiki/{quote(keyword.replace(' ', '_'))}"
        resp = http_get(wiki_url, headers=DEFAULT_HEADERS, timeout=10)
        
        if resp.status_code == 200:
            html = resp.text
            # 提取文章标题
            title_match = re.search(r'<h1[^>]*id="firstHeading"[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
            title = keyword
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
            
            # 提取文章内容
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
                # 即使无法提取内容，也返回Wikipedia链接
                return {
                    "title": f"{title} - Wikipedia",
                    "content": f"Wikipedia文章: {title}\n\n链接: {wiki_url}\n\n请访问链接查看完整的详细介绍。",
                    "source": "Wikipedia",
                    "url": wiki_url,
                    "type": "txt"
                }
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
    
    return None


def fetch_google_scholar_results(keyword: str, max_results: int = 5) -> List[Dict]:
    """
    从Google Scholar搜索学术论文
    返回学术论文链接和摘要
    """
    results = []
    
    try:
        # Google Scholar搜索URL
        search_url = f"https://scholar.google.com/scholar?q={quote(keyword)}"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            # 提取论文结果
            # Google Scholar的结果通常在 <div class="gs_ri"> 中
            paper_pattern = r'<div class="gs_ri"[^>]*>(.*?)</div>\s*</div>'
            papers = re.findall(paper_pattern, html, re.DOTALL | re.IGNORECASE)
            
            for paper_html in papers[:max_results * 2]:
                if len(results) >= max_results:
                    break
                
                # 提取标题和链接
                title_match = re.search(r'<h3[^>]*class="gs_rt"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', paper_html, re.DOTALL | re.IGNORECASE)
                if title_match:
                    url = title_match.group(1)
                    title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
                    
                    # 提取摘要
                    abstract_match = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', paper_html, re.DOTALL | re.IGNORECASE)
                    abstract = ""
                    if abstract_match:
                        abstract = re.sub(r'<[^>]+>', '', abstract_match.group(1)).strip()
                    
                    # 提取作者和来源信息
                    authors_match = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', paper_html, re.DOTALL | re.IGNORECASE)
                    authors = ""
                    if authors_match:
                        authors = re.sub(r'<[^>]+>', '', authors_match.group(1)).strip()
                    
                    if title and url:
                        content = f"论文标题: {title}\n\n"
                        if authors:
                            content += f"作者/来源: {authors}\n\n"
                        if abstract:
                            content += f"摘要: {abstract}\n\n"
                        content += f"论文链接: {url}\n\n请访问链接查看完整论文。"
                        
                        results.append({
                            "title": title,
                            "content": content,
                            "source": "Google Scholar",
                            "url": url,
                            "type": "txt"
                        })
            
            # 如果正则提取失败，至少返回搜索链接
            if not results:
                results.append({
                    "title": f"{keyword} - Google Scholar",
                    "content": f"Google Scholar搜索结果: {search_url}\n\n请访问链接查看相关学术论文。",
                    "source": "Google Scholar",
                    "url": search_url,
                    "type": "txt"
                })
    except Exception as e:
        print(f"Error fetching Google Scholar results: {e}")
        # 即使失败也返回搜索链接
        try:
            search_url = f"https://scholar.google.com/scholar?q={quote(keyword)}"
            results.append({
                "title": f"{keyword} - Google Scholar",
                "content": f"Google Scholar搜索链接: {search_url}\n\n请访问链接查看相关学术论文。",
                "source": "Google Scholar",
                "url": search_url,
                "type": "txt"
            })
        except:
            pass
    
    return results


def fetch_arxiv_results(keyword: str, max_results: int = 3) -> List[Dict]:
    """
    从arXiv搜索预印本论文
    返回arXiv论文链接和摘要
    """
    results = []
    
    try:
        # arXiv搜索API（不需要API key）
        search_url = f"http://export.arxiv.org/api/query?search_query=all:{quote(keyword)}&start=0&max_results={max_results}"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            xml_content = resp.text
            
            # 解析XML结果
            entries = re.findall(r'<entry>(.*?)</entry>', xml_content, re.DOTALL)
            
            for entry in entries:
                # 提取标题
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                title = ""
                if title_match:
                    title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                    title = title.replace('\n', ' ').strip()
                
                # 提取链接
                id_match = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                url = ""
                if id_match:
                    url = id_match.group(1).strip()
                
                # 提取摘要
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                abstract = ""
                if summary_match:
                    abstract = re.sub(r'<[^>]+>', '', summary_match.group(1)).strip()
                    abstract = abstract.replace('\n', ' ').strip()
                
                # 提取作者
                authors = []
                author_matches = re.findall(r'<name>(.*?)</name>', entry, re.DOTALL)
                for author_match in author_matches:
                    author = re.sub(r'<[^>]+>', '', author_match).strip()
                    if author:
                        authors.append(author)
                
                if title and url:
                    content = f"论文标题: {title}\n\n"
                    if authors:
                        content += f"作者: {', '.join(authors[:5])}\n\n"  # 最多显示5个作者
                    if abstract:
                        content += f"摘要: {abstract[:1000]}{'...' if len(abstract) > 1000 else ''}\n\n"
                    content += f"arXiv链接: {url}\n\n请访问链接查看完整论文。"
                    
                    results.append({
                        "title": title,
                        "content": content,
                        "source": "arXiv",
                        "url": url,
                        "type": "txt"
                    })
    except Exception as e:
        print(f"Error fetching arXiv results: {e}")
        # 即使失败也返回搜索链接
        try:
            search_url = f"https://arxiv.org/search/?query={quote(keyword)}&searchtype=all"
            results.append({
                "title": f"{keyword} - arXiv",
                "content": f"arXiv搜索链接: {search_url}\n\n请访问链接查看相关预印本论文。",
                "source": "arXiv",
                "url": search_url,
                "type": "txt"
            })
        except:
            pass
    
    return results


def is_irrelevant_url(url: str, title: str = "") -> bool:
    """
    检查URL和标题是否明显不相关（如报告、政策文档等）
    
    Args:
        url: URL地址
        title: 标题（可选）
    
    Returns:
        True 如果不相关，False 如果可能相关
    """
    url_lower = url.lower()
    title_lower = title.lower() if title else ""
    combined = f"{url_lower} {title_lower}"
    
    # 明显不相关的URL模式
    irrelevant_url_patterns = [
        r'raac',  # RAAC报告
        r'report[_-]?for[_-]?publishing',  # 报告类PDF
        r'/[^/]+report[^/]*\.pdf',  # PDF报告
        r'school[_-]?(closure|building|infrastructure)',  # 学校建筑问题
        r'pandemic[_-]?impact',  # 疫情影响
        r'ministerial[_-]?(visit|report)',  # 部长访问报告
        r'covid[_-]?19[_-]?(impact|effect)',  # 新冠疫情影响
    ]
    
    # 检查URL和标题是否包含不相关模式
    for pattern in irrelevant_url_patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return True
    
    # 如果URL包含"report"或"policy"等词，且不包含AI相关关键词，可能不相关
    if re.search(r'\b(report|policy|impact|disruption|building|infrastructure)\b', combined, re.IGNORECASE):
        # 检查是否包含AI相关关键词
        has_ai_keywords = any(keyword in combined for keyword in AI_RELEVANT_KEYWORDS)
        if not has_ai_keywords:
            return True
    
    return False


def fetch_academic_web_results(keyword: str, max_results: int = 3) -> List[Dict]:
    """
    （已弃用）学术网站搜索函数。
    
    说明：
    - 早期版本通过 DuckDuckGo HTML 搜索学术网站再二次爬取，逻辑复杂且不稳定；
    - 目前高质量学术文本资源主要由 Wikipedia / Google Scholar / arXiv 提供；
    - 为减少外部依赖并提升整体效率，这里不再额外发起 DuckDuckGo 搜索。
    
    返回：
        空列表（保留函数签名以兼容旧代码，后续如需扩展学术源可在此处接入）。
    """
    return []


def fetch_ddg_instant_answer(keyword: str) -> Dict:
    """
    （已弃用）DuckDuckGo 即时答案函数。
    
    说明：
    - 为了提高稳定性和效率，本项目已不再调用 DuckDuckGo 的即时答案接口；
    - 文本说明信息由 Wikipedia / Google Scholar / arXiv 等来源提供；
    - 本函数仅为兼容旧代码，如被调用将直接返回 None。
    """
    return None


def clean_title(title: str) -> str:
    """
    清理标题，移除联系方式、部门信息、网址等无关信息
    
    Args:
        title: 原始标题
    
    Returns:
        清理后的标题
    """
    if not title:
        return title
    
    # 移除邮箱地址（包含@符号的部分）
    title = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', title)
    
    # 移除网址（http://, https://, www.）
    title = re.sub(r'https?://[^\s]+', '', title, flags=re.IGNORECASE)
    title = re.sub(r'www\.[^\s]+', '', title, flags=re.IGNORECASE)
    
    # 移除学校域名（如 durham.ac.uk, .edu 等）
    title = re.sub(r'\b[a-zA-Z0-9.-]+\.(ac\.uk|edu\.|edu\b)\b', '', title, flags=re.IGNORECASE)
    
    # 移除部门信息（如果标题只包含部门名称）
    title = re.sub(r'^Department\s+of\s+[A-Za-z\s]+$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^Faculty\s+of\s+[A-Za-z\s]+$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^School\s+of\s+[A-Za-z\s]+$', '', title, flags=re.IGNORECASE)
    
    # 移除开头和结尾的部门名称（保留主要内容）
    title = re.sub(r'^(Department|Faculty|School|Institute|College)\s+of\s+', '', title, flags=re.IGNORECASE)
    
    # 清理多余的空白和标点
    title = re.sub(r'\s+', ' ', title)  # 多个空格变为一个
    title = re.sub(r'^\s*[,\-–—]\s*', '', title)  # 移除开头的逗号、破折号等
    title = re.sub(r'\s*[,\-–—]\s*$', '', title)  # 移除结尾的逗号、破折号等
    title = title.strip()
    
    # 如果清理后标题太短或为空，返回原始标题的简化版本
    if len(title) < 5:
        # 尝试提取标题的第一部分（通常在破折号、冒号或括号前）
        original = title if title else ""
        if not original:
            return "Untitled"
        # 提取第一个有意义的短语
        parts = re.split(r'[—–\-:]', original)
        if parts:
            title = parts[0].strip()
        if len(title) < 5:
            return original[:50] if original else "Untitled"
    
    # 限制标题长度
    if len(title) > 100:
        title = title[:97] + "..."
    
    return title


def clean_extracted_content(content: str) -> str:
    """
    清理提取的文章内容，移除联系方式、部门信息、地址等无关信息
    
    Args:
        content: 原始内容
    
    Returns:
        清理后的内容
    """
    if not content:
        return content
    
    # 按行分割，逐行过滤
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 跳过太短的行（可能是导航或标题）
        if len(line) < 10:
            continue
        
        # 移除包含邮箱的行
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line):
            continue
        
        # 移除包含常见联系方式的行（但如果行很长且包含学术内容，保留）
        contact_patterns = [
            r'^Contact\s*:',
            r'^Email\s*:',
            r'^Phone\s*:',
            r'^Tel\s*:',
            r'^Fax\s*:',
            r'^Address\s*:',
        ]
        
        # 检查是否以联系方式开头（且行很短）
        is_contact_header = any(re.match(pattern, line, re.IGNORECASE) for pattern in contact_patterns)
        if is_contact_header and len(line) < 150:
            continue
        
        # 移除以 @ 符号和域名开头的短行（纯联系信息行）
        if re.match(r'^\s*@\s*[a-zA-Z0-9.-]+\.(ac\.uk|edu|org|com)\s*$', line, re.IGNORECASE):
            continue
        
        # 移除包含学校网址的行（如 durham.ac.uk, .edu, .ac.uk 等）
        # 但如果这一行很长且包含学术内容，移除网址但保留内容
        if re.search(r'\b[a-zA-Z0-9.-]+\.(ac\.uk|edu\.|edu\b)', line):
            url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.(ac\.uk|edu\.|edu\b)'
            urls_in_line = re.findall(url_pattern, line)
            
            # 如果整行主要是网址且很短，跳过
            if urls_in_line and len(line) < 80:
                continue
            
            # 如果包含网址但行很长，移除网址但保留其他内容
            if urls_in_line and len(line) > 80:
                # 移除网址，保留其他内容
                line_without_urls = re.sub(url_pattern, '', line).strip()
                # 如果移除网址后仍有足够内容，保留它
                if len(line_without_urls) > 30:
                    line = line_without_urls
                else:
                    # 如果移除网址后内容太少，跳过整行
                    continue
        
        # 移除只包含部门名称的短行（不包含AI相关内容）
        if re.search(r'^(Department|Faculty|School|Institute|College)\s+of\s+[A-Za-z\s]{0,50}$', line, re.IGNORECASE):
            # 如果是短行且不包含AI相关关键词，跳过
            # 使用简化的关键词列表进行快速检查
            quick_ai_keywords = ['study', 'research', 'paper', 'article', 'method', 'theory', 
                               'algorithm', 'model', 'data', 'analysis', 'learning', 'system', 
                               'course', 'program', 'curriculum', 'ai', 'ml', 'neural', 'deep learning']
            if not any(keyword in line.lower() for keyword in quick_ai_keywords):
                continue
        
        # 移除纯导航文本
        navigation_keywords = [
            'home', 'about', 'contact', 'login', 'register', 'menu', 'search',
            'skip to', 'back to', 'next page', 'previous page', 'breadcrumb'
        ]
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in navigation_keywords) and len(line) < 100:
            continue
        
        # 移除只包含联系信息的短行
        if len(line) < 60 and re.search(r'(@|email|phone|tel|fax|address|contact)', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    # 合并清理后的行
    cleaned_content = '\n'.join(cleaned_lines)
    
    # 移除重复的空行
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    # 移除段落中残留的联系信息（使用更严格的模式）
    # 移除包含 @ 符号后跟域名的小段文字
    cleaned_content = re.sub(r'[^\n]{0,100}@[^\n]{0,100}(ac\.uk|edu|org|com)[^\n]{0,100}\n?', '', cleaned_content, flags=re.IGNORECASE)
    
    # 移除独立的网址行
    cleaned_content = re.sub(r'^https?://[^\s]+\s*$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_content = re.sub(r'^www\.[^\s]+\s*$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    
    # 移除只包含部门名称的短行
    cleaned_content = re.sub(r'^Department of [A-Za-z\s]{0,50}$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_content = re.sub(r'^Faculty of [A-Za-z\s]{0,50}$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_content = re.sub(r'^School of [A-Za-z\s]{0,50}$', '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    
    # 移除多余空行
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content


def extract_article_content(url: str, max_length: int = 3000) -> str:
    """
    从网页URL提取文章内容
    使用多种策略确保能提取到实际文本内容
    返回文章文本内容（前max_length个字符）
    """
    try:
        resp = http_get(url, headers=DEFAULT_HEADERS, timeout=12)
        if resp.status_code == 200:
            html = resp.text
            
            # 移除script、style、noscript等标签
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            # 移除常见的导航和无关内容
            html = re.sub(r'<div[^>]*class="[^"]*(?:nav|menu|sidebar|ad|advertisement|cookie|banner)[^"]*"[^>]*>.*?</div>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            # 策略1: 优先提取article, main等语义化标签
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
                    # 合并所有匹配的内容
                    text = " ".join(matches)
                    # 移除HTML标签
                    text = re.sub(r'<[^>]+>', ' ', text)
                    # 清理空白和特殊字符
                    text = re.sub(r'&[a-z]+;', ' ', text)  # HTML实体
                    text = re.sub(r'&#\d+;', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # 过滤掉太短或明显是导航的内容
                    if len(text) > 200 and not re.search(r'^(home|about|contact|login|register|menu|search)', text[:50], re.IGNORECASE):
                        if len(text) > len(extracted_text):
                            extracted_text = text
                            best_text = text
            
            # 策略2: 如果策略1失败，提取所有段落
            if not extracted_text or len(extracted_text) < 200:
                # 提取所有p标签
                p_matches = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
                if p_matches:
                    paragraphs = []
                    for p in p_matches:
                        text = re.sub(r'<[^>]+>', ' ', p)
                        text = re.sub(r'&[a-z]+;', ' ', text)
                        text = re.sub(r'&#\d+;', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        # 过滤掉太短的段落（可能是导航）
                        if len(text) > 50:
                            paragraphs.append(text)
                    
                    if paragraphs:
                        extracted_text = " ".join(paragraphs)
            
            # 策略3: 如果还是失败，提取body中的所有文本（但过滤导航）
            if not extracted_text or len(extracted_text) < 200:
                # 提取body标签内容
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
                if body_match:
                    body_html = body_match.group(1)
                    # 移除导航、菜单等
                    body_html = re.sub(r'<nav[^>]*>.*?</nav>', '', body_html, flags=re.DOTALL | re.IGNORECASE)
                    body_html = re.sub(r'<header[^>]*>.*?</header>', '', body_html, flags=re.DOTALL | re.IGNORECASE)
                    body_html = re.sub(r'<footer[^>]*>.*?</footer>', '', body_html, flags=re.DOTALL | re.IGNORECASE)
                    
                    # 提取所有文本
                    text = re.sub(r'<[^>]+>', ' ', body_html)
                    text = re.sub(r'&[a-z]+;', ' ', text)
                    text = re.sub(r'&#\d+;', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    extracted_text = text
            
            # 清理和过滤文本
            if extracted_text:
                # 移除常见的无关文本
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
                
                # 按句子分割，过滤掉明显是导航的句子
                sentences = re.split(r'[.!?]\s+', extracted_text)
                filtered_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    # 过滤太短的句子
                    if len(sentence) < 20:
                        continue
                    # 过滤明显是导航的句子
                    is_noise = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in noise_patterns)
                    if not is_noise:
                        filtered_sentences.append(sentence)
                
                if filtered_sentences:
                    extracted_text = ". ".join(filtered_sentences)
                else:
                    # 如果过滤后没有内容，使用原始文本
                    pass
            
            # 如果提取的文本太短，尝试提取所有文本
            if not extracted_text or len(extracted_text) < 100:
                # 最后手段：提取所有文本
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'&[a-z]+;', ' ', text)
                text = re.sub(r'&#\d+;', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                extracted_text = text
            
            # 限制长度
            if len(extracted_text) > max_length:
                # 尝试在句子边界截断
                truncated = extracted_text[:max_length]
                last_period = truncated.rfind('.')
                if last_period > max_length * 0.8:  # 如果最后一句在80%位置之后
                    extracted_text = truncated[:last_period + 1]
                else:
                    extracted_text = truncated + "..."
            
            # 最终清理：移除联系方式、部门信息、地址等无关内容
            extracted_text = extracted_text.strip()
            
            # 清理无关内容
            cleaned_text = clean_extracted_content(extracted_text)
            
            # 如果清理后的内容有意义（至少50个字符且不是纯链接），返回它
            if cleaned_text and len(cleaned_text) >= 50:
                # 检查是否主要是链接
                url_count = len(re.findall(r'https?://', cleaned_text))
                if url_count < len(cleaned_text) / 20:  # 链接数量不超过文本的5%
                    return cleaned_text
            
    except requests.exceptions.Timeout:
        print(f"Timeout extracting content from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Request error extracting content from {url}: {e}")
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    
    return ""


def fetch_ddg_web_results(keyword: str, max_results: int = 10, academic_only: bool = False) -> List[Dict]:
    """
    （已弃用）DuckDuckGo 网页搜索函数。
    
    说明：
    - 旧实现依赖 DuckDuckGo HTML 搜索结果并再次爬取目标网页，逻辑复杂且易受反爬限制影响；
    - 当前文本资源主要由 Wikipedia / Google Scholar / arXiv 提供，高质量学术内容已足够覆盖；
    - 为简化依赖并提升整体稳定性，本函数不再执行任何网络请求。
    
    返回：
        空列表（保留函数签名以兼容旧代码）。
    """
    return []


def fetch_google_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """从Google Images搜索中获取图片"""
    results = []
    
    try:
        # Google Images搜索URL
        search_url = f"https://www.google.com/search?tbm=isch&q={quote(keyword)}&safe=images"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            # 从HTML中提取图片URL（Google Images使用多种格式）
            # 尝试从JSON数据中提取
            json_pattern = r'AF_initDataCallback\([^)]+\)'
            json_matches = re.findall(json_pattern, html)
            
            # 更通用的图片URL提取模式
            img_patterns = [
                r'"ou":"([^"]+)"',  # Google Images的原始URL字段（最可靠）
                r'"ow":\d+,"oh":\d+,"ou":"([^"]+)"',  # 带尺寸信息的URL
                r'data-src="(https://[^"]+\.(jpg|jpeg|png|gif|webp))"',
                r'src="(https://encrypted-tbn[^"]+)"',  # Google缩略图（可以作为备用）
                r'https://[^"\s]+\.(jpg|jpeg|png|gif|webp|svg)',  # 直接匹配图片URL
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
                        # 过滤掉明显不是图片的URL（如JavaScript、CSS等）
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/search', '/webhp']):
                            continue
                        # 验证是图片URL
                        if any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', 'image', 'img', 'photo', 'picture']):
                            seen_urls.add(img_url)
                            results.append({
                                "title": f"图片: {keyword}",
                                "url": img_url,
                                "thumbnail": img_url,
                                "source": "Google Images",
                                "type": "image"
                            })
    except Exception as e:
        print(f"Error fetching Google Images: {e}")
    
    # 如果无法提取实际图片URL，至少返回搜索链接
    if not results:
        results.append({
            "title": f"Google图片搜索: {keyword}",
            "url": f"https://www.google.com/search?tbm=isch&q={quote(keyword)}",
            "thumbnail": "",
            "source": "Google Images",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_bing_images(keyword: str, max_results: int = 10) -> List[Dict]:
    """从Bing Images搜索中获取图片"""
    results = []
    
    try:
        # Bing Images搜索URL
        search_url = f"https://www.bing.com/images/search?q={quote(keyword)}&safe=strict"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            # 从HTML中提取图片URL（Bing Images使用多种格式）
            img_patterns = [
                r'"murl":"([^"]+)"',  # Bing Images的媒体URL字段（最可靠）
                r'"thumb":"([^"]+)"',  # 缩略图URL
                r'data-src="(https://[^"]+\.(jpg|jpeg|png|gif|webp))"',
                r'src="(https://[^"]+th\.bing\.com[^"]+)"',  # Bing缩略图服务器
                r'https://[^"\s]+\.(jpg|jpeg|png|gif|webp|svg)',  # 直接匹配图片URL
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
                        # 过滤掉明显不是图片的URL
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/search']):
                            continue
                        # 验证是图片URL
                        if any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', 'image', 'img', 'photo', 'picture', 'th.bing.com']):
                            seen_urls.add(img_url)
                            results.append({
                                "title": f"图片: {keyword}",
                                "url": img_url,
                                "thumbnail": img_url,
                                "source": "Bing Images",
                                "type": "image"
                            })
    except Exception as e:
        print(f"Error fetching Bing Images: {e}")
    
    # 如果无法提取实际图片URL，至少返回搜索链接
    if not results:
        results.append({
            "title": f"Bing图片搜索: {keyword}",
            "url": f"https://www.bing.com/images/search?q={quote(keyword)}",
            "thumbnail": "",
            "source": "Bing Images",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_unsplash_images(keyword: str, max_results: int = 5) -> List[Dict]:
    """从Unsplash免费图片库获取图片（高质量）"""
    results = []
    
    try:
        # Unsplash搜索URL（不需要API key，直接访问搜索页面）
        search_url = f"https://unsplash.com/s/photos/{quote(keyword)}"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            # 从HTML中提取图片URL（Unsplash使用多种格式）
            img_patterns = [
                r'"raw":"([^"]+)"',  # Unsplash原始图片URL（最可靠）
                r'"regular":"([^"]+)"',  # Unsplash常规尺寸
                r'"small":"([^"]+)"',  # 小尺寸
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
                        # 过滤掉非图片URL
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/s/photos']):
                            continue
                        seen_urls.add(img_url)
                        results.append({
                            "title": f"Unsplash图片: {keyword}",
                            "url": img_url,
                            "thumbnail": img_url,
                            "source": "Unsplash",
                            "type": "image"
                        })
    except Exception as e:
        print(f"Error fetching Unsplash images: {e}")
    
    # 如果无法提取实际图片URL，至少返回搜索链接
    if not results:
        results.append({
            "title": f"Unsplash图片搜索: {keyword}",
            "url": f"https://unsplash.com/s/photos/{quote(keyword)}",
            "thumbnail": "",
            "source": "Unsplash",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_pexels_images(keyword: str, max_results: int = 5) -> List[Dict]:
    """从Pexels免费图片库获取图片（高质量）"""
    results = []
    
    try:
        # Pexels搜索URL（不需要API key，直接访问搜索页面）
        search_url = f"https://www.pexels.com/search/{quote(keyword)}/"
        resp = http_get(search_url, headers=DEFAULT_HEADERS, timeout=15)
        
        if resp.status_code == 200:
            html = resp.text
            
            # 从HTML中提取图片URL（Pexels使用多种格式）
            img_patterns = [
                r'data-bg="([^"]+)"',  # Pexels背景图片
                r'data-src="(https://images\.pexels\.com[^"]+)"',  # 延迟加载的图片
                r'src="(https://images\.pexels\.com[^"]+)"',  # 直接src
                r'url\(["\']?(https://images\.pexels\.com[^"\']+)["\']?\)',  # CSS背景图片
                r'"original":"([^"]+)"',  # Pexels原始图片URL
                r'"large":"([^"]+)"',  # 大尺寸图片
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
                        # 过滤掉非图片URL
                        if any(skip in img_url.lower() for skip in ['javascript:', 'data:', 'about:', '/search']):
                            continue
                        seen_urls.add(img_url)
                        results.append({
                            "title": f"Pexels图片: {keyword}",
                            "url": img_url,
                            "thumbnail": img_url,
                            "source": "Pexels",
                            "type": "image"
                        })
    except Exception as e:
        print(f"Error fetching Pexels images: {e}")
    
    # 如果无法提取实际图片URL，至少返回搜索链接
    if not results:
        results.append({
            "title": f"Pexels图片搜索: {keyword}",
            "url": f"https://www.pexels.com/search/{quote(keyword)}/",
            "thumbnail": "",
            "source": "Pexels",
            "type": "image"
        })
    
    return results[:max_results]


def fetch_generic_image_links(keyword: str, max_results: int = 3) -> List[Dict]:
    """获取通用图片搜索链接（作为备用方案）"""
    results = []
    
    # 构建多个图片搜索源的链接
    image_search_sources = [
        {
            "title": f"Google图片搜索: {keyword}",
            "url": f"https://www.google.com/search?tbm=isch&q={quote(keyword)}",
            "source": "Google Images"
        },
        {
            "title": f"Bing图片搜索: {keyword}",
            "url": f"https://www.bing.com/images/search?q={quote(keyword)}",
            "source": "Bing Images"
        },
        {
            "title": f"Unsplash图片库: {keyword}",
            "url": f"https://unsplash.com/s/photos/{quote(keyword)}",
            "source": "Unsplash"
        },
        {
            "title": f"Pexels图片库: {keyword}",
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
    """
    （已弃用）DuckDuckGo 图片搜索函数。
    
    说明：
    - 图片资源已经由 Google Images / Bing Images / Unsplash / Pexels 等源提供；
    - 为减少第三方依赖并提升效率，本函数不再访问 DuckDuckGo。
    
    返回：
        空列表（如需额外图片源，可在其他专用函数中接入）。
    """
    return []


# ====== 通用网页搜索 ======
# 注意：此函数已不再使用，改为直接获取实际文章内容


# ====== YouTube相关函数 ======

def extract_youtube_videos_from_json(data: dict, max_results: int) -> List[Dict]:
    """从YouTube的JSON数据中提取视频信息"""
    videos = []
    
    try:
        # YouTube的数据结构比较复杂，需要递归查找
        def find_videos(obj, depth=0):
            if depth > 10:  # 防止无限递归
                return
            
            if isinstance(obj, dict):
                # 查找videoRenderer
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
                            "description": f"YouTube视频: {title}",
                            "video_id": video_id,
                            "thumbnail": thumbnail,
                            "type": "video"
                        })
                
                # 递归查找
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
    """从YouTube HTML中提取视频信息（备用方法）"""
    videos = []
    
    try:
        # 查找视频链接
        pattern = r'/watch\?v=([a-zA-Z0-9_-]{11})'
        video_ids = list(set(re.findall(pattern, html)))[:max_results]
        
        for video_id in video_ids:
            videos.append({
                "title": f"YouTube视频 {video_id}",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": f"YouTube视频ID: {video_id}",
                "video_id": video_id,
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/default.jpg",
                "type": "video"
            })
    except Exception as e:
        print(f"Error extracting YouTube videos from HTML: {e}")
    
    return videos


# ====== 主搜索函数 ======

def search_all_resources(keywords: List[str], max_per_type: int = 10, progress_callback=None) -> Dict[str, List[Dict]]:
    """
    为所有关键词搜索所有类型的资源
    不依赖特定API，使用通用搜索方法
    返回: {
        "txt": [...],
        "video": [...],
        "code": [...]
    }
    """
    all_txt = []
    all_video = []
    all_code = []
    
    for i, keyword in enumerate(keywords):
        print(f"搜索关键词 {i+1}/{len(keywords)}: {keyword}")
        
        # 通知进度回调：开始搜索关键词
        if progress_callback:
            progress_callback({
                "type": "keyword_start",
                "keyword": keyword,
                "index": i + 1,
                "total": len(keywords)
            })
        
        # 搜索文本资源（增加搜索数量以提高结果）
        try:
            txt_results = search_text_resources(keyword, max_per_type * 2)
            all_txt.extend(txt_results)
            print(f"  找到 {len(txt_results)} 个文本资源")
            
            # 通知进度回调：文本搜索完成
            if progress_callback:
                progress_callback({
                    "type": "keyword_text_done",
                    "keyword": keyword,
                    "count": len(txt_results)
                })
        except Exception as e:
            print(f"  文本搜索错误: {e}")
        
        # 搜索视频
        try:
            video_results = search_youtube_videos(keyword, max_per_type)
            all_video.extend(video_results)
            print(f"  找到 {len(video_results)} 个视频资源")
            
            # 通知进度回调：视频搜索完成
            if progress_callback:
                progress_callback({
                    "type": "keyword_video_done",
                    "keyword": keyword,
                    "count": len(video_results)
                })
        except Exception as e:
            print(f"  视频搜索错误: {e}")
        
        # 搜索代码资源
        try:
            code_results = search_code_resources(keyword, max_per_type)
            all_code.extend(code_results)
            print(f"  找到 {len(code_results)} 个代码资源")
            
            # 通知进度回调：代码搜索完成
            if progress_callback:
                progress_callback({
                    "type": "keyword_code_done",
                    "keyword": keyword,
                    "count": len(code_results)
                })
        except Exception as e:
            print(f"  代码搜索错误: {e}")
        
        # 通知进度回调：关键词搜索完成
        if progress_callback:
            progress_callback({
                "type": "keyword_done",
                "keyword": keyword,
                "index": i + 1,
                "total": len(keywords)
            })
        
        # 避免请求过快
        if i < len(keywords) - 1:
            time.sleep(random.uniform(0.5, 1.5))
    
    # 对所有资源进行去重（特别是视频资源）
    # 文本资源：基于URL去重
    seen_txt_urls = set()
    unique_txt = []
    for txt in all_txt:
        url = txt.get("url", "")
        if url and url not in seen_txt_urls:
            seen_txt_urls.add(url)
            unique_txt.append(txt)
    
    # 视频资源：基于video_id或URL去重
    seen_video_ids = set()
    seen_video_urls = set()
    unique_video = []
    for video in all_video:
        video_id = video.get("video_id")
        url = video.get("url", "")
        
        # 优先使用video_id去重
        if video_id and video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            seen_video_urls.add(url)
            unique_video.append(video)
        elif url and url not in seen_video_urls:
            # 如果没有video_id，使用URL去重
            seen_video_urls.add(url)
            unique_video.append(video)
    
    # 代码资源：基于URL去重
    seen_code_urls = set()
    unique_code = []
    for code in all_code:
        url = code.get("url", "")
        if url and url not in seen_code_urls:
            seen_code_urls.add(url)
            unique_code.append(code)
    
    # 不去重时清理标题，标题清理应该在保存时进行
    # 这样推荐算法可以使用原始标题进行相似度计算
    
    print(f"去重后: 文本 {len(all_txt)} -> {len(unique_txt)}, 视频 {len(all_video)} -> {len(unique_video)}, 代码 {len(all_code)} -> {len(unique_code)}")
    
    return {
        "txt": unique_txt,
        "video": unique_video,
        "code": unique_code
    }
