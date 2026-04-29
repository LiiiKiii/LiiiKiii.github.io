#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content-based recommendation module (CBF).
Select the best resources through similarity scoring.
"""

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

VERBOSE_LOGS = os.environ.get("AI_PEDIA_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}


def debug_print(*args, force: bool = False, **kwargs) -> None:
    """Stay quiet by default; print only when verbose mode or critical errors are enabled."""
    if force or VERBOSE_LOGS:
        print(*args, **kwargs)

try:
    from backend.core.resource_searcher import clean_extracted_content, clean_title
except ImportError:
    def clean_extracted_content(content: str) -> str:
        """Fallback cleaner used when the primary function cannot be imported."""
        return content
    
    def clean_title(title: str) -> str:
        """Fallback title cleaner."""
        return title

def read_txt_files(folder_path: str) -> List[str]:
    """
    Read the content of all TXT files in a folder.
    Return: List[str], where each item is one document.
    Exclude macOS resource-fork files (starting with ._) and other hidden files.
    """
    texts = []
    if not os.path.isdir(folder_path):
        return texts
    
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".txt"):
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return texts


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F\'\"\"''',.:;!?()[\]{}<>~`•…–—/_+=*^%$#@\\|-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_relevant_resource(resource: Dict, user_docs: List[str]) -> bool:
    """Return whether relevant resource."""
    irrelevant_patterns = [
        r'\braac\b',  # RAAC report (building issue)
        r'\breport\s+(on|about|of)\s+(the|impact|disruption)',  # Report-style document
        r'\bpandemic\s+impact',  # Pandemic impact report
        r'\bschool\s+(closure|building|infrastructure)',  # School building issue
        r'\bcovid-?19\s+(impact|effect)',  # COVID-19 impact
        r'\bministerial\s+(visit|report)',  # Ministerial visit report
    ]
    
    try:
        from backend.core.resource_searcher import AI_RELEVANT_KEYWORDS as academic_keywords
    except ImportError:
        academic_keywords = [
            'machine learning', 'deep learning', 'neural network', 'algorithm',
            'model', 'training', 'data', 'classification', 'regression',
            'supervised', 'unsupervised', 'reinforcement', 'gradient',
            'optimization', 'loss function', 'activation', 'backpropagation',
            'convolutional', 'recurrent', 'transformer', 'attention',
            'natural language processing', 'computer vision', 'speech recognition',
            'artificial intelligence', 'ai', 'ml', 'dl', 'nlp', 'cv',
        ]
    
    title = resource.get("title", "").lower()
    url = resource.get("url", "").lower()
    content = resource.get("content", "").lower()
    description = resource.get("description", "").lower()
    
    combined_text = f"{title} {url} {content[:500]} {description}"  # Check only the first 500 characters of content
    
    for pattern in irrelevant_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            has_academic_keywords = any(keyword in combined_text for keyword in academic_keywords)
            if not has_academic_keywords:
                debug_print(f"  Filtered irrelevant resource: {resource.get('title', 'Unknown')[:50]} (matched an irrelevant pattern)")
                return False
    
    irrelevant_url_patterns = [
        r'/report', r'/policy', r'/impact', r'/disruption',
        r'raac', r'building', r'infrastructure', r'school-closure',
    ]
    
    for pattern in irrelevant_url_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            has_academic_keywords = any(keyword in combined_text for keyword in academic_keywords)
            if not has_academic_keywords:
                debug_print(f"  Filtered irrelevant resource: {resource.get('title', 'Unknown')[:50]} (URL matched an irrelevant path)")
                return False
    
    if any(keyword in combined_text for keyword in academic_keywords):
        return True
    
    learning_keywords = ['learning', 'tutorial', 'course', 'lecture', 'guide', 'introduction', 
                         'explained', 'overview', 'basics', 'fundamentals', 'concept', 'theory',
                         'neural', 'network', 'model', 'algorithm', 'data', 'training']
    if any(keyword in combined_text for keyword in learning_keywords):
        return True
    
    return True


def compute_domain_bonus(resource: Dict) -> float:
    """Apply a light reranking bonus to resources that better match AI learning scenarios."""
    try:
        from backend.core.resource_searcher import AI_RELEVANT_KEYWORDS as ai_keywords
    except ImportError:
        ai_keywords = [
            'machine learning', 'deep learning', 'neural network', 'transformer',
            'attention', 'computer vision', 'natural language processing', 'reinforcement learning',
            'pytorch', 'tensorflow', 'github', 'arxiv', 'wikipedia'
        ]

    title = resource.get("title", "").lower()
    description = resource.get("description", "").lower()
    url = resource.get("url", "").lower()
    source = resource.get("source", "").lower()
    combined = f"{title} {description} {url} {source}"

    bonus = 0.0
    if any(keyword in combined for keyword in ai_keywords):
        bonus += 0.05
    if any(authority in url for authority in ["arxiv.org", "wikipedia.org", "scholar.google.com", "github.com"]):
        bonus += 0.03
    if any(keyword in title for keyword in ai_keywords):
        bonus += 0.02
    return bonus


def compute_similarity(user_docs: List[str], resources: List[Dict], resource_type: str) -> List[Tuple[Dict, float]]:
    """Compute similarity."""
    if not user_docs or not resources:
        return []
    
    cleaned_user_docs = [clean_text(doc) for doc in user_docs]
    user_text = " ".join(cleaned_user_docs)  # Merge all user documents
    
    resource_texts = []
    resource_objs = []
    
    for res in resources:
        if resource_type == "txt":
            text = res.get("content", res.get("title", ""))
        elif resource_type == "video":
            text = res.get("description", res.get("title", ""))
        elif resource_type == "code":
            text = res.get("description", res.get("title", ""))
        else:
            text = str(res.get("title", ""))
        
        if text:
            resource_texts.append(clean_text(text))
            resource_objs.append(res)
    
    if not resource_texts:
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
            norm="l2"
        )
        
        all_texts = [user_text] + resource_texts
        vectors = vectorizer.fit_transform(all_texts)
        
        user_vector = vectors[0:1]
        resource_vectors = vectors[1:]
        
        similarities = cosine_similarity(user_vector, resource_vectors)[0]
        
        results = list(zip(resource_objs, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return [(res, 0.0) for res in resource_objs]


def recommend_best_resources(
    user_folder_path: str,
    all_resources: Dict[str, List[Dict]],
    top_k_per_type: int = 5
) -> Dict[str, List[Dict]]:
    """Recommend best resources."""
    user_docs = read_txt_files(user_folder_path)
    
    if not user_docs:
        debug_print("Warning: No user documents found, returning all resources", force=True)
        return all_resources
    
    recommended = {}
    
    for resource_type, resources in all_resources.items():
        if not resources:
            recommended[resource_type] = []
            continue
        
        similarity_results = compute_similarity(user_docs, resources, resource_type)
        debug_print(f"  [{resource_type}] Computed similarity for {len(similarity_results)} resources")
        
        if similarity_results:
            max_score = max(score for _, score in similarity_results)
            min_score = min(score for _, score in similarity_results)
            avg_score = sum(score for _, score in similarity_results) / len(similarity_results)
            debug_print(f"  [{resource_type}] Similarity range: {min_score:.4f} - {max_score:.4f}, average: {avg_score:.4f}")
        
        reranked_results = [
            (res, score, score + compute_domain_bonus(res))
            for res, score in similarity_results
        ]
        sorted_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)
        debug_print(f"  [{resource_type}] Similarity and domain-bonus ranking complete; selecting the top {top_k_per_type}")
        
        top_resources = []
        relevance_filtered = 0
        for res, score, final_score in sorted_results[:top_k_per_type * 2]:  # Keep a slightly larger candidate pool
            if is_relevant_resource(res, user_docs):
                res["similarity_score"] = float(score)
                res["ranking_score"] = float(final_score)
                top_resources.append(res)
                if len(top_resources) >= top_k_per_type:
                    break
            else:
                relevance_filtered += 1
        
        debug_print(f"  [{resource_type}] Relevance-filtered: {relevance_filtered}, final recommendations: {len(top_resources)}")
        
        recommended[resource_type] = top_resources
    
    return recommended


def save_recommended_resources(
    recommended: Dict[str, List[Dict]],
    output_folder: str
):
    """Save recommended resources."""
    os.makedirs(output_folder, exist_ok=True)
    
    for resource_type, resources in recommended.items():
        type_folder = os.path.join(output_folder, resource_type)
        os.makedirs(type_folder, exist_ok=True)
        
        for i, res in enumerate(resources):
            if resource_type == "txt":
                cleaned_title = clean_title(res.get('title', 'resource'))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)}.txt"
                filepath = os.path.join(type_folder, filename)
                content = res.get("content", "")
                cleaned_content = clean_extracted_content(content)
                metadata = f"Source: {res.get('source', 'Unknown')}\n"
                metadata += f"URL: {res.get('url', '')}\n"
                metadata += f"Similarity Score: {res.get('similarity_score', 0.0):.4f}\n"
                metadata += "\n" + "="*50 + "\n\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(metadata + cleaned_content)
            
            elif resource_type == "video":
                cleaned_title = clean_title(res.get('title', 'video'))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)}.txt"
                filepath = os.path.join(type_folder, filename)
                content = f"Title: {cleaned_title}\n"
                content += f"URL: {res.get('url', '')}\n"
                description = res.get('description', '')
                if description:
                    content += f"Description: {description}\n"
                content += f"Similarity Score: {res.get('similarity_score', 0.0):.4f}\n"
                if res.get("thumbnail"):
                    content += f"Thumbnail: {res.get('thumbnail')}\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
            
            elif resource_type == "code":
                cleaned_title = clean_title(res.get('title', 'code'))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)}.txt"
                filepath = os.path.join(type_folder, filename)
                content = f"Title: {cleaned_title}\n"
                content += f"URL: {res.get('url', '')}\n"
                content += f"Source: {res.get('source', 'Unknown')}\n"
                description = res.get('description', '')
                if description:
                    content += f"Description: {description}\n"
                content += f"Similarity Score: {res.get('similarity_score', 0.0):.4f}\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)


def sanitize_filename(filename: str) -> str:
    """Handle sanitize filename."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename[:100]  # Limit length
    return filename
