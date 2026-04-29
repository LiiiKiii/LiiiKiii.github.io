#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword extraction module.
Extract key topics and phrases from documents.
"""

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(path: str) -> str:
    """Read file content."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def basic_clean(text: str) -> str:
    """Basic text cleaning: lowercase, strip HTML, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F\'\"\"''',.:;!?()[\]{}<>~`•…–—/_+=*^%$#@\\|-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phrase(s: str) -> str:
    """Normalize a phrase: hyphens to spaces, trim edges, and strip simple plurals."""
    s = s.replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s)
    tokens = s.split()
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return ""
    if len(tokens) >= 1:
        last = tokens[-1]
        if len(last) > 3 and last.endswith("s"):
            tokens[-1] = last[:-1]
    return " ".join(tokens)


def is_noise_phrase(phrase: str) -> bool:
    """
    Decide whether a phrase is noise (irrelevant terminology).
    Return True if it should be filtered out.
    """
    phrase_lower = phrase.lower().strip()
    tokens = phrase_lower.split()
    
    if len(tokens) >= 2:
        word_counts = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
        if any(count >= 2 for count in word_counts.values()):
            return True
    
    url_patterns = [
        r'www\.',
        r'http',
        r'https',
        r'\.com',
        r'\.org',
        r'\.edu',
        r'\.net',
        r'\.uk',
        r'\.cn',
        r'\.ac\.uk',  # University domains (for example, durham.ac.uk)
        r'\.edu\.',   # .edu. domain
        r'email',
        r'@',
        r'\.gov',
        r'\.mil',
        r'doi\s+org',  # DOI link
        r'dx\s+doi',   # DOI link variant
        r'doi\s+org',  # DOI link
    ]
    for pattern in url_patterns:
        if re.search(pattern, phrase_lower):
            return True
    
    citation_patterns = [
        r'\barxiv\s+preprint',
        r'\bpreprint\s+arxiv',
        r'\barxiv\s+\d+',
        r'\bet\s+al\s+(proposed|introduced|presented|showed|demonstrated|developed)',
        r'\bet\s+al\s+\d+',
        r'\bdoi\s+org',
        r'\bdx\s+doi',
        r'\bvol\s+\d+',
        r'\bpp\s+\d+',
        r'\bpages\s+\d+',
        r'\bvolume\s+\d+',
    ]
    for pattern in citation_patterns:
        if re.search(pattern, phrase_lower):
            return True
    
    university_patterns = [
        r'\b\w+\.ac\.uk\b',  # Match domains such as "durham.ac.uk" and "oxford.ac.uk"
        r'\b\w+\.edu\b',     # Match domains such as "mit.edu" and "stanford.edu"
        r'\bdurham\s+(university|ac|uk)\b',
        r'\buniversity\s+of\s+\w+\s+(ac|uk|edu)\b',
        r'\bdepartment\s+of\s+[a-z\s]+\s+(university|ac|uk|edu)\b',
        r'\bfaculty\s+of\s+[a-z\s]+\s+(university|ac|uk|edu)\b',
        r'\bschool\s+of\s+[a-z\s]+\s+(university|ac|uk|edu)\b',
    ]
    for pattern in university_patterns:
        if re.search(pattern, phrase_lower):
            return True
    
    if re.search(r'\d+.*(tel|phone|fax|mobile)', phrase_lower):
        return True
    if re.search(r'(tel|phone|fax|mobile).*\d+', phrase_lower):
        return True
    
    address_keywords = [
        'centre', 'center', 'street', 'road', 'avenue', 'lane',
        'building', 'floor', 'room', 'office', 'address',
        'postcode', 'zip', 'code', 'location',
        'durham', 'stockton', 'palatine',  # Common place names
        'department', 'faculty', 'school', 'institute', 'college',  # Institution names
        'campus', 'headquarters', 'head office',  # Office locations
    ]
    address_count = sum(1 for kw in address_keywords if kw in phrase_lower)
    institutional_keywords = ['department', 'faculty', 'school', 'institute', 'college', 'university']
    has_institutional = any(kw in phrase_lower for kw in institutional_keywords)
    
    academic_context = any(academic in phrase_lower for academic in [
        'mass', 'gravity', 'distribution', 'cluster', 'point',
        'matrix', 'vector', 'space', 'dimension',
        'thought', 'theory', 'method', 'approach', 'algorithm',
        'model', 'learning', 'network', 'data', 'analysis'
    ])
    
    if has_institutional and not academic_context:
        return True
    if address_count >= 2 and not academic_context:
        return True
    
    us_state_abbrevs = ['ny', 'ca', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi', 
                       'nj', 'va', 'wa', 'az', 'ma', 'tn', 'in', 'mo', 'md', 'wi',
                       'co', 'mn', 'sc', 'al', 'la', 'ky', 'or', 'ok', 'ct', 'ia',
                       'ut', 'ar', 'nv', 'ms', 'ks', 'nm', 'ne', 'wv', 'id', 'hi',
                       'nh', 'me', 'mt', 'ri', 'de', 'sd', 'nd', 'ak', 'dc', 'vt', 'wy']
    country_abbrevs = ['usa', 'uk', 'us', 'ca', 'au', 'de', 'fr', 'it', 'es', 'nl', 'be', 'ch', 'at', 'se', 'no', 'dk', 'fi', 'pl', 'cz', 'ie']
    
    if len(tokens) >= 2:
        has_place_name = any(len(t) > 3 for t in tokens)  # At least one longer token, possibly a place name
        has_abbrev = any(t in us_state_abbrevs or t in country_abbrevs for t in tokens)
        if has_place_name and has_abbrev:
            if not academic_context:
                return True
    
    contact_keywords = [
        'telephone', 'phone', 'tel', 'fax', 'mobile',
        'contact', 'call', 'reach',
    ]
    academic_keywords = [
        'classification', 'algorithm', 'model', 'data', 'analysis',
        'method', 'approach', 'technique', 'theory', 'concept',
        'learning', 'network', 'system', 'process', 'function',
        'matrix', 'vector', 'curve', 'score', 'metric', 'measure',
        'evaluation', 'performance', 'accuracy', 'precision', 'recall',
        'roc', 'auc', 'component', 'feature', 'sample', 'dataset'
    ]
    
    if any(kw in phrase_lower for kw in contact_keywords):
        if not any(academic in phrase_lower for academic in academic_keywords):
            return True
    
    if re.search(r'\d{4,}', phrase_lower):  # Numbers with 4 or more digits
        return True
    
    if re.search(r'[a-z]+\s+\d+\s+[a-z]+', phrase_lower):  # Phrases such as "chapter 4 section"
        if not any(academic in phrase_lower for academic in [
            'chapter', 'section', 'figure', 'table', 'equation',
            'algorithm', 'method', 'model'
        ]):
            return True
    
    if len(tokens) > 5:  # Phrases longer than 5 tokens are usually noise
        return True
    
    noise_patterns = [
        r'^\d+\s*$',  # Pure numbers
        r'^[a-z]\s+[a-z]\s+[a-z]$',  # Three single-letter tokens
        r'page\s+\d+',  # Page number
        r'figure\s+\d+',  # Figure number (while keeping the word "figure" itself)
    ]
    for pattern in noise_patterns:
        if re.match(pattern, phrase_lower):
            return True
    
    if re.search(r'(department|faculty|school|institute|college)\s+of\s+[^,]+,\s+(university|institute)', phrase_lower):
        return True
    
    common_universities = ['durham', 'oxford', 'cambridge', 'harvard', 'mit', 'stanford', 'yale', 'princeton']
    if any(uni in phrase_lower for uni in common_universities):
        if re.search(r'\b(durham|oxford|cambridge|harvard|mit|stanford|yale|princeton)\s+(university|college|institute)\b', phrase_lower):
            if not academic_context:
                return True
    
    copyright_keywords = [
        'copyright', 'licensed', 'license', 'reserved', 'rights',
        'permission', 'reproduce', 'reproduction', 'prohibited',
        'limited use', 'use limited', 'all rights', 'rights reserved'
    ]
    if any(kw in phrase_lower for kw in copyright_keywords):
        return True
    
    short_word_count = sum(1 for t in tokens if len(t) <= 2)
    if len(tokens) >= 2 and short_word_count >= len(tokens) * 0.5:  # More than half of the tokens are short
        academic_abbrevs = ['ai', 'ml', 'dl', 'nlp', 'cv', 'cnn', 'rnn', 'lstm', 'gan', 'svm', 
                           'pca', 'ica', 'knn', 'rf', 'gbm', 'xgb', 'bert', 'gpt', 'api', 'url',
                           'http', 'html', 'xml', 'json', 'sql', 'db', 'id', 'ui', 'ux']
        if not any(t in academic_abbrevs for t in tokens):
            return True
    
    url_like_phrases = [
        'world wide web', 'www', 'http', 'https', 'ftp', 'smtp'
    ]
    if phrase_lower in url_like_phrases:
        return True
    
    return False


def build_vectorizer():
    """Handle build vectorizer."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        norm=None,
        sublinear_tf=True
    )


def mmr_select(candidates, cand_vectors, query_vec, top_k=6, lambda_div=0.7):
    """Handle mmr select."""
    selected = []
    if len(candidates) == 0:
        return selected

    rep = cosine_similarity(cand_vectors, query_vec).ravel()

    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        if not selected:
            best = int(np.argmax(rep[remaining]))
            selected.append(remaining.pop(best))
        else:
            max_score, max_idx = -1e9, -1
            for idx_pos, idx in enumerate(remaining):
                sim_to_query = rep[idx]
                if selected:
                    sim_to_selected = cosine_similarity(
                        cand_vectors[idx:idx+1], cand_vectors[selected]
                    ).max()
                else:
                    sim_to_selected = 0.0
                score = lambda_div * sim_to_query - (1 - lambda_div) * sim_to_selected
                if score > max_score:
                    max_score, max_idx = score, idx_pos
            selected.append(remaining.pop(max_idx))
    return [candidates[i] for i in selected]


def compute_semantic_score(phrase: str) -> float:
    """Compute semantic score."""
    phrase_lower = phrase.lower()
    score = 0.0
    
    academic_terms = {
        'machine learning': 3.0, 'deep learning': 3.0, 'neural network': 3.0,
        'artificial intelligence': 3.0, 'ai': 2.5, 'ml': 2.0, 'dl': 2.0,
        'neural networks': 2.5, 'deep neural': 2.5, 'convolutional': 2.0,
        'recurrent neural': 2.0, 'rnn': 2.0, 'lstm': 2.0, 'cnn': 2.0,
        'transformer': 2.5, 'attention mechanism': 2.5, 'bert': 2.0, 'gpt': 2.0,
        'generative model': 2.5, 'gan': 2.0, 'variational': 2.0,
        'recommendation system': 2.5, 'content based': 2.5, 'collaborative filtering': 2.5,
        'recommender system': 2.5, 'content based filtering': 2.5,
        'natural language': 2.5, 'nlp': 2.0, 'language model': 2.5,
        'large language model': 3.0, 'llm': 2.5, 'text processing': 2.0,
        'data mining': 2.0, 'feature extraction': 2.0, 'dimensionality reduction': 2.0,
        'principal component': 2.0, 'pca': 1.5, 'clustering': 2.0, 'classification': 2.0,
        'regression': 2.0, 'supervised learning': 2.0, 'unsupervised learning': 2.0,
        'probability': 1.5, 'statistical': 1.5, 'optimization': 1.5,
        'gradient descent': 2.0, 'backpropagation': 2.0, 'loss function': 2.0,
        'computer vision': 2.5, 'cv': 2.0, 'image processing': 2.0,
        'object detection': 2.0, 'semantic segmentation': 2.0,
        'algorithm': 1.5, 'method': 1.0, 'approach': 1.0, 'technique': 1.0,
        'framework': 1.5, 'architecture': 1.5, 'model': 1.5, 'system': 1.0,
        'training': 1.5, 'evaluation': 1.5, 'performance': 1.0, 'accuracy': 1.0,
    }
    
    for term, weight in academic_terms.items():
        if term in phrase_lower:
            score += weight
            break  # Match only once to avoid repeated scoring
    
    academic_keywords = ['learning', 'network', 'model', 'algorithm', 'method', 
                        'data', 'feature', 'training', 'neural', 'deep']
    keyword_count = sum(1 for kw in academic_keywords if kw in phrase_lower)
    if keyword_count >= 2:
        score += 0.5
    
    if ' ' in phrase:
        score += 0.3
    
    word_count = len(phrase.split())
    if 2 <= word_count <= 4:
        score += 0.2
    
    return score


def extract_keywords_from_folder(folder_path: str, top_k: int = 10, min_docs: int = 3) -> list:
    """Extract keywords from folder."""
    txt_paths = []
    if os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for fname in files:
                if fname.startswith('._') or fname.startswith('.DS_Store'):
                    continue
                if fname.lower().endswith(".txt"):
                    txt_paths.append(os.path.join(root, fname))
    
    txt_paths = sorted(txt_paths)
    
    if len(txt_paths) < 2:
        raise ValueError("At least 2 TXT documents are required for keyword extraction")
    
    texts = []
    for p in txt_paths:
        t = basic_clean(read_file(p))
        texts.append(t)
    
    n_docs = len(texts)
    if n_docs < 2:
        raise ValueError("At least 2 documents are required")
    
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    
    tfidf_sum = X.sum(axis=0).A1
    doc_freq = (X > 0).sum(axis=0).A1
    tfidf_mean = tfidf_sum / np.maximum(doc_freq, 1)
    
    mask_coverage = doc_freq >= max(1, min_docs)
    
    is_phrase_like = np.array([" " in term or "-" in term for term in vocab])
    
    mean_threshold = np.quantile(tfidf_mean[mask_coverage], 0.85) if np.any(mask_coverage) else 0.0
    candidate_mask = mask_coverage & (is_phrase_like | (tfidf_mean >= mean_threshold * 1.2))
    
    cand_terms = vocab[candidate_mask]
    cand_scores = tfidf_mean[candidate_mask]
    
    normalized = [normalize_phrase(t) for t in cand_terms]
    keep_idx = [i for i, s in enumerate(normalized) if s]
    cand_terms = cand_terms[keep_idx]
    cand_scores = cand_scores[keep_idx]
    normalized = [normalized[i] for i in keep_idx]
    
    noise_filtered_idx = []
    for i, term in enumerate(cand_terms):
        if not is_noise_phrase(term):
            noise_filtered_idx.append(i)
    
    if len(noise_filtered_idx) == 0:
        noise_filtered_idx = list(range(min(50, len(cand_terms))))
    
    cand_terms = cand_terms[noise_filtered_idx]
    cand_scores = cand_scores[noise_filtered_idx]
    normalized = [normalized[i] for i in noise_filtered_idx]
    
    best_for_norm = {}
    for raw, norm, score in zip(cand_terms, normalized, cand_scores):
        if norm not in best_for_norm or score > best_for_norm[norm][1]:
            best_for_norm[norm] = (raw, score)
    
    final_raws = [best_for_norm[n][0] for n in best_for_norm]
    final_scores = np.array([best_for_norm[n][1] for n in best_for_norm])
    
    if len(final_raws) == 0:
        return []
    
    semantic_scores = np.array([compute_semantic_score(term) for term in final_raws])
    if final_scores.max() > 0:
        normalized_tfidf = final_scores / final_scores.max()
    else:
        normalized_tfidf = final_scores
    
    if semantic_scores.max() > 0:
        normalized_semantic = semantic_scores / semantic_scores.max()
    else:
        normalized_semantic = semantic_scores
    
    combined_scores = 0.6 * normalized_tfidf + 0.4 * normalized_semantic
    
    phrase_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        norm="l2",
        sublinear_tf=True,
    )
    phrase_vectors = phrase_vec.fit_transform(final_raws)
    query_vector = phrase_vec.transform([" ".join(texts)])
    
    order = np.argsort(-combined_scores)
    sorted_terms = [final_raws[i] for i in order]
    sorted_vecs = phrase_vectors[order]
    
    selected = mmr_select(
        candidates=sorted_terms,
        cand_vectors=sorted_vecs,
        query_vec=query_vector,
        top_k=top_k,
        lambda_div=0.7,
    )

    def is_redundant_term(candidate: str, chosen: list) -> bool:
        cand_norm = normalize_phrase(candidate)
        cand_tokens = set(cand_norm.split())
        if not cand_tokens:
            return True

        for existing in chosen:
            existing_norm = normalize_phrase(existing)
            existing_tokens = set(existing_norm.split())
            if not existing_tokens:
                continue

            if cand_norm == existing_norm:
                return True
            if cand_norm in existing_norm or existing_norm in cand_norm:
                return True

            overlap = len(cand_tokens & existing_tokens) / max(1, min(len(cand_tokens), len(existing_tokens)))
            if overlap >= 0.8:
                return True

        return False

    candidate_pool = selected + [term for term in sorted_terms if term not in selected]
    pruned_selected = []
    for term in candidate_pool:
        if not is_redundant_term(term, pruned_selected):
            pruned_selected.append(term)
        if len(pruned_selected) >= top_k:
            break

    return pruned_selected
