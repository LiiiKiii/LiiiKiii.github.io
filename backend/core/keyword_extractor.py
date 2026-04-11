#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词提取模块
从文档中提取关键主题和关键词
"""

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def basic_clean(text: str) -> str:
    """基础文本清洗：小写、去HTML标签、去多余空白"""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F\'\"\"''',.:;!?()[\]{}<>~`•…–—/_+=*^%$#@\\|-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phrase(s: str) -> str:
    """规范化短语：连字符->空格、收尾清理、简单的复数去尾"""
    s = s.replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s)
    # 去掉极短词
    tokens = s.split()
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return ""
    # 粗略单复数合并（仅最后一个词）
    if len(tokens) >= 1:
        last = tokens[-1]
        if len(last) > 3 and last.endswith("s"):
            tokens[-1] = last[:-1]
    return " ".join(tokens)


def is_noise_phrase(phrase: str) -> bool:
    """
    判断短语是否为噪声（无关术语）
    返回True表示应该过滤掉
    """
    phrase_lower = phrase.lower().strip()
    tokens = phrase_lower.split()
    
    # 0. 检测重复词（如 "cid cid cid"）
    if len(tokens) >= 2:
        word_counts = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
        # 如果有词出现2次以上，很可能是噪声
        if any(count >= 2 for count in word_counts.values()):
            return True
    
    # 1. 包含网址相关（更严格的过滤）
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
        r'\.ac\.uk',  # 大学域名（如 durham.ac.uk）
        r'\.edu\.',   # .edu. 域名
        r'email',
        r'@',
        r'\.gov',
        r'\.mil',
        r'doi\s+org',  # DOI链接
        r'dx\s+doi',   # DOI链接变体
        r'doi\s+org',  # DOI链接
    ]
    for pattern in url_patterns:
        if re.search(pattern, phrase_lower):
            return True
    
    # 1.5. 引用格式相关（arxiv, preprint, et al等）
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
    
    # 1.6. 大学域名和机构名称（更全面的过滤）
    university_patterns = [
        r'\b\w+\.ac\.uk\b',  # 匹配 "durham.ac.uk", "oxford.ac.uk" 等
        r'\b\w+\.edu\b',     # 匹配 "mit.edu", "stanford.edu" 等
        r'\bdurham\s+(university|ac|uk)\b',
        r'\buniversity\s+of\s+\w+\s+(ac|uk|edu)\b',
        r'\bdepartment\s+of\s+[a-z\s]+\s+(university|ac|uk|edu)\b',
        r'\bfaculty\s+of\s+[a-z\s]+\s+(university|ac|uk|edu)\b',
        r'\bschool\s+of\s+[a-z\s]+\s+(university|ac|uk|edu)\b',
    ]
    for pattern in university_patterns:
        if re.search(pattern, phrase_lower):
            return True
    
    # 2. 包含电话号码格式（数字+常见电话词汇）
    if re.search(r'\d+.*(tel|phone|fax|mobile)', phrase_lower):
        return True
    if re.search(r'(tel|phone|fax|mobile).*\d+', phrase_lower):
        return True
    
    # 3. 地址相关词汇和机构信息（更严格的过滤）
    address_keywords = [
        'centre', 'center', 'street', 'road', 'avenue', 'lane',
        'building', 'floor', 'room', 'office', 'address',
        'postcode', 'zip', 'code', 'location',
        'durham', 'stockton', 'palatine',  # 常见地名
        'department', 'faculty', 'school', 'institute', 'college',  # 机构名称
        'campus', 'headquarters', 'head office',  # 办公地点
    ]
    # 如果短语中包含地址关键词，且同时包含机构相关词汇，很可能是机构地址
    address_count = sum(1 for kw in address_keywords if kw in phrase_lower)
    institutional_keywords = ['department', 'faculty', 'school', 'institute', 'college', 'university']
    has_institutional = any(kw in phrase_lower for kw in institutional_keywords)
    
    # 但如果是学术术语（如"center of mass", "school of thought"），保留
    academic_context = any(academic in phrase_lower for academic in [
        'mass', 'gravity', 'distribution', 'cluster', 'point',
        'matrix', 'vector', 'space', 'dimension',
        'thought', 'theory', 'method', 'approach', 'algorithm',
        'model', 'learning', 'network', 'data', 'analysis'
    ])
    
    # 如果包含机构关键词且没有学术上下文，很可能是机构信息
    if has_institutional and not academic_context:
        return True
    # 如果包含多个地址关键词且没有学术上下文，很可能是地址
    if address_count >= 2 and not academic_context:
        return True
    
    # 3.5. 地名+缩写模式（如 "newyork ny usa"）
    us_state_abbrevs = ['ny', 'ca', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi', 
                       'nj', 'va', 'wa', 'az', 'ma', 'tn', 'in', 'mo', 'md', 'wi',
                       'co', 'mn', 'sc', 'al', 'la', 'ky', 'or', 'ok', 'ct', 'ia',
                       'ut', 'ar', 'nv', 'ms', 'ks', 'nm', 'ne', 'wv', 'id', 'hi',
                       'nh', 'me', 'mt', 'ri', 'de', 'sd', 'nd', 'ak', 'dc', 'vt', 'wy']
    country_abbrevs = ['usa', 'uk', 'us', 'ca', 'au', 'de', 'fr', 'it', 'es', 'nl', 'be', 'ch', 'at', 'se', 'no', 'dk', 'fi', 'pl', 'cz', 'ie']
    
    # 检测地名+州/国家缩写模式
    if len(tokens) >= 2:
        has_place_name = any(len(t) > 3 for t in tokens)  # 至少有一个较长的词（可能是地名）
        has_abbrev = any(t in us_state_abbrevs or t in country_abbrevs for t in tokens)
        if has_place_name and has_abbrev:
            # 如果没有学术上下文，很可能是地址
            if not academic_context:
                return True
    
    # 4. 联系方式相关
    contact_keywords = [
        'telephone', 'phone', 'tel', 'fax', 'mobile',
        'contact', 'call', 'reach',
    ]
    # 学术相关关键词（即使包含联系方式词汇，如果是学术术语也保留）
    academic_keywords = [
        'classification', 'algorithm', 'model', 'data', 'analysis',
        'method', 'approach', 'technique', 'theory', 'concept',
        'learning', 'network', 'system', 'process', 'function',
        'matrix', 'vector', 'curve', 'score', 'metric', 'measure',
        'evaluation', 'performance', 'accuracy', 'precision', 'recall',
        'roc', 'auc', 'component', 'feature', 'sample', 'dataset'
    ]
    
    if any(kw in phrase_lower for kw in contact_keywords):
        # 如果包含联系方式词汇且看起来不像学术术语
        if not any(academic in phrase_lower for academic in academic_keywords):
            return True
    
    # 5. 纯数字或包含过多数字
    if re.search(r'\d{4,}', phrase_lower):  # 4位以上数字
        return True
    
    # 6. 包含特殊字符组合（可能是格式化的信息）
    if re.search(r'[a-z]+\s+\d+\s+[a-z]+', phrase_lower):  # 类似 "chapter 4 section"
        # 但保留学术相关的
        if not any(academic in phrase_lower for academic in [
            'chapter', 'section', 'figure', 'table', 'equation',
            'algorithm', 'method', 'model'
        ]):
            return True
    
    # 7. 过短或过长的短语（可能是噪声）
    if len(tokens) > 5:  # 超过5个词的短语通常是噪声
        return True
    
    # 8. 包含常见无意义词汇组合
    noise_patterns = [
        r'^\d+\s*$',  # 纯数字
        r'^[a-z]\s+[a-z]\s+[a-z]$',  # 三个单字母
        r'page\s+\d+',  # 页码
        r'figure\s+\d+',  # 图号（但保留figure本身）
    ]
    for pattern in noise_patterns:
        if re.match(pattern, phrase_lower):
            return True
    
    # 9. 机构联系信息模式（如 "Department of X, University of Y"）
    if re.search(r'(department|faculty|school|institute|college)\s+of\s+[^,]+,\s+(university|institute)', phrase_lower):
        return True
    
    # 10. 包含常见大学名称（如果只是地名+大学，很可能是机构信息）
    common_universities = ['durham', 'oxford', 'cambridge', 'harvard', 'mit', 'stanford', 'yale', 'princeton']
    if any(uni in phrase_lower for uni in common_universities):
        # 如果只是 "durham university" 或类似结构，且没有学术上下文，过滤掉
        if re.search(r'\b(durham|oxford|cambridge|harvard|mit|stanford|yale|princeton)\s+(university|college|institute)\b', phrase_lower):
            if not academic_context:
                return True
    
    # 11. 版权信息相关
    copyright_keywords = [
        'copyright', 'licensed', 'license', 'reserved', 'rights',
        'permission', 'reproduce', 'reproduction', 'prohibited',
        'limited use', 'use limited', 'all rights', 'rights reserved'
    ]
    if any(kw in phrase_lower for kw in copyright_keywords):
        return True
    
    # 12. 包含过多单字母或双字母词（可能是缩写或噪声）
    short_word_count = sum(1 for t in tokens if len(t) <= 2)
    if len(tokens) >= 2 and short_word_count >= len(tokens) * 0.5:  # 超过一半是短词
        # 但保留一些常见的学术缩写
        academic_abbrevs = ['ai', 'ml', 'dl', 'nlp', 'cv', 'cnn', 'rnn', 'lstm', 'gan', 'svm', 
                           'pca', 'ica', 'knn', 'rf', 'gbm', 'xgb', 'bert', 'gpt', 'api', 'url',
                           'http', 'html', 'xml', 'json', 'sql', 'db', 'id', 'ui', 'ux']
        if not any(t in academic_abbrevs for t in tokens):
            return True
    
    # 13. 检测"world wide web"等URL相关但可能被误识别为学术术语的短语
    url_like_phrases = [
        'world wide web', 'www', 'http', 'https', 'ftp', 'smtp'
    ]
    if phrase_lower in url_like_phrases:
        return True
    
    return False


def build_vectorizer():
    """构建TF-IDF向量化器"""
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
    """
    Maximal Marginal Relevance 选择：在与"全书向量"相似（代表性）的同时，
    保证彼此之间不太相似（多样性）。
    """
    selected = []
    if len(candidates) == 0:
        return selected

    # 与全书的相似度（代表性）
    rep = cosine_similarity(cand_vectors, query_vec).ravel()

    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        if not selected:
            # 首个选"最代表"的
            best = int(np.argmax(rep[remaining]))
            selected.append(remaining.pop(best))
        else:
            # 代表性与多样性加权
            max_score, max_idx = -1e9, -1
            for idx_pos, idx in enumerate(remaining):
                sim_to_query = rep[idx]
                # 与已选的最大相似度（要惩罚）
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
    """
    计算短语的语义重要性得分
    基于是否包含学术相关术语、技术术语等
    """
    phrase_lower = phrase.lower()
    score = 0.0
    
    # 学术和技术术语词典（按重要性加权）
    academic_terms = {
        # 机器学习核心概念
        'machine learning': 3.0, 'deep learning': 3.0, 'neural network': 3.0,
        'artificial intelligence': 3.0, 'ai': 2.5, 'ml': 2.0, 'dl': 2.0,
        # 模型和算法
        'neural networks': 2.5, 'deep neural': 2.5, 'convolutional': 2.0,
        'recurrent neural': 2.0, 'rnn': 2.0, 'lstm': 2.0, 'cnn': 2.0,
        'transformer': 2.5, 'attention mechanism': 2.5, 'bert': 2.0, 'gpt': 2.0,
        'generative model': 2.5, 'gan': 2.0, 'variational': 2.0,
        # 推荐系统
        'recommendation system': 2.5, 'content based': 2.5, 'collaborative filtering': 2.5,
        'recommender system': 2.5, 'content based filtering': 2.5,
        # 自然语言处理
        'natural language': 2.5, 'nlp': 2.0, 'language model': 2.5,
        'large language model': 3.0, 'llm': 2.5, 'text processing': 2.0,
        # 数据科学
        'data mining': 2.0, 'feature extraction': 2.0, 'dimensionality reduction': 2.0,
        'principal component': 2.0, 'pca': 1.5, 'clustering': 2.0, 'classification': 2.0,
        'regression': 2.0, 'supervised learning': 2.0, 'unsupervised learning': 2.0,
        # 统计和数学
        'probability': 1.5, 'statistical': 1.5, 'optimization': 1.5,
        'gradient descent': 2.0, 'backpropagation': 2.0, 'loss function': 2.0,
        # 计算机视觉
        'computer vision': 2.5, 'cv': 2.0, 'image processing': 2.0,
        'object detection': 2.0, 'semantic segmentation': 2.0,
        # 其他重要术语
        'algorithm': 1.5, 'method': 1.0, 'approach': 1.0, 'technique': 1.0,
        'framework': 1.5, 'architecture': 1.5, 'model': 1.5, 'system': 1.0,
        'training': 1.5, 'evaluation': 1.5, 'performance': 1.0, 'accuracy': 1.0,
    }
    
    # 检查是否包含学术术语
    for term, weight in academic_terms.items():
        if term in phrase_lower:
            score += weight
            break  # 只匹配一次，避免重复加分
    
    # 如果包含多个学术关键词，额外加分
    academic_keywords = ['learning', 'network', 'model', 'algorithm', 'method', 
                        'data', 'feature', 'training', 'neural', 'deep']
    keyword_count = sum(1 for kw in academic_keywords if kw in phrase_lower)
    if keyword_count >= 2:
        score += 0.5
    
    # 如果是短语（包含空格），通常更有意义
    if ' ' in phrase:
        score += 0.3
    
    # 如果长度适中（2-4个词），通常更有意义
    word_count = len(phrase.split())
    if 2 <= word_count <= 4:
        score += 0.2
    
    return score


def extract_keywords_from_folder(folder_path: str, top_k: int = 10, min_docs: int = 3) -> list:
    """
    从文件夹中提取关键词/主题
    
    Args:
        folder_path: 包含txt文件的文件夹路径（包括PDF转换后的txt）
        top_k: 提取的关键词数量
        min_docs: 关键词至少出现在多少个文档中
    
    Returns:
        关键词列表
    """
    # 获取所有txt文件路径（包括PDF转换后的txt）
    # 排除macOS系统文件（以._开头的资源分叉文件）和其他隐藏文件
    txt_paths = []
    if os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for fname in files:
                # 过滤掉macOS资源分叉文件（以._开头）和其他系统隐藏文件
                if fname.startswith('._') or fname.startswith('.DS_Store'):
                    continue
                if fname.lower().endswith(".txt"):
                    txt_paths.append(os.path.join(root, fname))
    
    txt_paths = sorted(txt_paths)
    
    if len(txt_paths) < 2:
        raise ValueError("需要至少2个txt文档才能提取关键词")
    
    # 读取和清洗文档
    texts = []
    for p in txt_paths:
        t = basic_clean(read_file(p))
        texts.append(t)
    
    n_docs = len(texts)
    if n_docs < 2:
        raise ValueError("需要至少2个文档")
    
    # 向量化（章节为文档）
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    
    # 计算每个短语的"全书得分"和覆盖度
    tfidf_sum = X.sum(axis=0).A1
    doc_freq = (X > 0).sum(axis=0).A1
    tfidf_mean = tfidf_sum / np.maximum(doc_freq, 1)
    
    # 覆盖度过滤
    mask_coverage = doc_freq >= max(1, min_docs)
    
    # 短语长度过滤（优先选择短语，但也不完全排除单词）
    is_phrase_like = np.array([" " in term or "-" in term for term in vocab])
    
    # 候选集合：提高阈值，更严格筛选
    mean_threshold = np.quantile(tfidf_mean[mask_coverage], 0.85) if np.any(mask_coverage) else 0.0
    # 优先选择短语，或者TF-IDF得分较高的单词
    candidate_mask = mask_coverage & (is_phrase_like | (tfidf_mean >= mean_threshold * 1.2))
    
    cand_terms = vocab[candidate_mask]
    cand_scores = tfidf_mean[candidate_mask]
    
    # 规范化短语、去重
    normalized = [normalize_phrase(t) for t in cand_terms]
    keep_idx = [i for i, s in enumerate(normalized) if s]
    cand_terms = cand_terms[keep_idx]
    cand_scores = cand_scores[keep_idx]
    normalized = [normalized[i] for i in keep_idx]
    
    # 过滤噪声短语（地址、联系方式、网址等）
    noise_filtered_idx = []
    for i, term in enumerate(cand_terms):
        if not is_noise_phrase(term):
            noise_filtered_idx.append(i)
    
    if len(noise_filtered_idx) == 0:
        # 如果全部被过滤，至少保留一些
        noise_filtered_idx = list(range(min(50, len(cand_terms))))
    
    cand_terms = cand_terms[noise_filtered_idx]
    cand_scores = cand_scores[noise_filtered_idx]
    normalized = [normalized[i] for i in noise_filtered_idx]
    
    # 合并同义/变体
    best_for_norm = {}
    for raw, norm, score in zip(cand_terms, normalized, cand_scores):
        if norm not in best_for_norm or score > best_for_norm[norm][1]:
            best_for_norm[norm] = (raw, score)
    
    final_raws = [best_for_norm[n][0] for n in best_for_norm]
    final_scores = np.array([best_for_norm[n][1] for n in best_for_norm])
    
    if len(final_raws) == 0:
        return []
    
    # 计算语义得分并综合TF-IDF得分
    semantic_scores = np.array([compute_semantic_score(term) for term in final_raws])
    # 归一化TF-IDF得分和语义得分
    if final_scores.max() > 0:
        normalized_tfidf = final_scores / final_scores.max()
    else:
        normalized_tfidf = final_scores
    
    if semantic_scores.max() > 0:
        normalized_semantic = semantic_scores / semantic_scores.max()
    else:
        normalized_semantic = semantic_scores
    
    # 综合得分：TF-IDF权重0.6，语义得分权重0.4
    combined_scores = 0.6 * normalized_tfidf + 0.4 * normalized_semantic
    
    # 用词袋向量把"短语文本"向量化（用于MMR多样性）
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
    
    # 先按综合得分排序，再用MMR挑选多样化 Top-K
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

            # 完全相同或明显的包含关系，视为冗余
            if cand_norm == existing_norm:
                return True
            if cand_norm in existing_norm or existing_norm in cand_norm:
                return True

            # 如果较短短语的大部分token都被已有短语覆盖，也视为冗余
            overlap = len(cand_tokens & existing_tokens) / max(1, min(len(cand_tokens), len(existing_tokens)))
            if overlap >= 0.8:
                return True

        return False

    # MMR之后再做一轮冗余短语清理，避免出现“natural language processing”
    # 与“natural language”“language processing”同时入选。
    candidate_pool = selected + [term for term in sorted_terms if term not in selected]
    pruned_selected = []
    for term in candidate_pool:
        if not is_redundant_term(term, pruned_selected):
            pruned_selected.append(term)
        if len(pruned_selected) >= top_k:
            break

    return pruned_selected

