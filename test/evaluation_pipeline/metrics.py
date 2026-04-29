#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia Evaluation Metrics

Implements quantitative metrics for evaluating the pipeline:
- Keyword quality: coverage, diversity, AI relevance
- Resource quality: AI relevance, noise reduction, authority, valid URLs
- Resource mix: cross-platform diversity, per-type counts
"""

import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_AI_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "algorithm",
    "transformer", "attention", "cnn", "rnn", "lstm", "gradient",
    "classification", "regression", "supervised", "unsupervised",
    "reinforcement learning", "nlp", "computer vision", "ml", "dl",
    "ai", "artificial intelligence", "tensorflow", "pytorch", "keras",
    "backpropagation", "optimizer", "loss function", "embedding",
    "diffusion", "llm", "large language model", "retrieval",
    "recommendation", "content based filtering", "tf idf", "cosine similarity",
]


class EvaluationMetrics:
    """Calculate evaluation metrics for AI-Pedia pipeline."""

    AUTHORITY_SOURCES = [
        "arxiv.org", "wikipedia.org", "scholar.google.com",
        "github.com", "pytorch.org", "tensorflow.org",
        "keras.io", "scikit-learn.org", "jupyter.org",
        "huggingface.co", "colab.research.google.com",
    ]

    SOURCE_TYPES = {
        "text": ["wikipedia.org", "arxiv.org", "scholar.google.com", "researchgate.net"],
        "video": ["youtube.com", "youtu.be", "bilibili.com"],
        "code": ["github.com", "gitlab.com", "bitbucket.org", "colab.research.google.com"],
    }

    def __init__(self, ai_keywords: List[str] = None):
        self.ai_keywords = [k.lower() for k in (ai_keywords or DEFAULT_AI_KEYWORDS)]

    def _resource_text(self, resource: Dict[str, Any]) -> str:
        title = resource.get("title", "")
        description = resource.get("description", "")
        url = resource.get("url", "")
        content = resource.get("content", "")
        source = resource.get("source", "")
        text = f"{title} {description} {url} {content} {source}".lower()
        text = re.sub(r"[-_/]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def flatten_resources(self, resource_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for resource_type, items in (resource_groups or {}).items():
            for item in items or []:
                resource = dict(item)
                resource.setdefault("resource_type", resource_type)
                flattened.append(resource)
        return flattened

    def count_resource_types(self, resources: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter(resource.get("resource_type", "unknown") for resource in resources)
        return dict(counts)

    def calculate_ai_relevance(self, resources: List[Dict[str, Any]]) -> float:
        if not resources:
            return 0.0
        relevant_count = sum(
            1 for resource in resources
            if any(keyword in self._resource_text(resource) for keyword in self.ai_keywords)
        )
        return round((relevant_count / len(resources)) * 100, 2)

    def calculate_input_relevance(
        self,
        original_corpus: List[str],
        resources: List[Dict[str, Any]],
        similarity_threshold: float = 0.08,
    ) -> Dict[str, float]:
        """
        Measure topic alignment to learner input corpus.

        A resource is counted as input-relevant when cosine similarity between
        the merged learner corpus and the resource text exceeds a threshold.
        """
        if not original_corpus or not resources:
            return {
                "input_relevance": 0.0,
                "avg_similarity": 0.0,
                "threshold": similarity_threshold,
            }

        corpus_text = " ".join((doc or "").strip() for doc in original_corpus if doc and doc.strip())
        resource_texts = [self._resource_text(resource) for resource in resources]
        resource_texts = [text for text in resource_texts if text]
        if not corpus_text or not resource_texts:
            return {
                "input_relevance": 0.0,
                "avg_similarity": 0.0,
                "threshold": similarity_threshold,
            }

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=1,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
                norm="l2",
            )
            matrix = vectorizer.fit_transform([corpus_text] + resource_texts)
            similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
            if similarities.size == 0:
                return {
                    "input_relevance": 0.0,
                    "avg_similarity": 0.0,
                    "threshold": similarity_threshold,
                }

            relevant_count = int(np.sum(similarities >= similarity_threshold))
            input_relevance = round((relevant_count / len(similarities)) * 100, 2)
            avg_similarity = round(float(np.mean(similarities)), 4)
            return {
                "input_relevance": input_relevance,
                "avg_similarity": avg_similarity,
                "threshold": similarity_threshold,
            }
        except Exception:
            return {
                "input_relevance": 0.0,
                "avg_similarity": 0.0,
                "threshold": similarity_threshold,
            }

    def calculate_noise_reduction(self, initial_count: int, final_count: int) -> float:
        if initial_count <= 0:
            return 0.0
        return round(((initial_count - final_count) / initial_count) * 100, 2)

    def calculate_cross_platform_diversity(self, resources: List[Dict[str, Any]]) -> int:
        source_types_found = set()
        for resource in resources:
            url = resource.get("url", "").lower()
            for source_type, patterns in self.SOURCE_TYPES.items():
                if any(pattern in url for pattern in patterns):
                    source_types_found.add(source_type)
        return len(source_types_found)

    def calculate_authority_score(self, resources: List[Dict[str, Any]]) -> float:
        if not resources:
            return 0.0
        authority_count = sum(
            1 for resource in resources
            if any(source in resource.get("url", "").lower() for source in self.AUTHORITY_SOURCES)
        )
        return round((authority_count / len(resources)) * 100, 2)

    def calculate_novelty(self, original_corpus: List[str], resources: List[Dict[str, Any]]) -> float:
        if not resources:
            return 0.0

        corpus_keywords = set()
        for doc in original_corpus:
            corpus_keywords.update(re.findall(r"\b\w+\b", doc.lower()))

        novel_count = 0
        for resource in resources:
            resource_words = set(re.findall(r"\b\w+\b", self._resource_text(resource)))
            if not resource_words:
                continue
            new_words = resource_words - corpus_keywords
            if len(new_words) > len(resource_words) * 0.3:
                novel_count += 1

        return round((novel_count / len(resources)) * 100, 2)

    def validate_urls(self, resources: List[Dict[str, Any]]) -> Dict[str, float]:
        valid_count = 0
        invalid_count = 0
        for resource in resources:
            url = resource.get("url", "")
            if url and (url.startswith("http://") or url.startswith("https://")):
                valid_count += 1
            else:
                invalid_count += 1

        total = valid_count + invalid_count
        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "valid_percentage": round((valid_count / total * 100), 2) if total > 0 else 0.0,
        }

    def calculate_keyword_coverage(self, documents: List[str], keywords: List[str]) -> float:
        if not documents:
            return 0.0
        normalized_keywords = [keyword.lower().strip() for keyword in keywords if keyword and keyword.strip()]
        if not normalized_keywords:
            return 0.0

        represented_docs = 0
        for doc in documents:
            doc_lower = doc.lower()
            if any(keyword in doc_lower for keyword in normalized_keywords):
                represented_docs += 1

        return round((represented_docs / len(documents)) * 100, 2)

    def calculate_keyword_ai_relevance(self, keywords: List[str]) -> float:
        if not keywords:
            return 0.0

        strong_single_terms = {
            "ai", "ml", "dl", "nlp", "cnn", "rnn", "lstm",
            "neural", "transformer", "attention", "gradient",
            "backpropagation", "reinforcement", "pytorch", "tensorflow", "keras",
        }

        relevant = 0
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            tokens = set(keyword_lower.split())

            phrase_match = any(
                ai_keyword in keyword_lower
                for ai_keyword in self.ai_keywords
                if len(ai_keyword.split()) >= 2
            )
            strong_token_match = any(token in strong_single_terms for token in tokens)

            if phrase_match or strong_token_match:
                relevant += 1

        return round((relevant / len(keywords)) * 100, 2)

    def calculate_keyword_diversity(self, keywords: List[str]) -> float:
        normalized = [keyword.strip().lower() for keyword in keywords if keyword and keyword.strip()]
        if len(normalized) <= 1:
            return 1.0 if normalized else 0.0

        try:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            matrix = vectorizer.fit_transform(normalized)
            similarity = cosine_similarity(matrix)
            upper = similarity[np.triu_indices_from(similarity, k=1)]
            if upper.size == 0:
                return 1.0
            diversity = 1.0 - float(np.mean(upper))
            return round(max(0.0, min(1.0, diversity)), 4)
        except Exception:
            return 0.0

    def generate_keyword_report(self, documents: List[str], keywords: List[str]) -> Dict[str, Any]:
        return {
            "keyword_count": len(keywords),
            "coverage": self.calculate_keyword_coverage(documents, keywords),
            "ai_relevance": self.calculate_keyword_ai_relevance(keywords),
            "diversity": self.calculate_keyword_diversity(keywords),
            "keywords": keywords,
        }

    def generate_resource_report(
        self,
        resources: List[Dict[str, Any]],
        initial_count: int = 0,
        original_corpus: List[str] = None,
    ) -> Dict[str, Any]:
        report = {
            "total_resources": len(resources),
            "ai_relevance": self.calculate_ai_relevance(resources),
            "cross_platform_diversity": self.calculate_cross_platform_diversity(resources),
            "authority_score": self.calculate_authority_score(resources),
            "resource_type_counts": self.count_resource_types(resources),
            "url_validation": self.validate_urls(resources),
        }

        if initial_count > 0:
            report["noise_reduction"] = self.calculate_noise_reduction(initial_count, len(resources))

        if original_corpus:
            report["novelty"] = self.calculate_novelty(original_corpus, resources)
            input_relevance = self.calculate_input_relevance(original_corpus, resources)
            report["input_relevance"] = input_relevance["input_relevance"]
            report["avg_input_similarity"] = input_relevance["avg_similarity"]
            report["input_relevance_threshold"] = input_relevance["threshold"]

        return report
