#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia Evaluation Metrics

Implements quantitative metrics for evaluating the pipeline:
- AI Relevance: percentage of AI/ML-related resources
- Noise Reduction: percentage of irrelevant resources filtered
- Cross-Platform Diversity: number of different source types
- Novelty: percentage of resources beyond original corpus
- Authority Score: percentage from trusted sources
- Valid URLs: percentage of accessible links
"""

import re
from typing import List, Dict, Any
from collections import Counter


class EvaluationMetrics:
    """Calculate evaluation metrics for AI-Pedia pipeline."""

    # Trusted authority sources
    AUTHORITY_SOURCES = [
        "arxiv.org", "wikipedia.org", "scholar.google.com",
        "github.com", "pytorch.org", "tensorflow.org",
        "keras.io", "scikit-learn.org", "jupyter.org"
    ]

    # Resource type patterns
    SOURCE_TYPES = {
        "text": ["wikipedia.org", "arxiv.org", "scholar.google.com", "researchgate.net"],
        "video": ["youtube.com", "youtu.be", "bilibili.com"],
        "code": ["github.com", "gitlab.com", "bitbucket.org", "colab.research.google.com"]
    }

    def __init__(self, ai_keywords: List[str] = None):
        """Initialize with AI domain keywords."""
        self.ai_keywords = ai_keywords or []

    def calculate_ai_relevance(self, resources: List[Dict]) -> float:
        """Calculate percentage of AI/ML-related resources."""
        if not resources:
            return 0.0

        relevant_count = 0
        for resource in resources:
            title = resource.get("title", "").lower()
            description = resource.get("description", "").lower()
            url = resource.get("url", "").lower()
            content = f"{title} {description} {url}"

            if any(keyword in content for keyword in self.ai_keywords):
                relevant_count += 1

        return (relevant_count / len(resources)) * 100

    def calculate_noise_reduction(self, initial_count: int, final_count: int) -> float:
        """Calculate percentage of noise filtered out."""
        if initial_count == 0:
            return 0.0
        return ((initial_count - final_count) / initial_count) * 100

    def calculate_cross_platform_diversity(self, resources: List[Dict]) -> int:
        """Calculate number of different source platform types."""
        source_types_found = set()

        for resource in resources:
            url = resource.get("url", "").lower()
            for source_type, patterns in self.SOURCE_TYPES.items():
                if any(pattern in url for pattern in patterns):
                    source_types_found.add(source_type)

        return len(source_types_found)

    def calculate_authority_score(self, resources: List[Dict]) -> float:
        """Calculate percentage of resources from trusted sources."""
        if not resources:
            return 0.0

        authority_count = 0
        for resource in resources:
            url = resource.get("url", "").lower()
            if any(source in url for source in self.AUTHORITY_SOURCES):
                authority_count += 1

        return (authority_count / len(resources)) * 100

    def calculate_novelty(self, original_corpus: List[str], resources: List[Dict]) -> float:
        """Calculate percentage of resources beyond original corpus."""
        if not resources:
            return 0.0

        # Create a set of keywords from original corpus
        corpus_keywords = set()
        for doc in original_corpus:
            words = re.findall(r'\b\w+\b', doc.lower())
            corpus_keywords.update(words)

        novel_count = 0
        for resource in resources:
            title = resource.get("title", "").lower()
            url = resource.get("url", "").lower()
            resource_text = f"{title} {url}"

            # Check if resource introduces new concepts
            resource_words = set(re.findall(r'\b\w+\b', resource_text))
            new_words = resource_words - corpus_keywords

            # If resource has significant new content, count as novel
            if len(new_words) > len(resource_words) * 0.3:
                novel_count += 1

        return (novel_count / len(resources)) * 100

    def validate_urls(self, resources: List[Dict]) -> Dict[str, int]:
        """Validate URL accessibility (basic check)."""
        valid_count = 0
        invalid_count = 0

        for resource in resources:
            url = resource.get("url", "")
            # Basic URL format validation
            if url and (url.startswith("http://") or url.startswith("https://")):
                valid_count += 1
            else:
                invalid_count += 1

        total = valid_count + invalid_count
        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "valid_percentage": (valid_count / total * 100) if total > 0 else 0
        }

    def generate_report(self, resources: List[Dict], initial_count: int = 0,
                       original_corpus: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            "total_resources": len(resources),
            "ai_relevance": self.calculate_ai_relevance(resources),
            "cross_platform_diversity": self.calculate_cross_platform_diversity(resources),
            "authority_score": self.calculate_authority_score(resources),
            "url_validation": self.validate_urls(resources),
        }

        # Add noise reduction if initial count provided
        if initial_count > 0:
            report["noise_reduction"] = self.calculate_noise_reduction(
                initial_count, len(resources)
            )

        # Add novelty if corpus provided
        if original_corpus:
            report["novelty"] = self.calculate_novelty(original_corpus, resources)

        return report


# Example usage
if __name__ == "__main__":
    # Example resources
    sample_resources = [
        {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762",
         "description": "Transformer architecture paper"},
        {"title": "Neural Networks Playlist", "url": "https://youtube.com/watch?v=aircAruvnKk",
         "description": "3Blue1Brown neural networks"},
        {"title": "TensorFlow Examples", "url": "https://github.com/tensorflow/examples",
         "description": "TensorFlow tutorial code"},
    ]

    # Calculate metrics
    metrics = EvaluationMetrics()
    report = metrics.generate_report(sample_resources, initial_count=100)

    print("📊 Evaluation Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")