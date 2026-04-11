#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia Evaluation Configuration

Centralises filesystem paths and metric configuration for the evaluation pipeline.
"""

import os


class EvalConfig:
    """Evaluation configuration for reproducible local runs."""

    TEST_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_corpus")
    TEST_CORPORA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_corpora")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

    AI_RELEVANCE_KEYWORDS = [
        "machine learning", "deep learning", "neural network", "algorithm",
        "transformer", "attention", "cnn", "rnn", "lstm", "gradient",
        "classification", "regression", "supervised", "unsupervised",
        "reinforcement learning", "nlp", "computer vision", "ml", "dl",
        "ai", "artificial intelligence", "tensorflow", "pytorch", "keras",
    ]

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        return True


def load_config() -> EvalConfig:
    """Load and validate evaluation configuration."""
    config = EvalConfig()
    config.validate()
    return config
