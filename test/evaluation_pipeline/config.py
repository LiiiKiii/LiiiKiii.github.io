#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia Evaluation Configuration

Manages API keys and configuration via environment variables for security.
"""

import os
from typing import Optional


class EvalConfig:
    """Evaluation configuration managed via environment variables."""

    # OpenAI API Key (from environment variable for security)
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")

    # Evaluation settings
    DEFAULT_FOUNDATION_MODEL = "gpt-3.5-turbo"
    TEST_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_corpus")
    TEST_CORPORA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_corpora")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

    # Evaluation metrics
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
        if cls.OPENAI_API_KEY is None:
            print("⚠️  Warning: OPENAI_API_KEY not set. LLM features will use fallback.")
        return True

    @classmethod
    def get_model(cls, model_name: Optional[str] = None) -> str:
        """Get the foundation model to use for evaluation."""
        return model_name or cls.DEFAULT_FOUNDATION_MODEL


def load_config() -> EvalConfig:
    """Load and validate evaluation configuration."""
    config = EvalConfig()
    config.validate()
    return config
