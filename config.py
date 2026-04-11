# -*- coding: utf-8 -*-
"""Project-wide paths and tunables (single source of truth for app + tests)."""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MiB
MIN_VALID_DOCUMENTS = 10

# Recommendation / search pipeline
KEYWORD_TOP_K = 10
SEARCH_MAX_PER_TYPE = 20
RECOMMEND_TOP_K_PER_TYPE = 20
