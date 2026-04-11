#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance tests for AI-Pedia core local pipeline stages.

Outputs timing and optional CPU/memory usage for:
- keyword extraction
- ranking / recommendation
- a combined local pipeline

Run from project root:
    python3 test/performance/run_performance_tests.py
"""

import os
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_pythonpath() -> None:
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


setup_pythonpath()

from backend.core.keyword_extractor import extract_keywords_from_folder
from backend.core.recommender import recommend_best_resources


SAMPLE_DOCS = [
    "Machine learning focuses on models, optimization, data, and generalization. Neural networks learn layered representations from examples.",
    "Deep learning systems use backpropagation, gradient descent, transformers, attention, and embeddings for modern AI tasks.",
    "Computer vision studies convolutional neural networks, image classification, object detection, and representation learning.",
    "Natural language processing uses transformers, tokenization, language models, retrieval, and sequence learning.",
    "Reinforcement learning optimizes policies with rewards, value functions, exploration, and environment feedback.",
]


def make_temp_corpus() -> str:
    temp_dir = tempfile.mkdtemp(prefix="ai_pedia_perf_")
    for idx, text in enumerate(SAMPLE_DOCS, start=1):
        path = os.path.join(temp_dir, f"doc_{idx}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return temp_dir


def sample_resources() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "txt": [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "content": "Transformers rely on self-attention and sequence modeling for NLP tasks.",
                "source": "arXiv",
            },
            {
                "title": "Neural network tutorial",
                "url": "https://wikipedia.org/wiki/Artificial_neural_network",
                "content": "Artificial neural networks are learning systems inspired by biological neurons.",
                "source": "Wikipedia",
            },
        ],
        "video": [
            {
                "title": "Transformer basics",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "description": "Educational overview of transformers, attention, and NLP foundations.",
                "source": "YouTube",
            },
            {
                "title": "CNN explanation",
                "url": "https://www.youtube.com/watch?v=aircAruvnKk",
                "description": "Explains convolutional neural networks and image classification.",
                "source": "YouTube",
            },
        ],
        "code": [
            {
                "title": "TensorFlow examples",
                "url": "https://github.com/tensorflow/examples",
                "description": "Example implementations for neural networks, transformers, and vision models.",
                "source": "GitHub",
            },
            {
                "title": "PyTorch tutorials",
                "url": "https://github.com/pytorch/tutorials",
                "description": "PyTorch tutorials covering deep learning, NLP, and reinforcement learning.",
                "source": "GitHub",
            },
        ],
    }


def measure_with_resources(func: Callable[[], Any], repeat: int = 3) -> Dict[str, Any]:
    times: List[float] = []
    all_cpu: List[float] = []
    all_mem: List[float] = []

    for _ in range(repeat):
        start = time.perf_counter()
        if _HAS_PSUTIL:
            proc = psutil.Process(os.getpid())
            cpu_samples: List[float] = []
            mem_samples: List[float] = []
            end_evt = threading.Event()

            def sampler() -> None:
                while not end_evt.is_set():
                    try:
                        cpu_samples.append(proc.cpu_percent())
                        mem_samples.append(proc.memory_info().rss / (1024 * 1024))
                    except Exception:
                        break
                    time.sleep(0.05)

            thread = threading.Thread(target=sampler, daemon=True)
            thread.start()

        try:
            func()
        finally:
            if _HAS_PSUTIL:
                end_evt.set()
                time.sleep(0.1)
                all_cpu.extend(cpu_samples)
                all_mem.extend(mem_samples)

        times.append(time.perf_counter() - start)

    total = sum(times)
    avg = total / len(times) if times else 0.0
    result: Dict[str, Any] = {
        "runs": repeat,
        "total_time_s": round(total, 4),
        "avg_time_s": round(avg, 4),
        "t_full_s": round(avg, 4),
    }

    if _HAS_PSUTIL and all_cpu and all_mem:
        result["cpu_avg_pct"] = round(sum(all_cpu) / len(all_cpu), 2)
        result["cpu_peak_pct"] = round(max(all_cpu), 2)
        result["mem_peak_mb"] = round(max(all_mem), 2)
    else:
        result["cpu_avg_pct"] = None
        result["cpu_peak_pct"] = None
        result["mem_peak_mb"] = None

    return result


def test_keyword_extraction(corpus_dir: str) -> Dict[str, Any]:
    result = measure_with_resources(lambda: extract_keywords_from_folder(corpus_dir, top_k=8), repeat=3)
    result["module"] = "keyword_extraction"
    result["status"] = "ok"
    return result


def test_ranking_pipeline(corpus_dir: str) -> Dict[str, Any]:
    resources = sample_resources()
    result = measure_with_resources(
        lambda: recommend_best_resources(corpus_dir, resources, top_k_per_type=2),
        repeat=3,
    )
    result["module"] = "ranking_pipeline"
    result["status"] = "ok"
    return result


def test_local_pipeline(corpus_dir: str) -> Dict[str, Any]:
    resources = sample_resources()

    def _run() -> None:
        extract_keywords_from_folder(corpus_dir, top_k=8)
        recommend_best_resources(corpus_dir, resources, top_k_per_type=2)

    result = measure_with_resources(_run, repeat=3)
    result["module"] = "local_pipeline"
    result["status"] = "ok"
    result["note"] = "Measures deterministic local stages only, excluding external web search latency."
    return result


def main() -> None:
    print("=== 性能测试（AI-Pedia 本地核心阶段）===")
    if not _HAS_PSUTIL:
        print("(未安装 psutil，仅输出耗时；安装后可得到 CPU/内存: pip install psutil)")

    corpus_dir = make_temp_corpus()
    results: List[Dict[str, Any]] = []

    try:
        for runner in (test_keyword_extraction, test_ranking_pipeline, test_local_pipeline):
            try:
                result = runner(corpus_dir)
                results.append(result)
                print(f"\n--- {result['module']} ---")
                print(result)
            except Exception as exc:
                failure = {
                    "module": runner.__name__,
                    "status": "error",
                    "error": str(exc),
                }
                results.append(failure)
                print(f"\n[ERROR] {runner.__name__}: {exc}")
    finally:
        import shutil
        shutil.rmtree(corpus_dir, ignore_errors=True)

    print("\n=== 性能测试结束 ===")
    print("建议在论文中分别报告 keyword_extraction、ranking_pipeline、local_pipeline 三组结果。")


if __name__ == "__main__":
    main()
