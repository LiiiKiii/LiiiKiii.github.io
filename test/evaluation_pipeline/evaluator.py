#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia evaluation runner.

This script supports two evaluation modes:
1. single-corpus evaluation over one folder of TXT notes
2. focused multi-corpus evaluation over a root containing multiple 10+ document corpora

Outputs JSON, CSV, LaTeX tables, and paper-ready chart assets.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, PROJECT_ROOT)

from backend.core.keyword_extractor import extract_keywords_from_folder
from backend.core.recommender import recommend_best_resources
from backend.core.resource_searcher import search_all_resources
from config import EvalConfig, load_config
from metrics import EvaluationMetrics


class EvaluationRunner:
    """Run practical, code-backed evaluation for AI-Pedia."""

    def __init__(self):
        self.config = load_config()
        self.metrics = EvaluationMetrics(EvalConfig.AI_RELEVANCE_KEYWORDS)

    def load_documents(self, corpus_path: str) -> List[str]:
        documents: List[str] = []
        for root, _, files in os.walk(corpus_path):
            for name in sorted(files):
                if not name.lower().endswith(".txt") or name.startswith("._") or name.startswith(".DS_Store"):
                    continue
                file_path = os.path.join(root, name)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read().strip()
                    if content:
                        documents.append(content)
        return documents

    def list_corpora(self, corpora_root: str) -> List[Tuple[str, str]]:
        corpora: List[Tuple[str, str]] = []
        if not os.path.isdir(corpora_root):
            return corpora
        for name in sorted(os.listdir(corpora_root)):
            path = os.path.join(corpora_root, name)
            if not os.path.isdir(path):
                continue
            txt_files = [f for f in os.listdir(path) if f.lower().endswith('.txt')]
            if txt_files:
                corpora.append((name, path))
        return corpora

    def extract_simple_keywords(self, documents: List[str], top_k: int = 10) -> List[str]:
        import re

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "just", "also", "very", "too",
            "quite", "rather", "this", "that", "these", "those", "it",
        }

        word_counts = Counter()
        for doc in documents:
            words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", doc.lower())
            words = [word for word in words if word not in stop_words]
            word_counts.update(words)

        return [word for word, _ in word_counts.most_common(top_k)]

    def take_first_k_per_type(self, all_resources: Dict[str, List[Dict[str, Any]]], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        return {resource_type: list(items[:top_k]) for resource_type, items in all_resources.items()}

    def save_json(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_csv(self, path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _summary_view(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return results.get("aggregate", results)

    def _evaluate_corpus(
        self,
        corpus_path: str,
        cache_path: str,
        top_k_keywords: int,
        search_max_per_type: int,
        recommend_top_k: int,
        reuse_cache: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        documents = self.load_documents(corpus_path)
        if len(documents) < 2:
            raise ValueError(f"Evaluation needs at least 2 TXT documents in {corpus_path}.")

        simple_keywords = self.extract_simple_keywords(documents, top_k=top_k_keywords)
        tfidf_keywords = extract_keywords_from_folder(corpus_path, top_k=top_k_keywords)

        keyword_reports = {
            "simple_frequency": self.metrics.generate_keyword_report(documents, simple_keywords),
            "tfidf_mmr": self.metrics.generate_keyword_report(documents, tfidf_keywords),
        }

        if reuse_cache and os.path.exists(cache_path):
            raw_resources = self.load_json(cache_path)
        else:
            raw_resources = search_all_resources(tfidf_keywords, max_per_type=search_max_per_type)
            self.save_json(cache_path, raw_resources)

        raw_flat = self.metrics.flatten_resources(raw_resources)
        topk_unranked = self.take_first_k_per_type(raw_resources, recommend_top_k)
        topk_unranked_flat = self.metrics.flatten_resources(topk_unranked)
        ranked_resources = recommend_best_resources(corpus_path, raw_resources, top_k_per_type=recommend_top_k)
        ranked_flat = self.metrics.flatten_resources(ranked_resources)

        resource_reports = {
            "raw_search": self.metrics.generate_resource_report(raw_flat, original_corpus=documents),
            "topk_unranked": self.metrics.generate_resource_report(
                topk_unranked_flat,
                initial_count=len(raw_flat),
                original_corpus=documents,
            ),
            "topk_ranked": self.metrics.generate_resource_report(
                ranked_flat,
                initial_count=len(raw_flat),
                original_corpus=documents,
            ),
        }

        results = {
            "corpus": {
                "path": corpus_path,
                "document_count": len(documents),
                "word_count_total": sum(len(doc.split()) for doc in documents),
            },
            "parameters": {
                "top_k_keywords": top_k_keywords,
                "search_max_per_type": search_max_per_type,
                "recommend_top_k": recommend_top_k,
            },
            "keyword_reports": keyword_reports,
            "resource_reports": resource_reports,
            "resource_type_counts": {
                "raw_search": dict(Counter(resource.get("resource_type", "unknown") for resource in raw_flat)),
                "topk_unranked": dict(Counter(resource.get("resource_type", "unknown") for resource in topk_unranked_flat)),
                "topk_ranked": dict(Counter(resource.get("resource_type", "unknown") for resource in ranked_flat)),
            },
        }
        results["improvements"] = {
            "keyword_coverage_delta": round(
                keyword_reports["tfidf_mmr"]["coverage"] - keyword_reports["simple_frequency"]["coverage"],
                2,
            ),
            "keyword_ai_relevance_delta": round(
                keyword_reports["tfidf_mmr"]["ai_relevance"] - keyword_reports["simple_frequency"]["ai_relevance"],
                2,
            ),
            "keyword_diversity_delta": round(
                keyword_reports["tfidf_mmr"]["diversity"] - keyword_reports["simple_frequency"]["diversity"],
                4,
            ),
            "ranking_ai_relevance_delta": round(
                resource_reports["topk_ranked"]["ai_relevance"] - resource_reports["topk_unranked"]["ai_relevance"],
                2,
            ),
            "ranking_authority_delta": round(
                resource_reports["topk_ranked"]["authority_score"] - resource_reports["topk_unranked"]["authority_score"],
                2,
            ),
            "ranking_noise_reduction": resource_reports["topk_ranked"].get("noise_reduction", 0.0),
        }

        aux = {
            "documents": documents,
            "raw_flat": raw_flat,
            "unranked_flat": topk_unranked_flat,
            "ranked_flat": ranked_flat,
        }
        return results, aux

    def run_single(
        self,
        corpus_path: str,
        output_dir: str,
        top_k_keywords: int = 10,
        search_max_per_type: int = 8,
        recommend_top_k: int = 5,
        reuse_cache: bool = False,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        cache_path = os.path.join(output_dir, "raw_search_results.json")
        results, _ = self._evaluate_corpus(
            corpus_path=corpus_path,
            cache_path=cache_path,
            top_k_keywords=top_k_keywords,
            search_max_per_type=search_max_per_type,
            recommend_top_k=recommend_top_k,
            reuse_cache=reuse_cache,
        )
        self.save_json(os.path.join(output_dir, "evaluation_results.json"), results)
        return results

    def run_batch(
        self,
        corpora_root: str,
        output_dir: str,
        top_k_keywords: int = 10,
        search_max_per_type: int = 8,
        recommend_top_k: int = 5,
        reuse_cache: bool = False,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        corpora = self.list_corpora(corpora_root)
        if not corpora:
            raise ValueError(f"No corpus folders with TXT files found in {corpora_root}")

        corpus_results: Dict[str, Any] = {}
        corpus_rows: List[Dict[str, Any]] = []
        all_documents: List[str] = []
        all_raw_flat: List[Dict[str, Any]] = []
        all_unranked_flat: List[Dict[str, Any]] = []
        all_ranked_flat: List[Dict[str, Any]] = []
        keyword_simple_reports: List[Dict[str, Any]] = []
        keyword_full_reports: List[Dict[str, Any]] = []

        for corpus_name, corpus_path in corpora:
            corpus_output = os.path.join(output_dir, corpus_name)
            os.makedirs(corpus_output, exist_ok=True)
            cache_path = os.path.join(corpus_output, "raw_search_results.json")
            results, aux = self._evaluate_corpus(
                corpus_path=corpus_path,
                cache_path=cache_path,
                top_k_keywords=top_k_keywords,
                search_max_per_type=search_max_per_type,
                recommend_top_k=recommend_top_k,
                reuse_cache=reuse_cache,
            )
            self.save_json(os.path.join(corpus_output, "evaluation_results.json"), results)
            corpus_results[corpus_name] = results
            corpus_rows.append(
                {
                    "corpus": corpus_name,
                    "document_count": results["corpus"]["document_count"],
                    "word_count_total": results["corpus"]["word_count_total"],
                    "raw_resources": results["resource_reports"]["raw_search"]["total_resources"],
                    "ranked_resources": results["resource_reports"]["topk_ranked"]["total_resources"],
                }
            )
            all_documents.extend(aux["documents"])
            all_raw_flat.extend(aux["raw_flat"])
            all_unranked_flat.extend(aux["unranked_flat"])
            all_ranked_flat.extend(aux["ranked_flat"])
            keyword_simple_reports.append(results["keyword_reports"]["simple_frequency"])
            keyword_full_reports.append(results["keyword_reports"]["tfidf_mmr"])

        def avg(report_list: List[Dict[str, Any]], key: str, digits: int = 2) -> float:
            return round(sum(item[key] for item in report_list) / len(report_list), digits)

        aggregate = {
            "corpus": {
                "mode": "focused_multi_corpus",
                "corpus_count": len(corpora),
                "corpora": [name for name, _ in corpora],
                "document_count_total": sum(row["document_count"] for row in corpus_rows),
                "word_count_total": sum(row["word_count_total"] for row in corpus_rows),
            },
            "parameters": {
                "top_k_keywords": top_k_keywords,
                "search_max_per_type": search_max_per_type,
                "recommend_top_k": recommend_top_k,
            },
            "keyword_reports": {
                "simple_frequency": {
                    "keyword_count": round(avg(keyword_simple_reports, "keyword_count", 0)),
                    "coverage": avg(keyword_simple_reports, "coverage"),
                    "ai_relevance": avg(keyword_simple_reports, "ai_relevance"),
                    "diversity": avg(keyword_simple_reports, "diversity", 4),
                },
                "tfidf_mmr": {
                    "keyword_count": round(avg(keyword_full_reports, "keyword_count", 0)),
                    "coverage": avg(keyword_full_reports, "coverage"),
                    "ai_relevance": avg(keyword_full_reports, "ai_relevance"),
                    "diversity": avg(keyword_full_reports, "diversity", 4),
                },
            },
            "resource_reports": {
                "raw_search": self.metrics.generate_resource_report(all_raw_flat, original_corpus=all_documents),
                "topk_unranked": self.metrics.generate_resource_report(
                    all_unranked_flat,
                    initial_count=len(all_raw_flat),
                    original_corpus=all_documents,
                ),
                "topk_ranked": self.metrics.generate_resource_report(
                    all_ranked_flat,
                    initial_count=len(all_raw_flat),
                    original_corpus=all_documents,
                ),
            },
            "resource_type_counts": {
                "raw_search": dict(Counter(resource.get("resource_type", "unknown") for resource in all_raw_flat)),
                "topk_unranked": dict(Counter(resource.get("resource_type", "unknown") for resource in all_unranked_flat)),
                "topk_ranked": dict(Counter(resource.get("resource_type", "unknown") for resource in all_ranked_flat)),
            },
        }
        aggregate["improvements"] = {
            "keyword_coverage_delta": round(
                aggregate["keyword_reports"]["tfidf_mmr"]["coverage"] - aggregate["keyword_reports"]["simple_frequency"]["coverage"],
                2,
            ),
            "keyword_ai_relevance_delta": round(
                aggregate["keyword_reports"]["tfidf_mmr"]["ai_relevance"] - aggregate["keyword_reports"]["simple_frequency"]["ai_relevance"],
                2,
            ),
            "keyword_diversity_delta": round(
                aggregate["keyword_reports"]["tfidf_mmr"]["diversity"] - aggregate["keyword_reports"]["simple_frequency"]["diversity"],
                4,
            ),
            "ranking_ai_relevance_delta": round(
                aggregate["resource_reports"]["topk_ranked"]["ai_relevance"] - aggregate["resource_reports"]["topk_unranked"]["ai_relevance"],
                2,
            ),
            "ranking_authority_delta": round(
                aggregate["resource_reports"]["topk_ranked"]["authority_score"] - aggregate["resource_reports"]["topk_unranked"]["authority_score"],
                2,
            ),
            "ranking_noise_reduction": aggregate["resource_reports"]["topk_ranked"].get("noise_reduction", 0.0),
        }

        results = {
            "mode": "focused_multi_corpus",
            "aggregate": aggregate,
            "corpora": corpus_results,
        }
        self.save_json(os.path.join(output_dir, "evaluation_results.json"), results)
        self.save_csv(
            os.path.join(output_dir, "corpus_overview.csv"),
            ["corpus", "document_count", "word_count_total", "raw_resources", "ranked_resources"],
            corpus_rows,
        )
        return results

    def export_tables(self, results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        exported: Dict[str, str] = {}
        summary = self._summary_view(results)

        keyword_simple = summary["keyword_reports"]["simple_frequency"]
        keyword_full = summary["keyword_reports"]["tfidf_mmr"]
        raw = summary["resource_reports"]["raw_search"]
        unranked = summary["resource_reports"]["topk_unranked"]
        ranked = summary["resource_reports"]["topk_ranked"]
        batch_mode = results.get("mode") == "focused_multi_corpus"

        keyword_rows = [
            {
                "method": "Simple frequency baseline",
                "keyword_count": keyword_simple["keyword_count"],
                "coverage_pct": keyword_simple["coverage"],
                "ai_relevance_pct": keyword_simple["ai_relevance"],
                "diversity": keyword_simple["diversity"],
            },
            {
                "method": "TF-IDF + MMR",
                "keyword_count": keyword_full["keyword_count"],
                "coverage_pct": keyword_full["coverage"],
                "ai_relevance_pct": keyword_full["ai_relevance"],
                "diversity": keyword_full["diversity"],
            },
        ]
        keyword_csv = os.path.join(output_dir, "keyword_metrics.csv")
        self.save_csv(keyword_csv, ["method", "keyword_count", "coverage_pct", "ai_relevance_pct", "diversity"], keyword_rows)
        exported["keyword_csv"] = keyword_csv

        resource_rows = [
            {
                "stage": "Raw search",
                "total_resources": raw["total_resources"],
                "ai_relevance_pct": raw["ai_relevance"],
                "authority_score_pct": raw["authority_score"],
                "valid_url_pct": raw["url_validation"]["valid_percentage"],
                "noise_reduction_pct": raw.get("noise_reduction", 0.0),
                "cross_platform_diversity": raw["cross_platform_diversity"],
            },
            {
                "stage": "Top-k unranked",
                "total_resources": unranked["total_resources"],
                "ai_relevance_pct": unranked["ai_relevance"],
                "authority_score_pct": unranked["authority_score"],
                "valid_url_pct": unranked["url_validation"]["valid_percentage"],
                "noise_reduction_pct": unranked.get("noise_reduction", 0.0),
                "cross_platform_diversity": unranked["cross_platform_diversity"],
            },
            {
                "stage": "Top-k ranked",
                "total_resources": ranked["total_resources"],
                "ai_relevance_pct": ranked["ai_relevance"],
                "authority_score_pct": ranked["authority_score"],
                "valid_url_pct": ranked["url_validation"]["valid_percentage"],
                "noise_reduction_pct": ranked.get("noise_reduction", 0.0),
                "cross_platform_diversity": ranked["cross_platform_diversity"],
            },
        ]
        resource_csv = os.path.join(output_dir, "resource_metrics.csv")
        self.save_csv(
            resource_csv,
            ["stage", "total_resources", "ai_relevance_pct", "authority_score_pct", "valid_url_pct", "noise_reduction_pct", "cross_platform_diversity"],
            resource_rows,
        )
        exported["resource_csv"] = resource_csv

        keyword_caption = "Keyword Extraction Aggregated Across Focused 10-Document Corpora" if batch_mode else "Keyword Extraction on the Pilot Corpus"
        pipeline_caption = "Resource Quality Aggregated Across Focused Retrieval Corpora" if batch_mode else "Resource Quality Across Retrieval and Ranking Stages"
        latex_path = os.path.join(output_dir, "evaluation_tables.tex")
        latex_content = f"""%% Auto-generated by test/evaluation_pipeline/evaluator.py
\\begin{{table}}[!t]
\\renewcommand{{\\arraystretch}}{{1.2}}
\\caption{{{keyword_caption}}}
\\label{{tab:keyword_quality_auto}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\hline\\hline
Method & Keywords & Coverage (\\%) & AI rel. (\\%)\\\\
\\hline
Simple frequency baseline & {keyword_simple['keyword_count']} & {keyword_simple['coverage']:.2f} & {keyword_simple['ai_relevance']:.2f}\\\\
TF-IDF + MMR & {keyword_full['keyword_count']} & {keyword_full['coverage']:.2f} & {keyword_full['ai_relevance']:.2f}\\\\
\\hline\\hline
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[!t]
\\renewcommand{{\\arraystretch}}{{1.2}}
\\caption{{{pipeline_caption}}}
\\label{{tab:pipeline_comparison_auto}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\hline\\hline
Metric & Raw search & Top-$K$ unranked & Top-$K$ ranked\\\\
\\hline
Total resources & {raw['total_resources']} & {unranked['total_resources']} & {ranked['total_resources']}\\\\
AI relevance (\\%) & {raw['ai_relevance']:.2f} & {unranked['ai_relevance']:.2f} & {ranked['ai_relevance']:.2f}\\\\
Authority score (\\%) & {raw['authority_score']:.2f} & {unranked['authority_score']:.2f} & {ranked['authority_score']:.2f}\\\\
Valid URLs (\\%) & {raw['url_validation']['valid_percentage']:.2f} & {unranked['url_validation']['valid_percentage']:.2f} & {ranked['url_validation']['valid_percentage']:.2f}\\\\
Noise reduction (\\%) & {raw.get('noise_reduction', 0.0):.2f} & {unranked.get('noise_reduction', 0.0):.2f} & {ranked.get('noise_reduction', 0.0):.2f}\\\\
Cross-platform diversity & {raw['cross_platform_diversity']} & {unranked['cross_platform_diversity']} & {ranked['cross_platform_diversity']}\\\\
\\hline\\hline
\\end{{tabular}}
\\end{{table}}
"""
        self.save_text(latex_path, latex_content)
        exported["latex_tables"] = latex_path
        return exported

    def generate_plots(self, results: Dict[str, Any], plots_dir: str) -> Dict[str, str]:
        os.makedirs(plots_dir, exist_ok=True)
        generated: Dict[str, str] = {}
        summary = self._summary_view(results)

        raw = summary["resource_reports"]["raw_search"]
        unranked = summary["resource_reports"]["topk_unranked"]
        ranked = summary["resource_reports"]["topk_ranked"]
        labels = ["Raw search", "Top-k unranked", "CBF ranked"]
        metrics = ["ai_relevance", "authority_score", "valid_url_pct"]
        values = {
            "Raw search": [raw["ai_relevance"], raw["authority_score"], raw["url_validation"]["valid_percentage"]],
            "Top-k unranked": [unranked["ai_relevance"], unranked["authority_score"], unranked["url_validation"]["valid_percentage"]],
            "CBF ranked": [ranked["ai_relevance"], ranked["authority_score"], ranked["url_validation"]["valid_percentage"]],
        }

        x = range(len(metrics))
        width = 0.24
        plt.figure(figsize=(9, 5))
        for idx, label in enumerate(labels):
            offset = [pos + (idx - 1) * width for pos in x]
            plt.bar(offset, values[label], width=width, label=label)
        plt.xticks(list(x), ["AI relevance", "Authority", "Valid URLs"])
        plt.ylabel("Percentage")
        plt.ylim(0, 100)
        plt.title("AI-Pedia pipeline quality comparison")
        plt.legend()
        plt.tight_layout()
        pipeline_plot = os.path.join(plots_dir, "evaluation_pipeline_comparison.png")
        plt.savefig(pipeline_plot, dpi=220)
        plt.close()
        generated["pipeline_comparison"] = pipeline_plot

        keyword_simple = summary["keyword_reports"]["simple_frequency"]
        keyword_full = summary["keyword_reports"]["tfidf_mmr"]
        plt.figure(figsize=(8, 5))
        keyword_labels = ["Coverage", "AI relevance", "Diversity"]
        simple_values = [keyword_simple["coverage"], keyword_simple["ai_relevance"], keyword_simple["diversity"] * 100]
        full_values = [keyword_full["coverage"], keyword_full["ai_relevance"], keyword_full["diversity"] * 100]
        x2 = range(len(keyword_labels))
        plt.bar([pos - 0.18 for pos in x2], simple_values, width=0.36, label="Simple frequency")
        plt.bar([pos + 0.18 for pos in x2], full_values, width=0.36, label="TF-IDF + MMR")
        plt.xticks(list(x2), keyword_labels)
        plt.ylabel("Percentage / normalized score")
        plt.ylim(0, 100)
        plt.title("Keyword extraction quality comparison")
        plt.legend()
        plt.tight_layout()
        keyword_plot = os.path.join(plots_dir, "evaluation_precision_comparison.png")
        plt.savefig(keyword_plot, dpi=220)
        plt.close()
        generated["keyword_comparison"] = keyword_plot

        raw_counts = summary["resource_type_counts"]["raw_search"]
        ranked_counts = summary["resource_type_counts"]["topk_ranked"]
        type_labels = ["txt", "video", "code"]
        raw_values = [raw_counts.get(label, 0) for label in type_labels]
        ranked_values = [ranked_counts.get(label, 0) for label in type_labels]
        plt.figure(figsize=(8, 5))
        x3 = range(len(type_labels))
        plt.bar([pos - 0.18 for pos in x3], raw_values, width=0.36, label="Raw search")
        plt.bar([pos + 0.18 for pos in x3], ranked_values, width=0.36, label="Final ranked output")
        plt.xticks(list(x3), ["Text", "Video", "Code"])
        plt.ylabel("Resource count")
        plt.title("Resource type distribution before and after ranking")
        plt.legend()
        plt.tight_layout()
        distribution_plot = os.path.join(plots_dir, "resource_type_distribution.png")
        plt.savefig(distribution_plot, dpi=220)
        plt.close()
        generated["resource_type_distribution"] = distribution_plot
        return generated


def default_paper_figures_dir() -> str:
    candidate = os.path.join(PROJECT_ROOT, "Paper", "L3-CS Project Paper Template (LaTeX)", "figures")
    return candidate if os.path.isdir(candidate) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Pedia evaluation runner")
    parser.add_argument("--corpus", type=str, default=EvalConfig.TEST_CORPUS_PATH)
    parser.add_argument("--corpora-root", type=str, default=EvalConfig.TEST_CORPORA_ROOT)
    parser.add_argument("--use-focused-corpora", action="store_true")
    parser.add_argument("--output", type=str, default=EvalConfig.OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--tables-dir", type=str, default=None)
    parser.add_argument("--top-k-keywords", type=int, default=10)
    parser.add_argument("--search-max-per-type", type=int, default=8)
    parser.add_argument("--recommend-top-k", type=int, default=5)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    runner = EvaluationRunner()
    if args.use_focused_corpora:
        results = runner.run_batch(
            corpora_root=args.corpora_root,
            output_dir=args.output,
            top_k_keywords=args.top_k_keywords,
            search_max_per_type=args.search_max_per_type,
            recommend_top_k=args.recommend_top_k,
            reuse_cache=args.reuse_cache,
        )
    else:
        results = runner.run_single(
            corpus_path=args.corpus,
            output_dir=args.output,
            top_k_keywords=args.top_k_keywords,
            search_max_per_type=args.search_max_per_type,
            recommend_top_k=args.recommend_top_k,
            reuse_cache=args.reuse_cache,
        )

    plots_dir = args.plots_dir or default_paper_figures_dir() or args.output
    tables_dir = args.tables_dir or args.output
    generated = runner.generate_plots(results, plots_dir)
    exported_tables = runner.export_tables(results, tables_dir)

    summary = results.get("aggregate", results)
    print("=" * 60)
    print("AI-Pedia evaluation complete")
    print("=" * 60)
    print(json.dumps(summary["improvements"], indent=2, ensure_ascii=False))
    print("Generated plots:")
    for name, path in generated.items():
        print(f"- {name}: {path}")
    print("Exported tables:")
    for name, path in exported_tables.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
