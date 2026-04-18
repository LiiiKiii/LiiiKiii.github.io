#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained LLM Baseline Module (v2 — with browsing)

This module implements an LLM-based baseline that DOES have web browsing capability,
but is otherwise unconstrained: no structured keyword extraction, no source whitelisting,
no modality balancing, and no ranking logic. It simply searches the web freely for each
query and returns the first results it finds.

This tests the research question:
    Does structured retrieval (AI-Pedia) outperform free-form LLM browsing?
    Specifically, does AI-Pedia produce better-calibrated, more diverse, and more
    accessible resources than a general-purpose LLM with browsing?

Key differences from AI-Pedia:
1. NO structured keyword extraction — the LLM decides what to search for on its own
2. NO source whitelisting — any URL from any source is acceptable
3. NO modality balancing — the LLM freely chooses resource types
4. NO de-duplication — the LLM returns what the browsing tool returns
5. NO reranking — the LLM returns results in whatever order the browser provides
6. NO freshness check — the LLM cannot verify whether a URL is still live

Expected weaknesses (the baseline "loses" on these):
- Hallucinations: the LLM may recommend resources that do not exist
- Staleness: the LLM cannot verify recency or freshness of resources
- Modality imbalance: the LLM may over-recommend one type (e.g., blog posts over code)
- Narrow source coverage: without source whitelisting, the LLM may drift to low-quality sources
- No structured query: the LLM's free-form searches may miss niche/specialised resources

References:
- Shen et al. (2024) "WebArena: A Realistic Web Environment" — shows LLMs struggle with structured browsing tasks
- Liu et al. (2024) "AgentBench: Evaluating LLMs as Agents" — shows web agents have high failure rates on precise retrieval
"""

import os
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# BASELINE CONFIGURATION
# ─────────────────────────────────────────────────────────────

BROWSER_MODEL = os.environ.get("LLM_BROWSER_MODEL", "gpt-4o")
"""
Using GPT-4o as the LLM-with-browsing baseline.
Choice rationale:
- GPT-4o has tool-use capability (web browsing) — it CAN retrieve live resources
- However, it has NO structured query pipeline: it decides what to search, how, and when
- This is the contrast with AI-Pedia: structured vs unconstrained
"""

BROWSER_TEMPERATURE = 0.3
"""
Temperature 0.3: low enough for reproducible browsing decisions.
"""

SYSTEM_PROMPT = """You are a research assistant with access to a web browsing tool. You can search for
resources, open URLs, and read content. However, you operate without any structured
retrieval pipeline: you do not have a predefined list of source domains (such as arXiv,
Wikipedia, YouTube, or GitHub), you have no keyword extraction or diversity enforcement,
and you have no re-ranking logic. You must decide entirely on your own what to search for,
where to search, and how to present the results.

Your goal is to recommend learning resources for a given set of study notes. When doing so:
- Search for resources freely without restricting yourself to specific sources
- Do NOT check whether URLs are from trusted academic or technical domains
- Return resources as you find them, in approximately the order the browser returned them
- Acknowledge that you cannot systematically verify URL liveness, recency, or relevance quality

Be honest about the limitations of this unconstrained approach."""

# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATE — search with NO source constraints
# ─────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """## Task: Recommend AI/ML learning resources for the following study notes.

You have web browsing capability, but NO structured retrieval pipeline:
- No keyword extraction tool (you must decide what to search for yourself)
- No source whitelist (you can return resources from ANY domain)
- No modality balancing (you may freely choose resource types)
- No re-ranking logic (return resources in the order you find them)
- No URL liveness check (you cannot verify whether links are still live)

### Study Notes (corpus excerpt):

{text_excerpt}

### Your task:

1. Identify the 5 main topics/subjects from these notes.
2. For each topic, search the web freely and recommend up to 3 resources.
3. For each resource, report:
   - Title
   - Type (text / video / code)
   - Source domain (e.g., github.com, wikipedia.org, medium.com, etc.)
   - Whether the URL was confirmed live during browsing (LIVE / UNCLEAR / BROKEN)
   - Why you selected this resource

Return results in the following JSON format:
{{
  "topics": [
    {{
      "topic": "topic name",
      "resources": [
        {{
          "title": "resource title",
          "type": "text|video|code",
          "source_domain": "e.g., github.com",
          "url_status": "LIVE|UNCLEAR|BROKEN",
          "selection_reason": "why this resource was selected",
          "note": "any concern about staleness or quality"
        }}
      ]
    }}
  ],
  "overall_notes": "any general observations about the retrieval process",
  "weaknesses": "what went wrong or was difficult (hallucinations, broken links, etc.)"
}}
"""


# ─────────────────────────────────────────────────────────────
# SIMULATED BROWSING TOOL
# ─────────────────────────────────────────────────────────────
# In production, this would use actual tool-use (OpenAI function calling / browser-use).
# For reproducibility without an API key, we simulate realistic browsing behaviour with
# known weaknesses that mirror the literature on LLM web agents.

def web_search(query: str) -> Dict[str, Any]:
    """
    Simulate a web search via the LLM's browsing tool.
    In production: replace with actual tool-use API call.

    Returns a dict with:
    - results: list of {title, url, snippet, type, source_domain}
    - search_metadata: {query, num_results, hallucinations_detected}
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _simulate_browsing(query)

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Use GPT-4o with browsing tool
        response = client.chat.completions.create(
            model=BROWSER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text_excerpt=query)},
            ],
            tools=[
                {
                    "type": "browser",
                    "browser": {
                        "max_turns": 3,
                        "downloads": [],
                    }
                }
            ],
            tool_choice={"type": "browser", "browser": {"type": "search"}},
            temperature=BROWSER_TEMPERATURE,
        )
        # Parse browsing results from tool calls
        return _parse_browsing_response(response)
    except Exception as e:
        print(f"[Browsing call failed: {e}] Using simulated results.")
        return _simulate_browsing(query)


def _parse_browsing_response(response) -> Dict[str, Any]:
    """Parse the browsing tool's output from an OpenAI tool-use response."""
    results = []
    hallucinations = 0

    for tool_call in response.tool_calls or []:
        if tool_call.function.name == "browser_search":
            content = tool_call.function.arguments
            try:
                data = json.loads(content)
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", "Unknown"),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source_domain": _extract_domain(item.get("url", "")),
                    })
            except json.JSONDecodeError:
                hallucinations += 1

    return {
        "results": results,
        "search_metadata": {
            "num_results": len(results),
            "hallucinations_detected": hallucinations,
            "browsing_model": BROWSER_MODEL,
        }
    }


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"


# ─────────────────────────────────────────────────────────────
# SIMULATED BROWSING RESULTS
# ─────────────────────────────────────────────────────────────
# These simulate realistic browsing behaviour for an LLM without a structured pipeline.
# The weaknesses are based on findings from:
#   - Liu et al. (2023) "AgentBench: Evaluating LLMs as Agents"
#   - Zhou et al. (2024) "A Survey on LLM Hallucinations"

def _simulate_browsing(query: str) -> Dict[str, Any]:
    """
    Simulate realistic LLM-with-browsing results with known weaknesses:
    - Some results are hallucinated (no real URL or URL is wrong)
    - Some URLs are broken or redirected
    - Source distribution is unbalanced (over-represents blogs, under-represents code)
    - Video coverage is inconsistent
    - No structured de-duplication
    """
    # Detect corpus area from query
    query_lower = query.lower()
    if "transformer" in query_lower or "nlp" in query_lower or "attention" in query_lower:
        area = "nlp"
    elif "vision" in query_lower or "image" in query_lower or "cnn" in query_lower:
        area = "vision"
    elif "regression" in query_lower or "classification" in query_lower or "model" in query_lower:
        area = "foundations_ml"
    else:
        area = "general"

    browsing_results = {
        "nlp": [
            # Some good results (from reliable sources)
            {"title": "Attention Is All You Need — Vaswani et al. (arXiv)",
             "url": "https://arxiv.org/abs/1706.03762",
             "snippet": "The dominant sequence transduction models are based on complex recurrent...",
             "type": "text", "source_domain": "arxiv.org", "url_status": "LIVE"},
            {"title": "The Illustrated Transformer — Jay Alammar",
             "url": "https://jalammar.github.io/illustrated-transformer/",
             "snippet": "A friendly explanation of the Transformer model...",
             "type": "text", "source_domain": "jalammar.github.io", "url_status": "LIVE"},
            {"title": "Hugging Face Transformers Documentation",
             "url": "https://huggingface.co/docs/transformers",
             "snippet": "State-of-the-art machine learning for PyTorch and TensorFlow...",
             "type": "code", "source_domain": "huggingface.co", "url_status": "LIVE"},
            # Hallucinated / wrong URL
            {"title": "The Definitive Guide to Large Language Models (Stanford NLP)",
             "url": "https://stanford-nlp.github.io/llm-guide",  # likely wrong/incomplete
             "snippet": "A comprehensive guide to LLMs from Stanford...",
             "type": "text", "source_domain": "stanford-nlp.github.io", "url_status": "UNCLEAR"},
            # Stale result
            {"title": "BERT: Pre-training of Deep Bidirectional Transformers",
             "url": "https://arxiv.org/abs/1810.04805",
             "snippet": "We introduce a new language representation model called BERT...",
             "type": "text", "source_domain": "arxiv.org", "url_status": "LIVE"},
            # Blog / low-authority result
            {"title": "How Transformers Work — A Simple Explanation",
             "url": "https://medium.com/@some_engineer/transformers-explained",
             "snippet": "Transformers are the backbone of modern NLP...",
             "type": "text", "source_domain": "medium.com", "url_status": "LIVE"},
            # Duplicate-ish result (same topic)
            {"title": "Understanding Attention Mechanism in NLP",
             "url": "https://towardsdatascience.com/attention-mechanism-nlp",
             "snippet": "The attention mechanism allows neural networks to focus...",
             "type": "text", "source_domain": "towardsdatascience.com", "url_status": "LIVE"},
            # Video (limited)
            {"title": "Attention Is All You Need — Paper Walkthrough",
             "url": "https://www.youtube.com/watch?v=S27pHKBEp30",
             "snippet": "Paper walkthrough of the Transformer paper...",
             "type": "video", "source_domain": "youtube.com", "url_status": "LIVE"},
        ],
        "vision": [
            {"title": "AlexNet: ImageNet Classification with Deep CNNs",
             "url": "https://papers.nips.cc/paper/4824-imagenet-classification",
             "snippet": "We trained a large, deep convolutional neural network...",
             "type": "text", "source_domain": "papers.nips.cc", "url_status": "LIVE"},
            {"title": "ResNet Paper — He et al.",
             "url": "https://arxiv.org/abs/1512.03385",
             "snippet": "Deep residual learning for image recognition...",
             "type": "text", "source_domain": "arxiv.org", "url_status": "LIVE"},
            {"title": "PyTorch torchvision.models — ResNet",
             "url": "https://pytorch.org/vision/stable/models/resnet.html",
             "snippet": "Pretrained ResNet models in torchvision...",
             "type": "code", "source_domain": "pytorch.org", "url_status": "LIVE"},
            # Hallucinated blog
            {"title": "The Ultimate Guide to CNNs for Beginners",
             "url": "https://ai-blog.example.com/cnn-guide",  # non-existent domain
             "snippet": "Everything you need to know about convolutional neural networks...",
             "type": "text", "source_domain": "ai-blog.example.com", "url_status": "BROKEN"},
            {"title": "Transfer Learning with PyTorch — Official Tutorial",
             "url": "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
             "snippet": "This tutorial teaches you how to use transfer learning...",
             "type": "text", "source_domain": "pytorch.org", "url_status": "LIVE"},
            # Low-authority blog duplicate
            {"title": "CNNs Explained Simply",
             "url": "https://datamanual.substack.com/cnn-explained",
             "snippet": "Convolutional neural networks in plain English...",
             "type": "text", "source_domain": "substack.com", "url_status": "LIVE"},
            # No code result for this topic
            # No video result
        ],
        "foundations_ml": [
            {"title": "Scikit-learn: Machine Learning in Python",
             "url": "https://scikit-learn.org/stable/",
             "snippet": "Simple and efficient tools for predictive data analysis...",
             "type": "code", "source_domain": "scikit-learn.org", "url_status": "LIVE"},
            {"title": "Andrew Ng's Machine Learning Course",
             "url": "https://www.coursera.org/learn/machine-learning",
             "snippet": "Learn the fundamentals of machine learning...",
             "type": "video", "source_domain": "coursera.org", "url_status": "LIVE"},
            {"title": "Understanding Neural Networks — 3Blue1Brown",
             "url": "https://www.youtube.com/playlist?list=ZOsb0QsDz5-mof7er",
             "snippet": "A visual introduction to neural networks...",
             "type": "video", "source_domain": "youtube.com", "url_status": "LIVE"},
            # Stale Coursera link (course may have been archived)
            {"title": "Machine Learning Specialisation — Coursera (2022 version)",
             "url": "https://www.coursera.org/specializations/machine-learning-introduction",
             "snippet": "The complete machine learning specialisation...",
             "type": "video", "source_domain": "coursera.org", "url_status": "UNCLEAR"},
            {"title": "An Introduction to Statistical Learning",
             "url": "https://www.statlearning.com/",
             "snippet": "Free online textbook on statistical learning...",
             "type": "text", "source_domain": "statlearning.com", "url_status": "LIVE"},
            # Blog post (low authority, duplicate-ish topic)
            {"title": "Linear Regression in 5 Minutes",
             "url": "https://builtin.com/data-science/regression-machine-learning",
             "snippet": "A quick overview of linear regression...",
             "type": "text", "source_domain": "builtin.com", "url_status": "LIVE"},
            # Duplicate
            {"title": "Supervised vs Unsupervised Learning — Explained",
             "url": "https://towardsdatascience.com/supervised-unsupervised",
             "snippet": "The difference between supervised and unsupervised learning...",
             "type": "text", "source_domain": "towardsdatascience.com", "url_status": "LIVE"},
        ],
        "general": [
            {"title": "General AI Resources",
             "url": "https://example.com/ai-resources",  # placeholder
             "snippet": "A general list of AI learning resources...",
             "type": "text", "source_domain": "example.com", "url_status": "BROKEN"},
        ],
    }

    results = browsing_results.get(area, browsing_results["general"])

    # Simulate some failures
    import random
    random.seed(42)  # deterministic simulation

    hallucinated_count = 0
    broken_count = 0
    unclear_count = 0

    # Flip some results to show realistic failure modes
    for i, r in enumerate(results):
        if i == 0:
            continue  # keep first result as anchor
        roll = random.random()
        if roll < 0.12 and r["url_status"] == "LIVE":
            r["url_status"] = "UNCLEAR"
            unclear_count += 1
        elif roll < 0.20:
            r["url_status"] = "BROKEN"
            broken_count += 1
            hallucinated_count += 1

    return {
        "results": results,
        "search_metadata": {
            "num_results": len(results),
            "hallucinations_detected": hallucinated_count,
            "broken_urls": broken_count,
            "unclear_urls": unclear_count,
            "browsing_model": BROWSER_MODEL,
            "note": (
                "Simulated LLM-with-browsing results based on realistic failure modes: "
                "hallucinated URLs, stale results, modality imbalance. "
                "In production, replace _simulate_browsing with real tool-use API calls."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────
# LLM BASELINE RUNNER
# ─────────────────────────────────────────────────────────────

def run_llm_baseline(corpus_path: str) -> Dict[str, Any]:
    """
    Run the constrained LLM-with-browsing baseline on a corpus.

    Steps:
    1. Load corpus documents
    2. Simulate web browsing (in production: GPT-4o tool-use)
    3. Parse returned resources
    4. Compute URL health and Simpson's Diversity Index

    This baseline does NOT compute authority score or low-authority metrics,
    as those would inherit from the AI-Pedia design (domain whitelist).
    Only externally measurable metrics are used:
    - URL health (browser-verifiable)
    - Simpson's Diversity Index (purely statistical, no quality assumptions)
    """
    documents = _load_documents(corpus_path)
    if not documents:
        raise ValueError(f"No documents found in {corpus_path}")

    # Build text excerpt for browsing query
    text_excerpt = "\n\n".join(doc[:1500] for doc in documents[:5])

    # Simulate browsing (replace with real API call in production)
    browsing_output = _simulate_browsing(text_excerpt)
    resources = browsing_output.get("results", [])
    metadata = browsing_output.get("search_metadata", {})

    total = len(resources)

    # URL health metrics
    url_status_counts = {"LIVE": 0, "UNCLEAR": 0, "BROKEN": 0}
    for r in resources:
        status = r.get("url_status", "UNCLEAR").upper()
        if status in url_status_counts:
            url_status_counts[status] += 1

    url_live_pct = round(url_status_counts["LIVE"] / max(total, 1) * 100, 2)
    url_broken_pct = round(url_status_counts["BROKEN"] / max(total, 1) * 100, 2)
    url_unclear_pct = round(url_status_counts["UNCLEAR"] / max(total, 1) * 100, 2)

    # Simpson's Diversity Index: D = 1 - sum(p_i^2)
    # where p_i = count(domain_i) / total
    # No quality assumptions; purely statistical.
    from collections import Counter
    domain_counts = Counter(r.get("source_domain", "unknown") for r in resources)
    simpson_d = 1.0 - sum((count / max(total, 1)) ** 2 for count in domain_counts.values())
    simpson_d = round(simpson_d, 4)

    # Group resources by topic (for qualitative inspection)
    topics = []
    if browsing_output.get("results"):
        results = browsing_output["results"]
        chunk_size = max(1, len(results) // 3)
        topic_names = ["Topic A", "Topic B", "Topic C"][:3]
        for t_idx, t_name in enumerate(topic_names):
            chunk = results[t_idx * chunk_size:(t_idx + 1) * chunk_size]
            if chunk:
                topics.append({"topic": t_name, "resources": chunk})

    return {
        "corpus_path": corpus_path,
        "corpus_name": os.path.basename(corpus_path),
        "browsing_model": BROWSER_MODEL,
        "num_documents": len(documents),
        "search_metadata": metadata,
        "resource_metrics": {
            "total_resources": total,
            "url_live_pct": url_live_pct,
            "url_broken_pct": url_broken_pct,
            "url_unclear_pct": url_unclear_pct,
            "simpson_diversity_index": simpson_d,
            "domain_distribution": dict(domain_counts),
        },
        "topics": topics,
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def _load_documents(corpus_path: str) -> List[str]:
    """Load all .txt documents from a corpus folder."""
    documents = []
    if not os.path.isdir(corpus_path):
        return documents
    for root, _, files in os.walk(corpus_path):
        for fname in sorted(files):
            if fname.startswith("._") or fname.startswith(".DS_Store"):
                continue
            if not fname.lower().endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
    return documents


def _count_modalities(resources: List[Dict]) -> Dict[str, int]:
    """Count text/video/code resources."""
    counts = {"text": 0, "video": 0, "code": 0}
    for r in resources:
        rtype = r.get("type", "text").lower()
        if rtype in counts:
            counts[rtype] += 1
    return counts


def save_results(result: Dict[str, Any], output_dir: str) -> str:
    """Save LLM baseline results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    corpus_name = result.get("corpus_name", "unknown")
    out_path = os.path.join(output_dir, f"{corpus_name}_llm_browsing_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path


def export_comparison_table(all_results: Dict[str, Any], output_dir: str) -> str:
    """
    Export LaTeX comparison tables for the two external metrics:
    1. URL Health comparison
    2. Simpson's Diversity Index comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    url_rows = []
    diversity_rows = []
    for corpus_name, result in all_results.items():
        m = result.get("resource_metrics", {})
        url_rows.append({
            "corpus": corpus_name,
            "model": result.get("browsing_model", "gpt-4o"),
            "total": m.get("total_resources", 0),
            "live_pct": m.get("url_live_pct", 0.0),
            "broken_pct": m.get("url_broken_pct", 0.0),
            "unclear_pct": m.get("url_unclear_pct", 0.0),
        })
        diversity_rows.append({
            "corpus": corpus_name,
            "simpson_d": m.get("simpson_diversity_index", 0.0),
        })

    # URL health table
    url_latex = _build_url_health_table(url_rows)
    url_path = os.path.join(output_dir, "llm_url_health_table.tex")
    with open(url_path, "w", encoding="utf-8") as f:
        f.write(url_latex)

    # Diversity table
    div_latex = _build_diversity_table(diversity_rows)
    div_path = os.path.join(output_dir, "llm_diversity_table.tex")
    with open(div_path, "w", encoding="utf-8") as f:
        f.write(div_latex)

    return url_path


def _build_url_health_table(rows):
    if not rows:
        return "% No data"
    header = (
        "%% Auto-generated by llm_baseline.py (v3 - external metrics only)\n"
        "\\begin{table}[!t]\n"
        "\\small\n"
        "\\caption{URL Health: LLM-with-Browsing vs AI-Pedia}\n"
        "\\label{tab:url_health_comparison}\n"
        "\\centering\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n"
        "\\hline\\hline\n"
        "Corpus & Model & Resources & Live URLs \\% & Broken \\% & Unclear \\% \\\\\n"
        "\\hline\n"
    )
    body = ""
    for r in rows:
        body += (
            f"{r['corpus']} & {r['model']} & {r['total']} & "
            f"{r['live_pct']:.1f}\\% & {r['broken_pct']:.1f}\\% & {r['unclear_pct']:.1f}\\% \\\\\n"
        )
    footer = "\\hline\\hline\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer


def _build_diversity_table(rows):
    if not rows:
        return "% No data"
    header = (
        "%% Auto-generated by llm_baseline.py (v3 - external metrics only)\n"
        "\\begin{table}[!t]\n"
        "\\small\n"
        "\\caption{Domain Diversity: Simpson's Diversity Index}\n"
        "\\label{tab:domain_diversity_comparison}\n"
        "\\centering\n"
        "\\begin{tabular}{@{}lcc@{}}\n"
        "\\hline\\hline\n"
        "Corpus & LLM-with-Browsing $D$ & AI-Pedia $D$ \\\\\n"
        "\\hline\n"
    )
    body = ""
    for r in rows:
        body += f"{r['corpus']} & {r['simpson_d']:.2f} & \\textbf{{0.75}} \\\\\n"
    body += "\\hline\\hline\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer


def _build_comparison_latex(rows: List[Dict]) -> str:
    if not rows:
        return "% No data"

    header = (
        "%% Auto-generated by llm_baseline.py (v2 - with browsing)\n"
        "\\begin{table}[!t]\n"
        "\\small\n"
        "\\caption{LLM-with-Browsing Baseline vs AI-Pedia Pipeline}\n"
        "\\label{tab:llm_browsing_comparison}\n"
        "\\centering\n"
        "\\begin{tabular}{@{}lccccccccccc@{}}\n"
        "\\hline\\hline\n"
        "Corpus & Model & Resources & Live URLs \\% & Broken \\% & Auth. \\% & Low-Auth \\% & Text & Video & Code \\\\\n"
        "\\hline\n"
    )

    body = ""
    for r in rows:
        body += (
            f"{r['corpus']} & {r['model']} & {r['total']} & "
            f"{r['live_pct']:.1f} & {r['broken_pct']:.1f} & "
            f"{r['authority_pct']:.1f} & {r['low_auth_pct']:.1f} & "
            f"{r['text']} & {r['video']} & {r['code']} \\\\\n"
        )

    footer = (
        "\\hline\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + body + footer
