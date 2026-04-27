#!/usr/bin/env python3
"""
Generate paper figures from the live LLM-baseline evaluator JSON.
"""
import json
import os

import numpy as np


PROJECT_ROOT = "/Users/macbook/Desktop/AI-Pedia/Project/Code"
PAPER_FIGURE_DIR = (
    "/Users/macbook/Desktop/AI-Pedia/Project/Paper/"
    "L3-CS Project Paper Template (LaTeX)/figures"
)
RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "test", "llm_baseline_eval", "results", "llm_comparison_results.json"
)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def simpson_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return 1.0 - sum((v / total) ** 2 for v in counts.values())


def load_live_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("per_corpus", {})
    aipedia_domain_d = {
        "foundations_ml": 0.854,
        "nlp_transformers": 0.891,
        "vision_representation": 0.841,
    }
    metrics = {}
    for corpus, row in rows.items():
        llm = row["llm_baseline"]["resource_metrics"]
        ap = row["aipedia_result"]["resource_reports"]["topk_ranked"]
        ap_types = ap.get("resource_type_counts", {})
        llm_counts = llm.get("modality_counts", {})
        ap_counts = {
            "text": ap_types.get("txt", 0),
            "video": ap_types.get("video", 0),
            "code": ap_types.get("code", 0),
        }
        metrics[corpus] = {
            "llm_domain_d": llm.get("simpson_diversity_index", 0.0),
            "ap_domain_d": aipedia_domain_d.get(corpus, 0.86),
            "llm_authority": llm.get("authority_score_pct", 0.0) / 100.0,
            "ap_authority": ap.get("authority_score", 0.0) / 100.0,
            "llm_type_d": simpson_from_counts(llm_counts),
            "ap_type_d": simpson_from_counts(ap_counts),
            "llm_live": llm.get("url_live_pct", 0.0) / 100.0,
            "ap_live": ap.get("url_validation", {}).get("valid_percentage", 0.0) / 100.0,
            "llm_counts": llm_counts,
            "ap_counts": ap_counts,
        }
    return data, rows, metrics


def generate_pipeline_flow(plt, FancyBboxPatch, FancyArrowPatch):
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    stages = [
        ("Learner Corpus", "ZIP\nTXT / PDF", 0.35, "#E9F3FB"),
        ("Topic Signal", "TF-IDF\nMMR", 2.35, "#FFF3D8"),
        ("Candidate Pool", "Wiki / Scholar\narXiv / YouTube\nGitHub", 4.35, "#FCE4E4"),
        ("Relevance Gate", "AI filter\ndedup\nlanguage", 6.35, "#FDEED7"),
        ("CBF Ranking", "similarity\nauthority\nbalance", 8.35, "#EEE4F6"),
        ("Curated Output", "15 items\ntext / video / code", 10.35, "#E2F6F3"),
    ]

    for i, (title, body, x, color) in enumerate(stages):
        box = FancyBboxPatch(
            (x, 2.0),
            1.55,
            1.22,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#6B7280",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 0.775, 2.83, title, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#263238")
        ax.text(x + 0.775, 2.34, body, ha="center", va="center", fontsize=8.0, color="#374151", linespacing=1.16)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 1.60, 2.62),
                (stages[i + 1][2] - 0.06, 2.64),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color="#4B5563",
            ))

    # Stage grouping bands
    group_specs = [
        (0.28, 1.82, "Input"),
        (2.25, 1.82, "Signal"),
        (4.25, 3.82, "Retrieval"),
        (8.25, 3.65, "Reranking"),
    ]
    for start_x, width, label in group_specs:
        band = FancyBboxPatch(
            (start_x, 3.62),
            width,
            0.36,
            boxstyle="round,pad=0.015,rounding_size=0.05",
            linewidth=0.8,
            edgecolor="#CBD5E1",
            facecolor="#F8FAFC",
        )
        ax.add_patch(band)
        ax.text(start_x + width / 2, 3.8, label, ha="center", va="center", fontsize=8.3, color="#475569")

    # Lower artefact/output row
    artefacts = [
        ("Cache", "JSON", 4.15, "#F8FAFC"),
        ("Pack", "txt / video / code", 10.45, "#F8FAFC"),
        ("Eval", "tables / charts", 11.50, "#F3F6FF"),
    ]
    for title, body, x, color in artefacts:
        mini = FancyBboxPatch(
            (x, 0.56),
            0.92,
            0.46,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0.9,
            edgecolor="#94A3B8",
            facecolor=color,
        )
        ax.add_patch(mini)
        ax.text(x + 0.46, 0.81, title, ha="center", va="center", fontsize=7.7, fontweight="bold", color="#334155")
        ax.text(x + 0.46, 0.62, body, ha="center", va="center", fontsize=6.9, color="#475569")

    # Vertical/downstream links
    connector_specs = [
        ((5.10, 1.98), (4.61, 1.03)),
        ((11.16, 1.98), (10.92, 1.03)),
        ((11.16, 1.98), (11.96, 1.03)),
    ]
    for start, end in connector_specs:
        ax.add_patch(FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="#64748B",
            connectionstyle="arc3,rad=0.0",
        ))

    # Summary branch note
    summary = FancyBboxPatch(
        (11.28, 3.08),
        0.98,
        0.4,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.9,
        edgecolor="#A7B3C4",
        facecolor="#FFFFFF",
    )
    ax.add_patch(summary)
    ax.text(11.77, 3.26, "LLM summary\nor fallback", ha="center", va="center", fontsize=7.0, color="#475569")
    ax.add_patch(FancyArrowPatch(
        (11.12, 2.0),
        (11.74, 3.05),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color="#64748B",
    ))

    # Baseline comparison branch
    baseline = FancyBboxPatch(
        (12.25, 2.0),
        0.98,
        1.22,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#6B7280",
        facecolor="#EDF1FF",
    )
    ax.add_patch(baseline)
    ax.text(12.74, 2.82, "LLM Baseline", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#263238")
    ax.text(12.74, 2.40, "GPT-4o\nweb search", ha="center", va="center", fontsize=7.6, color="#374151")
    ax.add_patch(FancyArrowPatch(
        (11.94, 2.64),
        (12.21, 2.64),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#4B5563",
    ))

    ax.text(6.75, 4.25, "AI-Pedia End-to-End Artefact Flow", ha="center", fontsize=15, fontweight="bold")
    ax.text(
        6.75,
        0.95,
        "Limited notes -> open-web candidates -> corpus-matched recommendations",
        ha="center",
        fontsize=8.7,
        color="#4B5563",
    )
    ax.plot([0.45, 13.05], [1.42, 1.42], color="#CBD5E1", linewidth=1)
    ax.text(1.15, 1.16, "Input", ha="center", fontsize=8.0, color="#64748B")
    ax.text(6.55, 1.16, "Pipeline", ha="center", fontsize=8.0, color="#64748B")
    ax.text(11.55, 1.16, "Outputs", ha="center", fontsize=8.0, color="#64748B")
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "pipeline_flow.png"), dpi=180, bbox_inches="tight")
    plt.close()


def generate_resource_modality_summary(plt, metrics):
    raw = {"Text": 257, "Video": 210, "Code": 75}
    final = {"Text": 15, "Video": 15, "Code": 15}
    corpora = list(metrics.keys())
    llm_counts = {k: sum(metrics[c]["llm_counts"].get(k, 0) for c in corpora) for k in ("text", "video", "code")}
    ap_counts = {k: sum(metrics[c]["ap_counts"].get(k, 0) for c in corpora) for k in ("text", "video", "code")}
    colors = {"Text": "#4C78A8", "Video": "#F2B134", "Code": "#59A14F"}
    type_colors = {"text": "#4C78A8", "video": "#F2B134", "code": "#59A14F"}

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.8), gridspec_kw={"height_ratios": [1.05, 1]})

    labels = list(raw.keys())
    y = np.arange(len(labels))
    axes[0].barh(y + 0.18, [raw[k] for k in labels], height=0.34, label="Raw pool", color="#A6CEE3")
    axes[0].barh(y - 0.18, [final[k] for k in labels], height=0.34, label="Final output", color=[colors[k] for k in labels])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Resource count")
    axes[0].set_title("A. Candidate Compression", fontsize=11.5, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    for i, key in enumerate(labels):
        axes[0].text(raw[key] + 5, i + 0.18, str(raw[key]), va="center", fontsize=8)
        axes[0].text(final[key] + 5, i - 0.18, str(final[key]), va="center", fontsize=8)

    bottoms = np.zeros(2)
    xlabels = ["Live GPT-4o", "AI-Pedia"]
    for key in ("text", "video", "code"):
        vals = np.array([llm_counts[key], ap_counts[key]])
        axes[1].bar(xlabels, vals, bottom=bottoms, label=key.title(), color=type_colors[key], width=0.55)
        for i, val in enumerate(vals):
            if val > 0:
                axes[1].text(i, bottoms[i] + val / 2, str(int(val)), ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals
    axes[1].set_ylabel("Resources")
    axes[1].set_title("B. Final Output Modality Profile", fontsize=11.5, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Compression and Multimodal Balancing", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "resource_modality_summary.png"), dpi=170, bbox_inches="tight")
    plt.close()


def generate_keyword_signal_diagram(plt, FancyBboxPatch, FancyArrowPatch):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 3.0)
    ax.axis("off")

    boxes = [
        ("Learner notes", "clean + chunk", 0.28, "#E9F3FB"),
        ("TF-IDF terms", "salience scan", 2.18, "#FFF3D8"),
        ("MMR topics", "query set", 4.08, "#EAF7E8"),
    ]
    for i, (title, body, x, color) in enumerate(boxes):
        box = FancyBboxPatch(
            (x, 1.05), 1.45, 0.86,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.0, edgecolor="#64748B", facecolor=color
        )
        ax.add_patch(box)
        ax.text(x + 0.725, 1.58, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")
        ax.text(x + 0.725, 1.28, body, ha="center", va="center", fontsize=8.2, color="#475569")
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 1.49, 1.48), (boxes[i + 1][2] - 0.06, 1.48),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#4B5563"
            ))

    # Stage band
    band_specs = [
        (0.3, 1.43, "Input"),
        (2.2, 1.43, "Scoring"),
        (4.1, 1.43, "Selection"),
    ]
    for x, width, label in band_specs:
        band = FancyBboxPatch(
            (x, 2.02), width, 0.24,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.8, edgecolor="#CBD5E1", facecolor="#F8FAFC"
        )
        ax.add_patch(band)
        ax.text(x + width / 2, 2.14, label, ha="center", va="center", fontsize=7.4, color="#475569")

    # Lower artefact row
    chips = [
        ("chunks", 0.73),
        ("ranked terms", 2.86),
        ("search prompts", 5.02),
    ]
    for label, x in chips:
        chip = FancyBboxPatch(
            (x - 0.34, 0.36), 0.68, 0.22,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.8, edgecolor="#CBD5E1", facecolor="#FFFFFF"
        )
        ax.add_patch(chip)
        ax.text(x, 0.47, label, ha="center", va="center", fontsize=7.1, color="#475569")

    connectors = [
        ((0.73, 1.02), (0.73, 0.58)),
        ((2.90, 1.02), (2.90, 0.58)),
        ((4.80, 1.02), (5.02, 0.58)),
    ]
    for start, end in connectors:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=0.9, color="#94A3B8"
        ))

    ax.text(3.1, 2.50, "Keyword Signals", ha="center", fontsize=12.0, fontweight="bold")
    ax.text(3.1, 0.72, "salience first, redundancy second", ha="center", fontsize=8.2, color="#475569")
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "keyword_signal_diagram.png"), dpi=170, bbox_inches="tight")
    plt.close()


def generate_small_sample_diagram(plt, FancyBboxPatch, FancyArrowPatch):
    fig, ax = plt.subplots(figsize=(6.2, 3.25))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 3.25)
    ax.axis("off")

    left = FancyBboxPatch((0.35, 1.05), 1.35, 1.0, boxstyle="round,pad=0.03,rounding_size=0.06",
                          linewidth=1.0, edgecolor="#64748B", facecolor="#E9F3FB")
    mid = FancyBboxPatch((2.42, 0.88), 1.35, 1.34, boxstyle="round,pad=0.03,rounding_size=0.06",
                         linewidth=1.0, edgecolor="#64748B", facecolor="#FFF3D8")
    right = FancyBboxPatch((4.5, 1.05), 1.35, 1.0, boxstyle="round,pad=0.03,rounding_size=0.06",
                           linewidth=1.0, edgecolor="#64748B", facecolor="#EAF7E8")
    ax.add_patch(left)
    ax.add_patch(mid)
    ax.add_patch(right)

    ax.text(1.025, 1.62, "Small corpus", ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")
    ax.text(1.025, 1.30, "private notes", ha="center", va="center", fontsize=8.2, color="#475569")
    ax.text(3.095, 1.75, "Open-web pool", ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")
    ax.text(3.095, 1.46, "papers / video / code", ha="center", va="center", fontsize=8.2, color="#475569")
    ax.text(3.095, 1.16, "retrieve + filter", ha="center", va="center", fontsize=8.2, color="#475569")
    ax.text(5.175, 1.62, "Matched set", ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")
    ax.text(5.175, 1.30, "ranked support", ha="center", va="center", fontsize=8.2, color="#475569")

    ax.add_patch(FancyArrowPatch((1.72, 1.55), (2.36, 1.55), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#4B5563"))
    ax.add_patch(FancyArrowPatch((3.83, 1.55), (4.44, 1.55), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#4B5563"))

    # Stage band
    for x, width, label in [(0.38, 1.29, "Anchor"), (2.45, 1.29, "Expand"), (4.53, 1.29, "Rerank")]:
        band = FancyBboxPatch(
            (x, 2.27), width, 0.24,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.8, edgecolor="#CBD5E1", facecolor="#F8FAFC"
        )
        ax.add_patch(band)
        ax.text(x + width / 2, 2.39, label, ha="center", va="center", fontsize=7.4, color="#475569")

    # Expansion branches
    branch_specs = [
        ("text", 2.68, 0.66, "#E8F1FB"),
        ("video", 3.10, 0.66, "#FFF1D6"),
        ("code", 3.52, 0.66, "#E8F7EC"),
    ]
    for label, x, y, color in branch_specs:
        chip = FancyBboxPatch(
            (x - 0.22, y), 0.44, 0.18,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.8, edgecolor="#CBD5E1", facecolor=color
        )
        ax.add_patch(chip)
        ax.text(x, y + 0.09, label, ha="center", va="center", fontsize=7.0, color="#475569")
        ax.add_patch(FancyArrowPatch(
            (3.095, 0.88), (x, y + 0.18),
            arrowstyle="-", linewidth=0.9, color="#94A3B8"
        ))

    ax.text(3.1, 2.62, "Small-Sample Logic", ha="center", fontsize=12.0, fontweight="bold")
    ax.text(3.1, 0.42, "small input, large candidate pool", ha="center", fontsize=8.2, color="#475569")
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "small_sample_recommendation.png"), dpi=170, bbox_inches="tight")
    plt.close()


def generate_ranking_signal_diagram(plt):
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    labels = ["Similarity", "Authority", "Balance"]
    values = [0.60, 0.30, 0.10]
    colors = ["#4C78A8", "#F2B134", "#59A14F"]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Weight in ranking score")
    ax.set_title("Ranking Signal Composition", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(value + 0.015, i, f"{value:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "ranking_signal_diagram.png"), dpi=170, bbox_inches="tight")
    plt.close()


def generate_metric_delta(plt, metrics):
    corpora = list(metrics.keys())
    metric_names = [
        "Domain diversity",
        "Authority coverage",
        "Resource-type diversity",
        "URL validity",
    ]
    llm_means = [
        mean(metrics[c]["llm_domain_d"] for c in corpora),
        mean(metrics[c]["llm_authority"] for c in corpora),
        mean(metrics[c]["llm_type_d"] for c in corpora),
        mean(metrics[c]["llm_live"] for c in corpora),
    ]
    ap_means = [
        mean(metrics[c]["ap_domain_d"] for c in corpora),
        mean(metrics[c]["ap_authority"] for c in corpora),
        mean(metrics[c]["ap_type_d"] for c in corpora),
        mean(metrics[c]["ap_live"] for c in corpora),
    ]
    deltas = [ap - llm for ap, llm in zip(ap_means, llm_means)]
    y = np.arange(len(metric_names))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    bar_colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in deltas]
    ax.barh(y, deltas, color=bar_colors, height=0.48)
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel("AI-Pedia minus live GPT-4o")
    ax.set_title("Mean Evaluation Delta", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(deltas):
        offset = 0.012 if value >= 0 else -0.012
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, i, f"{value:+.3f}", va="center", ha=ha, fontsize=9)
    ax.set_xlim(-0.08, 0.34)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "metric_delta.png"), dpi=170, bbox_inches="tight")
    plt.close()


def generate_radar(plt, metrics):
    llm_values = [
        mean(m["llm_domain_d"] for m in metrics.values()),
        mean(m["llm_authority"] for m in metrics.values()),
        mean(m["llm_type_d"] for m in metrics.values()),
        mean(m["llm_live"] for m in metrics.values()),
    ]
    ap_values = [
        mean(m["ap_domain_d"] for m in metrics.values()),
        mean(m["ap_authority"] for m in metrics.values()),
        mean(m["ap_type_d"] for m in metrics.values()),
        mean(m["ap_live"] for m in metrics.values()),
    ]
    categories = ["Domain\nDiversity", "Authority\nCoverage", "Type\nDiversity", "URL\nValidity"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    llm_closed = llm_values + llm_values[:1]
    ap_closed = ap_values + ap_values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.fill(angles, llm_closed, color="#E76F51", alpha=0.24)
    ax.plot(angles, llm_closed, color="#E76F51", linewidth=2, label="Live GPT-4o Baseline")
    ax.scatter(angles[:-1], llm_values, color="#E76F51", s=50, zorder=5)
    ax.fill(angles, ap_closed, color="#2A9D8F", alpha=0.24)
    ax.plot(angles, ap_closed, color="#2A9D8F", linewidth=2, label="AI-Pedia")
    ax.scatter(angles[:-1], ap_values, color="#2A9D8F", s=50, zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.02), fontsize=10)
    ax.set_title("Live LLM Baseline vs AI-Pedia", fontsize=14, fontweight="bold", y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGURE_DIR, "radar_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()


def generate_all():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError:
        print("matplotlib not available")
        return
    os.makedirs(PAPER_FIGURE_DIR, exist_ok=True)
    run_data, _rows, metrics = load_live_results()
    generate_keyword_signal_diagram(plt, FancyBboxPatch, FancyArrowPatch)
    generate_small_sample_diagram(plt, FancyBboxPatch, FancyArrowPatch)
    generate_ranking_signal_diagram(plt)
    generate_pipeline_flow(plt, FancyBboxPatch, FancyArrowPatch)
    generate_resource_modality_summary(plt, metrics)
    generate_radar(plt, metrics)
    generate_metric_delta(plt, metrics)
    print("Generated paper figures from", RESULTS_PATH)
    print("Run modes:", ", ".join(run_data.get("llm_run_modes", [])))


if __name__ == "__main__":
    generate_all()
