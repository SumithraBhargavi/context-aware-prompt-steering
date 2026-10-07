"""
generate_figures.py — CAPS Experiment Figure Generation

Reads CSV files from results/analysis/ and produces 8 publication-ready
figures into results/analysis/figures/.

All data comes directly from the aggregated CSVs — no numbers are
hard-coded. If insufficient data is available for a figure, it is skipped
with an informative message rather than crashing.

FIGURES PRODUCED
----------------
fig1_overall_conformance.png  – Baseline vs CAPS structural conformance (bar)
fig2_overall_functional.png   – Baseline vs CAPS functional correctness (bar)
fig3_violations.png           – Baseline vs CAPS violation count (bar)
fig4_pattern_conformance.png  – Conformance by design pattern
fig5_pattern_functional.png   – Functional correctness by design pattern
fig6_task_conformance.png     – Conformance by task (all 10)
fig7_model_conformance.png    – Conformance by model (skipped if only 1 model)
fig8_joint_matrix.png         – Struct×Func 2×2 outcome distribution

USAGE
-----
    python generate_figures.py

    # Use a custom analysis directory:
    python generate_figures.py --analysis-dir path/to/analysis

    # Restrict to a single model:
    python generate_figures.py --model gpt-4o
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

ANALYSIS_DIR = os.path.join("results", "analysis")
FIGURES_DIR  = os.path.join(ANALYSIS_DIR, "figures")
ENCODING     = "utf-8"

BASELINE_COLOR = "#4C72B0"   # muted blue
CAPS_COLOR     = "#DD8452"   # warm orange
FIGSIZE_BAR    = (7, 5)
FIGSIZE_WIDE   = (12, 5)
FIGSIZE_MATRIX = (7, 6)
DPI            = 150


# ============================================================
# DATA LOADING
# ============================================================

def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding=ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def _float(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _int(val: Optional[str]) -> int:
    if val is None or val == "":
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


# ============================================================
# MATPLOTLIB SETUP
# ============================================================

def _init_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        return plt, mticker
    except ImportError:
        print(
            "ERROR: matplotlib is not installed.\n"
            "Install it with:  pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)


# ============================================================
# COMMON CHART HELPERS
# ============================================================

def _bar_pair(
    ax,
    categories: List[str],
    baseline_vals: List[Optional[float]],
    caps_vals: List[Optional[float]],
    ylabel: str,
    title: str,
    pct: bool = False,
    lower_is_better: bool = False,
) -> None:
    """Draw a grouped bar chart comparing baseline and CAPS."""
    import numpy as np

    x   = np.arange(len(categories))
    w   = 0.35

    b_vals = [v * 100 if (pct and v is not None) else (v if v is not None else 0)
              for v in baseline_vals]
    c_vals = [v * 100 if (pct and v is not None) else (v if v is not None else 0)
              for v in caps_vals]

    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline",
                    color=BASELINE_COLOR, alpha=0.88, edgecolor="white")
    bars_c = ax.bar(x + w / 2, c_vals, w, label="CAPS",
                    color=CAPS_COLOR, alpha=0.88, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar in list(bars_b) + list(bars_c):
        h = bar.get_height()
        if h == 0:
            continue
        label = f"{h:.1f}%" if pct else f"{h:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (max(b_vals + c_vals, default=1) * 0.01),
            label,
            ha="center", va="bottom", fontsize=8,
        )


def _save(fig, path: str, plt) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(path)}")


# ============================================================
# FIGURE 1 — Overall Structural Conformance
# ============================================================

def fig1_overall_conformance(
    overall: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    rows = {r["condition"]: r for r in overall}
    if "baseline" not in rows or "caps" not in rows:
        print("  fig1: insufficient data (need both conditions). Skipped.")
        return False

    b = _float(rows["baseline"].get("interface_conformance_mean"))
    c = _float(rows["caps"].get("interface_conformance_mean"))
    if b is None or c is None:
        print("  fig1: conformance values missing. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    _bar_pair(ax, ["All Tasks"], [b], [c],
              ylabel="Interface Conformance (%)",
              title="Overall Structural Conformance\nBaseline vs CAPS",
              pct=True)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig1_overall_conformance.png"), plt)
    return True


# ============================================================
# FIGURE 2 — Overall Functional Correctness
# ============================================================

def fig2_overall_functional(
    overall: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    rows = {r["condition"]: r for r in overall}
    if "baseline" not in rows or "caps" not in rows:
        print("  fig2: insufficient data. Skipped.")
        return False

    b = _float(rows["baseline"].get("functional_valid_rate"))
    c = _float(rows["caps"].get("functional_valid_rate"))
    if b is None or c is None:
        print("  fig2: functional_valid_rate missing. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    _bar_pair(ax, ["All Tasks"], [b], [c],
              ylabel="Functional Correctness Rate (%)",
              title="Overall Functional Correctness\nBaseline vs CAPS",
              pct=True)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig2_overall_functional.png"), plt)
    return True


# ============================================================
# FIGURE 3 — Structural Violations
# ============================================================

def fig3_violations(
    overall: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    rows = {r["condition"]: r for r in overall}
    if "baseline" not in rows or "caps" not in rows:
        print("  fig3: insufficient data. Skipped.")
        return False

    b = _float(rows["baseline"].get("violation_count_mean"))
    c = _float(rows["caps"].get("violation_count_mean"))
    if b is None or c is None:
        print("  fig3: violation_count_mean missing. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    _bar_pair(ax, ["All Tasks"], [b], [c],
              ylabel="Mean Structural Violations (count)",
              title="Mean Structural Violations\nBaseline vs CAPS",
              pct=False, lower_is_better=True)
    ax.set_ylim(bottom=0)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig3_violations.png"), plt)
    return True


# ============================================================
# FIGURE 4 — Pattern Conformance
# ============================================================

def fig4_pattern_conformance(
    pattern: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    patterns_present = sorted({r["pattern"] for r in pattern if r["pattern"]})
    if not patterns_present:
        print("  fig4: no pattern data. Skipped.")
        return False

    b_vals, c_vals = [], []
    for pat in patterns_present:
        rows = [r for r in pattern if r["pattern"] == pat]
        b_row = next((r for r in rows if r["condition"] == "baseline"), None)
        c_row = next((r for r in rows if r["condition"] == "caps"), None)
        b_vals.append(_float(b_row.get("interface_conformance_mean")) if b_row else None)
        c_vals.append(_float(c_row.get("interface_conformance_mean")) if c_row else None)

    if all(v is None for v in b_vals + c_vals):
        print("  fig4: all conformance values None. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _bar_pair(ax, patterns_present, b_vals, c_vals,
              ylabel="Interface Conformance (%)",
              title="Structural Conformance by Design Pattern\nBaseline vs CAPS",
              pct=True)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig4_pattern_conformance.png"), plt)
    return True


# ============================================================
# FIGURE 5 — Pattern Functional Correctness
# ============================================================

def fig5_pattern_functional(
    pattern: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    patterns_present = sorted({r["pattern"] for r in pattern if r["pattern"]})
    if not patterns_present:
        print("  fig5: no pattern data. Skipped.")
        return False

    b_vals, c_vals = [], []
    for pat in patterns_present:
        rows = [r for r in pattern if r["pattern"] == pat]
        b_row = next((r for r in rows if r["condition"] == "baseline"), None)
        c_row = next((r for r in rows if r["condition"] == "caps"), None)
        b_vals.append(_float(b_row.get("functional_valid_rate")) if b_row else None)
        c_vals.append(_float(c_row.get("functional_valid_rate")) if c_row else None)

    if all(v is None for v in b_vals + c_vals):
        print("  fig5: all functional values None. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _bar_pair(ax, patterns_present, b_vals, c_vals,
              ylabel="Functional Correctness Rate (%)",
              title="Functional Correctness by Design Pattern\nBaseline vs CAPS",
              pct=True)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig5_pattern_functional.png"), plt)
    return True


# ============================================================
# FIGURE 6 — Task-Level Conformance
# ============================================================

def fig6_task_conformance(
    task: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    from src.experiment_runner import TASK_IDS

    # All 10 tasks in canonical order
    task_ids = TASK_IDS

    b_vals, c_vals = [], []
    for tid in task_ids:
        rows = [r for r in task if r["task_id"] == tid]
        b_row = next((r for r in rows if r["condition"] == "baseline"), None)
        c_row = next((r for r in rows if r["condition"] == "caps"), None)
        b_vals.append(_float(b_row.get("interface_conformance_mean")) if b_row else None)
        c_vals.append(_float(c_row.get("interface_conformance_mean")) if c_row else None)

    if all(v is None for v in b_vals + c_vals):
        print("  fig6: no task data. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=(13, 5))
    _bar_pair(ax, task_ids, b_vals, c_vals,
              ylabel="Interface Conformance (%)",
              title="Structural Conformance by Task\nBaseline vs CAPS",
              pct=True)
    ax.tick_params(axis="x", rotation=30)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig6_task_conformance.png"), plt)
    return True


# ============================================================
# FIGURE 7 — Model Comparison (skipped if only 1 model)
# ============================================================

def fig7_model_conformance(
    model_rows: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
) -> bool:
    models = sorted({r["model"] for r in model_rows if r["model"]})
    if len(models) < 2:
        print(f"  fig7: only {len(models)} model(s) present. Skipped.")
        return False

    b_vals, c_vals = [], []
    for m in models:
        rows = [r for r in model_rows if r["model"] == m]
        b_row = next((r for r in rows if r["condition"] == "baseline"), None)
        c_row = next((r for r in rows if r["condition"] == "caps"), None)
        b_vals.append(_float(b_row.get("interface_conformance_mean")) if b_row else None)
        c_vals.append(_float(c_row.get("interface_conformance_mean")) if c_row else None)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _bar_pair(ax, models, b_vals, c_vals,
              ylabel="Interface Conformance (%)",
              title="Structural Conformance by Model\nBaseline vs CAPS",
              pct=True)
    _save(fig, os.path.join(figures_dir, "fig7_model_conformance.png"), plt)
    return True


# ============================================================
# FIGURE 8 — Joint Struct × Func Matrix
# ============================================================

def fig8_joint_matrix(
    overall: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    import numpy as np

    baseline_row = next((r for r in overall if r["condition"] == "baseline"), None)
    caps_row     = next((r for r in overall if r["condition"] == "caps"), None)

    if not baseline_row or not caps_row:
        print("  fig8: need both conditions. Skipped.")
        return False

    def _counts(row: Dict[str, str]):
        return [
            _int(row.get("SS_FP")),
            _int(row.get("SS_FF")),
            _int(row.get("SN_FP")),
            _int(row.get("SN_FF")),
        ]

    b_counts = _counts(baseline_row)
    c_counts = _counts(caps_row)

    if sum(b_counts) == 0 and sum(c_counts) == 0:
        print("  fig8: all joint counts are zero. Skipped.")
        return False

    labels = [
        "Struct✓\nFunc✓",
        "Struct✓\nFunc✗",
        "Struct✗\nFunc✓",
        "Struct✗\nFunc✗",
    ]

    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, counts, title in [
        (ax1, b_counts, "Baseline"),
        (ax2, c_counts, "CAPS"),
    ]:
        total = sum(counts)
        pcts = [c / total * 100 if total > 0 else 0 for c in counts]
        bars = ax.bar(labels, pcts, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("% of runs", fontsize=11)
        ax.set_ylim(0, max(pcts + [10]) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, pct, cnt in zip(bars, pcts, counts):
            if cnt == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%\n(n={cnt})",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(
        "Structural × Functional Outcome Distribution",
        fontsize=14, fontweight="bold",
    )
    if model_filter:
        fig.text(0.5, 0.01, f"Model: {model_filter}",
                 ha="center", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig8_joint_matrix.png"), plt)
    return True


# ============================================================
# FIGURE 9 — Model × Pattern × Condition Heatmap
# ============================================================

def fig9_model_pattern_heatmap(
    pattern_rows: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
) -> bool:
    import numpy as np

    models = sorted({r["model"] for r in pattern_rows if r.get("model")})
    patterns = ["Factory", "Observer", "Strategy", "Singleton", "Adapter"]
    if not models or not pattern_rows:
        print("  fig9: insufficient pattern data. Skipped.")
        return False

    row_labels = []
    matrix_data = []

    for m in models:
        for cond in ["baseline", "caps"]:
            label = f"{m} ({cond})"
            row_labels.append(label)
            row_vals = []
            for pat in patterns:
                r = next((x for x in pattern_rows if x.get("model") == m and x.get("pattern") == pat and x.get("condition") == cond), None)
                val = _float(r.get("interface_conformance_mean")) if r else 0.0
                row_vals.append((val or 0.0) * 100)
            matrix_data.append(row_vals)

    data = np.array(matrix_data)
    fig, ax = plt.subplots(figsize=(9, max(4, len(row_labels) * 0.8)))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(patterns)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(patterns, fontsize=10, fontweight="bold")
    ax.set_yticklabels(row_labels, fontsize=10)

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(patterns)):
            val = data[i, j]
            color = "white" if val > 70 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_title("Model × Pattern × Condition Conformance Heatmap", fontsize=12, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Interface Conformance (%)", rotation=-90, va="bottom", fontsize=10)

    _save(fig, os.path.join(figures_dir, "fig9_model_pattern_heatmap.png"), plt)
    return True


# ============================================================
# FIGURE 10 — Task-Level Conformance Heatmap
# ============================================================

def fig10_task_conformance_heatmap(
    task_rows: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
) -> bool:
    import numpy as np
    from src.experiment_runner import TASK_IDS

    models = sorted({r["model"] for r in task_rows if r.get("model")})
    if not models or not task_rows:
        print("  fig10: insufficient task data. Skipped.")
        return False

    model_names = {
        "claude-sonnet-5": "Claude",
        "deepseek-v4.1-flash": "DeepSeek",
    }
    col_labels = []
    for m in models:
        col_labels.append(f"{model_names.get(m, m)}\nBaseline")
        col_labels.append(f"{model_names.get(m, m)}\nCAPS")

    matrix_data = []
    for tid in TASK_IDS:
        row_vals = []
        for m in models:
            b_r = next((x for x in task_rows if x.get("model") == m and x.get("task_id") == tid and x.get("condition") == "baseline"), None)
            c_r = next((x for x in task_rows if x.get("model") == m and x.get("task_id") == tid and x.get("condition") == "caps"), None)
            bv = (_float(b_r.get("interface_conformance_mean")) if b_r else 0.0) or 0.0
            cv = (_float(c_r.get("interface_conformance_mean")) if c_r else 0.0) or 0.0
            row_vals.append(bv * 100)
            row_vals.append(cv * 100)
        matrix_data.append(row_vals)

    data = np.array(matrix_data)
    fig, ax = plt.subplots(figsize=(8.5, 9.5), dpi=200)
    im = ax.imshow(data, cmap="OrRd", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(TASK_IDS)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(TASK_IDS, fontsize=10)
    ax.tick_params(axis="x", pad=8)

    for i in range(len(TASK_IDS)):
        for j in range(len(col_labels)):
            val = data[i, j]
            color = "white" if val > 75 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    ax.set_title("Task-Level Structural Conformance Heatmap", fontsize=12, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cbar.ax.set_ylabel("Interface Conformance (%)", rotation=-90, va="bottom", fontsize=10)

    fig.tight_layout()
    _save(fig, os.path.join(figures_dir, "fig10_task_conformance_heatmap.png"), plt)
    return True


# ============================================================
# FIGURE 11 — AST Validity
# ============================================================

def fig11_ast_validity(
    overall: List[Dict[str, str]],
    figures_dir: str,
    plt, mticker,
    model_filter: Optional[str],
) -> bool:
    rows = {r["condition"]: r for r in overall}
    if "baseline" not in rows or "caps" not in rows:
        print("  fig11: insufficient data. Skipped.")
        return False

    b = _float(rows["baseline"].get("ast_valid_rate"))
    c = _float(rows["caps"].get("ast_valid_rate"))
    if b is None or c is None:
        print("  fig11: ast_valid_rate missing. Skipped.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    _bar_pair(ax, ["All Tasks"], [b], [c],
              ylabel="AST Parse Validity Rate (%)",
              title="AST Parse Validity\nBaseline vs CAPS",
              pct=True)
    ax.set_ylim(0, 110)
    if model_filter:
        ax.set_xlabel(f"Model: {model_filter}", fontsize=9, color="gray")
    _save(fig, os.path.join(figures_dir, "fig11_ast_validity.png"), plt)
    return True


# ============================================================
# MAIN
# ============================================================

def run_figures(
    analysis_dir: str,
    figures_dir: str,
    model_filter: Optional[str],
) -> int:

    print("=" * 70)
    print("CAPS — FIGURE GENERATION")
    print("=" * 70)
    print(f"  Analysis dir : {analysis_dir}")
    print(f"  Figures dir  : {figures_dir}")
    if model_filter:
        print(f"  Model filter : {model_filter}")
    print()

    plt, mticker = _init_matplotlib()

    # ---- Load CSVs ----
    overall_rows = _load_csv(os.path.join(analysis_dir, "overall_summary.csv"))
    pattern_rows = _load_csv(os.path.join(analysis_dir, "pattern_summary.csv"))
    task_rows    = _load_csv(os.path.join(analysis_dir, "task_summary.csv"))
    model_rows   = _load_csv(os.path.join(analysis_dir, "model_summary.csv"))

    if not overall_rows:
        print(
            "No data found in results/analysis/.\n"
            "Run aggregate_results.py first after completing formal generations.\n"
        )
        return 0

    # Apply model filter
    if model_filter:
        pattern_rows = [r for r in pattern_rows if r.get("model") == model_filter]
        task_rows    = [r for r in task_rows    if r.get("model") == model_filter]

    os.makedirs(figures_dir, exist_ok=True)
    produced = 0

    fig_fns = [
        ("fig1", lambda: fig1_overall_conformance(overall_rows, figures_dir, plt, mticker, model_filter)),
        ("fig2", lambda: fig2_overall_functional(overall_rows, figures_dir, plt, mticker, model_filter)),
        ("fig3", lambda: fig3_violations(overall_rows, figures_dir, plt, mticker, model_filter)),
        ("fig4", lambda: fig4_pattern_conformance(pattern_rows, figures_dir, plt, mticker, model_filter)),
        ("fig5", lambda: fig5_pattern_functional(pattern_rows, figures_dir, plt, mticker, model_filter)),
        ("fig6", lambda: fig6_task_conformance(task_rows, figures_dir, plt, mticker, model_filter)),
        ("fig7", lambda: fig7_model_conformance(model_rows, figures_dir, plt, mticker)),
        ("fig8", lambda: fig8_joint_matrix(overall_rows, figures_dir, plt, mticker, model_filter)),
        ("fig9", lambda: fig9_model_pattern_heatmap(pattern_rows, figures_dir, plt, mticker)),
        ("fig10", lambda: fig10_task_conformance_heatmap(task_rows, figures_dir, plt, mticker)),
        ("fig11", lambda: fig11_ast_validity(overall_rows, figures_dir, plt, mticker, model_filter)),
    ]

    for name, fn in fig_fns:
        try:
            if fn():
                produced += 1
        except Exception as exc:
            print(f"  {name}: ERROR — {exc}")

    print()
    print(f"Figures produced: {produced}/11")
    print(f"Saved to        : {os.path.abspath(figures_dir)}")
    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_figures",
        description="CAPS figure generation from aggregated CSVs",
    )
    parser.add_argument(
        "--analysis-dir",
        default=ANALYSIS_DIR,
        help="Directory containing the aggregated CSVs (default: results/analysis)",
    )
    parser.add_argument(
        "--figures-dir",
        default=FIGURES_DIR,
        help="Output directory for figures (default: results/analysis/figures)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Restrict figures to this model only",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    return run_figures(
        analysis_dir=args.analysis_dir,
        figures_dir=args.figures_dir,
        model_filter=args.model,
    )


if __name__ == "__main__":
    sys.exit(main())
