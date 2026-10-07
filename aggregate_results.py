"""
aggregate_results.py — CAPS Experiment Results Aggregation

Reads all completed experiment records from results/experiments/ and
produces a structured results/analysis/ directory with:

    all_runs.csv          – one row per completed generation
    task_summary.csv      – per (model, task, condition) mean metrics
    pattern_summary.csv   – per (model, pattern, condition) aggregated
    model_summary.csv     – per (model, condition) overall
    overall_summary.csv   – grand total across all models
    statistical_tests.csv – Wilcoxon + McNemar results
    error_summary.csv     – structural/functional violation category counts

Also produces:
    aggregation.json      – full nested aggregation (machine-readable)
    statistical_tests.json – full statistical analysis (machine-readable)

PILOT DATA EXCLUSION
---------------------
Data in results/ (top level) is pilot data and is NOT included.
Only records under results/experiments/<model>/<task>/<condition>_run_<NN>/
are processed.

USAGE
-----
    python aggregate_results.py

    # Restrict to specific model(s):
    python aggregate_results.py --model gpt-4o

    # Specify alternate experiments directory:
    python aggregate_results.py --experiments-dir path/to/experiments
"""

import argparse
import csv
import json
import os
import sys
import statistics
from typing import Any, Dict, List, Optional, Tuple

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.experiment_runner import ExperimentRunner, TASK_IDS
from src.statistical_analysis import StatisticalAnalyzer
from src.error_analyzer import ErrorAnalyzer

ENCODING   = "utf-8"
ANALYSIS_DIR = os.path.join("results", "analysis")

PATTERNS = ["Factory", "Observer", "Strategy", "Singleton", "Adapter"]

# Map task_id → pattern name (canonical capitalisation)
TASK_PATTERN_MAP: Dict[str, str] = {
    "factory_01":  "Factory",
    "factory_02":  "Factory",
    "observer_01": "Observer",
    "observer_02": "Observer",
    "strategy_01": "Strategy",
    "strategy_02": "Strategy",
    "singleton_01":"Singleton",
    "singleton_02":"Singleton",
    "adapter_01":  "Adapter",
    "adapter_02":  "Adapter",
}

NUMERIC_METRICS = [
    "interface_conformance",
    "violation_count",
    "functional_score",
]

BINARY_METRICS = [
    "ast_valid",
    "functional_valid",
]


# ============================================================
# FLAT ROW BUILDER
# ============================================================

def _to_flat_row(record: Dict[str, Any]) -> Dict[str, Any]:
    ev = record.get("evaluation", {})
    task_id = record.get("task_id", "")
    return {
        "model":                   record.get("model"),
        "model_version":           record.get("model_version"),
        "task_id":                 task_id,
        "pattern":                 TASK_PATTERN_MAP.get(task_id, record.get("pattern", "")),
        "condition":               record.get("condition"),
        "run":                     record.get("run"),
        "timestamp":               record.get("timestamp"),
        "ast_valid":               ev.get("ast_valid"),
        "interface_conformance":   ev.get("interface_conformance"),
        "violation_count":         ev.get("violation_count"),
        "functional_valid":        ev.get("functional_valid"),
        "functional_score":        ev.get("functional_score"),
        "functional_tests_total":  ev.get("functional_tests_total"),
        "functional_tests_passed": ev.get("functional_tests_passed"),
        "data_complete":           record.get("data_complete", True),
    }


# ============================================================
# DESCRIPTIVE STATS
# ============================================================

def _mean(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.mean(clean) if clean else None


def _median(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def _stdev(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.stdev(clean) if len(clean) >= 2 else (0.0 if clean else None)


def _proportion_true(vals: List[Optional[bool]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return sum(1 for v in clean if v) / len(clean) if clean else None


def _summarise_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n": len(rows)}
    for m in NUMERIC_METRICS:
        vals = [r.get(m) for r in rows]
        summary[f"{m}_mean"]   = _mean(vals)
        summary[f"{m}_median"] = _median(vals)
        summary[f"{m}_stdev"]  = _stdev(vals)
    for m in BINARY_METRICS:
        vals = [r.get(m) for r in rows]
        summary[f"{m}_rate"] = _proportion_true(vals)
    return summary


# ============================================================
# JOINT STRUCT × FUNC MATRIX
# ============================================================

def _joint_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute 2×2 matrix:
      struct_conforming   = interface_conformance >= 0.8 (or None treated as False)
      func_valid          = functional_valid == True

    Categories:
      SS_FP: struct conforming + func pass
      SS_FF: struct conforming + func fail
      SN_FP: struct non-conforming + func pass
      SN_FF: struct non-conforming + func fail
    """
    threshold = 0.8  # treat ≥80% conformance as "structurally conforming"
    counts = {"SS_FP": 0, "SS_FF": 0, "SN_FP": 0, "SN_FF": 0}
    for row in rows:
        ic = row.get("interface_conformance")
        fv = row.get("functional_valid")
        if ic is None:
            continue
        struct_ok = float(ic) >= threshold
        func_ok   = bool(fv) if fv is not None else False
        if struct_ok and func_ok:
            counts["SS_FP"] += 1
        elif struct_ok and not func_ok:
            counts["SS_FF"] += 1
        elif not struct_ok and func_ok:
            counts["SN_FP"] += 1
        else:
            counts["SN_FF"] += 1
    n = sum(counts.values())
    proportions = {
        k: (v / n if n > 0 else None)
        for k, v in counts.items()
    }
    return {"counts": counts, "proportions": proportions, "n_with_data": n}


# ============================================================
# GROUP HELPERS
# ============================================================

def _group_rows(
    rows: List[Dict[str, Any]],
    keys: List[str],
) -> Dict[Tuple, List[Dict[str, Any]]]:
    groups: Dict[Tuple, List] = {}
    for row in rows:
        k = tuple(str(row.get(key, "")) for key in keys)
        groups.setdefault(k, []).append(row)
    return groups


# ============================================================
# CSV WRITERS
# ============================================================

def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding=ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


# ============================================================
# SUMMARY BUILDERS
# ============================================================

def build_task_summary(flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (model, task, condition) summary."""
    groups = _group_rows(flat_rows, ["model", "task_id", "condition"])
    out = []
    for (model, task_id, condition), rows in sorted(groups.items()):
        s = _summarise_group(rows)
        joint = _joint_matrix(rows)
        out.append({
            "model":           model,
            "task_id":         task_id,
            "pattern":         TASK_PATTERN_MAP.get(task_id, ""),
            "condition":       condition,
            "n_runs":          len(rows),
            "ast_valid_rate":          _fmt(s.get("ast_valid_rate")),
            "interface_conformance_mean":   _fmt(s.get("interface_conformance_mean")),
            "interface_conformance_median": _fmt(s.get("interface_conformance_median")),
            "interface_conformance_stdev":  _fmt(s.get("interface_conformance_stdev")),
            "violation_count_mean":        _fmt(s.get("violation_count_mean")),
            "violation_count_median":      _fmt(s.get("violation_count_median")),
            "functional_valid_rate":       _fmt(s.get("functional_valid_rate")),
            "functional_score_mean":       _fmt(s.get("functional_score_mean")),
            "SS_FP": joint["counts"]["SS_FP"],
            "SS_FF": joint["counts"]["SS_FF"],
            "SN_FP": joint["counts"]["SN_FP"],
            "SN_FF": joint["counts"]["SN_FF"],
        })
    return out


def build_pattern_summary(flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (model, pattern, condition) summary."""
    groups = _group_rows(flat_rows, ["model", "pattern", "condition"])
    out = []
    for (model, pattern, condition), rows in sorted(groups.items()):
        s = _summarise_group(rows)
        joint = _joint_matrix(rows)
        out.append({
            "model":           model,
            "pattern":         pattern,
            "condition":       condition,
            "n_runs":          len(rows),
            "ast_valid_rate":          _fmt(s.get("ast_valid_rate")),
            "interface_conformance_mean":   _fmt(s.get("interface_conformance_mean")),
            "interface_conformance_median": _fmt(s.get("interface_conformance_median")),
            "violation_count_mean":        _fmt(s.get("violation_count_mean")),
            "functional_valid_rate":       _fmt(s.get("functional_valid_rate")),
            "functional_score_mean":       _fmt(s.get("functional_score_mean")),
            "SS_FP": joint["counts"]["SS_FP"],
            "SS_FF": joint["counts"]["SS_FF"],
            "SN_FP": joint["counts"]["SN_FP"],
            "SN_FF": joint["counts"]["SN_FF"],
        })
    return out


def build_model_summary(flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (model, condition) summary."""
    groups = _group_rows(flat_rows, ["model", "condition"])
    out = []
    for (model, condition), rows in sorted(groups.items()):
        s = _summarise_group(rows)
        joint = _joint_matrix(rows)
        out.append({
            "model":           model,
            "condition":       condition,
            "n_runs":          len(rows),
            "ast_valid_rate":          _fmt(s.get("ast_valid_rate")),
            "interface_conformance_mean":   _fmt(s.get("interface_conformance_mean")),
            "interface_conformance_median": _fmt(s.get("interface_conformance_median")),
            "violation_count_mean":        _fmt(s.get("violation_count_mean")),
            "functional_valid_rate":       _fmt(s.get("functional_valid_rate")),
            "functional_score_mean":       _fmt(s.get("functional_score_mean")),
            "SS_FP": joint["counts"]["SS_FP"],
            "SS_FF": joint["counts"]["SS_FF"],
            "SN_FP": joint["counts"]["SN_FP"],
            "SN_FF": joint["counts"]["SN_FF"],
        })
    return out


def build_overall_summary(flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Grand summary across all models, split only by condition."""
    groups = _group_rows(flat_rows, ["condition"])
    out = []
    for (condition,), rows in sorted(groups.items()):
        s = _summarise_group(rows)
        joint = _joint_matrix(rows)
        out.append({
            "condition":       condition,
            "n_runs":          len(rows),
            "ast_valid_rate":          _fmt(s.get("ast_valid_rate")),
            "interface_conformance_mean":   _fmt(s.get("interface_conformance_mean")),
            "interface_conformance_median": _fmt(s.get("interface_conformance_median")),
            "violation_count_mean":        _fmt(s.get("violation_count_mean")),
            "functional_valid_rate":       _fmt(s.get("functional_valid_rate")),
            "functional_score_mean":       _fmt(s.get("functional_score_mean")),
            "SS_FP": joint["counts"]["SS_FP"],
            "SS_FF": joint["counts"]["SS_FF"],
            "SN_FP": joint["counts"]["SN_FP"],
            "SN_FF": joint["counts"]["SN_FF"],
        })
    return out


# ============================================================
# STATISTICAL TESTS CSV
# ============================================================

def build_stats_csv(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the statistical_analysis.json into CSV rows."""
    rows = []
    for model_id, model_data in analysis.get("per_model", {}).items():
        for metric, result in model_data.get("wilcoxon", {}).items():
            rows.append({
                "model":           model_id,
                "test":            result.get("test", "Wilcoxon signed-rank"),
                "metric":          metric,
                "sample_size":     result.get("sample_size"),
                "n_discordant":    result.get("n_discordant"),
                "statistic":       result.get("statistic"),
                "p_value":         result.get("p_value"),
                "effect_size_r":   result.get("effect_size_r"),
                "direction":       result.get("direction"),
                "mean_difference": result.get("mean_difference"),
                "limitation_note": result.get("limitation_note", ""),
            })
    return rows


# ============================================================
# MAIN AGGREGATION
# ============================================================

def run_aggregation(
    experiments_dir: str,
    analysis_dir: str,
    model_filter: Optional[str],
) -> int:

    print("=" * 70)
    print("CAPS — RESULTS AGGREGATION")
    print("=" * 70)
    print(f"  Experiments dir : {experiments_dir}")
    print(f"  Output dir      : {analysis_dir}")
    if model_filter:
        print(f"  Model filter    : {model_filter}")
    print()

    # ---- Load records ----
    runner = ExperimentRunner(
        dataset_path=os.path.join("dataset", "test_tasks.json"),
        base_dir=experiments_dir,
    )
    all_records = runner.load_all_records()

    if model_filter:
        all_records = [
            r for r in all_records
            if r.get("model") == model_filter
        ]

    print(f"Records loaded: {len(all_records)}")

    if not all_records:
        print(
            "\nNo formal experiment records found.\n"
            "Run formal generations first using collect_generation.py.\n"
            "Output directory will NOT be created with empty data."
        )
        return 0

    # ---- Warn about incomplete records ----
    incomplete = [r for r in all_records if not r.get("data_complete", True)]
    if incomplete:
        print(
            f"WARNING: {len(incomplete)} record(s) are incomplete "
            "(missing raw_code.txt or evaluation.json). "
            "They are included but flagged."
        )

    # ---- Build flat rows ----
    flat_rows = [_to_flat_row(r) for r in all_records]

    os.makedirs(analysis_dir, exist_ok=True)

    # ---- all_runs.csv ----
    all_runs_fields = [
        "model", "model_version", "task_id", "pattern",
        "condition", "run", "timestamp",
        "ast_valid", "interface_conformance", "violation_count",
        "functional_valid", "functional_score",
        "functional_tests_total", "functional_tests_passed",
        "data_complete",
    ]
    all_runs_path = os.path.join(analysis_dir, "all_runs.csv")
    _write_csv(all_runs_path, all_runs_fields, flat_rows)
    print(f"  Written: all_runs.csv          ({len(flat_rows)} rows)")

    # ---- task_summary.csv ----
    task_rows = build_task_summary(flat_rows)
    task_fields = [
        "model", "task_id", "pattern", "condition", "n_runs",
        "ast_valid_rate",
        "interface_conformance_mean", "interface_conformance_median",
        "interface_conformance_stdev",
        "violation_count_mean", "violation_count_median",
        "functional_valid_rate", "functional_score_mean",
        "SS_FP", "SS_FF", "SN_FP", "SN_FF",
    ]
    task_path = os.path.join(analysis_dir, "task_summary.csv")
    _write_csv(task_path, task_fields, task_rows)
    print(f"  Written: task_summary.csv      ({len(task_rows)} rows)")

    # ---- pattern_summary.csv ----
    pattern_rows = build_pattern_summary(flat_rows)
    pattern_fields = [
        "model", "pattern", "condition", "n_runs",
        "ast_valid_rate",
        "interface_conformance_mean", "interface_conformance_median",
        "violation_count_mean",
        "functional_valid_rate", "functional_score_mean",
        "SS_FP", "SS_FF", "SN_FP", "SN_FF",
    ]
    pattern_path = os.path.join(analysis_dir, "pattern_summary.csv")
    _write_csv(pattern_path, pattern_fields, pattern_rows)
    print(f"  Written: pattern_summary.csv   ({len(pattern_rows)} rows)")

    # ---- model_summary.csv ----
    model_rows = build_model_summary(flat_rows)
    model_fields = [
        "model", "condition", "n_runs",
        "ast_valid_rate",
        "interface_conformance_mean", "interface_conformance_median",
        "violation_count_mean",
        "functional_valid_rate", "functional_score_mean",
        "SS_FP", "SS_FF", "SN_FP", "SN_FF",
    ]
    model_path = os.path.join(analysis_dir, "model_summary.csv")
    _write_csv(model_path, model_fields, model_rows)
    print(f"  Written: model_summary.csv     ({len(model_rows)} rows)")

    # ---- overall_summary.csv ----
    overall_rows = build_overall_summary(flat_rows)
    overall_fields = [
        "condition", "n_runs",
        "ast_valid_rate",
        "interface_conformance_mean", "interface_conformance_median",
        "violation_count_mean",
        "functional_valid_rate", "functional_score_mean",
        "SS_FP", "SS_FF", "SN_FP", "SN_FF",
    ]
    overall_path = os.path.join(analysis_dir, "overall_summary.csv")
    _write_csv(overall_path, overall_fields, overall_rows)
    print(f"  Written: overall_summary.csv   ({len(overall_rows)} rows)")

    # ---- statistical_tests.csv ----
    # Build flat dicts from records for StatisticalAnalyzer
    stat_records = []
    for rec in all_records:
        ev = rec.get("evaluation", {})
        stat_records.append({
            "task_id":               rec.get("task_id"),
            "pattern":               TASK_PATTERN_MAP.get(rec.get("task_id", ""), rec.get("pattern", "")),
            "model":                 rec.get("model"),
            "condition":             rec.get("condition"),
            "run":                   rec.get("run"),
            "ast_valid":             ev.get("ast_valid"),
            "interface_conformance": ev.get("interface_conformance"),
            "violation_count":       ev.get("violation_count"),
            "functional_valid":      ev.get("functional_valid"),
            "functional_score":      ev.get("functional_score"),
        })

    analyzer = StatisticalAnalyzer(stat_records)
    analysis = analyzer.analyze()

    # Save full JSON
    stats_json_path = os.path.join(analysis_dir, "statistical_tests.json")
    with open(stats_json_path, "w", encoding=ENCODING) as f:
        json.dump(analysis, f, indent=4, default=str)

    # Save flat CSV
    stats_csv_rows = build_stats_csv(analysis)
    stats_csv_fields = [
        "model", "test", "metric", "sample_size", "n_discordant",
        "statistic", "p_value", "effect_size_r",
        "direction", "mean_difference", "limitation_note",
    ]
    stats_path = os.path.join(analysis_dir, "statistical_tests.csv")
    _write_csv(stats_path, stats_csv_fields, stats_csv_rows)
    print(f"  Written: statistical_tests.csv ({len(stats_csv_rows)} rows)")

    # ---- error_summary.csv ----
    # ErrorAnalyzer needs evaluation_file paths from the raw metadata records
    ea = ErrorAnalyzer(all_records)
    err_path = os.path.join(analysis_dir, "error_summary.csv")
    ea.save(path=err_path)
    viol_rows = ea.extract_all_violations()
    print(f"  Written: error_summary.csv     ({len(viol_rows)} violation rows)")

    # ---- aggregation.json (full nested) ----
    from src.results_aggregator import ResultsAggregator
    agg = ResultsAggregator(
        experiment_runner=runner,
        results_dir=analysis_dir,
    )
    agg._records = all_records   # inject already-loaded records
    agg_data = agg.aggregate()
    agg_path = os.path.join(analysis_dir, "aggregation.json")
    with open(agg_path, "w", encoding=ENCODING) as f:
        json.dump(agg_data, f, indent=4, default=str)
    print(f"  Written: aggregation.json")

    print()
    print(f"All output written to: {os.path.abspath(analysis_dir)}")
    print("=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)

    models = sorted({r.get("model") for r in all_records if r.get("model")})
    print(f"  Models found   : {', '.join(models) if models else 'none'}")
    print(f"  Total records  : {len(all_records)}")
    print(f"  Incomplete     : {len(incomplete)}")
    print()
    print("Next step:")
    print("  python generate_figures.py")
    print("=" * 70)

    return 0


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggregate_results",
        description="CAPS experiment results aggregation",
    )
    parser.add_argument(
        "--experiments-dir",
        default=os.path.join("results", "experiments"),
        help="Path to the experiments directory (default: results/experiments)",
    )
    parser.add_argument(
        "--analysis-dir",
        default=ANALYSIS_DIR,
        help="Output directory for analysis files (default: results/analysis)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Restrict aggregation to this model ID only",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    return run_aggregation(
        experiments_dir=args.experiments_dir,
        analysis_dir=args.analysis_dir,
        model_filter=args.model,
    )


if __name__ == "__main__":
    sys.exit(main())
