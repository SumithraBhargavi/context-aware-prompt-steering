"""
ResultsAggregator – Phase 7 of the CAPS research infrastructure.

Loads all completed experiment metadata records and produces:
  - results/summary.csv    – flat per-observation CSV
  - results/summary.json   – flat per-observation JSON
  - results/statistical_analysis.json  (via StatisticalAnalyzer)

Also provides in-memory breakdowns by:
  - pattern
  - model
  - condition (baseline vs. caps)
  - pattern × condition
  - model × condition

IMPORTANT:
  - Pilot data (results/factory_01_*) is explicitly excluded from
    aggregation. It lives at the top level of results/ and is
    clearly labeled as pilot data only.
  - Records with data_complete=False are flagged but included so
    partial results are not silently dropped.
  - No metrics are fabricated. Missing values are represented as None.
"""

import os
import csv
import json
import statistics
from typing import Dict, Any, List, Optional

from src.experiment_runner import ExperimentRunner


# ============================================================
# CONSTANTS
# ============================================================

RESULTS_DIR = "results"
ENCODING = "utf-8"

# Fields to include in the flat CSV/JSON summary
FLAT_FIELDS = [
    "task_id",
    "pattern",
    "model",
    "model_version",
    "condition",
    "run",
    "timestamp",
    "ast_valid",
    "interface_conformance",
    "violation_count",
    "functional_valid",
    "functional_score",
    "functional_tests_total",
    "functional_tests_passed",
    "data_complete",
]

# Numeric metrics for descriptive statistics
NUMERIC_METRICS = [
    "interface_conformance",
    "violation_count",
    "functional_score",
    "functional_tests_passed",
]

# Binary metrics for proportion summaries
BINARY_METRICS = [
    "ast_valid",
    "functional_valid",
]


# ============================================================
# RESULTS AGGREGATOR
# ============================================================

class ResultsAggregator:
    """
    Aggregates completed experiment records into summary files.

    Args:
        experiment_runner: An ExperimentRunner instance (provides
                           load_all_records()).
        results_dir:       Directory where summary files are written.
    """

    def __init__(
        self,
        experiment_runner: Optional[ExperimentRunner] = None,
        results_dir: str = RESULTS_DIR,
    ):
        self.results_dir = results_dir

        if experiment_runner is None:
            experiment_runner = ExperimentRunner()

        self._runner = experiment_runner
        self._records: List[Dict[str, Any]] = []

    # =========================================================
    # LOAD
    # =========================================================

    def load(self) -> int:
        """
        Load all completed experiment records.

        Returns:
            Number of records loaded.
        """

        self._records = self._runner.load_all_records()
        return len(self._records)

    # =========================================================
    # FLAT ROW BUILDING
    # =========================================================

    def _to_flat_row(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert a metadata record to a flat dict for CSV/JSON output.
        """

        ev = record.get("evaluation", {})

        return {
            "task_id":                 record.get("task_id"),
            "pattern":                 record.get("pattern"),
            "model":                   record.get("model"),
            "model_version":           record.get("model_version"),
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

    # =========================================================
    # DESCRIPTIVE STATISTICS
    # =========================================================

    @staticmethod
    def _describe(values: List[Optional[float]]) -> Dict[str, Any]:
        """
        Compute descriptive statistics for a list of numeric values.
        None values are excluded.

        Returns dict with: n, mean, median, stdev, min, max
        """

        clean = [v for v in values if v is not None]

        if not clean:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "stdev": None,
                "min": None,
                "max": None,
            }

        return {
            "n": len(clean),
            "mean": statistics.mean(clean),
            "median": statistics.median(clean),
            "stdev": (
                statistics.stdev(clean) if len(clean) >= 2 else 0.0
            ),
            "min": min(clean),
            "max": max(clean),
        }

    @staticmethod
    def _binary_proportion(
        values: List[Optional[bool]],
    ) -> Dict[str, Any]:
        """
        Compute proportion True for a binary metric.
        None values are excluded.
        """

        clean = [v for v in values if v is not None]

        if not clean:
            return {"n": 0, "proportion_true": None, "count_true": 0}

        n_true = sum(1 for v in clean if v)

        return {
            "n": len(clean),
            "count_true": n_true,
            "proportion_true": n_true / len(clean),
        }

    def _summarize_group(
        self, rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Produce a descriptive summary for a group of flat rows.
        """

        summary: Dict[str, Any] = {"n": len(rows)}

        for metric in NUMERIC_METRICS:
            vals = [r.get(metric) for r in rows]
            summary[metric] = self._describe(vals)

        for metric in BINARY_METRICS:
            vals = [r.get(metric) for r in rows]
            summary[metric] = self._binary_proportion(vals)

        return summary

    # =========================================================
    # AGGREGATE
    # =========================================================

    def aggregate(self) -> Dict[str, Any]:
        """
        Build the complete aggregation.

        If no records are loaded, calls load() automatically.

        Returns:
            dict with overall, by_pattern, by_model, by_condition,
            by_pattern_condition, by_model_condition breakdowns.
        """

        if not self._records:
            self.load()

        flat_rows = [self._to_flat_row(r) for r in self._records]

        result = {
            "total_records": len(flat_rows),
            "overall": self._summarize_group(flat_rows),
            "by_condition": self._group_by(flat_rows, "condition"),
            "by_pattern": self._group_by(flat_rows, "pattern"),
            "by_model": self._group_by(flat_rows, "model"),
            "by_pattern_condition": self._group_by_two(
                flat_rows, "pattern", "condition"
            ),
            "by_model_condition": self._group_by_two(
                flat_rows, "model", "condition"
            ),
        }

        return result

    def _group_by(
        self,
        rows: List[Dict[str, Any]],
        key: str,
    ) -> Dict[str, Any]:
        """Group rows by a single key and summarize each group."""

        groups: Dict[str, List] = {}

        for row in rows:
            k = str(row.get(key, "unknown"))
            groups.setdefault(k, []).append(row)

        return {
            k: self._summarize_group(v)
            for k, v in sorted(groups.items())
        }

    def _group_by_two(
        self,
        rows: List[Dict[str, Any]],
        key1: str,
        key2: str,
    ) -> Dict[str, Any]:
        """Group rows by two keys and summarize each group."""

        groups: Dict[str, List] = {}

        for row in rows:
            k = f"{row.get(key1, 'unknown')}__{row.get(key2, 'unknown')}"
            groups.setdefault(k, []).append(row)

        return {
            k: self._summarize_group(v)
            for k, v in sorted(groups.items())
        }

    # =========================================================
    # SAVE
    # =========================================================

    def save_summary_csv(
        self,
        path: Optional[str] = None,
    ) -> str:
        """
        Write flat per-observation CSV.

        Args:
            path: Output path. Defaults to results/summary.csv.

        Returns:
            Absolute path to the written file.
        """

        if path is None:
            path = os.path.join(self.results_dir, "summary.csv")

        if not self._records:
            self.load()

        flat_rows = [self._to_flat_row(r) for r in self._records]

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="", encoding=ENCODING) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=FLAT_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(flat_rows)

        return os.path.abspath(path)

    def save_summary_json(
        self,
        path: Optional[str] = None,
    ) -> str:
        """
        Write flat per-observation JSON.

        Args:
            path: Output path. Defaults to results/summary.json.

        Returns:
            Absolute path to the written file.
        """

        if path is None:
            path = os.path.join(self.results_dir, "summary.json")

        if not self._records:
            self.load()

        flat_rows = [self._to_flat_row(r) for r in self._records]

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding=ENCODING) as f:
            json.dump(flat_rows, f, indent=4, default=str)

        return os.path.abspath(path)

    def save_all(self) -> Dict[str, str]:
        """
        Save summary.csv, summary.json, and aggregation.json.

        Returns:
            dict mapping file type to absolute path.
        """

        csv_path = self.save_summary_csv()
        json_path = self.save_summary_json()

        agg = self.aggregate()
        agg_path = os.path.join(self.results_dir, "aggregation.json")
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)

        with open(agg_path, "w", encoding=ENCODING) as f:
            json.dump(agg, f, indent=4, default=str)

        return {
            "summary_csv": csv_path,
            "summary_json": json_path,
            "aggregation_json": os.path.abspath(agg_path),
        }

    # =========================================================
    # PILOT DATA NOTE
    # =========================================================

    @staticmethod
    def pilot_data_note() -> str:
        """
        Return a human-readable note about the pilot data exclusion.
        """

        return (
            "PILOT DATA NOTE:\n"
            "  Pilot data for factory_01 (1 baseline + 1 CAPS run) is stored\n"
            "  in results/factory_01_* and results/factory_01_experiment_result.json.\n"
            "  This data is NOT included in the final aggregation and statistics.\n"
            "  It was collected before the formal experimental protocol was finalized.\n"
            "  It is preserved for transparency but must not be cited as final results.\n"
        )
