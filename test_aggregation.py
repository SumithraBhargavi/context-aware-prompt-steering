"""
test_aggregation.py — Synthetic-data tests for the CAPS aggregation pipeline.

IMPORTANT:
  - All data in this file is SYNTHETIC (fabricated for testing only).
  - This test NEVER reads from or writes to results/experiments/.
  - Synthetic records are written to a temporary directory and deleted
    after each test.
  - This file does NOT test the correctness of LLM-generated code or
    any experimental results.

Tests cover:
  - aggregate_results.py produces all 7 required CSVs
  - generate_figures.py produces figures without crashing
  - StatisticalAnalyzer handles small/empty datasets gracefully
  - ErrorAnalyzer categorizes known violation strings correctly
  - collect_generation.py CLI can be imported without error
"""

import json
import os
import sys
import csv
import tempfile
import unittest
from typing import Any, Dict, List

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# SYNTHETIC DATA FACTORY
# ============================================================

def _make_synthetic_record(
    model: str,
    task_id: str,
    pattern: str,
    condition: str,
    run: int,
    ast_valid: bool = True,
    conformance: float = 0.9,
    violations: int = 1,
    functional_valid: bool = True,
    functional_score: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a minimal metadata.json-compatible record.
    All values are synthetic — NOT real experiment data.
    """
    return {
        "task_id":       task_id,
        "pattern":       pattern,
        "model":         model,
        "model_version": None,
        "condition":     condition,
        "run":           run,
        "timestamp":     "2026-01-01T00:00:00Z",
        "prompt_file":   None,
        "raw_code_file": None,
        "normalized_code_file": None,
        "evaluation": {
            "ast_valid":               ast_valid,
            "interface_conformance":   conformance,
            "violation_count":         violations,
            "functional_valid":        functional_valid,
            "functional_score":        functional_score,
            "functional_tests_total":  5,
            "functional_tests_passed": 5 if functional_valid else 3,
        },
        "evaluation_file": None,
        "data_complete":   True,
    }


SYNTHETIC_TASKS = [
    ("factory_01",  "Factory"),
    ("factory_02",  "Factory"),
    ("observer_01", "Observer"),
    ("observer_02", "Observer"),
    ("strategy_01", "Strategy"),
    ("strategy_02", "Strategy"),
    ("singleton_01","Singleton"),
    ("singleton_02","Singleton"),
    ("adapter_01",  "Adapter"),
    ("adapter_02",  "Adapter"),
]


def _make_synthetic_records(
    models: List[str] = ("test_model_A",),
    runs: int = 3,
) -> List[Dict[str, Any]]:
    """
    Create a full synthetic dataset: all 10 tasks × 2 conditions × N runs.
    Baseline has 80% conformance; CAPS has 95% conformance.
    NEVER use for any paper result.
    """
    records = []
    for model in models:
        for task_id, pattern in SYNTHETIC_TASKS:
            for run in range(1, runs + 1):
                records.append(_make_synthetic_record(
                    model=model, task_id=task_id, pattern=pattern,
                    condition="baseline", run=run,
                    conformance=0.80, violations=3,
                    functional_valid=False, functional_score=0.6,
                ))
                records.append(_make_synthetic_record(
                    model=model, task_id=task_id, pattern=pattern,
                    condition="caps", run=run,
                    conformance=0.95, violations=1,
                    functional_valid=True, functional_score=1.0,
                ))
    return records


# ============================================================
# HELPERS
# ============================================================

def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# TEST: ErrorAnalyzer categorization
# ============================================================

class TestErrorAnalyzerCategorization(unittest.TestCase):

    def test_structural_categories(self):
        from src.error_analyzer import ErrorAnalyzer
        cases = [
            ("AST parse error: invalid syntax at line 3",   "syntax_ast_failure"),
            ("Missing required class: NotificationFactory", "missing_class"),
            ("EmailNotification should inherit from Notification", "incorrect_inheritance"),
            ("Notification should be abstract",              "incorrect_class_abstraction"),
            ("NotificationFactory should be concrete",       "incorrect_class_abstraction"),
            ("Missing method: Notification.send",            "missing_method"),
            ("Notification.send should be abstract",         "incorrect_method_abstraction"),
            ("Factory.get_instance should be static",        "incorrect_static_declaration"),
            ("Parameter count mismatch in Notification.send: expected 0, found 2",
                                                             "parameter_count_mismatch"),
            ("Parameter name mismatch in Adapter.request",   "parameter_name_mismatch"),
            ("Parameter type mismatch in Adapter.request(target): expected 'str', found '<missing>'",
                                                             "parameter_type_mismatch"),
            ("Return type mismatch in NotificationFactory.create_notification",
                                                             "return_type_mismatch"),
        ]
        for message, expected_category in cases:
            with self.subTest(message=message):
                result = ErrorAnalyzer.categorize_structural_violation(message)
                self.assertEqual(
                    result, expected_category,
                    f"Expected '{expected_category}' for: {message!r}, got '{result}'"
                )

    def test_functional_categories(self):
        from src.error_analyzer import ErrorAnalyzer
        cases = [
            ("Execution timed out after 30 seconds", "execution_timeout"),
            ("SyntaxError: invalid syntax",          "syntax_runtime_error"),
            ("ImportError: cannot import name",      "import_error"),
            ("TypeError: __init__ got wrong args",   "instantiation_error"),
            ("AttributeError: method not callable",  "method_not_callable"),
            ("FAIL: assertion_failed",               "assertion_failure"),
            ("Something entirely unexpected",        "functional_other"),
        ]
        for message, expected_category in cases:
            with self.subTest(message=message):
                result = ErrorAnalyzer.categorize_functional_failure(message)
                self.assertEqual(result, expected_category)

    def test_unknown_structural_returns_uncategorized(self):
        from src.error_analyzer import ErrorAnalyzer
        result = ErrorAnalyzer.categorize_structural_violation(
            "Some completely novel violation message XYZ"
        )
        self.assertEqual(result, "uncategorized_structural")


# ============================================================
# TEST: StatisticalAnalyzer with synthetic data
# ============================================================

class TestStatisticalAnalyzerSynthetic(unittest.TestCase):

    def _make_flat_records(self):
        """Flat dict records suitable for StatisticalAnalyzer."""
        records = []
        for task_id, pattern in SYNTHETIC_TASKS:
            records.append({
                "task_id": task_id, "pattern": pattern,
                "model": "test_model_A", "condition": "baseline", "run": 1,
                "ast_valid": True, "interface_conformance": 0.80,
                "violation_count": 3, "functional_valid": False, "functional_score": 0.6,
            })
            records.append({
                "task_id": task_id, "pattern": pattern,
                "model": "test_model_A", "condition": "caps", "run": 1,
                "ast_valid": True, "interface_conformance": 0.95,
                "violation_count": 1, "functional_valid": True, "functional_score": 1.0,
            })
        return records

    def test_build_pairs_returns_correct_count(self):
        from src.statistical_analysis import StatisticalAnalyzer
        records = self._make_flat_records()
        sa = StatisticalAnalyzer(records)
        pairs = sa.build_pairs(model="test_model_A")
        # 10 tasks × 1 run = 10 pairs
        self.assertEqual(len(pairs), 10)
        self.assertTrue(all(p["complete"] for p in pairs))

    def test_build_task_pairs_aggregates_runs(self):
        from src.statistical_analysis import StatisticalAnalyzer
        # 3 runs per condition
        records = []
        for task_id, pattern in SYNTHETIC_TASKS[:3]:
            for run in range(1, 4):
                records.append({
                    "task_id": task_id, "pattern": pattern,
                    "model": "test_model_A", "condition": "baseline", "run": run,
                    "ast_valid": True, "interface_conformance": 0.80,
                    "violation_count": 3, "functional_valid": False, "functional_score": 0.6,
                })
                records.append({
                    "task_id": task_id, "pattern": pattern,
                    "model": "test_model_A", "condition": "caps", "run": run,
                    "ast_valid": True, "interface_conformance": 0.95,
                    "violation_count": 1, "functional_valid": True, "functional_score": 1.0,
                })
        sa = StatisticalAnalyzer(records)
        task_pairs = sa.build_task_pairs(model="test_model_A")
        self.assertEqual(len(task_pairs), 3)
        for p in task_pairs:
            self.assertAlmostEqual(p["baseline_interface_conformance"], 0.80)
            self.assertAlmostEqual(p["caps_interface_conformance"], 0.95)

    def test_wilcoxon_direction_and_fields(self):
        from src.statistical_analysis import StatisticalAnalyzer
        records = self._make_flat_records()
        sa = StatisticalAnalyzer(records)
        pairs = sa.build_pairs(model="test_model_A")
        result = sa.wilcoxon_test(pairs, "interface_conformance")
        self.assertIn("test", result)
        self.assertIn("p_value", result)
        self.assertIn("direction", result)
        # CAPS (0.95) > baseline (0.80)
        self.assertEqual(result["direction"], "CAPS > baseline")

    def test_analyze_returns_per_model_structure(self):
        from src.statistical_analysis import StatisticalAnalyzer
        records = self._make_flat_records()
        sa = StatisticalAnalyzer(records)
        analysis = sa.analyze(models=["test_model_A"])
        self.assertIn("per_model", analysis)
        self.assertIn("test_model_A", analysis["per_model"])

    def test_empty_records_does_not_crash(self):
        from src.statistical_analysis import StatisticalAnalyzer
        sa = StatisticalAnalyzer([])
        analysis = sa.analyze(models=[])
        self.assertEqual(analysis["per_model"], {})


# ============================================================
# TEST: aggregate_results.py with synthetic data
# ============================================================

class TestAggregationPipeline(unittest.TestCase):

    def setUp(self):
        """Create a fully synthetic experiments directory structure."""
        self._tmpdir = tempfile.mkdtemp(prefix="caps_test_agg_")
        self._experiments_dir = os.path.join(self._tmpdir, "experiments")
        self._analysis_dir    = os.path.join(self._tmpdir, "analysis")
        os.makedirs(self._experiments_dir)

        # Write synthetic metadata.json files
        records = _make_synthetic_records(
            models=["test_model_A"],
            runs=3,
        )
        for rec in records:
            model_id  = rec["model"]
            task_id   = rec["task_id"]
            condition = rec["condition"]
            run       = rec["run"]
            run_dir   = os.path.join(
                self._experiments_dir,
                model_id, task_id,
                f"{condition}_run_{run:02d}",
            )
            os.makedirs(run_dir, exist_ok=True)
            meta_path = os.path.join(run_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=4)
            # Write stub raw_code.txt and evaluation.json so data_complete=True
            with open(os.path.join(run_dir, "raw_code.txt"), "w") as f:
                f.write("# synthetic\n")
            eval_data = {
                "structural": {
                    "ast_valid": rec["evaluation"]["ast_valid"],
                    "violations": [],
                },
                "functional": {
                    "failures": [],
                    "timed_out": False,
                    "execution_error": None,
                },
            }
            with open(os.path.join(run_dir, "evaluation.json"), "w") as f:
                json.dump(eval_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run_aggregation(self):
        """Import and run aggregate_results directly with synthetic dirs."""
        # We call the aggregation function directly to avoid subprocess overhead
        import aggregate_results as agg_module
        return agg_module.run_aggregation(
            experiments_dir=self._experiments_dir,
            analysis_dir=self._analysis_dir,
            model_filter=None,
        )

    def test_aggregation_exits_zero(self):
        rc = self._run_aggregation()
        self.assertEqual(rc, 0)

    def test_all_required_csv_files_produced(self):
        self._run_aggregation()
        required_files = [
            "all_runs.csv",
            "task_summary.csv",
            "pattern_summary.csv",
            "model_summary.csv",
            "overall_summary.csv",
            "statistical_tests.csv",
            "error_summary.csv",
        ]
        for fname in required_files:
            path = os.path.join(self._analysis_dir, fname)
            self.assertTrue(
                os.path.exists(path),
                f"Missing required output file: {fname}",
            )

    def test_all_runs_csv_row_count(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "all_runs.csv"))
        # 10 tasks × 2 conditions × 3 runs = 60 rows
        self.assertEqual(len(rows), 60)

    def test_task_summary_has_both_conditions(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "task_summary.csv"))
        conditions = {r["condition"] for r in rows}
        self.assertIn("baseline", conditions)
        self.assertIn("caps", conditions)

    def test_pattern_summary_has_all_patterns(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "pattern_summary.csv"))
        patterns = {r["pattern"] for r in rows}
        expected = {"Factory", "Observer", "Strategy", "Singleton", "Adapter"}
        self.assertEqual(patterns, expected)

    def test_overall_summary_has_both_conditions(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "overall_summary.csv"))
        conditions = {r["condition"] for r in rows}
        self.assertIn("baseline", conditions)
        self.assertIn("caps", conditions)

    def test_statistical_tests_csv_has_expected_metrics(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "statistical_tests.csv"))
        metrics = {r["metric"] for r in rows}
        for m in ["interface_conformance", "violation_count", "functional_score"]:
            self.assertIn(m, metrics, f"Missing metric in statistical_tests.csv: {m}")

    def test_overall_summary_conformance_values_in_range(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "overall_summary.csv"))
        for row in rows:
            val = row.get("interface_conformance_mean")
            if val:
                self.assertGreaterEqual(float(val), 0.0)
                self.assertLessEqual(float(val), 1.0)

    def test_joint_matrix_columns_present(self):
        self._run_aggregation()
        rows = _read_csv(os.path.join(self._analysis_dir, "overall_summary.csv"))
        for row in rows:
            for col in ("SS_FP", "SS_FF", "SN_FP", "SN_FF"):
                self.assertIn(col, row, f"Missing joint matrix column: {col}")


# ============================================================
# TEST: generate_figures.py with synthetic analysis data
# ============================================================

class TestFigureGeneration(unittest.TestCase):
    """
    Verifies that figure generation runs without error on synthetic data
    and produces PNG files. Does NOT verify figure aesthetics.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="caps_test_figs_")
        self._analysis_dir = os.path.join(self._tmpdir, "analysis")
        self._figures_dir  = os.path.join(self._analysis_dir, "figures")
        os.makedirs(self._analysis_dir)

        # Write minimal synthetic CSVs
        ENCODING = "utf-8"

        overall_fields = [
            "condition", "n_runs",
            "ast_valid_rate",
            "interface_conformance_mean", "interface_conformance_median",
            "violation_count_mean",
            "functional_valid_rate", "functional_score_mean",
            "SS_FP", "SS_FF", "SN_FP", "SN_FF",
        ]
        overall_rows = [
            {"condition": "baseline", "n_runs": 30,
             "ast_valid_rate": 1.0,
             "interface_conformance_mean": 0.80,
             "interface_conformance_median": 0.80,
             "violation_count_mean": 3.0,
             "functional_valid_rate": 0.60,
             "functional_score_mean": 0.70,
             "SS_FP": 6, "SS_FF": 9, "SN_FP": 12, "SN_FF": 3},
            {"condition": "caps", "n_runs": 30,
             "ast_valid_rate": 1.0,
             "interface_conformance_mean": 0.95,
             "interface_conformance_median": 0.95,
             "violation_count_mean": 1.0,
             "functional_valid_rate": 0.80,
             "functional_score_mean": 0.90,
             "SS_FP": 20, "SS_FF": 5, "SN_FP": 3, "SN_FF": 2},
        ]
        self._write_csv(os.path.join(self._analysis_dir, "overall_summary.csv"),
                        overall_fields, overall_rows)

        # Pattern summary (all 5 patterns, 2 conditions)
        pattern_fields = [
            "model", "pattern", "condition", "n_runs",
            "ast_valid_rate", "interface_conformance_mean",
            "interface_conformance_median", "violation_count_mean",
            "functional_valid_rate", "functional_score_mean",
            "SS_FP", "SS_FF", "SN_FP", "SN_FF",
        ]
        pat_rows = []
        for pat in ["Factory", "Observer", "Strategy", "Singleton", "Adapter"]:
            pat_rows.append({
                "model": "test_model_A", "pattern": pat, "condition": "baseline",
                "n_runs": 6, "ast_valid_rate": 1.0,
                "interface_conformance_mean": 0.75,
                "interface_conformance_median": 0.75,
                "violation_count_mean": 4.0,
                "functional_valid_rate": 0.5,
                "functional_score_mean": 0.6,
                "SS_FP": 1, "SS_FF": 2, "SN_FP": 1, "SN_FF": 2,
            })
            pat_rows.append({
                "model": "test_model_A", "pattern": pat, "condition": "caps",
                "n_runs": 6, "ast_valid_rate": 1.0,
                "interface_conformance_mean": 0.92,
                "interface_conformance_median": 0.92,
                "violation_count_mean": 1.0,
                "functional_valid_rate": 0.8,
                "functional_score_mean": 0.9,
                "SS_FP": 4, "SS_FF": 1, "SN_FP": 1, "SN_FF": 0,
            })
        self._write_csv(os.path.join(self._analysis_dir, "pattern_summary.csv"),
                        pattern_fields, pat_rows)

        # Task summary (all 10 tasks)
        from src.experiment_runner import TASK_IDS
        task_fields = [
            "model", "task_id", "pattern", "condition", "n_runs",
            "ast_valid_rate", "interface_conformance_mean",
            "interface_conformance_median", "interface_conformance_stdev",
            "violation_count_mean", "violation_count_median",
            "functional_valid_rate", "functional_score_mean",
            "SS_FP", "SS_FF", "SN_FP", "SN_FF",
        ]
        task_rows = []
        pats = ["Factory", "Factory", "Observer", "Observer",
                "Strategy", "Strategy", "Singleton", "Singleton",
                "Adapter", "Adapter"]
        for tid, pat in zip(TASK_IDS, pats):
            for cond, ic in [("baseline", 0.78), ("caps", 0.94)]:
                task_rows.append({
                    "model": "test_model_A", "task_id": tid, "pattern": pat,
                    "condition": cond, "n_runs": 3,
                    "ast_valid_rate": 1.0,
                    "interface_conformance_mean": ic,
                    "interface_conformance_median": ic,
                    "interface_conformance_stdev": 0.02,
                    "violation_count_mean": 3.0 if cond == "baseline" else 1.0,
                    "violation_count_median": 3.0 if cond == "baseline" else 1.0,
                    "functional_valid_rate": 0.5,
                    "functional_score_mean": 0.7,
                    "SS_FP": 1, "SS_FF": 1, "SN_FP": 1, "SN_FF": 0,
                })
        self._write_csv(os.path.join(self._analysis_dir, "task_summary.csv"),
                        task_fields, task_rows)

        # Model summary (1 model only → fig7 should be skipped)
        model_fields = [
            "model", "condition", "n_runs",
            "ast_valid_rate", "interface_conformance_mean",
            "interface_conformance_median", "violation_count_mean",
            "functional_valid_rate", "functional_score_mean",
            "SS_FP", "SS_FF", "SN_FP", "SN_FF",
        ]
        model_rows = [
            {"model": "test_model_A", "condition": "baseline", "n_runs": 30,
             "ast_valid_rate": 1.0, "interface_conformance_mean": 0.80,
             "interface_conformance_median": 0.80, "violation_count_mean": 3.0,
             "functional_valid_rate": 0.6, "functional_score_mean": 0.7,
             "SS_FP": 6, "SS_FF": 9, "SN_FP": 12, "SN_FF": 3},
            {"model": "test_model_A", "condition": "caps", "n_runs": 30,
             "ast_valid_rate": 1.0, "interface_conformance_mean": 0.95,
             "interface_conformance_median": 0.95, "violation_count_mean": 1.0,
             "functional_valid_rate": 0.8, "functional_score_mean": 0.9,
             "SS_FP": 20, "SS_FF": 5, "SN_FP": 3, "SN_FF": 2},
        ]
        self._write_csv(os.path.join(self._analysis_dir, "model_summary.csv"),
                        model_fields, model_rows)

    @staticmethod
    def _write_csv(path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_figure_generation_exits_zero(self):
        import generate_figures as gf
        rc = gf.run_figures(
            analysis_dir=self._analysis_dir,
            figures_dir=self._figures_dir,
            model_filter=None,
        )
        self.assertEqual(rc, 0)

    def test_at_least_seven_figures_produced(self):
        """fig7 is skipped (only 1 model); the other 7 should be produced."""
        import generate_figures as gf
        gf.run_figures(
            analysis_dir=self._analysis_dir,
            figures_dir=self._figures_dir,
            model_filter=None,
        )
        pngs = [
            f for f in os.listdir(self._figures_dir)
            if f.endswith(".png")
        ]
        self.assertGreaterEqual(len(pngs), 7, f"Expected ≥7 PNGs, got {len(pngs)}: {pngs}")

    def test_specific_figure_files_exist(self):
        import generate_figures as gf
        gf.run_figures(
            analysis_dir=self._analysis_dir,
            figures_dir=self._figures_dir,
            model_filter=None,
        )
        for fname in [
            "fig1_overall_conformance.png",
            "fig2_overall_functional.png",
            "fig3_violations.png",
            "fig4_pattern_conformance.png",
            "fig5_pattern_functional.png",
            "fig6_task_conformance.png",
            "fig8_joint_matrix.png",
        ]:
            self.assertTrue(
                os.path.exists(os.path.join(self._figures_dir, fname)),
                f"Expected figure not produced: {fname}",
            )


# ============================================================
# TEST: collect_generation.py importable
# ============================================================

class TestCollectGenerationImport(unittest.TestCase):

    def test_module_imports_without_error(self):
        import collect_generation
        self.assertTrue(hasattr(collect_generation, "build_parser"))
        self.assertTrue(hasattr(collect_generation, "cmd_prompt"))
        self.assertTrue(hasattr(collect_generation, "cmd_record"))
        self.assertTrue(hasattr(collect_generation, "cmd_status"))
        self.assertTrue(hasattr(collect_generation, "cmd_evaluate"))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CAPS PIPELINE — AGGREGATION + FIGURE TESTS (SYNTHETIC DATA ONLY)")
    print("=" * 70)
    print("WARNING: All data used by these tests is synthetic.")
    print("         These tests do NOT produce formal experiment results.")
    print("=" * 70)
    unittest.main(verbosity=2)
