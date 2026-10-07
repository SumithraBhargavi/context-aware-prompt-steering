"""
ErrorAnalyzer – Phase 9 of the CAPS research infrastructure.

Categorizes structural violations and functional failures from all
experiment records into research-meaningful groups.

STRUCTURAL VIOLATION CATEGORIES (from CodeEvaluator violations list):
  1. syntax_ast_failure          – AST parse error
  2. missing_class               – Missing required class
  3. incorrect_inheritance       – Class should inherit from X
  4. incorrect_class_abstraction – Class should be abstract/concrete
  5. missing_method              – Missing required method
  6. incorrect_method_abstraction – Method should be abstract
  7. incorrect_static_declaration – Method should be static
  8. parameter_count_mismatch    – Parameter count mismatch
  9. parameter_name_mismatch     – Parameter name mismatch
  10. parameter_type_mismatch    – Parameter type mismatch
  11. return_type_mismatch       – Return type mismatch

FUNCTIONAL FAILURE CATEGORIES:
  1. execution_timeout       – Subprocess timed out
  2. syntax_runtime_error    – Script crashed (SyntaxError or similar)
  3. import_error            – ImportError at runtime
  4. instantiation_error     – Class could not be instantiated
  5. method_not_callable     – Required method raised unexpected error
  6. assertion_failure       – A test assertion failed
  7. type_error              – Wrong return type
  8. functional_other        – Other functional failure

Produces:
  results/error_analysis.csv

with counts by model, pattern, condition, violation_type.
"""

import os
import re
import csv
import json
from typing import Dict, Any, List, Tuple

ENCODING = "utf-8"


# ============================================================
# CATEGORY MATCHERS
# ============================================================

# Each entry: (category_key, compiled regex pattern)
_STRUCTURAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("syntax_ast_failure",
     re.compile(r"AST parse error", re.IGNORECASE)),

    ("missing_class",
     re.compile(r"Missing required class", re.IGNORECASE)),

    ("incorrect_inheritance",
     re.compile(r"should inherit from", re.IGNORECASE)),

    ("missing_method",
     re.compile(r"Missing method", re.IGNORECASE)),

    ("incorrect_method_abstraction",
     re.compile(r"\.\w+ should be abstract", re.IGNORECASE)),

    ("incorrect_class_abstraction",
     re.compile(r"should be (abstract|concrete)", re.IGNORECASE)),

    ("incorrect_static_declaration",
     re.compile(r"should be static", re.IGNORECASE)),

    ("parameter_count_mismatch",
     re.compile(r"Parameter count mismatch", re.IGNORECASE)),

    ("parameter_name_mismatch",
     re.compile(r"Parameter name mismatch", re.IGNORECASE)),

    ("parameter_type_mismatch",
     re.compile(r"Parameter type mismatch", re.IGNORECASE)),

    ("return_type_mismatch",
     re.compile(r"Return type mismatch", re.IGNORECASE)),
]

_FUNCTIONAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("execution_timeout",
     re.compile(r"timed out", re.IGNORECASE)),

    ("syntax_runtime_error",
     re.compile(
         r"(SyntaxError|IndentationError|TabError|raised SyntaxError)",
         re.IGNORECASE
     )),

    ("import_error",
     re.compile(r"(ImportError|ModuleNotFoundError)", re.IGNORECASE)),

    ("instantiation_error",
     re.compile(r"(TypeError|__init__|instantiat)", re.IGNORECASE)),

    ("method_not_callable",
     re.compile(r"(AttributeError|callable|not callable)", re.IGNORECASE)),

    ("type_error",
     re.compile(r"wrong.*type|returns.*bool|returns.*str|isinstance",
                re.IGNORECASE)),

    ("assertion_failure",
     re.compile(r"(AssertionError|assert|FAIL:)", re.IGNORECASE)),
]


# ============================================================
# ERROR ANALYZER
# ============================================================

class ErrorAnalyzer:
    """
    Categorizes violations from experiment records for error analysis.

    Args:
        records: List of flat observation dicts (from ResultsAggregator)
                 OR list of raw metadata dicts from ExperimentRunner.
    """

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    # =========================================================
    # STRUCTURAL CATEGORIZATION
    # =========================================================

    @staticmethod
    def categorize_structural_violation(violation: str) -> str:
        """
        Map a single violation message to a category key.

        Matches patterns in order; returns 'uncategorized' if no match.
        """

        for category, pattern in _STRUCTURAL_PATTERNS:
            if pattern.search(violation):
                return category

        return "uncategorized_structural"

    # =========================================================
    # FUNCTIONAL CATEGORIZATION
    # =========================================================

    @staticmethod
    def categorize_functional_failure(failure: str) -> str:
        """
        Map a single functional failure message to a category key.
        """

        # Timeout is best detected from the timed_out flag, but also
        # catch the string in case it appears in a failure message.
        for category, pattern in _FUNCTIONAL_PATTERNS:
            if pattern.search(failure):
                return category

        return "functional_other"

    # =========================================================
    # EXTRACT VIOLATIONS FROM RECORDS
    # =========================================================

    def extract_all_violations(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Extract individual violation/failure rows from all records.

        Each row represents a single violation or failure instance:
            task_id, pattern, model, condition, run,
            violation_type ('structural' or 'functional'),
            category, message

        Returns:
            List of violation row dicts.
        """

        rows = []

        for rec in self._records:

            # Extract identifying fields
            task_id   = rec.get("task_id", "unknown")
            pattern   = rec.get("pattern", "unknown")
            model     = rec.get("model", "unknown")
            condition = rec.get("condition", "unknown")
            run       = rec.get("run", 0)

            # ---- Structural violations ----

            # Records may come from either flat rows (ResultsAggregator)
            # or full metadata (ExperimentRunner). Handle both.
            structural_violations = []

            # Full evaluation stored under evaluation_file or evaluation key
            eval_data = rec.get("_evaluation_detail")

            if eval_data is None:
                # Try loading from evaluation_file path
                eval_file = rec.get("evaluation_file")
                if eval_file and os.path.exists(eval_file):
                    try:
                        with open(eval_file, "r", encoding=ENCODING) as f:
                            eval_data = json.load(f)
                    except Exception:
                        pass

            if eval_data:
                structural = eval_data.get("structural", {})
                structural_violations = structural.get("violations", [])

                functional = eval_data.get("functional", {})
                func_failures = functional.get("failures", [])
                timed_out = functional.get("timed_out", False)
                exec_error = functional.get("execution_error")

                # Add execution_error as a failure if present
                if exec_error and exec_error not in func_failures:
                    func_failures = list(func_failures) + [exec_error]

                if timed_out:
                    func_failures = list(func_failures) + [
                        "Execution timed out"
                    ]

                for failure in func_failures:
                    category = ErrorAnalyzer.categorize_functional_failure(
                        failure
                    )
                    rows.append({
                        "task_id":        task_id,
                        "pattern":        pattern,
                        "model":          model,
                        "condition":      condition,
                        "run":            run,
                        "violation_type": "functional",
                        "category":       category,
                        "message":        failure[:300],
                    })

            # Structural violations from eval_data or skip
            for violation in structural_violations:
                category = ErrorAnalyzer.categorize_structural_violation(
                    violation
                )
                rows.append({
                    "task_id":        task_id,
                    "pattern":        pattern,
                    "model":          model,
                    "condition":      condition,
                    "run":            run,
                    "violation_type": "structural",
                    "category":       category,
                    "message":        violation[:300],
                })

        return rows

    # =========================================================
    # COUNT TABLE
    # =========================================================

    def count_by_category(
        self,
        violation_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Aggregate violation rows into counts by
        (model, pattern, condition, violation_type, category).

        Returns:
            List of count rows for CSV output.
        """

        counts: Dict[Tuple, int] = {}

        for row in violation_rows:
            key = (
                row["model"],
                row["pattern"],
                row["condition"],
                row["violation_type"],
                row["category"],
            )
            counts[key] = counts.get(key, 0) + 1

        result_rows = []

        for (model, pattern, condition, v_type, category), count in (
            sorted(counts.items())
        ):
            result_rows.append({
                "model":          model,
                "pattern":        pattern,
                "condition":      condition,
                "violation_type": v_type,
                "category":       category,
                "count":          count,
            })

        return result_rows

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        path: str = os.path.join("results", "error_analysis.csv"),
    ) -> str:
        """
        Extract violations, count them, and write to CSV.

        Args:
            path: Output CSV path.

        Returns:
            Absolute path to the written file.
        """

        violation_rows = self.extract_all_violations()
        count_rows = self.count_by_category(violation_rows)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        fieldnames = [
            "model",
            "pattern",
            "condition",
            "violation_type",
            "category",
            "count",
        ]

        with open(path, "w", newline="", encoding=ENCODING) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(count_rows)

        return os.path.abspath(path)

    # =========================================================
    # IN-MEMORY SUMMARY (for testing without files)
    # =========================================================

    def summarize(self) -> Dict[str, Any]:
        """
        Return a dict summary of violation counts by category.
        """

        violation_rows = self.extract_all_violations()
        count_rows = self.count_by_category(violation_rows)

        structural_total = sum(
            r["count"] for r in count_rows
            if r["violation_type"] == "structural"
        )

        functional_total = sum(
            r["count"] for r in count_rows
            if r["violation_type"] == "functional"
        )

        return {
            "total_violations": len(violation_rows),
            "structural_total": structural_total,
            "functional_total": functional_total,
            "counts": count_rows,
        }
