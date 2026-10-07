"""
StatisticalAnalyzer – Phase 8 of the CAPS research infrastructure.

Implements statistically defensible paired comparison between the
baseline and CAPS conditions.

STATISTICAL DESIGN
==================

The experiment has a paired structure:
  (task_id, model, run) → [baseline result, caps result]

For each (task, model, run) triplet there is one baseline observation
and one CAPS observation. This is a within-cell paired design.

Tests applied:
  - Wilcoxon signed-rank test (scipy.stats.wilcoxon):
      For paired continuous/ordinal metrics:
        * interface_conformance
        * violation_count
        * functional_score
      Used instead of paired t-test because normality cannot be assumed
      with small samples and bounded metrics.

  - McNemar's test (scipy.stats.mcnemar):
      For paired binary outcomes:
        * ast_valid
        * functional_valid
      Appropriate because both conditions are applied to the same
      (task, model, run) unit.

SAMPLE-SIZE LIMITATIONS
========================
With 10 tasks × 2 models × 3 runs = 60 pairs (maximum 30 per model),
statistical power will be limited. The module detects this and reports
the limitation honestly rather than manufacturing significance.

Minimum pairs required for each test:
  - Wilcoxon: at least 6 discordant pairs (otherwise reports limitation)
  - McNemar:  at least 5 discordant pairs (b + c >= 5)

UNIT OF ANALYSIS
================
Formal inferential comparisons are performed at the task level. For each
(model, task, condition), repeated runs are first summarized by their mean
for continuous metrics and by their observed proportion for binary metrics.
The resulting baseline/CAPS task summaries are then paired by task. This
avoids treating repeated generations of the same benchmark task as fully
independent experimental units.

Runs remain useful as independent replications for estimating variability
and for descriptive reporting. Different models are analyzed separately;
pooled model-level results are descriptive only because model identity can
confound the treatment effect.

EFFECT SIZES
============
Rank-biserial correlation r is reported for Wilcoxon tests.
Odds ratio is reported for McNemar's test.

IMPORTANT:
  - Results are computed from actual collected data only.
  - No results are fabricated or simulated.
  - If insufficient data is available, the limitation is reported.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

ENCODING = "utf-8"

# Minimum discordant pairs for tests to be informative
MIN_WILCOXON_PAIRS = 6
MIN_MCNEMAR_PAIRS = 5


# ============================================================
# SCIPY AVAILABILITY CHECK
# ============================================================

try:
    from scipy import stats as _scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# STATISTICAL ANALYZER
# ============================================================

class StatisticalAnalyzer:
    """
    Performs paired statistical analysis on CAPS experiment results.

    Args:
        records: List of flat observation dicts (from ResultsAggregator).
                 Each dict must have: task_id, model, condition, run,
                 and evaluation metrics.
    """

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    # =========================================================
    # PAIR BUILDING
    # =========================================================

    def build_pairs(
        self,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build paired observations: for each (task_id, model, run),
        join the baseline and caps records.

        Args:
            model: If provided, restrict to this model only.

        Returns:
            List of pair dicts with keys:
                task_id, pattern, model, run,
                baseline_<metric>, caps_<metric>  for each metric,
                complete (bool: both sides present)
        """

        # Index records by (task_id, model, run, condition)
        index: Dict[Tuple, Dict] = {}

        for rec in self._records:
            if model is not None and rec.get("model") != model:
                continue

            key = (
                rec.get("task_id"),
                rec.get("model"),
                rec.get("run"),
            )
            condition = rec.get("condition")

            if condition not in ("baseline", "caps"):
                continue

            index.setdefault(key, {})[condition] = rec

        pairs = []

        for (task_id, model_id, run), sides in sorted(index.items()):

            baseline = sides.get("baseline")
            caps = sides.get("caps")

            complete = baseline is not None and caps is not None

            pair: Dict[str, Any] = {
                "task_id": task_id,
                "pattern": (
                    baseline or caps or {}
                ).get("pattern"),
                "model": model_id,
                "run": run,
                "complete": complete,
            }

            metrics = [
                "ast_valid",
                "interface_conformance",
                "violation_count",
                "functional_valid",
                "functional_score",
            ]

            for metric in metrics:
                pair[f"baseline_{metric}"] = (
                    baseline.get(metric) if baseline else None
                )
                pair[f"caps_{metric}"] = (
                    caps.get(metric) if caps else None
                )

            pairs.append(pair)

        return pairs

    # =========================================================
    # TASK-LEVEL PAIR BUILDING
    # =========================================================

    def build_task_pairs(self, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Aggregate repeated runs within each task and pair baseline vs CAPS.

        Continuous metrics use the mean across runs. Binary metrics use the
        proportion of True outcomes across runs. The task is therefore the
        inferential unit, while runs provide replication within each task.
        """
        grouped: Dict[Tuple, Dict[str, List[Dict[str, Any]]]] = {}
        for rec in self._records:
            if model is not None and rec.get("model") != model:
                continue
            condition = rec.get("condition")
            if condition not in ("baseline", "caps"):
                continue
            key = (rec.get("task_id"), rec.get("model"))
            grouped.setdefault(key, {}).setdefault(condition, []).append(rec)

        continuous = ["interface_conformance", "violation_count", "functional_score"]
        binary = ["ast_valid", "functional_valid"]
        pairs = []
        for (task_id, model_id), sides in sorted(grouped.items()):
            b = sides.get("baseline", [])
            c = sides.get("caps", [])
            if not b or not c:
                continue
            pair = {
                "task_id": task_id,
                "model": model_id,
                "pattern": (b[0] if b else c[0]).get("pattern"),
                "baseline_runs": len(b),
                "caps_runs": len(c),
                "complete": True,
            }
            for metric in continuous:
                bv = [float(r[metric]) for r in b if r.get(metric) is not None]
                cv = [float(r[metric]) for r in c if r.get(metric) is not None]
                pair[f"baseline_{metric}"] = sum(bv)/len(bv) if bv else None
                pair[f"caps_{metric}"] = sum(cv)/len(cv) if cv else None
            for metric in binary:
                bv = [bool(r[metric]) for r in b if r.get(metric) is not None]
                cv = [bool(r[metric]) for r in c if r.get(metric) is not None]
                pair[f"baseline_{metric}"] = sum(bv)/len(bv) if bv else None
                pair[f"caps_{metric}"] = sum(cv)/len(cv) if cv else None
            pairs.append(pair)
        return pairs

    # =========================================================
    # WILCOXON SIGNED-RANK TEST
    # =========================================================

    def wilcoxon_test(
        self,
        pairs: List[Dict[str, Any]],
        metric: str,
    ) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test on paired differences for metric.

        Args:
            pairs:  List of pair dicts from build_pairs().
            metric: Metric name (without baseline_/caps_ prefix).

        Returns:
            dict with test results including sample_size, statistic,
            p_value, effect_size_r, interpretation, and limitation_note.
        """

        result: Dict[str, Any] = {
            "test": "Wilcoxon signed-rank",
            "metric": metric,
            "null_hypothesis": (
                f"No difference in {metric} between baseline and CAPS"
            ),
            "sample_size": 0,
            "n_discordant": 0,
            "statistic": None,
            "p_value": None,
            "effect_size_r": None,
            "direction": None,
            "limitation_note": None,
            "scipy_available": SCIPY_AVAILABLE,
        }

        # Extract complete pairs with non-None values
        b_vals = []
        c_vals = []

        for pair in pairs:
            if not pair.get("complete"):
                continue

            bv = pair.get(f"baseline_{metric}")
            cv = pair.get(f"caps_{metric}")

            if bv is None or cv is None:
                continue

            # Only numeric values
            try:
                b_vals.append(float(bv))
                c_vals.append(float(cv))
            except (TypeError, ValueError):
                continue

        result["sample_size"] = len(b_vals)

        if len(b_vals) == 0:
            result["limitation_note"] = (
                "No complete pairs available for this metric."
            )
            return result

        # Count discordant pairs (where values differ)
        differences = [c - b for b, c in zip(b_vals, c_vals)]
        n_discordant = sum(1 for d in differences if d != 0.0)
        result["n_discordant"] = n_discordant

        # Compute direction (mean difference)
        mean_diff = sum(differences) / len(differences)
        result["direction"] = (
            "CAPS > baseline" if mean_diff > 0
            else "baseline > CAPS" if mean_diff < 0
            else "no difference"
        )
        result["mean_difference"] = round(mean_diff, 6)

        if n_discordant < MIN_WILCOXON_PAIRS:
            result["limitation_note"] = (
                f"Only {n_discordant} discordant pairs "
                f"(minimum {MIN_WILCOXON_PAIRS} required). "
                "Wilcoxon test is unreliable; interpret with caution."
            )

        if not SCIPY_AVAILABLE:
            result["limitation_note"] = (
                "scipy is not installed. Install scipy to run "
                "statistical tests: pip install scipy"
            )
            return result

        if n_discordant == 0:
            result["limitation_note"] = (
                "All differences are zero. "
                "Wilcoxon test cannot be computed."
            )
            return result

        try:
            stat, pval = _scipy_stats.wilcoxon(
                b_vals,
                c_vals,
                alternative="two-sided",
                zero_method="wilcox",
            )
            result["statistic"] = float(stat)
            result["p_value"] = float(pval)

            # Rank-biserial correlation as effect size
            # r = 1 - (2W) / (n*(n+1)) for Wilcoxon W
            n = len(b_vals)
            if n > 0:
                r = 1.0 - (2.0 * stat) / (n * (n + 1))
                result["effect_size_r"] = round(r, 4)

        except Exception as exc:
            result["limitation_note"] = f"Wilcoxon test failed: {exc}"

        return result

    # =========================================================
    # MCNEMAR'S TEST
    # =========================================================

    def mcnemar_test(
        self,
        pairs: List[Dict[str, Any]],
        metric: str,
    ) -> Dict[str, Any]:
        """
        Perform McNemar's test on a paired binary metric.

        Contingency table:
          b = count(baseline=True,  caps=False)  → regression
          c = count(baseline=False, caps=True)   → improvement

        Args:
            pairs:  List of pair dicts from build_pairs().
            metric: Binary metric name (without baseline_/caps_ prefix).

        Returns:
            dict with contingency table, statistic, p_value,
            odds_ratio, and limitation notes.
        """

        result: Dict[str, Any] = {
            "test": "McNemar",
            "metric": metric,
            "null_hypothesis": (
                f"No difference in {metric} between baseline and CAPS"
            ),
            "sample_size": 0,
            "a": 0,  # both True
            "b": 0,  # baseline True, caps False (regression)
            "c": 0,  # baseline False, caps True (improvement)
            "d": 0,  # both False
            "statistic": None,
            "p_value": None,
            "odds_ratio": None,
            "direction": None,
            "limitation_note": None,
            "scipy_available": SCIPY_AVAILABLE,
        }

        for pair in pairs:
            if not pair.get("complete"):
                continue

            bv = pair.get(f"baseline_{metric}")
            cv = pair.get(f"caps_{metric}")

            if bv is None or cv is None:
                continue

            bv = bool(bv)
            cv = bool(cv)

            if bv and cv:
                result["a"] += 1
            elif bv and not cv:
                result["b"] += 1
            elif not bv and cv:
                result["c"] += 1
            else:
                result["d"] += 1

        result["sample_size"] = (
            result["a"] + result["b"]
            + result["c"] + result["d"]
        )

        b, c = result["b"], result["c"]
        discordant = b + c

        result["direction"] = (
            "CAPS improves" if c > b
            else "CAPS hurts" if b > c
            else "no discordant pairs"
        )

        if discordant < MIN_MCNEMAR_PAIRS:
            result["limitation_note"] = (
                f"Only {discordant} discordant pairs "
                f"(b+c={discordant}, minimum {MIN_MCNEMAR_PAIRS}). "
                "McNemar's test is unreliable; interpret with caution."
            )

        if not SCIPY_AVAILABLE:
            result["limitation_note"] = (
                "scipy is not installed. Install scipy to run "
                "statistical tests: pip install scipy"
            )
            return result

        if discordant == 0:
            result["limitation_note"] = (
                "No discordant pairs. McNemar's test cannot be computed."
            )
            return result

        try:
            table = [[result["a"], result["b"]],
                     [result["c"], result["d"]]]

            mc_result = _scipy_stats.mcnemar(
                table,
                exact=discordant < 25,  # exact for small samples
                correction=discordant >= 25,
            )

            result["statistic"] = float(mc_result.statistic)
            result["p_value"] = float(mc_result.pvalue)

            # Odds ratio: c / b (improvement / regression)
            if b > 0:
                result["odds_ratio"] = round(c / b, 4)
            elif c > 0:
                result["odds_ratio"] = float("inf")
            else:
                result["odds_ratio"] = None

        except Exception as exc:
            result["limitation_note"] = f"McNemar test failed: {exc}"

        return result

    # =========================================================
    # FULL ANALYSIS
    # =========================================================

    def analyze(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the primary task-level paired analysis.

        Repeated runs are averaged within each task before inferential tests.
        This is the primary analysis used for the paper.
        """
        if models is None:
            models = sorted({r.get("model") for r in self._records if r.get("model")})

        analysis: Dict[str, Any] = {
            "design": {
                "type": "paired task-level analysis with repeated-run aggregation",
                "inferential_unit": "task_id within model",
                "replication_unit": "run",
                "conditions": ["baseline", "caps"],
                "note": "Runs are summarized within task before inferential testing; models are analyzed separately.",
            },
            "scipy_available": SCIPY_AVAILABLE,
            "per_model": {},
        }

        continuous_metrics = ["interface_conformance", "violation_count", "functional_score", "ast_valid", "functional_valid"]

        for model_id in models:
            task_pairs = self.build_task_pairs(model=model_id)
            model_result = {
                "model": model_id,
                "task_pairs": len(task_pairs),
                "task_ids": [p["task_id"] for p in task_pairs],
                "wilcoxon": {},
            }
            for metric in continuous_metrics:
                model_result["wilcoxon"][metric] = self.wilcoxon_test(task_pairs, metric)
            analysis["per_model"][model_id] = model_result

        return analysis

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        analysis: Optional[Dict[str, Any]] = None,
        path: str = os.path.join("results", "statistical_analysis.json"),
    ) -> str:
        """
        Save the statistical analysis to JSON.

        Args:
            analysis: Pre-computed analysis dict. If None, calls analyze().
            path:     Output file path.

        Returns:
            Absolute path to the written file.
        """

        if analysis is None:
            analysis = self.analyze()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding=ENCODING) as f:
            json.dump(analysis, f, indent=4, default=str)

        return os.path.abspath(path)
