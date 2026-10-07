"""
ExperimentRunner – Phase 6 of the CAPS research infrastructure.

Manages the directory structure, prompt generation, and metadata
recording for all 10 CAPS benchmark tasks across multiple models,
conditions (baseline / caps), and independent runs.

DIRECTORY STRUCTURE
===================

results/
└── experiments/
    └── <model_id>/
        └── <task_id>/
            └── <condition>_run_<NN>/
                ├── prompt.txt          – exact prompt given to the LLM
                ├── raw_code.txt        – raw LLM output (never modified)
                ├── normalized_code.py  – fence-stripped copy for execution
                                          (absent if identical to raw)
                ├── evaluation.json     – structural + functional results
                └── metadata.json       – full experiment record

FAIRNESS GUARANTEE
==================
For each (task, model, run), BOTH conditions (baseline + caps) use:
  - same task description
  - same model and model_version
  - same generation environment
  - same evaluation procedure

Only the prompt content differs between conditions.

RESEARCH INTEGRITY NOTES
=========================
  - Generated code is NEVER repaired before evaluation.
  - Prompt content is deterministic for the same task.
  - Pilot data lives in results/ (not results/experiments/) and is
    never mixed with final experiment records.
  - Timestamps are real; never fabricated.
  - Model version is left as provided by the caller (or None/unknown).
"""

import os
import json
import datetime
from typing import Dict, Any, Optional, List

from src.llm_runner import LLMRunner
from src.evaluator import CodeEvaluator
from src.functional_evaluator import FunctionalEvaluator
from src.task_contracts import load_tasks


# ============================================================
# CONSTANTS
# ============================================================

EXPERIMENTS_BASE = os.path.join("results", "experiments")
CONDITIONS = ("baseline", "caps")
ENCODING = "utf-8"

TASK_IDS = [
    "factory_01",
    "factory_02",
    "observer_01",
    "observer_02",
    "strategy_01",
    "strategy_02",
    "singleton_01",
    "singleton_02",
    "adapter_01",
    "adapter_02",
]


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

class ExperimentRunner:
    """
    Orchestrates the CAPS experiment across tasks, models, and runs.

    Usage pattern (manual LLM generation workflow):
        1. runner.generate_prompts(task, model_id, run)
           → saves prompts; you copy them to the LLM manually
        2. runner.record_raw_output(task_id, model_id, condition, run, raw)
           → saves the raw LLM response
        3. runner.evaluate_and_save(task_id, model_id, condition, run)
           → runs structural + functional evaluation and saves metadata

    Args:
        dataset_path: Path to test_tasks.json.
        base_dir:     Root experiments directory.
    """

    def __init__(
        self,
        dataset_path: str = "dataset/test_tasks.json",
        base_dir: str = EXPERIMENTS_BASE,
    ):
        self.base_dir = base_dir
        self._tasks: Dict[str, Dict[str, Any]] = {}

        task_list = load_tasks(dataset_path)

        for task in task_list:
            self._tasks[task["task_id"]] = task

    # =========================================================
    # DIRECTORY MANAGEMENT
    # =========================================================

    def get_run_dir(
        self,
        model_id: str,
        task_id: str,
        condition: str,
        run: int,
    ) -> str:
        """Return the run directory path (does not create it)."""

        if condition not in CONDITIONS:
            raise ValueError(
                f"Invalid condition '{condition}'. "
                f"Must be one of: {CONDITIONS}"
            )

        return os.path.join(
            self.base_dir,
            model_id,
            task_id,
            f"{condition}_run_{run:02d}",
        )

    def ensure_run_dir(
        self,
        model_id: str,
        task_id: str,
        condition: str,
        run: int,
    ) -> str:
        """Create and return the run directory path."""

        run_dir = self.get_run_dir(model_id, task_id, condition, run)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def setup_all_dirs(
        self,
        model_ids: List[str],
        runs_per_condition: int = 3,
    ) -> None:
        """
        Create the full directory skeleton for all model/task/condition/run
        combinations. Does not create any data files.

        Args:
            model_ids:           List of model identifiers.
            runs_per_condition:  Number of independent runs per condition.
        """

        os.makedirs(self.base_dir, exist_ok=True)

        for model_id in model_ids:
            for task_id in TASK_IDS:
                for condition in CONDITIONS:
                    for run in range(1, runs_per_condition + 1):
                        self.ensure_run_dir(
                            model_id, task_id, condition, run
                        )

    # =========================================================
    # PROMPT GENERATION
    # =========================================================

    def generate_prompts(
        self,
        task_id: str,
        model_id: str,
        run: int,
        save: bool = True,
    ) -> Dict[str, str]:
        """
        Generate deterministic baseline and CAPS prompts for a task/run.

        The prompts are IDENTICAL for the same (task_id, run) regardless
        of model — model differences should come only from the LLM's
        response, not from the prompt content.

        Args:
            task_id:  Benchmark task identifier.
            model_id: Model identifier (for directory routing only).
            run:      Run number (1-indexed).
            save:     If True, write prompt.txt to both condition dirs.

        Returns:
            dict with keys 'baseline' and 'caps' mapping to prompt strings.
        """

        if task_id not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_id}")

        task = self._tasks[task_id]
        runner = LLMRunner(task, model=model_id)

        baseline_prompt = runner.build_baseline_prompt()
        caps_prompt = runner.build_steered_prompt()

        prompts = {
            "baseline": baseline_prompt,
            "caps": caps_prompt,
        }

        if save:
            for condition, prompt_text in prompts.items():
                run_dir = self.ensure_run_dir(
                    model_id, task_id, condition, run
                )
                prompt_path = os.path.join(run_dir, "prompt.txt")
                with open(prompt_path, "w", encoding=ENCODING) as f:
                    f.write(prompt_text)

        return prompts

    # =========================================================
    # RAW OUTPUT RECORDING
    # =========================================================

    def record_raw_output(
        self,
        task_id: str,
        model_id: str,
        condition: str,
        run: int,
        raw_code: str,
    ) -> str:
        """
        Save the raw LLM output for a specific experiment cell.

        The raw code is saved EXACTLY as provided — it is never repaired
        or altered. If the raw code contains markdown fences, a separate
        normalized_code.py is also written for inspection (but evaluation
        always works from normalized content internally).

        Args:
            task_id:   Benchmark task identifier.
            model_id:  Model identifier.
            condition: 'baseline' or 'caps'.
            run:       Run number (1-indexed).
            raw_code:  Raw string output from the LLM.

        Returns:
            Path to the saved raw_code.txt.
        """

        run_dir = self.ensure_run_dir(
            model_id, task_id, condition, run
        )

        # Save raw code (NEVER modified)
        raw_path = os.path.join(run_dir, "raw_code.txt")
        with open(raw_path, "w", encoding=ENCODING) as f:
            f.write(raw_code)

        # Save normalized copy only if it differs from raw
        from src.functional_evaluator import FunctionalEvaluator
        normalized = FunctionalEvaluator._normalize_code(raw_code)

        if normalized.strip() != raw_code.strip():
            norm_path = os.path.join(run_dir, "normalized_code.py")
            with open(norm_path, "w", encoding=ENCODING) as f:
                f.write(normalized)

        return raw_path

    # =========================================================
    # EVALUATION
    # =========================================================

    def evaluate_and_save(
        self,
        task_id: str,
        model_id: str,
        condition: str,
        run: int,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run structural + functional evaluation on a recorded raw output.

        Reads raw_code.txt from the run directory, evaluates it, and
        saves evaluation.json and metadata.json.

        IMPORTANT:
          - The code is NEVER repaired before evaluation.
          - If it fails, the failure is recorded honestly.
          - Functional and structural results are stored separately.

        Args:
            task_id:       Benchmark task identifier.
            model_id:      Model identifier.
            condition:     'baseline' or 'caps'.
            run:           Run number (1-indexed).
            model_version: Optional version string (e.g., "gpt-4o-2024-08").
                           Leave None if unknown; do NOT fabricate.

        Returns:
            The full metadata record dict.
        """

        run_dir = self.get_run_dir(model_id, task_id, condition, run)

        raw_path = os.path.join(run_dir, "raw_code.txt")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"raw_code.txt not found in {run_dir}. "
                "Run record_raw_output() first."
            )

        with open(raw_path, "r", encoding=ENCODING, errors="replace") as f:
            raw_code = f.read()

        task = self._tasks[task_id]

        # --- Structural evaluation ---
        evaluator = CodeEvaluator(task["contract"])
        structural = evaluator.evaluate(raw_code)

        # --- Functional evaluation ---
        try:
            fe = FunctionalEvaluator(task_id)
            functional = fe.evaluate(raw_code)
        except Exception as exc:
            functional = {
                "functional_valid": False,
                "tests_total": 0,
                "tests_passed": 0,
                "functional_score": 0.0,
                "failures": [str(exc)],
                "execution_error": str(exc),
                "timed_out": False,
                "normalized_code": "",
            }

        # --- Build evaluation record ---
        evaluation = {
            "structural": structural,
            "functional": functional,
        }

        eval_path = os.path.join(run_dir, "evaluation.json")
        with open(eval_path, "w", encoding=ENCODING) as f:
            json.dump(evaluation, f, indent=4)

        # --- Build metadata record ---
        prompt_path = os.path.join(run_dir, "prompt.txt")
        norm_path = os.path.join(run_dir, "normalized_code.py")

        metadata = {
            "task_id": task_id,
            "pattern": task["pattern"],
            "model": model_id,
            "model_version": model_version,  # None if unknown; do NOT fabricate
            "condition": condition,
            "run": run,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "prompt_file": os.path.relpath(prompt_path, start="."),
            "raw_code_file": os.path.relpath(raw_path, start="."),
            "normalized_code_file": (
                os.path.relpath(norm_path, start=".")
                if os.path.exists(norm_path)
                else None
            ),
            "evaluation": {
                "ast_valid": structural["ast_valid"],
                "interface_conformance": structural["interface_conformance"],
                "violation_count": structural["violation_count"],
                "functional_valid": functional["functional_valid"],
                "functional_score": functional["functional_score"],
                "functional_tests_total": functional["tests_total"],
                "functional_tests_passed": functional["tests_passed"],
            },
            "evaluation_file": os.path.relpath(eval_path, start="."),
        }

        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path, "w", encoding=ENCODING) as f:
            json.dump(metadata, f, indent=4)

        return metadata

    # =========================================================
    # RECORD LOADING
    # =========================================================

    def load_all_records(
        self,
        max_runs: Optional[int] = 3,
    ) -> List[Dict[str, Any]]:
        """
        Walk the experiments directory and load all completed metadata.json
        records.

        A record is considered complete if metadata.json exists.
        Records with missing raw_code.txt or evaluation.json are loaded
        but flagged with a 'data_complete' field set to False.

        Args:
            max_runs: Maximum run number to include (default: 3).
                      Set to None to include all runs.

        Returns:
            List of metadata dicts, sorted by (model, task_id, condition, run).
        """

        records = []

        if not os.path.exists(self.base_dir):
            return records

        for model_id in sorted(os.listdir(self.base_dir)):
            model_dir = os.path.join(self.base_dir, model_id)

            if not os.path.isdir(model_dir):
                continue

            for task_id in sorted(os.listdir(model_dir)):
                task_dir = os.path.join(model_dir, task_id)

                if not os.path.isdir(task_dir):
                    continue

                for run_name in sorted(os.listdir(task_dir)):
                    run_dir = os.path.join(task_dir, run_name)

                    if not os.path.isdir(run_dir):
                        continue

                    meta_path = os.path.join(run_dir, "metadata.json")

                    if not os.path.exists(meta_path):
                        continue

                    with open(meta_path, "r", encoding=ENCODING) as f:
                        record = json.load(f)

                    run_num = record.get("run")
                    if max_runs is not None and isinstance(run_num, int) and run_num > max_runs:
                        continue

                    raw_exists = os.path.exists(
                        os.path.join(run_dir, "raw_code.txt")
                    )
                    eval_exists = os.path.exists(
                        os.path.join(run_dir, "evaluation.json")
                    )

                    record["data_complete"] = raw_exists and eval_exists
                    records.append(record)

        return records

    # =========================================================
    # PROMPT VERIFICATION (for fairness validation)
    # =========================================================

    def verify_prompt_separation(
        self, task_id: str
    ) -> Dict[str, bool]:
        """
        Verify that baseline and CAPS prompts differ only in the
        structural contract section.

        Returns:
            dict with:
                baseline_has_no_contract (bool)
                caps_has_contract        (bool)
                prompts_differ           (bool)
        """

        task = self._tasks[task_id]
        runner = LLMRunner(task)

        baseline = runner.build_baseline_prompt()
        caps = runner.build_steered_prompt()

        return {
            "baseline_has_no_contract": (
                "STRUCTURAL CONTRACT" not in baseline
                and "Required Classes:" not in baseline
            ),
            "caps_has_contract": (
                "STRUCTURAL CONTRACT" in caps
            ),
            "prompts_differ": baseline != caps,
        }
