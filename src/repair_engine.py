"""
RepairEngine — Verifier-Guided Structural Repair (CAPS Phase 2).

Uses structural violation feedback from CodeEvaluator to generate targeted
repair prompts for LLMs and iteratively repair non-conforming Python code.

RESEARCH INTEGRITY:
- Operates strictly in results/repair_experiments/
- Does NOT modify original raw_code.txt or frozen experiment outputs
- Reuses existing CodeEvaluator, FunctionalEvaluator, ContextEngine, and task contracts
- Implements safety checks: accepts repaired versions only if AST is valid
  and structural conformance improves or violation count decreases.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.context_engine import ContextEngine
from src.evaluator import CodeEvaluator
from src.functional_evaluator import FunctionalEvaluator
from src.task_contracts import load_tasks

MAX_REPAIR_ITERATIONS = 3
REPAIR_BASE_DIR = os.path.join("results", "repair_experiments")
ENCODING = "utf-8"


class RepairEngine:
    """
    Orchestrates verifier-guided structural repair.
    """

    def __init__(
        self,
        dataset_path: str = "dataset/test_tasks.json",
        base_dir: str = REPAIR_BASE_DIR,
    ):
        self.base_dir = base_dir
        self._tasks: Dict[str, Dict[str, Any]] = {}
        for task in load_tasks(dataset_path):
            self._tasks[task["task_id"]] = task

    def get_task(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._tasks[task_id]

    @staticmethod
    def extract_violations(structural_result: Dict[str, Any]) -> List[str]:
        """Extract violation strings from structural evaluation result."""
        if not structural_result:
            return []
        return list(structural_result.get("violations", []))

    def construct_repair_prompt(
        self,
        task_id: str,
        original_code: str,
        violations: List[str],
    ) -> str:
        """
        Construct a targeted, minimal repair prompt incorporating contract requirements
        and detected violations.
        """
        task = self.get_task(task_id)
        contract = task["contract"]
        ce = ContextEngine(contract)
        structural_context = ce.generate_context()

        violation_lines = "\n".join(f"- {v}" for v in violations) if violations else "- No specific violations listed."

        return f"""You are repairing an existing Python implementation.

Do not redesign the implementation.
Do not remove required classes.
Do not change the task's intended behavior.

Fix ONLY the structural violations listed below.

Structural Contract:
{structural_context}

Detected Structural Violations:
{violation_lines}

Original Code to Repair:
{original_code}

Instructions:
- Preserve all required class names, inheritance relationships, and method names.
- Preserve all required parameter names, parameter types, and return types.
- Preserve abstract/static method decorators where required.
- Preserve the functional behavior and task logic.
- Do not introduce unrelated redesign or extra helper functions unless strictly necessary.
- Return ONLY the complete corrected Python source code.
- Do not use Markdown code fences.
- Do not include conversational text or explanations.
""".strip()

    def evaluate_code(
        self,
        task_id: str,
        code: str,
        run_functional: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate structural and (optionally) functional correctness of code.
        Reuses existing CodeEvaluator and FunctionalEvaluator.
        """
        task = self.get_task(task_id)
        evaluator = CodeEvaluator(task["contract"])
        structural = evaluator.evaluate(code)

        functional = None
        if run_functional:
            try:
                fe = FunctionalEvaluator(task_id)
                functional = fe.evaluate(code)
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

        return {
            "structural": structural,
            "functional": functional,
        }

    def run_repair_step(
        self,
        task_id: str,
        current_code: str,
        repair_llm_fn,  # callable: (prompt: str) -> str
        iteration: int,
        run_functional: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a single repair iteration step.
        """
        current_eval = self.evaluate_code(task_id, current_code, run_functional=run_functional)
        violations = self.extract_violations(current_eval["structural"])

        if not current_eval["structural"]["ast_valid"] or current_eval["structural"]["interface_conformance"] < 1.0 or violations:
            prompt = self.construct_repair_prompt(task_id, current_code, violations)
            repaired_code = repair_llm_fn(prompt)
            repaired_eval = self.evaluate_code(task_id, repaired_code, run_functional=run_functional)
        else:
            prompt = ""
            repaired_code = current_code
            repaired_eval = current_eval

        return {
            "iteration": iteration,
            "prompt": prompt,
            "repaired_code": repaired_code,
            "evaluation": repaired_eval,
        }

    def run_repair_loop(
        self,
        task_id: str,
        model_id: str,
        source_condition: str,
        source_run: int,
        original_code: str,
        repair_llm_fn,  # callable: (prompt: str) -> str
        max_iterations: int = MAX_REPAIR_ITERATIONS,
        run_functional: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full iterative repair loop with safety checking and early stopping.
        """
        initial_eval = self.evaluate_code(task_id, original_code, run_functional=run_functional)
        initial_struct = initial_eval["structural"]
        initial_func = initial_eval.get("functional") or {}

        history = []
        current_code = original_code
        current_eval = initial_eval

        # Early check if already valid
        stop_early = (
            initial_struct["ast_valid"]
            and initial_struct["interface_conformance"] >= 1.0
            and (not run_functional or initial_func.get("functional_valid", False))
        )

        if stop_early:
            iterations_completed = 0
            repair_success = True
            failure_reason = None
            final_code = original_code
            final_eval = initial_eval
        else:
            repair_success = False
            failure_reason = "Max iterations reached without complete repair"
            final_code = original_code
            final_eval = initial_eval

            for it in range(1, max_iterations + 1):
                step = self.run_repair_step(task_id, current_code, repair_llm_fn, it, run_functional)
                history.append(step)

                rep_struct = step["evaluation"]["structural"]
                rep_func = step["evaluation"].get("functional") or {}

                # Safety check: accept repaired code if AST valid and conformance improved or violations decreased
                curr_struct = current_eval["structural"]
                is_safe = (
                    rep_struct["ast_valid"]
                    and (
                        rep_struct["interface_conformance"] > curr_struct["interface_conformance"]
                        or rep_struct["violation_count"] < curr_struct["violation_count"]
                        or (rep_struct["interface_conformance"] == curr_struct["interface_conformance"] and rep_struct["ast_valid"])
                    )
                )

                if is_safe:
                    current_code = step["repaired_code"]
                    current_eval = step["evaluation"]
                    final_code = current_code
                    final_eval = current_eval

                # Check stopping condition
                is_fully_repaired = (
                    rep_struct["ast_valid"]
                    and rep_struct["interface_conformance"] >= 1.0
                    and (not run_functional or rep_func.get("functional_valid", False))
                )

                if is_fully_repaired:
                    iterations_completed = it
                    repair_success = True
                    failure_reason = None
                    break
                else:
                    iterations_completed = it

        final_struct = final_eval["structural"]
        final_func = final_eval.get("functional") or {}

        result = {
            "model": model_id,
            "task_id": task_id,
            "source_condition": source_condition,
            "source_run": source_run,
            "iterations_completed": iterations_completed,
            "max_iterations": max_iterations,
            "repair_success": repair_success,
            "failure_reason": failure_reason,
            "pre_repair": {
                "ast_valid": initial_struct["ast_valid"],
                "interface_conformance": initial_struct["interface_conformance"],
                "violation_count": initial_struct["violation_count"],
                "functional_valid": initial_func.get("functional_valid"),
                "functional_score": initial_func.get("functional_score"),
            },
            "post_repair": {
                "ast_valid": final_struct["ast_valid"],
                "interface_conformance": final_struct["interface_conformance"],
                "violation_count": final_struct["violation_count"],
                "functional_valid": final_func.get("functional_valid"),
                "functional_score": final_func.get("functional_score"),
            },
            "deltas": {
                "conformance_gain": final_struct["interface_conformance"] - initial_struct["interface_conformance"],
                "violation_reduction": initial_struct["violation_count"] - final_struct["violation_count"],
                "functional_score_gain": (
                    (final_func.get("functional_score") or 0.0) - (initial_func.get("functional_score") or 0.0)
                    if initial_func.get("functional_score") is not None and final_func.get("functional_score") is not None
                    else 0.0
                ),
            },
            "original_code": original_code,
            "final_code": final_code,
            "history": history,
        }

        if save_results:
            self.save_repair_record(result)

        return result

    def get_repair_run_dir(
        self,
        model_id: str,
        task_id: str,
        condition: str,
        source_run: int,
    ) -> str:
        return os.path.join(
            self.base_dir,
            model_id,
            task_id,
            condition,
            f"source_run_{source_run:02d}",
        )

    def save_repair_record(self, repair_result: Dict[str, Any]) -> str:
        model_id = repair_result["model"]
        task_id = repair_result["task_id"]
        condition = repair_result["source_condition"]
        source_run = repair_result["source_run"]

        run_dir = self.get_repair_run_dir(model_id, task_id, condition, source_run)
        os.makedirs(run_dir, exist_ok=True)

        # Save original code
        orig_path = os.path.join(run_dir, "original_code.txt")
        with open(orig_path, "w", encoding=ENCODING) as f:
            f.write(repair_result["original_code"])

        # Save iteration files
        for step in repair_result.get("history", []):
            it = step["iteration"]
            if step.get("prompt"):
                p_path = os.path.join(run_dir, f"repair_prompt_{it:02d}.txt")
                with open(p_path, "w", encoding=ENCODING) as f:
                    f.write(step["prompt"])
            if step.get("repaired_code"):
                rc_path = os.path.join(run_dir, f"repaired_code_{it:02d}.py")
                with open(rc_path, "w", encoding=ENCODING) as f:
                    f.write(step["repaired_code"])
            e_path = os.path.join(run_dir, f"repair_evaluation_{it:02d}.json")
            with open(e_path, "w", encoding=ENCODING) as f:
                json.dump(step["evaluation"], f, indent=4)

        # Save final result
        final_path = os.path.join(run_dir, "final_result.json")
        clean_result = {k: v for k, v in repair_result.items() if k not in ("original_code", "final_code", "history")}
        with open(final_path, "w", encoding=ENCODING) as f:
            json.dump(clean_result, f, indent=4)

        return run_dir
