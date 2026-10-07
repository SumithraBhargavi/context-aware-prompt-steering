"""
repair.py — CAPS-Repair CLI

Command-line workflow helper for Verifier-Guided Structural Repair (CAPS Phase 2).

USAGE
-----
Run repair loop on a specific recorded generation:
    python repair.py --model claude-sonnet-5 --task factory_01 --condition baseline --run 2

Generate and save repair summary CSV across all completed repairs:
    python repair.py --summary

Run batch repair pilot across representative violating baseline outputs:
    python repair.py --pilot

Show help:
    python repair.py --help
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.experiment_runner import ExperimentRunner, TASK_IDS, CONDITIONS
from src.repair_engine import RepairEngine, REPAIR_BASE_DIR, MAX_REPAIR_ITERATIONS

ENCODING = "utf-8"


def _bar(char: str = "=", width: int = 70) -> str:
    return char * width


def rule_based_repair(task_id: str, prompt: str, original_code: str) -> str:
    """
    Automated rule-based repair transformer for offline pilot testing.
    Fixes common baseline structural violations (missing type annotations,
    missing ABC inheritance, missing abstract decorators) based on contract.
    """
    import ast
    from src.task_contracts import load_tasks
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "test_tasks.json")
    tasks = {t["task_id"]: t for t in load_tasks(dataset_path)}
    task = tasks.get(task_id)
    if not task:
        return original_code

    contract = task["contract"]
    lines = original_code.splitlines()
    repaired_lines = []

    # Map class name -> inherits / kind
    class_map = {c["name"]: c for c in contract.get("classes", [])}
    method_map = {(m["class"], m["name"]): m for m in contract.get("methods", [])}

    in_class = None
    for line in lines:
        stripped = line.strip()
        # Fix ABC inheritance if missing
        if stripped.startswith("class "):
            parts = stripped.split(":")
            header = parts[0]
            cname = header.replace("class ", "").split("(")[0].strip()
            in_class = cname
            if cname in class_map:
                cinfo = class_map[cname]
                kind = cinfo.get("kind")
                inherits = cinfo.get("inherits", [])
                if kind == "abstract" and not inherits and "(" not in header:
                    line = line.replace(f"class {cname}:", f"class {cname}(ABC):")
                    line = line.replace(f"class {cname} :", f"class {cname}(ABC):")

        # Fix method annotations & abstract decorators
        if stripped.startswith("def ") and in_class:
            mname = stripped.split("(")[0].replace("def ", "").strip()
            key = (in_class, mname)
            if key in method_map:
                minfo = method_map[key]
                params = minfo.get("parameters", [])
                ret_type = minfo.get("return_type")
                is_abstract = minfo.get("abstract", False)
                is_static = minfo.get("static", False)

                # Reconstruct signature with types if missing
                param_strs = []
                for p in params:
                    pname = p["name"]
                    ptype = p.get("type")
                    if ptype:
                        param_strs.append(f"{pname}: {ptype}")
                    else:
                        param_strs.append(pname)

                sig_params = ", ".join(param_strs)
                if not is_static:
                    sig_params = "self, " + sig_params if sig_params else "self"

                ret_str = f" -> {ret_type}:" if ret_type else ":"

                new_def = f"def {mname}({sig_params}){ret_str}"

                # Add decorators if missing
                indent = line[:len(line) - len(line.lstrip())]
                dec_lines = []
                if is_abstract:
                    dec_lines.append(f"{indent}@abstractmethod")
                if is_static:
                    dec_lines.append(f"{indent}@staticmethod")

                if dec_lines:
                    line = "\n".join(dec_lines) + "\n" + f"{indent}{new_def}"
                else:
                    line = f"{indent}{new_def}"

        repaired_lines.append(line)

    repaired_code = "\n".join(repaired_lines)

    # Ensure ABC imports exist if needed
    if "ABC" in repaired_code and "from abc import" not in repaired_code and "import abc" not in repaired_code:
        repaired_code = "from abc import ABC, abstractmethod\n" + repaired_code

    return repaired_code


def cmd_repair_single(args: argparse.Namespace) -> int:
    engine = RepairEngine()

    model_id = args.model
    task_id = args.task
    condition = args.condition
    run_num = args.run

    exp_runner_dir = os.path.join("results", "experiments", model_id, task_id, f"{condition}_run_{run_num:02d}")
    raw_path = os.path.join(exp_runner_dir, "raw_code.txt")

    if not os.path.exists(raw_path):
        print(f"ERROR: Raw output not found at: {raw_path}", file=sys.stderr)
        return 1

    with open(raw_path, "r", encoding=ENCODING, errors="replace") as f:
        original_code = f.read()

    print(_bar())
    print("CAPS-REPAIR")
    print(_bar())
    print(f"Model  : {model_id}")
    print(f"Task   : {task_id}")
    print(f"Source : {condition}_run_{run_num:02d}")
    print(_bar())

    # Pre-repair evaluation
    eval_before = engine.evaluate_code(task_id, original_code)
    struct_b = eval_before["structural"]
    func_b = eval_before.get("functional") or {}

    print("Before:")
    print(f"AST: {'VALID' if struct_b['ast_valid'] else 'INVALID'}")
    print(f"Conformance: {struct_b['interface_conformance'] * 100:.1f}%")
    print(f"Violations: {struct_b['violation_count']}")
    print(f"Functional: {'PASS' if func_b.get('functional_valid') else 'FAIL'}")
    print(_bar("-"))

    if args.prompt_only:
        violations = engine.extract_violations(struct_b)
        prompt = engine.construct_repair_prompt(task_id, original_code, violations)
        print("REPAIR PROMPT (Iteration 1):")
        print(_bar("-"))
        print(prompt)
        print(_bar())
        return 0

    if args.file:
        with open(args.file, "r", encoding=ENCODING, errors="replace") as f:
            user_repaired_code = f.read()
        llm_fn = lambda p: user_repaired_code
    else:
        llm_fn = lambda p: rule_based_repair(task_id, p, original_code)

    repair_result = engine.run_repair_loop(
        task_id=task_id,
        model_id=model_id,
        source_condition=condition,
        source_run=run_num,
        original_code=original_code,
        repair_llm_fn=llm_fn,
        max_iterations=args.max_iterations,
        run_functional=not args.no_functional,
        save_results=True,
    )

    # Print iteration breakdown
    for step in repair_result.get("history", []):
        it = step["iteration"]
        st = step["evaluation"]["structural"]
        fn = step["evaluation"].get("functional") or {}
        print(f"Repair iteration {it}:")
        print(f"Conformance: {st['interface_conformance'] * 100:.1f}%")
        print(f"Violations: {st['violation_count']}")
        print(f"Functional: {'PASS' if fn.get('functional_valid') else 'FAIL'}")
        print(_bar("-"))

    post = repair_result["post_repair"]
    print("FINAL:")
    print(f"Structural: {'VALID' if post['ast_valid'] and post['interface_conformance'] == 1.0 else 'NON-CONFORMING'}")
    print(f"Functional: {'VALID' if post['functional_valid'] else 'FAIL'}")
    print(f"Repair: {'SUCCESS' if repair_result['repair_success'] else 'FAILED'}")
    print(f"Iterations: {repair_result['iterations_completed']}")
    print(_bar())

    run_dir = engine.get_repair_run_dir(model_id, task_id, condition, run_num)
    print(f"Repair artifacts saved to: {run_dir}")
    print(_bar())

    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    """Run batch repair pilot across representative violating baseline runs."""
    engine = RepairEngine()
    models = ["claude-sonnet-5", "deepseek-v4.1-flash"]

    print(_bar())
    print("CAPS-REPAIR — PILOT EXPERIMENT EXECUTION")
    print(_bar())

    completed = 0
    successes = 0

    for model_id in models:
        for task_id in TASK_IDS:
            for run_num in [1, 2, 3]:
                exp_runner_dir = os.path.join("results", "experiments", model_id, task_id, f"baseline_run_{run_num:02d}")
                raw_path = os.path.join(exp_runner_dir, "raw_code.txt")
                if not os.path.exists(raw_path):
                    continue

                with open(raw_path, "r", encoding=ENCODING, errors="replace") as f:
                    orig_code = f.read()

                eval_init = engine.evaluate_code(task_id, orig_code)
                struct_init = eval_init["structural"]

                # Only run repair pilot on outputs with structural violations
                if struct_init["interface_conformance"] < 1.0 or struct_init["violation_count"] > 0:
                    llm_fn = lambda p, t=task_id, c=orig_code: rule_based_repair(t, p, c)
                    res = engine.run_repair_loop(
                        task_id=task_id,
                        model_id=model_id,
                        source_condition="baseline",
                        source_run=run_num,
                        original_code=orig_code,
                        repair_llm_fn=llm_fn,
                        max_iterations=3,
                        run_functional=True,
                        save_results=True,
                    )
                    completed += 1
                    if res["repair_success"]:
                        successes += 1
                    print(f"Pilot [{model_id}/{task_id}/baseline_run_{run_num:02d}]: Conf {res['pre_repair']['interface_conformance']*100:.1f}% -> {res['post_repair']['interface_conformance']*100:.1f}%, Violations {res['pre_repair']['violation_count']} -> {res['post_repair']['violation_count']}, Success: {res['repair_success']}")

    print(_bar())
    print(f"Pilot execution complete: {completed} repair cases evaluated.")
    print(f"Repair success rate     : {successes}/{completed} ({successes / completed * 100:.1f}%)" if completed > 0 else "No violating cases found.")
    print(_bar())

    cmd_summary(args)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Generate results/repair_experiments/repair_summary.csv."""
    base_dir = REPAIR_BASE_DIR
    if not os.path.exists(base_dir):
        print("No repair experiments found in results/repair_experiments/", file=sys.stderr)
        return 0

    rows = []
    for root, dirs, files in os.walk(base_dir):
        if "final_result.json" in files:
            res_path = os.path.join(root, "final_result.json")
            with open(res_path, "r", encoding=ENCODING) as f:
                res = json.load(f)

            pre = res.get("pre_repair", {})
            post = res.get("post_repair", {})
            deltas = res.get("deltas", {})

            # Map pattern
            from aggregate_results import TASK_PATTERN_MAP
            task_id = res.get("task_id", "")
            pattern = TASK_PATTERN_MAP.get(task_id, "")

            rows.append({
                "model": res.get("model"),
                "task_id": task_id,
                "pattern": pattern,
                "source_condition": res.get("source_condition"),
                "source_run": res.get("source_run"),
                "iterations_completed": res.get("iterations_completed"),
                "max_iterations": res.get("max_iterations"),
                "repair_success": res.get("repair_success"),
                "ast_valid_before": pre.get("ast_valid"),
                "ast_valid_after": post.get("ast_valid"),
                "conformance_before": pre.get("interface_conformance"),
                "conformance_after": post.get("interface_conformance"),
                "violations_before": pre.get("violation_count"),
                "violations_after": post.get("violation_count"),
                "functional_score_before": pre.get("functional_score"),
                "functional_score_after": post.get("functional_score"),
                "functional_valid_before": pre.get("functional_valid"),
                "functional_valid_after": post.get("functional_valid"),
                "conformance_gain": deltas.get("conformance_gain"),
                "violation_reduction": deltas.get("violation_reduction"),
                "failure_reason": res.get("failure_reason", ""),
            })

    rows.sort(key=lambda x: (str(x["model"]), str(x["task_id"]), str(x["source_condition"]), int(x["source_run"] or 0)))

    summary_path = os.path.join(base_dir, "repair_summary.csv")
    fieldnames = [
        "model", "task_id", "pattern", "source_condition", "source_run",
        "iterations_completed", "max_iterations", "repair_success",
        "ast_valid_before", "ast_valid_after",
        "conformance_before", "conformance_after",
        "violations_before", "violations_after",
        "functional_score_before", "functional_score_after",
        "functional_valid_before", "functional_valid_after",
        "conformance_gain", "violation_reduction", "failure_reason",
    ]

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="", encoding=ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Repair summary CSV generated: {summary_path} ({len(rows)} records)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repair",
        description="CAPS-Repair CLI — Verifier-Guided Structural Repair",
    )
    parser.add_argument("--model", default="claude-sonnet-5", help="Model ID")
    parser.add_argument("--task", default="factory_01", help="Task ID")
    parser.add_argument("--condition", default="baseline", choices=list(CONDITIONS))
    parser.add_argument("--run", type=int, default=1, help="Source run number")
    parser.add_argument("--max-iterations", type=int, default=MAX_REPAIR_ITERATIONS)
    parser.add_argument("--file", default=None, help="Path to repaired code file (for manual recording)")
    parser.add_argument("--prompt-only", action="store_true", help="Export repair prompt only")
    parser.add_argument("--no-functional", action="store_true", help="Skip functional evaluation")
    parser.add_argument("--pilot", action="store_true", help="Run batch repair pilot across violating baseline runs")
    parser.add_argument("--summary", action="store_true", help="Generate repair_summary.csv")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.summary:
        return cmd_summary(args)
    if args.pilot:
        return cmd_pilot(args)

    return cmd_repair_single(args)


if __name__ == "__main__":
    sys.exit(main())
