"""
collect_generation.py — CAPS Formal Experiment CLI

Provides a command-line interface for the manual LLM generation workflow.
Does NOT call any LLM API. All generation is performed manually by the
researcher; this script manages prompts, raw output storage, and evaluation.

USAGE
-----

Step 1 — Generate and display both prompts for a task/run:

    python collect_generation.py prompt \\
        --model <model_id> --task <task_id> --run <N>

    Example:
        python collect_generation.py prompt \\
            --model gpt-4o --task observer_01 --run 1

Step 2 — After copying a prompt to the LLM and receiving its response,
save the raw output and run evaluation:

    python collect_generation.py record \\
        --model <model_id> --task <task_id> \\
        --condition <baseline|caps> --run <N> \\
        --file <path_to_raw_output.txt>

    Example:
        python collect_generation.py record \\
            --model gpt-4o --task observer_01 \\
            --condition baseline --run 1 \\
            --file /tmp/observer_01_baseline_r1.txt

    You can also pipe stdin instead of --file:
        python collect_generation.py record \\
            --model gpt-4o --task observer_01 \\
            --condition baseline --run 1

    (will prompt for raw output via stdin)

Step 3 — Check completion status:

    python collect_generation.py status --model <model_id>

    Example:
        python collect_generation.py status --model gpt-4o

RESEARCH INTEGRITY
------------------
- Generated code is NEVER repaired.
- Raw output is stored exactly as provided.
- If a run directory already contains raw_code.txt, the script will
  refuse to overwrite it (prevents accidental data loss).
- Use --force only when explicitly re-recording a run.
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple

# Allow running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.experiment_runner import ExperimentRunner, TASK_IDS, CONDITIONS

ENCODING = "utf-8"


# ============================================================
# HELPERS
# ============================================================

def _make_runner() -> ExperimentRunner:
    dataset_path = os.path.join(
        os.path.dirname(__file__), "dataset", "test_tasks.json"
    )
    return ExperimentRunner(dataset_path=dataset_path)


def _bar(char: str = "=", width: int = 70) -> str:
    return char * width


# ============================================================
# SUBCOMMAND: prompt
# ============================================================

def cmd_prompt(args: argparse.Namespace) -> int:
    """Generate and display (and save) both prompts for a task/run."""

    runner = _make_runner()

    task_id  = args.task
    model_id = args.model
    run      = args.run

    if task_id not in TASK_IDS:
        print(
            f"ERROR: Unknown task_id '{task_id}'.\n"
            f"Valid tasks: {', '.join(TASK_IDS)}",
            file=sys.stderr,
        )
        return 1

    print(_bar())
    print("CAPS — FORMAL EXPERIMENT PROMPT GENERATION")
    print(_bar())
    print(f"  Model : {model_id}")
    print(f"  Task  : {task_id}")
    print(f"  Run   : {run}")
    print(_bar())

    prompts = runner.generate_prompts(
        task_id=task_id,
        model_id=model_id,
        run=run,
        save=True,
    )

    # Show where prompts were saved
    for condition in CONDITIONS:
        run_dir = runner.get_run_dir(model_id, task_id, condition, run)
        prompt_file = os.path.join(run_dir, "prompt.txt")
        print(f"\n[{condition.upper()}] prompt saved to:")
        print(f"  {prompt_file}")

    if args.show:
        for condition, text in prompts.items():
            print(f"\n{'=' * 70}")
            print(f"PROMPT — {condition.upper()}")
            print("=" * 70)
            print(text)
            print()

    print(_bar())
    print("NEXT STEP:")
    print("  1. Copy the BASELINE prompt to the LLM.")
    print("  2. Copy the exact raw response to a text file.")
    print("  3. Run:  python collect_generation.py record \\")
    print(f"              --model {model_id} --task {task_id} \\")
    print(f"              --condition baseline --run {run} --file <path>")
    print("  4. Repeat for CAPS.")
    print(_bar())

    return 0


# ============================================================
# RESPONSE FORMAT VALIDATION (NON-DESTRUCTIVE AUDIT)
# ============================================================

def validate_response_format(raw_code: str) -> Dict[str, Any]:
    """
    Non-destructively inspect raw LLM output to audit formatting, detect
    Markdown fences, check AST syntax validity, and warn about suspicious
    unindented lines (potential copy/paste or shell redirection corruption).

    Research Integrity Guarantee:
      This function NEVER modifies, auto-indents, or repairs the code.
    """
    import ast
    from src.evaluator import CodeEvaluator

    normalized = CodeEvaluator._normalize_code(raw_code)

    has_fences = "```" in raw_code
    fence_count = raw_code.count("```")
    has_non_ascii = any(ord(c) > 127 for c in raw_code)

    if "\r\n" in raw_code:
        line_endings = "CRLF"
    elif "\n" in raw_code:
        line_endings = "LF"
    else:
        line_endings = "SINGLE_LINE"

    ast_valid = False
    ast_error = None
    try:
        ast.parse(normalized.lstrip("\ufeff"))
        ast_valid = True
    except SyntaxError as err:
        ast_error = f"SyntaxError line {err.lineno}: {err.msg}"

    suspicious_lines = []
    lines = normalized.splitlines()
    expecting_indent = False
    prev_header_line_no = 0
    prev_header_content = ""

    block_openers = (
        "class ", "def ", "if ", "elif ", "else:", "try:",
        "except", "finally:", "for ", "while ", "with "
    )

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" \t"))

        if expecting_indent:
            if indent == 0 and not (
                stripped.startswith("class ") or
                stripped.startswith("def ") or
                stripped.startswith("import ") or
                stripped.startswith("from ") or
                stripped.startswith("if __name__")
            ):
                suspicious_lines.append({
                    "line_no": idx,
                    "line_content": stripped,
                    "after_line_no": prev_header_line_no,
                    "after_content": prev_header_content,
                })
            expecting_indent = False

        if stripped.startswith("@"):
            expecting_indent = True
            prev_header_line_no = idx
            prev_header_content = stripped
        elif stripped.endswith(":"):
            if any(stripped.startswith(b) or stripped == b for b in block_openers):
                expecting_indent = True
                prev_header_line_no = idx
                prev_header_content = stripped

    return {
        "raw_length": len(raw_code),
        "line_count": len(raw_code.splitlines()),
        "has_markdown_fences": has_fences,
        "fence_count": fence_count,
        "has_non_ascii": has_non_ascii,
        "line_endings": line_endings,
        "ast_valid": ast_valid,
        "ast_error": ast_error,
        "suspicious_unindented_lines": suspicious_lines,
        "normalized_length": len(normalized),
    }


def print_format_validation_report(report: Dict[str, Any]) -> None:
    """Print format validation report to stdout."""
    print(_bar("-"))
    print("RESPONSE FORMAT AUDIT REPORT")
    print(_bar("-"))
    print(f"  Raw Character Count  : {report['raw_length']}")
    print(f"  Line Count           : {report['line_count']}")
    print(f"  Line Endings         : {report['line_endings']}")
    print(f"  Markdown Fences      : {'Detected (' + str(report['fence_count']) + ' delimiters)' if report['has_markdown_fences'] else 'None'}")
    print(f"  Non-ASCII Characters : {report['has_non_ascii']}")
    print(f"  AST Parse Status     : {'VALID' if report['ast_valid'] else 'INVALID (' + str(report['ast_error']) + ')'}")

    suspicious = report["suspicious_unindented_lines"]
    if suspicious:
        print()
        print("  [WARNING] SUSPICIOUS UNINDENTED LINES DETECTED:")
        print("  Lines at column 0 immediately following a block header statement:")
        for s in suspicious[:5]:
            print(f"    - Line {s['line_no']}: '{s['line_content']}' (after line {s['after_line_no']}: '{s['after_content']}')")
        if len(suspicious) > 5:
            print(f"    ... and {len(suspicious) - 5} more line(s).")
        print()
        print("  RESEARCH INTEGRITY AUDIT CHECK:")
        print("    Did the LLM web interface ACTUALLY render this code without indentation?")
        print("    [YES] -> Real model failure. Proceed with recording.")
        print("    [NO]  -> Indentation was stripped by copy/paste or shell redirection.")
        print("             Re-save response file using VS Code or Notepad (UTF-8) before recording.")
    else:
        print("  Indentation Status   : NORMAL (No suspicious column-0 line breaks detected)")
    print(_bar("-"))


# ============================================================
# SUBCOMMAND: record
# ============================================================

def cmd_record(args: argparse.Namespace) -> int:
    """Save raw LLM output and run structural + functional evaluation."""

    runner   = _make_runner()
    task_id  = args.task
    model_id = args.model
    condition = args.condition
    run      = args.run
    force    = getattr(args, "force", False)

    if task_id not in TASK_IDS:
        print(f"ERROR: Unknown task_id '{task_id}'.", file=sys.stderr)
        return 1

    if condition not in CONDITIONS:
        print(
            f"ERROR: condition must be one of: {CONDITIONS}",
            file=sys.stderr,
        )
        return 1

    # ---- Load raw code ----
    if args.file:
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}", file=sys.stderr)
            return 1
        with open(file_path, "r", encoding=ENCODING) as fh:
            raw_code = fh.read()
    else:
        print(
            "Paste the raw LLM output below.\n"
            "When done, press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows):",
            file=sys.stderr,
        )
        raw_code = sys.stdin.read()

    if not raw_code.strip():
        print("ERROR: Raw code is empty. Nothing saved.", file=sys.stderr)
        return 1

    # ---- Guard against overwrite ----
    run_dir  = runner.get_run_dir(model_id, task_id, condition, run)
    raw_path = os.path.join(run_dir, "raw_code.txt")

    if os.path.exists(raw_path) and not force:
        print(
            f"ERROR: raw_code.txt already exists in:\n  {run_dir}\n"
            "Use --force to overwrite (only if you are intentionally "
            "re-recording this run).",
            file=sys.stderr,
        )
        return 1

    print(_bar())
    print("CAPS — RECORDING RAW OUTPUT")
    print(_bar())
    print(f"  Model     : {model_id}")
    print(f"  Task      : {task_id}")
    print(f"  Condition : {condition}")
    print(f"  Run       : {run}")
    print(f"  Characters: {len(raw_code)}")

    # Audit format non-destructively
    report = validate_response_format(raw_code)
    print_format_validation_report(report)

    # ---- Save raw output ----
    saved_path = runner.record_raw_output(
        task_id=task_id,
        model_id=model_id,
        condition=condition,
        run=run,
        raw_code=raw_code,
    )
    print(f"\nRaw output saved: {saved_path}")

    # ---- Evaluate ----
    print("\nRunning structural + functional evaluation …")
    try:
        metadata = runner.evaluate_and_save(
            task_id=task_id,
            model_id=model_id,
            condition=condition,
            run=run,
            model_version=getattr(args, "model_version", None),
        )
    except Exception as exc:
        print(f"ERROR during evaluation: {exc}", file=sys.stderr)
        return 1

    ev = metadata.get("evaluation", {})

    print(_bar("-"))
    print("EVALUATION RESULT")
    print(_bar("-"))
    print(f"  AST Valid            : {ev.get('ast_valid')}")
    print(
        f"  Interface Conformance: "
        f"{ev.get('interface_conformance', 0.0) * 100:.1f}%"
    )
    print(f"  Violation Count      : {ev.get('violation_count')}")
    print(f"  Functional Valid     : {ev.get('functional_valid')}")
    print(
        f"  Functional Score     : "
        f"{ev.get('functional_score', 0.0) * 100:.1f}%"
        f"  ({ev.get('functional_tests_passed')}/{ev.get('functional_tests_total')})"
    )
    print(_bar())

    meta_path = os.path.join(run_dir, "metadata.json")
    print(f"Metadata saved : {meta_path}")
    print(f"Evaluation saved: {os.path.join(run_dir, 'evaluation.json')}")
    print(_bar())

    return 0


# ============================================================
# SUBCOMMAND: check
# ============================================================

def cmd_check(args: argparse.Namespace) -> int:
    """Non-destructively inspect a response file without recording it."""
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        return 1
    with open(file_path, "r", encoding=ENCODING) as fh:
        raw_code = fh.read()

    print(_bar())
    print(f"CAPS — RESPONSE FORMAT CHECK: {file_path}")
    print(_bar())
    report = validate_response_format(raw_code)
    print_format_validation_report(report)
    print(_bar())
    return 0


# ============================================================
# SUBCOMMAND: status
# ============================================================

def cmd_status(args: argparse.Namespace) -> int:
    """Show completion status for a model across all 10 tasks × 2 conditions × 5 runs."""

    runner   = _make_runner()
    model_id = args.model
    runs     = args.runs

    print(_bar())
    print(f"CAPS — EXPERIMENT STATUS  [{model_id}]")
    print(_bar())
    print(f"{'Task':<16} ", end="")

    for cond in CONDITIONS:
        for r in range(1, runs + 1):
            print(f"{cond[:1].upper()}{r}", end=" ")
    print()
    print(_bar("-"))

    total_cells   = len(TASK_IDS) * len(CONDITIONS) * runs
    complete_cells = 0
    evaluated_cells = 0

    for task_id in TASK_IDS:
        print(f"{task_id:<16} ", end="")
        for cond in CONDITIONS:
            for r in range(1, runs + 1):
                run_dir  = runner.get_run_dir(model_id, task_id, cond, r)
                raw_ok   = os.path.exists(os.path.join(run_dir, "raw_code.txt"))
                eval_ok  = os.path.exists(os.path.join(run_dir, "evaluation.json"))

                if raw_ok and eval_ok:
                    print("✓ ", end="")
                    complete_cells += 1
                    evaluated_cells += 1
                elif raw_ok:
                    print("R ", end="")   # raw saved, not yet evaluated
                    complete_cells += 1
                else:
                    print(". ", end="")   # empty
        print()

    print(_bar("-"))
    print(f"Legend: ✓=complete  R=raw only (run evaluate)  .=empty")
    print(_bar("-"))
    print(f"Evaluated : {evaluated_cells}/{total_cells}")
    print(f"Remaining : {total_cells - complete_cells}/{total_cells}")
    print(_bar())

    return 0


# ============================================================
# SUBCOMMAND: evaluate
# ============================================================

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Re-run evaluation on a raw_code.txt that is already saved."""

    runner = _make_runner()

    task_id   = args.task
    model_id  = args.model
    condition = args.condition
    run       = args.run

    print(f"Re-evaluating: {model_id}/{task_id}/{condition}_run_{run:02d} …")

    try:
        metadata = runner.evaluate_and_save(
            task_id=task_id,
            model_id=model_id,
            condition=condition,
            run=run,
            model_version=getattr(args, "model_version", None),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR during evaluation: {exc}", file=sys.stderr)
        return 1

    ev = metadata.get("evaluation", {})
    print(f"  AST Valid           : {ev.get('ast_valid')}")
    print(
        f"  Interface Conformance: "
        f"{ev.get('interface_conformance', 0.0) * 100:.1f}%"
    )
    print(f"  Violation Count     : {ev.get('violation_count')}")
    print(f"  Functional Valid    : {ev.get('functional_valid')}")
    print(
        f"  Functional Score    : "
        f"{ev.get('functional_score', 0.0) * 100:.1f}%"
    )
    return 0


# ============================================================
# SUBCOMMAND: prepare
# ============================================================

def cmd_prepare(args: argparse.Namespace) -> int:
    """Generate and save all missing prompt.txt files for a model (10 tasks × 2 conditions × N runs)."""

    runner   = _make_runner()
    model_id = args.model
    runs     = args.runs

    print(_bar())
    print(f"CAPS — PREPARING PROMPTS  [{model_id}]")
    print(_bar())

    total_expected = len(TASK_IDS) * len(CONDITIONS) * runs

    for task_id in TASK_IDS:
        for r in range(1, runs + 1):
            runner.generate_prompts(
                task_id=task_id,
                model_id=model_id,
                run=r,
                save=True,
            )

    print(f"Model          : {model_id}")
    print(f"Total expected : {total_expected} prompt files ({len(TASK_IDS)} tasks × {len(CONDITIONS)} conditions × {runs} runs)")
    print(f"Status         : All prompts generated and ready in results/experiments/{model_id}/")
    print(_bar())

    return 0


# ============================================================
# SUBCOMMAND: next
# ============================================================

def cmd_next(args: argparse.Namespace) -> int:
    """Identify and display the next pending generation in sequence."""

    runner   = _make_runner()
    model_id = args.model
    runs     = args.runs

    for task_id in TASK_IDS:
        for r in range(1, runs + 1):
            for cond in CONDITIONS:
                run_dir = runner.get_run_dir(model_id, task_id, cond, r)
                raw_path = os.path.join(run_dir, "raw_code.txt")
                if not os.path.exists(raw_path):
                    # Ensure prompt exists
                    prompt_path = os.path.join(run_dir, "prompt.txt")
                    if not os.path.exists(prompt_path):
                        runner.generate_prompts(
                            task_id=task_id,
                            model_id=model_id,
                            run=r,
                            save=True,
                        )

                    print(_bar())
                    print("NEXT GENERATION")
                    print(_bar())
                    print(f"Model    : {model_id}")
                    print(f"Task     : {task_id}")
                    print(f"Run      : {r}")
                    print(f"Condition: {cond}")
                    print(f"Prompt   : {prompt_path}")
                    print(_bar())
                    print("INSTRUCTIONS:")
                    print(f"  1. Copy prompt from: {prompt_path}")
                    print(f"  2. Submit to model '{model_id}'.")
                    print("  3. Save raw output to a text file (e.g. response.txt).")
                    print("  4. Record and evaluate:")
                    print(
                        f"     python collect_generation.py record \\\n"
                        f"         --model {model_id} --task {task_id} \\\n"
                        f"         --condition {cond} --run {r} --file response.txt"
                    )
                    print(_bar())
                    return 0

    print(_bar())
    print(f"ALL GENERATIONS COMPLETE FOR MODEL [{model_id}]")
    print(f"All {len(TASK_IDS) * len(CONDITIONS) * runs} runs have recorded raw output.")
    print(_bar())
    return 0


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_generation",
        description=(
            "CAPS formal experiment CLI — "
            "manual generation workflow helper"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- prepare ----
    p_prep = sub.add_parser(
        "prepare",
        help="Generate and save all missing prompt.txt files for a model.",
    )
    p_prep.add_argument("--model", required=True, help="Model identifier")
    p_prep.add_argument(
        "--runs", type=int, default=5,
        help="Expected runs per condition (default: 5)",
    )
    p_prep.set_defaults(func=cmd_prepare)

    # ---- next ----
    p_next = sub.add_parser(
        "next",
        help="Display the next pending generation in sequence.",
    )
    p_next.add_argument("--model", required=True, help="Model identifier")
    p_next.add_argument(
        "--runs", type=int, default=5,
        help="Expected runs per condition (default: 5)",
    )
    p_next.set_defaults(func=cmd_next)

    # ---- prompt ----
    p_prompt = sub.add_parser(
        "prompt",
        help="Generate and save both prompts for a task/run.",
    )
    p_prompt.add_argument("--model",  required=True, help="Model identifier")
    p_prompt.add_argument("--task",   required=True, help="Task ID (e.g. observer_01)")
    p_prompt.add_argument("--run",    required=True, type=int, help="Run number (1–5)")
    p_prompt.add_argument(
        "--show", action="store_true",
        help="Also print the prompt text to stdout",
    )
    p_prompt.set_defaults(func=cmd_prompt)

    # ---- record ----
    p_record = sub.add_parser(
        "record",
        help="Save raw LLM output and evaluate it.",
    )
    p_record.add_argument("--model",     required=True)
    p_record.add_argument("--task",      required=True)
    p_record.add_argument("--condition", required=True, choices=list(CONDITIONS))
    p_record.add_argument("--run",       required=True, type=int)
    p_record.add_argument(
        "--file", default=None,
        help="Path to a text file containing the raw LLM output. "
             "If omitted, read from stdin.",
    )
    p_record.add_argument(
        "--model-version", dest="model_version", default=None,
        help="Optional model version string (e.g. gpt-4o-2024-08). "
             "Do NOT fabricate; leave blank if unknown.",
    )
    p_record.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing raw_code.txt (use with caution).",
    )
    p_record.set_defaults(func=cmd_record)

    # ---- status ----
    p_status = sub.add_parser(
        "status",
        help="Show completion status for a model.",
    )
    p_status.add_argument("--model", required=True)
    p_status.add_argument(
        "--runs", type=int, default=5,
        help="Expected runs per condition (default: 5)",
    )
    p_status.set_defaults(func=cmd_status)

    # ---- check ----
    p_chk = sub.add_parser(
        "check",
        help="Inspect a response file for formatting issues without recording.",
    )
    p_chk.add_argument(
        "--file", required=True,
        help="Path to response file (e.g. response.txt)",
    )
    p_chk.set_defaults(func=cmd_check)

    # ---- evaluate ----
    p_eval = sub.add_parser(
        "evaluate",
        help="Re-run evaluation on an already-recorded raw_code.txt.",
    )
    p_eval.add_argument("--model",     required=True)
    p_eval.add_argument("--task",      required=True)
    p_eval.add_argument("--condition", required=True, choices=list(CONDITIONS))
    p_eval.add_argument("--run",       required=True, type=int)
    p_eval.add_argument(
        "--model-version", dest="model_version", default=None,
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

