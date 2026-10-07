"""
FunctionalEvaluator – Phase 3 of the CAPS research infrastructure.

Executes generated Python code against task-specific functional tests
using subprocess isolation with a hard timeout.

IMPORTANT RESEARCH-INTEGRITY NOTES:
  - Raw generated code is NEVER modified in place.
  - Markdown code fences are stripped only in a COPY for execution.
  - Generated code is NEVER repaired before evaluation.
  - Functional correctness is a SEPARATE metric; it does NOT influence
    the structural conformance score produced by CodeEvaluator.
  - Generated code executes in an isolated temporary directory.
    This is suitable for a local experimental environment only.
    Do NOT use in production or with untrusted code.

WARNING:
  This evaluator executes model-generated Python code locally.
  Run only in a controlled, isolated experimental environment.
  The caller is responsible for the security of the execution context.
"""

import os
import re
import sys
import json
import tempfile
import subprocess
import textwrap
from typing import Dict, Any, Optional, List

from src.functional_tests.harnesses import get_test_harness


# ============================================================
# CONSTANTS
# ============================================================

# Subprocess execution timeout in seconds.
# Prevents infinite loops from hanging the experiment.
EXECUTION_TIMEOUT_SECONDS = 30

# Encoding for all file I/O.
_ENCODING = "utf-8"

# Regex to strip markdown code fences.
# Handles ```python, ```py, ``` (any language tag or none).
_FENCE_PATTERN = re.compile(
    r"^```[a-zA-Z0-9_]*\s*\n(.*?)\n?```\s*$",
    re.DOTALL | re.MULTILINE,
)


# ============================================================
# FUNCTIONAL EVALUATOR
# ============================================================

class FunctionalEvaluator:
    """
    Evaluates the functional correctness of LLM-generated Python code
    for a given CAPS benchmark task.

    The evaluator:
      1. Normalizes the raw generated code (strips markdown fences).
      2. Appends the task-specific functional test harness.
      3. Writes the combined script to a temporary file.
      4. Executes it as a subprocess with a hard timeout.
      5. Parses stdout/stderr to extract pass/fail counts.
      6. Returns a structured result dict.

    The raw code is preserved; only a temporary copy is used for
    execution. The evaluator is intentionally independent of
    CodeEvaluator – functional and structural metrics do not
    influence each other.

    Args:
        task_id: One of the 10 CAPS benchmark task identifiers.
        timeout: Execution timeout in seconds (default: 30).
    """

    def __init__(
        self,
        task_id: str,
        timeout: int = EXECUTION_TIMEOUT_SECONDS,
    ):
        self.task_id = task_id
        self.timeout = timeout

        # Validate task_id early so callers get a clear error.
        self._harness = get_test_harness(task_id)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def evaluate(self, raw_code: str) -> Dict[str, Any]:
        """
        Evaluate functional correctness of raw_code for this task.

        Args:
            raw_code: The raw string exactly as produced by the LLM
                      (may contain markdown code fences).

        Returns:
            dict with keys:
                functional_valid  (bool)   – True iff all tests pass
                tests_total       (int)    – number of tests attempted
                tests_passed      (int)    – number of tests that passed
                functional_score  (float)  – tests_passed / tests_total
                                             or 0.0 if tests_total == 0
                failures          (list)   – list of failure message strings
                execution_error   (str|None) – set if subprocess crashed
                                               before tests completed
                timed_out         (bool)   – True if timeout was hit
                normalized_code   (str)    – code after fence stripping
        """

        result: Dict[str, Any] = {
            "functional_valid": False,
            "tests_total": 0,
            "tests_passed": 0,
            "functional_score": 0.0,
            "failures": [],
            "execution_error": None,
            "timed_out": False,
            "normalized_code": "",
        }

        # -------------------------------------------------
        # 1. Normalize code (strip markdown fences only)
        # -------------------------------------------------

        normalized = self._normalize_code(raw_code)
        result["normalized_code"] = normalized

        if not normalized.strip():
            result["execution_error"] = (
                "Normalized code is empty after fence stripping."
            )
            return result

        # -------------------------------------------------
        # 2. Build combined script
        # -------------------------------------------------

        combined_script = normalized + "\n\n" + self._harness

        # -------------------------------------------------
        # 3. Write to temp file and execute
        # -------------------------------------------------

        with tempfile.TemporaryDirectory(prefix="caps_eval_") as tmpdir:

            script_path = os.path.join(tmpdir, f"{self.task_id}_eval.py")

            with open(script_path, "w", encoding=_ENCODING) as f:
                f.write(combined_script)

            proc_result = self._run_subprocess(script_path)

        # -------------------------------------------------
        # 4. Handle timeout
        # -------------------------------------------------

        if proc_result["timed_out"]:
            result["timed_out"] = True
            result["execution_error"] = (
                f"Execution timed out after {self.timeout} seconds."
            )
            return result

        # -------------------------------------------------
        # 5. Parse output
        # -------------------------------------------------

        self._parse_output(proc_result, result)

        return result

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_code(raw_code: str) -> str:
        """
        Create a normalized COPY of the raw LLM response for execution.

        Only Markdown fence lines are removed or isolated.
        The raw LLM response is never modified or repaired.
        Line breaks are normalized to \n in the execution copy.
        """
        if not raw_code:
            return ""

        clean_code = raw_code.lstrip("\ufeff")

        if "```" in clean_code:
            pattern = re.compile(r"```[a-zA-Z0-9_]*\s*\n(.*?)\n?```", re.DOTALL)
            matches = pattern.findall(clean_code)
            if matches:
                body = "\n\n".join(matches)
                return "\n".join(body.splitlines()).strip()

            lines = clean_code.splitlines()
            normalized_lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            return "\n".join(normalized_lines).strip()

        return "\n".join(clean_code.splitlines()).strip()

    # =========================================================
    # SUBPROCESS EXECUTION
    # =========================================================

    @staticmethod
    def _decode_bytes(data: bytes) -> str:
        """
        Robustly decode subprocess stdout/stderr bytes into str.
        Tries UTF-8 first; if that fails (e.g., Windows code page bytes like degree symbol 0xb0),
        falls back to CP1252, and finally utf-8 with errors='replace'.
        """
        if not data:
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("cp1252")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="replace")

    def _run_subprocess(
        self, script_path: str
    ) -> Dict[str, Any]:
        """
        Run the script as a subprocess with the configured timeout.

        Returns dict with:
            stdout (str), stderr (str), returncode (int),
            timed_out (bool)
        """

        proc_result = {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "timed_out": False,
        }

        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                timeout=self.timeout,
            )

            proc_result["stdout"] = self._decode_bytes(completed.stdout or b"")
            proc_result["stderr"] = self._decode_bytes(completed.stderr or b"")
            proc_result["returncode"] = completed.returncode

        except subprocess.TimeoutExpired:
            proc_result["timed_out"] = True

        except Exception as exc:
            proc_result["stderr"] = (
                f"Subprocess launch error: {type(exc).__name__}: {exc}"
            )

        return proc_result

    # =========================================================
    # OUTPUT PARSING
    # =========================================================

    @staticmethod
    def _parse_output(
        proc_result: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """
        Parse subprocess stdout/stderr into the evaluation result dict.

        Expected stdout format (from harness):
            TESTS_TOTAL: <int>
            TESTS_PASSED: <int>
            ALL_TESTS_PASSED   (optional, only if all passed)

        Expected stderr format:
            FAIL: <message>   (one per failure)
        """

        stdout = proc_result.get("stdout", "")
        stderr = proc_result.get("stderr", "")
        returncode = proc_result.get("returncode", -1)

        # --- Parse TESTS_TOTAL ---
        total_match = re.search(
            r"^TESTS_TOTAL:\s*(\d+)",
            stdout,
            re.MULTILINE,
        )
        if total_match:
            result["tests_total"] = int(total_match.group(1))

        # --- Parse TESTS_PASSED ---
        passed_match = re.search(
            r"^TESTS_PASSED:\s*(\d+)",
            stdout,
            re.MULTILINE,
        )
        if passed_match:
            result["tests_passed"] = int(passed_match.group(1))

        # --- Parse individual FAIL messages from stderr ---
        failures: List[str] = []
        for line in stderr.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAIL:"):
                failures.append(stripped[5:].strip())

        # If there's no FAIL lines but returncode != 0, record raw stderr
        if returncode != 0 and not failures and stderr.strip():
            failures.append(
                f"Script exited with code {returncode}. "
                f"stderr: {stderr.strip()[:500]}"
            )

        result["failures"] = failures

        # --- Compute score ---
        total = result["tests_total"]
        passed = result["tests_passed"]

        if total > 0:
            result["functional_score"] = passed / total
        else:
            result["functional_score"] = 0.0

        # --- Determine validity ---
        result["functional_valid"] = (
            returncode == 0
            and total > 0
            and passed == total
            and len(failures) == 0
        )

        # --- Capture execution error if script crashed entirely ---
        if result["tests_total"] == 0 and stderr.strip():
            result["execution_error"] = stderr.strip()[:1000]
