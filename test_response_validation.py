"""
test_response_validation.py — Unit tests for raw response recording, normalization,
and non-destructive format auditing.

Covers research integrity rules:
  a. Correctly indented Python
  b. Genuinely unindented Python (AST invalid, no repair)
  c. Markdown fenced Python
  d. Markdown fenced Python with valid indentation
  e. Malformed Python (syntax errors preserved)
  f. Preservation of raw output (exact bytes/chars)
  g. Normalization without code repair (no auto-indenting)
  h. Responses containing blank lines
  i. Responses containing non-ASCII characters
  j. Responses with CRLF / LF line endings
"""

import os
import json
import tempfile
import unittest
from src.evaluator import CodeEvaluator
from src.functional_evaluator import FunctionalEvaluator
from src.experiment_runner import ExperimentRunner
from collect_generation import validate_response_format


class TestResponseValidationAndNormalization(unittest.TestCase):

    def test_a_correctly_indented_python(self):
        code = (
            "class Notification:\n"
            "    def send(self) -> None:\n"
            "        print('hello')\n"
        )
        norm = CodeEvaluator._normalize_code(code)
        self.assertEqual(norm, code.strip())
        report = validate_response_format(code)
        self.assertTrue(report["ast_valid"])
        self.assertEqual(len(report["suspicious_unindented_lines"]), 0)

    def test_b_genuinely_unindented_python(self):
        code = (
            "class Notification:\n"
            "@abstractmethod\n"
            "def send(self):\n"
            "pass\n"
        )
        norm = CodeEvaluator._normalize_code(code)
        # Normalization MUST NOT auto-indent code
        self.assertEqual(norm, code.strip())
        report = validate_response_format(code)
        self.assertFalse(report["ast_valid"])
        self.assertGreater(len(report["suspicious_unindented_lines"]), 0)

    def test_c_markdown_fenced_python(self):
        code = (
            "Here is the solution:\n"
            "```python\n"
            "class Notification:\n"
            "    pass\n"
            "```\n"
            "Hope this helps!"
        )
        norm = CodeEvaluator._normalize_code(code)
        expected = "class Notification:\n    pass"
        self.assertEqual(norm, expected)
        report = validate_response_format(code)
        self.assertTrue(report["has_markdown_fences"])

    def test_d_markdown_fenced_python_with_valid_indentation(self):
        code = (
            "```python\n"
            "class Factory:\n"
            "    def create(self):\n"
            "        return None\n"
            "```"
        )
        norm = CodeEvaluator._normalize_code(code)
        self.assertEqual(norm, "class Factory:\n    def create(self):\n        return None")
        report = validate_response_format(code)
        self.assertTrue(report["ast_valid"])
        self.assertEqual(len(report["suspicious_unindented_lines"]), 0)

    def test_e_malformed_python_syntax_error(self):
        code = "class Notification(\n    def send(self):\n        pass"
        norm = CodeEvaluator._normalize_code(code)
        self.assertEqual(norm, code.strip())
        report = validate_response_format(code)
        self.assertFalse(report["ast_valid"])

    def test_f_preservation_of_raw_output(self):
        with tempfile.TemporaryDirectory(prefix="caps_test_raw_") as tmpdir:
            runner = ExperimentRunner(base_dir=tmpdir)
            raw_input = (
                "```python\n"
                "# Raw output test\n"
                "class Test:\n"
                "    pass\n"
                "```"
            )
            saved_path = runner.record_raw_output(
                task_id="factory_01",
                model_id="test_model",
                condition="baseline",
                run=1,
                raw_code=raw_input,
            )
            with open(saved_path, "r", encoding="utf-8") as fh:
                stored = fh.read()
            self.assertEqual(stored, raw_input)

    def test_g_normalization_without_code_repair(self):
        code = "class Notification:\ndef send(self):\npass"
        evaluator = CodeEvaluator({})
        norm = CodeEvaluator._normalize_code(code)
        result = evaluator.evaluate(code)
        # Normalization MUST NOT fix syntax or indentation
        self.assertFalse(result["ast_valid"])
        self.assertIn("AST parse error", result["violations"][0])

    def test_h_responses_containing_blank_lines(self):
        code = (
            "class Notification:\n"
            "\n"
            "    def send(self):\n"
            "\n"
            "        pass\n"
        )
        norm = CodeEvaluator._normalize_code(code)
        self.assertIn("\n\n    def send(self):", norm)
        report = validate_response_format(code)
        self.assertTrue(report["ast_valid"])

    def test_i_responses_containing_non_ascii_characters(self):
        code = (
            "# Unicode test: α + β = γ\n"
            "class Notification:\n"
            "    def send(self) -> None:\n"
            "        print('Notification sent 👍')\n"
        )
        norm = CodeEvaluator._normalize_code(code)
        self.assertIn("Notification sent 👍", norm)
        report = validate_response_format(code)
        self.assertTrue(report["has_non_ascii"])

    def test_j_responses_with_crlf_and_lf_line_endings(self):
        crlf_code = "class Notification:\r\n    def send(self):\r\n        pass\r\n"
        lf_code = "class Notification:\n    def send(self):\n        pass\n"

        norm_crlf = CodeEvaluator._normalize_code(crlf_code)
        norm_lf = CodeEvaluator._normalize_code(lf_code)

        self.assertEqual(norm_crlf, norm_lf)

        report_crlf = validate_response_format(crlf_code)
        self.assertEqual(report_crlf["line_endings"], "CRLF")

        report_lf = validate_response_format(lf_code)
        self.assertEqual(report_lf["line_endings"], "LF")


if __name__ == "__main__":
    unittest.main(verbosity=2)
