import ast
from typing import Dict, Any, Optional


class CodeEvaluator:
    """
    Research-grade evaluator for structural conformance
    of LLM-generated Python code.

    Metrics:
        - AST parse validity
        - Interface conformance
        - Structural violation count

    Structural checks:
        - Required classes
        - Class kind (abstract/concrete)
        - Inheritance
        - Required methods
        - Abstract method status
        - Static method status
        - Parameter count
        - Parameter names
        - Parameter types
        - Return types
    """

    def __init__(self, contract: Dict[str, Any]):
        if contract is None:
            contract = {}
        if not isinstance(contract, dict):
            raise TypeError("contract must be a dictionary mapping to class and method metadata")
        self.contract = contract
    @staticmethod
    def _normalize_code(code: str) -> str:
        """
        Create a normalized COPY of the raw LLM response for execution/parsing.

        Only Markdown code-fence lines are removed or isolated.
        No Python code is repaired, re-indented, rewritten, or otherwise modified.
        Line breaks are normalized to \n in the execution copy.
        """
        if not code:
            return ""

        clean_code = code.lstrip("\ufeff")

        if "```" in clean_code:
            import re
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
    def evaluate(self, code: str) -> Dict[str, Any]:

        result = {
            "ast_valid": False,
            "interface_conformance": 0.0,
            "total_checks": 0,
            "passed_checks": 0,
            "violation_count": 0,
            "violations": []
        }

        if code is None:
            code = ""
        elif not isinstance(code, str):
            code = str(code)

        # =========================================================
        # 1. AST PARSING
        # =========================================================

        try:
            normalized_code = self._normalize_code(code)

            # Ignore a UTF-8 BOM for parsing only.
            # The original raw LLM output remains unchanged.
            parse_code = normalized_code.lstrip("\ufeff")

            tree = ast.parse(parse_code)
            result["ast_valid"] = True

        except SyntaxError as e:
            result["violations"].append(
                f"AST parse error: {e.msg} at line {e.lineno}"
            )

            result["violation_count"] = 1

            return result

        # =========================================================
        # 2. EXTRACT CLASSES
        # =========================================================

        classes = {}

        for node in ast.walk(tree):

            if isinstance(node, ast.ClassDef):
                classes[node.name] = node

        # =========================================================
        # 3. CLASS CHECKS
        # =========================================================

        for expected_class in self.contract.get("classes", []):

            class_name = expected_class["name"]

            # -----------------------------------------------------
            # Class existence
            # -----------------------------------------------------

            result["total_checks"] += 1

            if class_name not in classes:

                result["violations"].append(
                    f"Missing required class: {class_name}"
                )

                continue

            result["passed_checks"] += 1

            class_node = classes[class_name]

            # -----------------------------------------------------
            # Class inheritance
            # -----------------------------------------------------

            expected_bases = expected_class.get("inherits", [])

            if expected_bases:

                actual_bases = []

                for base in class_node.bases:

                    if isinstance(base, ast.Name):
                        actual_bases.append(base.id)

                    elif isinstance(base, ast.Attribute):
                        actual_bases.append(
                            ast.unparse(base)
                        )

                for expected_base in expected_bases:

                    result["total_checks"] += 1

                    if (
                        expected_base in actual_bases
                        or any(
                            base.endswith("." + expected_base)
                            for base in actual_bases
                        )
                    ):
                        result["passed_checks"] += 1

                    else:
                        result["violations"].append(
                            f"{class_name} should inherit from "
                            f"{expected_base}"
                        )

            # -----------------------------------------------------
            # Abstract / concrete class check
            # -----------------------------------------------------

            expected_kind = expected_class.get("kind")

            if expected_kind:

                result["total_checks"] += 1

                is_abstract = self._class_is_abstract(class_node)

                if expected_kind == "abstract":

                    if is_abstract:
                        result["passed_checks"] += 1

                    else:
                        result["violations"].append(
                            f"{class_name} should be abstract"
                        )

                elif expected_kind == "concrete":

                    if not is_abstract:
                        result["passed_checks"] += 1

                    else:
                        result["violations"].append(
                            f"{class_name} should be concrete"
                        )

        # =========================================================
        # 4. METHOD CHECKS
        # =========================================================

        for expected_method in self.contract.get("methods", []):

            class_name = expected_method["class"]
            method_name = expected_method["name"]

            # -----------------------------------------------------
            # Class must exist
            # -----------------------------------------------------

            if class_name not in classes:
                continue

            class_node = classes[class_name]

            # -----------------------------------------------------
            # Find method
            # -----------------------------------------------------

            actual_method = None

            for node in class_node.body:

                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef)
                ):

                    if node.name == method_name:
                        actual_method = node
                        break

            # -----------------------------------------------------
            # Method existence
            # -----------------------------------------------------

            result["total_checks"] += 1

            if actual_method is None:

                result["violations"].append(
                    f"Missing method: "
                    f"{class_name}.{method_name}"
                )

                continue

            result["passed_checks"] += 1

            # =====================================================
            # METHOD MODIFIERS
            # =====================================================

            expected_abstract = expected_method.get(
                "abstract",
                False
            )

            expected_static = expected_method.get(
                "static",
                False
            )

            actual_abstract = self._has_decorator(
                actual_method,
                "abstractmethod"
            )

            actual_static = self._has_decorator(
                actual_method,
                "staticmethod"
            )

            # -----------------------------------------------------
            # Abstract method check
            # -----------------------------------------------------

            if expected_abstract:

                result["total_checks"] += 1

                if actual_abstract:
                    result["passed_checks"] += 1

                else:
                    result["violations"].append(
                        f"{class_name}.{method_name} "
                        f"should be abstract"
                    )

            # -----------------------------------------------------
            # Static method check
            # -----------------------------------------------------

            if expected_static:

                result["total_checks"] += 1

                if actual_static:
                    result["passed_checks"] += 1

                else:
                    result["violations"].append(
                        f"{class_name}.{method_name} "
                        f"should be static"
                    )

            # =====================================================
            # PARAMETERS
            # =====================================================

            expected_parameters = expected_method.get(
                "parameters",
                []
            )

            actual_parameters = [
                parameter
                for parameter in actual_method.args.args
                if parameter.arg != "self"
            ]

            # -----------------------------------------------------
            # Parameter count
            # -----------------------------------------------------

            result["total_checks"] += 1

            if len(actual_parameters) != len(expected_parameters):

                result["violations"].append(
                    f"Parameter count mismatch in "
                    f"{class_name}.{method_name}: "
                    f"expected {len(expected_parameters)}, "
                    f"found {len(actual_parameters)}"
                )

            else:

                result["passed_checks"] += 1

                # -------------------------------------------------
                # Individual parameters
                # -------------------------------------------------

                for expected_param, actual_param in zip(
                    expected_parameters,
                    actual_parameters
                ):

                    expected_name = expected_param["name"]

                    # ---------------------------------------------
                    # Parameter name
                    # ---------------------------------------------

                    result["total_checks"] += 1

                    if actual_param.arg == expected_name:

                        result["passed_checks"] += 1

                    else:

                        result["violations"].append(
                            f"Parameter name mismatch in "
                            f"{class_name}.{method_name}: "
                            f"expected '{expected_name}', "
                            f"found '{actual_param.arg}'"
                        )

                    # ---------------------------------------------
                    # Parameter type
                    # ---------------------------------------------

                    expected_type = expected_param.get("type")

                    if expected_type:

                        result["total_checks"] += 1

                        actual_type = self._annotation_to_string(
                            actual_param.annotation
                        )

                        if actual_type == expected_type:

                            result["passed_checks"] += 1

                        else:

                            result["violations"].append(
                                f"Parameter type mismatch in "
                                f"{class_name}.{method_name}"
                                f"({expected_name}): "
                                f"expected '{expected_type}', "
                                f"found '{actual_type}'"
                            )

            # =====================================================
            # RETURN TYPE
            # =====================================================

            expected_return = expected_method.get(
                "return_type"
            )

            if expected_return:

                result["total_checks"] += 1

                actual_return = self._annotation_to_string(
                    actual_method.returns
                )

                if actual_return == expected_return:

                    result["passed_checks"] += 1

                else:

                    result["violations"].append(
                        f"Return type mismatch in "
                        f"{class_name}.{method_name}: "
                        f"expected '{expected_return}', "
                        f"found '{actual_return}'"
                    )

        # =========================================================
        # 5. FINAL METRICS
        # =========================================================

        if result["total_checks"] > 0:

            result["interface_conformance"] = (
                result["passed_checks"]
                / result["total_checks"]
            )

        result["violation_count"] = len(
            result["violations"]
        )

        return result

    # =========================================================
    # COMBINED EVALUATION (Phase 5)
    # =========================================================

    def evaluate_full(
        self,
        code: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run structural evaluation and (optionally) functional evaluation.

        Structural and functional correctness remain SEPARATE metrics.
        The functional score does NOT influence interface_conformance
        or violation_count.

        Args:
            code:    Raw generated code string.
            task_id: If provided, also run FunctionalEvaluator for
                     this task. If None, functional fields are absent.

        Returns:
            dict with:
                structural: full result from evaluate()
                functional: full result from FunctionalEvaluator, or None
        """

        structural = self.evaluate(code)

        functional = None

        if task_id is not None:

            # Import here to avoid circular dependency at module level.
            from src.functional_evaluator import FunctionalEvaluator

            try:
                fe = FunctionalEvaluator(task_id)
                functional = fe.evaluate(code)

            except KeyError as exc:
                functional = {
                    "functional_valid": False,
                    "tests_total": 0,
                    "tests_passed": 0,
                    "functional_score": 0.0,
                    "failures": [str(exc)],
                    "execution_error": f"Unsupported task_id: {task_id}",
                    "timed_out": False,
                    "normalized_code": "",
                }

        return {
            "structural": structural,
            "functional": functional,
        }



    @staticmethod
    def _class_is_abstract(class_node: ast.ClassDef) -> bool:

        # Check for abstract methods
        for node in class_node.body:

            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):

                if CodeEvaluator._has_decorator(
                    node,
                    "abstractmethod"
                ):
                    return True

        # Check whether class inherits from ABC
        for base in class_node.bases:

            if isinstance(base, ast.Name):

                if base.id == "ABC":
                    return True

            elif isinstance(base, ast.Attribute):

                if base.attr == "ABC":
                    return True

        return False

    # =============================================================
    # HELPER: CHECK DECORATORS
    # =============================================================

    @staticmethod
    def _has_decorator(
        node,
        decorator_name: str
    ) -> bool:

        for decorator in node.decorator_list:

            if isinstance(decorator, ast.Name):

                if decorator.id == decorator_name:
                    return True

            elif isinstance(decorator, ast.Attribute):

                if decorator.attr == decorator_name:
                    return True

        return False

    # =============================================================
    # HELPER: CONVERT AST ANNOTATION TO STRING
    # =============================================================

    @staticmethod
    def _annotation_to_string(annotation):

        # No annotation
        if annotation is None:
            return "<missing>"

        # Simple annotation:
        # str, float, bool, Notification, etc.
        if isinstance(annotation, ast.Name):
            return annotation.id

        # Attribute annotation:
        # typing.List, module.Type, etc.
        if isinstance(annotation, ast.Attribute):
            return ast.unparse(annotation)

        # String annotation:
        # "Notification"
        if isinstance(annotation, ast.Constant):

            if isinstance(annotation.value, str):
                return annotation.value

            # Actual Python annotation: -> None
            if annotation.value is None:
                return "None"

        # Generic / complex annotation
        try:
            return ast.unparse(annotation)

        except Exception:
            return "<missing>"