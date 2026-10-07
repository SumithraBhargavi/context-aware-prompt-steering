from typing import Dict, Any


class ContextEngine:
    """
    Generates compact structural constraints from a design-pattern
    contract for Context-Aware Prompt Steering (CAPS).

    The generated context explicitly represents:
        - Required classes
        - Class kind
        - Inheritance
        - Required methods
        - Parameter names and types
        - Return types
        - Abstract methods
        - Static methods
        - Structural requirements
    """

    def __init__(self, contract: Dict[str, Any]):
        if contract is None:
            contract = {}
        if not isinstance(contract, dict):
            raise TypeError("contract must be a dictionary mapping to class and method metadata")
        self.contract = contract

    # =========================================================
    # MAIN METHOD
    # =========================================================

    def generate_context(self) -> str:
        """
        Generate a compact structural contract that faithfully reflects the
        task contract without inventing extra requirements.
        """
        sections = ["STRUCTURAL CONTRACT"]

        pattern = self.contract.get("pattern")
        if pattern:
            sections.append(f"Pattern: {pattern}")

        classes = self.contract.get("classes") or []
        if classes:
            sections.append("Classes:")
            for class_info in classes:
                if not isinstance(class_info, dict):
                    continue

                class_name = class_info.get("name")
                if not class_name:
                    continue

                kind = class_info.get("kind", "concrete")
                inherits = class_info.get("inherits") or []
                sections.append(f"- {class_name}")
                sections.append(f"  - kind: {kind}")
                sections.append(
                    "  - inherits: " + (", ".join(inherits) if inherits else "none")
                )

        methods = self.contract.get("methods") or []
        if methods:
            sections.append("Methods:")
            for method in methods:
                if not isinstance(method, dict):
                    continue

                class_name = method.get("class")
                method_name = method.get("name")
                if not class_name or not method_name:
                    continue

                parameters = method.get("parameters") or []
                param_parts = []
                for parameter in parameters:
                    if not isinstance(parameter, dict):
                        continue
                    name = parameter.get("name")
                    if not name:
                        continue
                    parameter_type = parameter.get("type")
                    param_parts.append(
                        f"{name}: {parameter_type}" if parameter_type else name
                    )

                signature = (
                    f"- {class_name}.{method_name}({', '.join(param_parts) if param_parts else ''})"
                    f" -> {method.get('return_type', 'None')}"
                )
                sections.append(signature)
                sections.append(
                    "  - abstract: " + ("yes" if method.get("abstract", False) else "no")
                )
                sections.append(
                    "  - static: " + ("yes" if method.get("static", False) else "no")
                )

        return "\n".join(sections)