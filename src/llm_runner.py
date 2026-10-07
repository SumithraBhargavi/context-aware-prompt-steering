from typing import Dict, Any

from src.context_engine import ContextEngine
from src.task_contracts import partial_skeleton


class LLMRunner:
    """
    Builds baseline and CAPS-steered prompts for manual LLM evaluation.

    No API is used here. The generated code can be copied manually
    from an LLM and evaluated using CodeEvaluator.
    """

    def __init__(self, task: Dict[str, Any], model: str = ""):

        if task is None:
            raise ValueError("task cannot be None")
        if not isinstance(task, dict):
            raise TypeError("task must be a dictionary")
        if "contract" not in task:
            raise KeyError("task must include a 'contract' entry")

        self.task = task
        self.model = model
        self.contract = task["contract"]

        self.context_engine = ContextEngine(
            self.contract
        )

    # =========================================================
    # BASELINE PROMPT
    # =========================================================

    def _build_common_prompt(self) -> str:
        description = self.task["description"]
        skeleton = partial_skeleton(self.contract)
        return f"""You are an expert Python developer.

Implement the following software design pattern task in Python.

Task:
{description}

{skeleton}

Requirements:
- Return only executable Python source code.
- Do not use Markdown code fences.
- Do not provide explanations outside the code.
- Implement the requested task completely.
- Follow the stated requirements exactly.
- Write valid Python that can be executed without editing.
- Keep the implementation consistent with the requested design pattern.
""".strip()

    def build_baseline_prompt(self) -> str:
        """Build the baseline prompt without structural contract context."""
        return self._build_common_prompt()

    def build_caps_prompt(self) -> str:
        """Build the same prompt as baseline with only the generated structural contract inserted."""
        common = self._build_common_prompt()
        structural_context = self.context_engine.generate_context()
        return (
            common
            + """\n\nIMPORTANT: The structural contract below is the authoritative structural specification for this task.
Before writing the code, use it as an exact checklist. Preserve every required class name, inheritance relationship, method name, parameter name, parameter type, return type, and abstract/static requirement. Do not add, remove, or change parameters of required methods. Keep the generated Python syntactically valid and correctly indented. If the task description is ambiguous about structure, follow the structural contract.

"""
            + structural_context
        )

    def build_steered_prompt(self) -> str:
        """Backward-compatible alias for the CAPS prompt."""
        return self.build_caps_prompt()

    # =========================================================
    # SAVE PROMPTS
    # =========================================================

    def save_prompts(
        self,
        baseline_path: str,
        steered_path: str
    ) -> None:

        baseline_prompt = self.build_baseline_prompt()
        steered_prompt = self.build_steered_prompt()

        with open(
            baseline_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(baseline_prompt)

        with open(
            steered_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(steered_prompt)