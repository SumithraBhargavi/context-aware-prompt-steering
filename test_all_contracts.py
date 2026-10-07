import json

from src.context_engine import ContextEngine
from src.evaluator import CodeEvaluator


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    "dataset/test_tasks.json",
    "r",
    encoding="utf-8"
) as f:
    tasks = json.load(f)


print("=" * 70)
print("FULL DATASET CONTRACT VALIDATION")
print("=" * 70)

print(f"\nTotal tasks: {len(tasks)}")


# ============================================================
# VALIDATE EACH CONTRACT
# ============================================================

all_passed = True

for index, task in enumerate(tasks, start=1):

    # Your dataset does not contain an "id" field.
    # Use the known task naming convention from the dataset.
    pattern_index = (index - 1) // 2
    variation = ((index - 1) % 2) + 1

    patterns = [
        "factory",
        "observer",
        "strategy",
        "singleton",
        "adapter"
    ]

    task_id = (
        f"{patterns[pattern_index]}_{variation:02d}"
    )

    contract = task["contract"]

    print("\n" + "-" * 70)
    print(f"Task {index}: {task_id}")
    print("-" * 70)

    try:

        # --------------------------------------------------------
        # Context Engine
        # --------------------------------------------------------

        context_engine = ContextEngine(contract)

        context = context_engine.generate_context()

        if not context.strip():
            raise ValueError(
                "Context Engine produced empty context."
            )

        # --------------------------------------------------------
        # Evaluator
        # --------------------------------------------------------

        evaluator = CodeEvaluator(contract)

        # Minimal syntactically valid Python.
        # We are only testing that the evaluator can process
        # every contract without crashing.
        test_code = "pass\n"

        result = evaluator.evaluate(test_code)

        # --------------------------------------------------------
        # Validate evaluator result structure
        # --------------------------------------------------------

        required_fields = [
            "ast_valid",
            "interface_conformance",
            "total_checks",
            "passed_checks",
            "violation_count",
            "violations"
        ]

        for field in required_fields:

            if field not in result:
                raise ValueError(
                    f"Missing evaluator field: {field}"
                )

        print("Context Engine: PASS")
        print("Evaluator: PASS")
        print(
            f"Structural checks: "
            f"{result['total_checks']}"
        )

        print(
            f"Initial conformance: "
            f"{result['interface_conformance'] * 100:.2f}%"
        )

    except Exception as e:

        all_passed = False

        print("STATUS: FAIL")
        print(f"Error: {e}")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if all_passed:

    print("RESULT: ALL CONTRACTS PASSED")
    print(
        "The dataset is compatible with the "
        "Context Engine and Evaluator."
    )

else:

    print("RESULT: VALIDATION FAILED")
    print(
        "At least one contract requires investigation."
    )

print("=" * 70)