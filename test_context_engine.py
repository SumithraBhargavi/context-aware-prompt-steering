import json

from src.context_engine import ContextEngine


# ---------------------------------------------------------
# Load benchmark dataset
# ---------------------------------------------------------

with open(
    "dataset/test_tasks.json",
    "r",
    encoding="utf-8"
) as f:

    tasks = json.load(f)


# ---------------------------------------------------------
# Test first task
# ---------------------------------------------------------

task = tasks[0]

contract = task["contract"]

engine = ContextEngine(contract)

context = engine.generate_context()


# ---------------------------------------------------------
# Display generated context
# ---------------------------------------------------------

print("=" * 60)
print("CONTEXT ENGINE TEST")
print("=" * 60)

print("\nTask ID:")
print(task["task_id"])

print("\nGenerated Structural Context:")
print("-" * 60)

print(context)

print("-" * 60)

print("\nContext generated successfully.")