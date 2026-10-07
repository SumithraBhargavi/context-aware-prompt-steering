from src.llm_runner import LLMRunner
from src.evaluator import CodeEvaluator


# =========================================================
# LOAD DATASET
# =========================================================

from src.task_contracts import load_tasks

tasks = load_tasks("dataset/test_tasks.json")


print("=" * 70)
print("CONTEXT-AWARE PROMPT STEERING")
print("EXPERIMENT RUNNER")
print("=" * 70)

print(f"\nTotal tasks loaded: {len(tasks)}")


# =========================================================
# SELECT TASK
# =========================================================

task = tasks[0]

task_id = task["task_id"]

print(f"\nSelected task: {task_id}")


# =========================================================
# INITIALIZE COMPONENTS
# =========================================================

runner = LLMRunner(task)

evaluator = CodeEvaluator(
    task["contract"]
)


# =========================================================
# BUILD PROMPTS
# =========================================================

baseline_prompt = runner.build_baseline_prompt()

steered_prompt = runner.build_steered_prompt()


# =========================================================
# DISPLAY EXPERIMENT OPTIONS
# =========================================================

print("\n" + "=" * 70)
print("EXPERIMENT PROMPTS")
print("=" * 70)

print("\nTwo experimental conditions have been prepared:")

print("\n[1] BASELINE")
print("Task description only")

print("\n[2] CAPS-STEERED")
print("Task description + structural contract")


# =========================================================
# SAVE PROMPTS
# =========================================================

baseline_file = (
    f"results/{task_id}_baseline_prompt.txt"
)

steered_file = (
    f"results/{task_id}_steered_prompt.txt"
)


runner.save_prompts(
    baseline_file,
    steered_file
)


print("\nPrompts saved successfully:")

print(f"- {baseline_file}")
print(f"- {steered_file}")


# =========================================================
# DISPLAY NEXT STEP
# =========================================================

print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)

print("""
Copy the BASELINE prompt into an LLM and generate Python code.

Then copy the CAPS-STEERED prompt into the same LLM
and generate Python code.

We will evaluate both generated outputs using the
same CodeEvaluator.
""")

print("=" * 70)