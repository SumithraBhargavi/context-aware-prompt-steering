import json

from src.llm_runner import LLMRunner


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

with open(
    "dataset/test_tasks.json",
    "r",
    encoding="utf-8"
) as f:

    tasks = json.load(f)


# ---------------------------------------------------------
# Select first task
# ---------------------------------------------------------

task = tasks[0]

runner = LLMRunner(task)


# ---------------------------------------------------------
# Generate prompts
# ---------------------------------------------------------

baseline_prompt = runner.build_baseline_prompt()

steered_prompt = runner.build_steered_prompt()


# ---------------------------------------------------------
# Display baseline prompt
# ---------------------------------------------------------

print("=" * 70)
print("BASELINE PROMPT")
print("=" * 70)

print(baseline_prompt)


# ---------------------------------------------------------
# Display CAPS prompt
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CAPS-STEERED PROMPT")
print("=" * 70)

print(steered_prompt)


# ---------------------------------------------------------
# Basic verification
# ---------------------------------------------------------

assert len(baseline_prompt) > 0
assert len(steered_prompt) > 0

assert "STRUCTURAL CONTRACT" in steered_prompt

print("\n" + "=" * 70)
print("LLM RUNNER TEST PASSED")
print("=" * 70)