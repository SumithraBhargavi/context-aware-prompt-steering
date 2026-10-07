import os


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

tasks = [
    "factory_01",
    "factory_02",
    "observer_01",
    "observer_02",
    "strategy_01",
    "strategy_02",
    "singleton_01",
    "singleton_02",
    "adapter_01",
    "adapter_02"
]

runs_per_condition = 5


# ============================================================
# CREATE EXPERIMENT DIRECTORIES
# ============================================================

base_directory = "results/experiments"

os.makedirs(base_directory, exist_ok=True)


for task in tasks:

    task_directory = os.path.join(
        base_directory,
        task
    )

    os.makedirs(
        task_directory,
        exist_ok=True
    )


# ============================================================
# DISPLAY STRUCTURE
# ============================================================

print("=" * 70)
print("EXPERIMENT DIRECTORY SETUP")
print("=" * 70)

print()

for task in tasks:

    print(f"{task}/")

    for run in range(1, runs_per_condition + 1):

        print(f"    baseline_run_{run}.py")
        print(f"    caps_run_{run}.py")

print()

print("=" * 70)
print("SETUP COMPLETE")
print("=" * 70)

print()
print(f"Tasks: {len(tasks)}")
print(f"Runs per condition: {runs_per_condition}")
print("Conditions: Baseline + CAPS")
print(
    f"Total expected programs: "
    f"{len(tasks) * runs_per_condition * 2}"
)

print()
print("Directory:")
print(base_directory)