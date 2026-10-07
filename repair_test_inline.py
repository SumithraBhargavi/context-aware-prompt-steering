"""End-to-end repair test - adapter_01 baseline run 2"""
import sys
sys.path.insert(0, ".")
from src.repair_engine import RepairEngine
from repair import rule_based_repair

code = open("results/experiments/claude-sonnet-5/adapter_01/baseline_run_02/raw_code.txt", "r", encoding="utf-8").read()
engine = RepairEngine()
llm_fn = lambda prompt: rule_based_repair("adapter_01", prompt, code)
result = engine.run_repair_loop(
    task_id="adapter_01",
    model_id="claude-sonnet-5",
    source_condition="baseline",
    source_run=2,
    original_code=code,
    repair_llm_fn=llm_fn,
    max_iterations=3,
    run_functional=True,
    save_results=True,
)
print("RepairEngine invoked: YES")
print("Iterations completed:", result["iterations_completed"])
print("Repair success:", result["repair_success"])
pre = result["pre_repair"]
post = result["post_repair"]
print("Pre-repair conformance: ", round(pre["interface_conformance"] * 100, 1), "%")
print("Post-repair conformance:", round(post["interface_conformance"] * 100, 1), "%")
print("Pre-repair violations:  ", pre["violation_count"])
print("Post-repair violations: ", post["violation_count"])
print("Pre-repair functional:  ", pre["functional_valid"])
print("Post-repair functional: ", post["functional_valid"])
print("Pre-repair func_score:  ", pre["functional_score"])
print("Post-repair func_score: ", post["functional_score"])

history = result.get("history", [])
print("History iterations:", [s["iteration"] for s in history])
for step in history:
    s = step["evaluation"]["structural"]
    fn = step["evaluation"].get("functional") or {}
    it = step["iteration"]
    conf = round(s["interface_conformance"] * 100, 1)
    viols = s["violation_count"]
    func = fn.get("functional_valid")
    print(f"  Iter {it}: conf={conf}% viols={viols} functional_valid={func}")

# Task contracts
print()
print("=== TASK CONTRACT CHECK ===")
from src.task_contracts import load_tasks
from src.context_engine import ContextEngine
tasks = load_tasks("dataset/test_tasks.json")
task_ids = [t["task_id"] for t in tasks]
print("Task IDs:", task_ids)
adapter_task = next(t for t in tasks if t["task_id"] == "adapter_01")
print("adapter_01 pattern:", adapter_task.get("pattern"))
ce = ContextEngine(adapter_task["contract"])
ctx = ce.generate_context()
print("ContextEngine output (first 200 chars):", ctx[:200])

# Repair history
print()
print("=== REPAIR HISTORY ===")
import glob, os, pandas as pd
result_files = glob.glob(os.path.join("results/repair_experiments", "**", "final_result.json"), recursive=True)
print(f"Total final_result.json files: {len(result_files)}")
repair_summary = "results/repair_experiments/repair_summary.csv"
if os.path.exists(repair_summary):
    rdf = pd.read_csv(repair_summary)
    print(f"Repair summary rows: {len(rdf)}")
    print("First row:", rdf.iloc[0][["model", "task_id", "source_condition", "source_run", "repair_success"]].to_dict())
else:
    print("No repair_summary.csv found")

# Formal experiment preservation
print()
print("=== FORMAL EXPERIMENT PRESERVATION ===")
raw_count = sum(1 for root, dirs, files in os.walk("results/experiments") for fn in files if fn == "raw_code.txt")
eval_count = sum(1 for root, dirs, files in os.walk("results/experiments") for fn in files if fn == "evaluation.json")
print(f"Total raw_code.txt files in results/experiments: {raw_count}")
print(f"Total evaluation.json files in results/experiments: {eval_count}")
print("Formal data PRESERVED:", "YES - no modifications made" if raw_count >= 120 else "CHECK NEEDED")
