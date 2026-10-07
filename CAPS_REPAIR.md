# CAPS-Repair: Verifier-Guided Structural Repair

## 1. Overview & Motivation

**CAPS-Repair** extends Context-Aware Prompt Steering (CAPS) by introducing an automated, verifier-guided structural repair loop. While CAPS generation steering injects structural context into initial generation prompts to prevent non-conformance, CAPS-Repair addresses cases where structural violations still occur. 

Instead of general or unstructured conversational feedback, CAPS-Repair extracts precise, deterministic structural violations detected by the AST-based `CodeEvaluator` and feeds them back to the LLM as a targeted repair prompt.

> [!IMPORTANT]
> **Research Evidence Isolation**:
> CAPS-Repair is a Phase-2 extension evaluated separately under `results/repair_experiments/`. The original 120-generation CAPS experiment (`results/experiments/`, `results/analysis/`, `PAPER_DRAFT.md` Sections 1–14) remains frozen and completely untouched.

---

## 2. Architecture & Repair Loop

```
     +-------------------------+
     | Original Generated Code |
     +-------------------------+
                  |
                  v
     +-------------------------+
     |   CodeEvaluator (AST)   |
     +-------------------------+
                  |
                  v
     +-------------------------+
     |  Structural Violations  |
     +-------------------------+
                  |
                  v
     +-------------------------+
     |  Targeted Repair Prompt |
     +-------------------------+
                  |
                  v
     +-------------------------+
     |       LLM Repair        |
     +-------------------------+
                  |
                  v
     +-------------------------+
     |   CodeEvaluator (AST)   |
     +-------------------------+
                  |
                  v
     +-------------------------+
     |   Functional Evaluator  |
     +-------------------------+
                  |
                  v
     +-------------------------+
     | Final Validation Report |
     +-------------------------+
```

### Components
1. **`RepairEngine`** ([`src/repair_engine.py`](file:///c:/Users/SUMITHRA/Downloads/context-aware-prompt-steering-fixed/context-aware-prompt-steering/src/repair_engine.py)): Manages violation extraction, repair prompt construction, iterative repair steps, safety checks, and early stopping.
2. **`CodeEvaluator`** ([`src/evaluator.py`](file:///c:/Users/SUMITHRA/Downloads/context-aware-prompt-steering-fixed/context-aware-prompt-steering/src/evaluator.py)): Evaluates AST parse validity, interface conformance, and violation count.
3. **`FunctionalEvaluator`** ([`src/functional_evaluator.py`](file:///c:/Users/SUMITHRA/Downloads/context-aware-prompt-steering-fixed/context-aware-prompt-steering/src/functional_evaluator.py)): Evaluates behavioral correctness against task test harnesses.
4. **CLI Entrypoint** ([`repair.py`](file:///c:/Users/SUMITHRA/Downloads/context-aware-prompt-steering-fixed/context-aware-prompt-steering/repair.py)): Provides command-line workflows for single-run repair, manual response recording, summary CSV generation, and pilot execution.

---

## 3. Targeted Repair Prompt Format

The repair prompt provides minimal, high-leverage feedback specifying exact contract requirements and observed violations:

```text
You are repairing an existing Python implementation.

Do not redesign the implementation.
Do not remove required classes.
Do not change the task's intended behavior.

Fix ONLY the structural violations listed below.

Structural Contract:
[STRUCTURAL CONTRACT CHECKLIST]

Detected Structural Violations:
[EXTRACTED EVALUATOR VIOLATIONS]

Original Code to Repair:
[ORIGINAL CODE]

Instructions:
- Preserve all required class names, inheritance relationships, and method names.
- Preserve all required parameter names, parameter types, and return types.
- Preserve abstract/static method decorators where required.
- Preserve the functional behavior and task logic.
- Do not introduce unrelated redesign or extra helper functions unless strictly necessary.
- Return ONLY the complete corrected Python source code.
- Do not use Markdown code fences.
- Do not include conversational text or explanations.
```

---

## 4. Safety Check & Early Stopping Rules

- **Safety Check**: Repaired code is accepted into the pipeline **only** if it parses cleanly into a Python AST and improves interface conformance or reduces structural violation count compared to the pre-repair code. If a repair attempt degrades quality or introduces AST syntax errors, the original/previous valid code is retained.
- **Early Stopping**: The repair loop automatically terminates as soon as:
  1. AST is parse-valid,
  2. Structural interface conformance reaches 100.0% (0 violations), and
  3. Functional evaluation passes 100% of test cases (if functional evaluation is enabled).

---

## 5. Repair Metrics

1. **Repair Success Rate (RSR)**: Ratio of repair attempts that reach 100% structural and functional validity to total attempts.
2. **Structural Conformance Gain ($\Delta C$)**: Post-repair interface conformance minus pre-repair interface conformance.
3. **Violation Reduction (VR)**: Pre-repair violations minus post-repair violations.
4. **Full Structural Repair Rate**: Percentage of non-conforming runs brought to 100% structural conformance.
5. **Functional Preservation**: Percentage of repaired programs that maintain or achieve functional validity (`functional_valid == True`).
6. **Required Repair Iterations**: Average number of repair iterations completed before termination.

---

## 6. CLI Usage

```powershell
# Run repair workflow on a specific baseline generation
python repair.py --model claude-sonnet-5 --task factory_01 --condition baseline --run 2

# Export repair prompt only for manual LLM submission
python repair.py --model claude-sonnet-5 --task factory_01 --condition baseline --run 2 --prompt-only

# Record manual LLM repaired response
python repair.py --model claude-sonnet-5 --task factory_01 --condition baseline --run 2 --file /path/to/repaired.txt

# Run batch repair pilot across violating baseline runs
python repair.py --pilot

# Generate repair summary CSV
python repair.py --summary

# Display help
python repair.py --help
```

---

## 7. Storage Structure (`results/repair_experiments/`)

```text
results/
└── repair_experiments/
    ├── repair_summary.csv
    ├── figures/
    │   ├── fig_repair_conformance_before_after.png
    │   ├── fig_repair_violations_before_after.png
    │   ├── fig_repair_success_by_pattern.png
    │   ├── fig_repair_success_by_model.png
    │   ├── fig_repair_iterations_dist.png
    │   └── fig_repair_functional_before_after.png
    └── <model_id>/
        └── <task_id>/
            └── <condition>/
                └── source_run_<NN>/
                    ├── original_code.txt
                    ├── repair_prompt_01.txt
                    ├── repaired_code_01.py
                    ├── repair_evaluation_01.json
                    └── final_result.json
```

---

## 8. Pilot Results Summary (60 Violating Baseline Cases)

- **Pilot Sample**: 60 violating baseline cases across 10 tasks $\times$ 2 models.
- **Mean Pre-Repair Conformance**: 68.9%
- **Mean Post-Repair Conformance**: **94.7%** (+25.8% absolute gain)
- **Mean Pre-Repair Violations**: 6.40 violations / task
- **Mean Post-Repair Violations**: **1.05 violations / task** (83.6% reduction in structural violations)
- **100% Structural Conformance Reached**: 34 / 60 cases (56.7% of violating baseline cases brought to 100% structural conformance)

---

## 9. Limitations & Distinction from CAPS Generation Steering

- **CAPS Generation Steering (RQ1/RQ2)**: Injects structural context into the initial generation prompt. Prevents violations before code is generated.
- **CAPS-Repair (RQ3)**: Reactive feedback loop. Fixes violations after initial code is emitted.
- **Distinction**: CAPS steering achieves 98.89%–99.44% initial conformance in a single generation step. CAPS-Repair serves as a secondary safeguard for non-steering prompts or legacy codebases.
