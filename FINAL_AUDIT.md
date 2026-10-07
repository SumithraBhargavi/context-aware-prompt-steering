# CAPS Experiment — Final Audit Report

**Date of Audit**: September 20, 2026  
**Project**: Context-Aware Prompt Steering (CAPS) Research Infrastructure  
**Audit Scope**: Dataset integrity, evaluator robustness, statistical validity, figure correctness, and experimental provenance.

---

## 1. Formal Sample Size Verification

- **Claude Sonnet 5**: 10 tasks $\times$ 2 conditions (Baseline vs CAPS) $\times$ 3 runs = **60 formal records**
- **DeepSeek V4.1 Flash**: 10 tasks $\times$ 2 conditions (Baseline vs CAPS) $\times$ 3 runs = **60 formal records**
- **Total Formal Experimental Dataset**: **120 formal records**
- **Record Completeness**: 120 / 120 records (100%) contain complete AST, structural interface, violation count, and functional evaluation data.

---

## 2. Pilot Data & Extra Run Exclusions

- **Claude `factory_01` Extra Runs**: Runs 4 and 5 (`baseline_run_04`, `baseline_run_05`, `caps_run_04`, `caps_run_05`) exist on disk as pilot/extra observations.
- **Audit Result**: Confirmed that `ExperimentRunner.load_all_records(max_runs=3)` strictly enforces `max_runs=3`. Runs 4 and 5 are **100% excluded** from `all_runs.csv`, summary CSVs, `aggregation.json`, and all statistical tests.
- **Raw Evidence Status**: Raw files (`raw_code.txt`, `prompt.txt`) for runs 4 and 5 remain completely untouched on disk.

---

## 3. Evaluator Integrity & Robustness Audit

- **Windows Subprocess Decoding Fix**: `FunctionalEvaluator` now captures raw binary output (`text=False`) from subprocess pipes and decodes via a multi-stage fallback strategy (`UTF-8` $\rightarrow$ `CP1252` $\rightarrow$ `errors="replace"`).
- **Evaluator Exceptions**: **0** exceptions across all 120 formal runs.
- **Subprocess Timeouts**: **0** timeouts.
- **Execution Crashes**: **0** crashes.
- **Structural Independence**: Verified that `CodeEvaluator` performs purely static AST parsing/inspection and is completely independent of `FunctionalEvaluator`. The encoding fix did not alter structural metrics.

---

## 4. Verified Overall Empirical Results

### Summary Table across 120 Formal Generations

| Model | Condition | $N$ | AST Valid | Interface Conformance (Mean) | Interface Conformance (Median) | Mean Violations / Task | Functional Valid Rate | Functional Score (Mean) | Joint Struct✓ / Func✓ Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Claude Sonnet 5** | Baseline | 30 | 100% | 65.50% | 66.26% | 6.93 | 30.00% | 54.88% | 3.33% (1/30) |
| **Claude Sonnet 5** | CAPS | 30 | 100% | **98.89%** | **100.00%** | **0.07** | **90.00%** | **90.00%** | **90.00%** (27/30) |
| **DeepSeek V4.1 Flash** | Baseline | 30 | 100% | 72.24% | 73.07% | 5.87 | 16.67% | 37.26% | 6.67% (2/30) |
| **DeepSeek V4.1 Flash** | CAPS | 30 | 100% | **99.44%** | **100.00%** | **0.03** | **100.00%** | **100.00%** | **100.00%** (30/30) |

---

## 5. Verified Statistical Test Results

- **Inferential Unit**: **Task** ($N = 10$ tasks per model).
- **Replication Unit**: Runs 1–3 aggregated to task-level means/proportions before inferential testing.
- **Model Isolation**: Claude and DeepSeek are analyzed separately to avoid confounding treatment effects with model identity.

### Wilcoxon Signed-Rank Test Results ($N = 10$ Tasks / Model)

#### Claude Sonnet 5
- **Interface Conformance**: $W = 0.0, p = 0.00195$ ($p < 0.002$), Effect Size $r = 1.000$ (**CAPS > Baseline**, $+33.39\%$ mean difference)
- **Violation Reduction**: $W = 0.0, p = 0.00195$ ($p < 0.002$), Effect Size $r = 1.000$ (**Baseline > CAPS**, $-6.87$ mean violations/task reduction)
- **Functional Score**: $W = 0.0, p = 0.00781$ ($p < 0.01$), Effect Size $r = 1.000$ (**CAPS > Baseline**, $+35.12\%$ mean difference)
- **Functional Valid Rate**: $W = 0.0, p = 0.00781$ ($p < 0.01$), Effect Size $r = 1.000$ (**CAPS > Baseline**, $+60.00\%$ point improvement)

#### DeepSeek V4.1 Flash
- **Interface Conformance**: $W = 0.0, p = 0.00195$ ($p < 0.002$), Effect Size $r = 1.000$ (**CAPS > Baseline**, $+27.21\%$ mean difference)
- **Violation Reduction**: $W = 0.0, p = 0.00195$ ($p < 0.002$), Effect Size $r = 1.000$ (**Baseline > CAPS**, $-5.83$ mean violations/task reduction)
- **Functional Score**: $W = 0.0, p = 0.00391$ ($p < 0.004$), Effect Size $r = 1.000$ (**CAPS > Baseline**, $+62.74\%$ mean difference)
- **Functional Valid Rate**: $W = 0.0, p = 0.00391$ ($p < 0.004$), Effect Size $r = 1.000$ (**CAPS > Baseline**, $+83.33\%$ point improvement)

---

## 6. Mathematical Explanation of Rank-Biserial Correlation $r = 1.000$

Audit task #6 requested an explicit investigation into why every reported effect size is $r = 1.000$:

- **Formula**: For the Wilcoxon signed-rank test on paired differences, rank-biserial correlation is computed as $r = 1 - \frac{2W}{N(N+1)}$, where $W$ is the sum of ranks for the smaller group.
- **Task-Level Monotonicity**: 
  - For Interface Conformance: 10 out of 10 tasks improved under CAPS for both Claude ($10/10 > 0$) and DeepSeek ($10/10 > 0$). Zero tasks regressed.
  - For Violation Reduction: 10 out of 10 tasks reduced violations under CAPS for both Claude ($10/10 < 0$) and DeepSeek ($10/10 < 0$). Zero tasks regressed.
  - For Functional Score: Every task with a non-zero difference improved under CAPS (8/8 tasks for Claude, 9/9 tasks for DeepSeek). Zero tasks regressed.
- **Conclusion**: Because **100% of non-zero task differences move strictly in the direction favoring CAPS** with zero reversals, the Wilcoxon test statistic $W$ is mathematically $0.0$, yielding $r = 1.000$. This is **100% empirically accurate and mathematically sound**.

---

## 7. Generated Figure Audit

All 11 figures in [`results/analysis/figures/`](file:///c:/Users/SUMITHRA/Downloads/context-aware-prompt-steering-fixed/context-aware-prompt-steering/results/analysis/figures) were verified:

1. `fig1_overall_conformance.png` — Overall Structural Conformance (Bar chart)
2. `fig2_overall_functional.png` — Overall Functional Correctness Rate (Bar chart)
3. `fig3_violations.png` — Structural Violation Reduction (Bar chart)
4. `fig4_pattern_conformance.png` — Pattern-Level Structural Conformance (Grouped bar)
5. `fig5_pattern_functional.png` — Pattern-Level Functional Correctness (Grouped bar)
6. `fig6_task_conformance.png` — Task-Level Structural Conformance (10 tasks)
7. `fig7_model_conformance.png` — Model-Level Comparison (Claude vs DeepSeek)
8. `fig8_joint_matrix.png` — Structural $\times$ Functional 2$\times$2 Outcome Distribution
9. `fig9_model_pattern_heatmap.png` — Model $\times$ Pattern $\times$ Condition Conformance Heatmap
10. `fig10_task_conformance_heatmap.png` — Task-Level Conformance Heatmap
11. `fig11_ast_validity.png` — AST Parse Validity Comparison

---

## 8. Final Audit Status & Outstanding Issues

- **Raw Evidence Modification**: None. All `raw_code.txt` and `prompt.txt` files are unmodified.
- **Evaluation Pipeline**: Fully functional, error-free, and Windows-compatible.
- **Remaining Issues**: **NONE**. The experimental results, statistical analysis, and figures are 100% verified and ready for paper manuscript inclusion.
