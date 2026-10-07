# PAPER_DRAFT.md Changelog

**Date of Update**: September 20, 2026  
**Target File**: [`PAPER_DRAFT.md`](file:///c:/Users/SUMITHRA/Downloads/context-aware-prompt-steering-fixed/context-aware-prompt-steering/PAPER_DRAFT.md)  
**Data Sources**:
- `results/analysis/overall_summary.csv`
- `results/analysis/model_summary.csv`
- `results/analysis/pattern_summary.csv`
- `results/analysis/task_summary.csv`
- `results/analysis/statistical_tests.csv`
- `results/analysis/all_runs.csv`
- `FINAL_AUDIT.md`
- `EXPERIMENT_PROTOCOL.md`
- `dataset/test_tasks.json`

---

## Sections Updated

1. **Title & Abstract**:
   - Removed placeholder status header.
   - Replaced future/planned phrasing with completed-study findings across 120 formal generations (Claude Sonnet 5 + DeepSeek V4.1 Flash).
   - Reported exact interface conformance, violation count, and functional valid rate numbers.

2. **1. Introduction & 2. Background**:
   - Updated language to reflect completed empirical evaluation while retaining RQ1, RQ2, and the four contributions.

3. **3. Proposed Approach & 4. Automated Evaluation**:
   - Ensured accurate description matching actual implementation: machine-readable contract, Context Engine checklist, baseline vs CAPS common prompt, AST parsing, and deterministic black-box test harness subprocess execution.

4. **5. Experimental Design & 6. Metrics**:
   - Replaced planned 5-run design with actual formal design: 10 tasks, 5 patterns, 2 conditions, 3 runs, 2 models (120 formal generations total).
   - Documented exclusion of Claude `factory_01` pilot runs 4–5.
   - Clarified that repeated runs are replications aggregated within task cells ($N = 10$ tasks per model as inferential unit).

5. **7. Statistical Analysis**:
   - Described Wilcoxon signed-rank test on task-aggregated paired means/proportions ($N=10$ tasks/model) and rank-biserial correlation $r$.

6. **8. Results**:
   - Completely replaced TBD placeholders with verified empirical data tables:
     - **8.1 Overall Results**: Table for Claude Sonnet 5 and DeepSeek V4.1 Flash across AST Valid, Interface Conformance, Violation Count, Functional Score, Functional Valid Rate, Joint Struct✓/Func✓ Rate.
     - **8.2 Pattern-Level Results**: Table for Factory, Observer, Strategy, Singleton, Adapter per model/condition.
     - **8.3 Task-Level Results**: Full table for all 10 tasks across both models.
     - **8.4 Statistical Results**: Table for Wilcoxon test statistics $W$, $p$-values, effect sizes $r$, and mean differences.
     - **Rank-Biserial Explanation**: Detailed mathematical explanation of $r = 1.000$ due to 100% monotonic task-level improvement with 0 task regressions.

7. **9. Error Analysis**:
   - Grouped 384 baseline violations vs 3 CAPS violations into 4 empirical categories (type annotation omissions, abstract modifier omissions, parameter count/name mismatches, missing classes/methods).

8. **10. Discussion**:
   - Evaluated support for RQ1 and RQ2.
   - Discussed relationship between structural conformance and functional test execution.
   - Analyzed remaining behavioral logic failures (e.g., `observer_02`).

9. **11. Threats to Validity & 12. Conclusion**:
   - Updated model, task, evaluator, functional, and statistical threats to validity.
   - Formulated empirical conclusions based on completed 120-run experiment.

10. **13. Figures**:
    - Added a dedicated figures section listing and describing `fig1`, `fig2`, `fig3`, `fig9`, `fig10`, and `fig11` with publication-style captions.
