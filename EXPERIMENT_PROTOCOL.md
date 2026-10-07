# CAPS Formal Experiment Protocol

## Purpose

This protocol freezes the implementation and evaluation procedure before formal data collection for the paper **Context-Aware Prompt Steering for Improving Structural Conformance in LLM-Generated Code**.

## Conditions

### Baseline
The baseline prompt contains the task description, common implementation
instructions, and the same automatically generated partial class skeleton used
by CAPS. It does not contain the structural contract checklist.

### CAPS
The CAPS prompt is identical to the baseline prompt except that the
Context Engine-generated structural contract is inserted as an explicit
checklist. Prompt files are generated separately for the two conditions.

This isolates structural-contract context as the experimental intervention
while keeping the partial skeleton constant. The Factory contract explicitly
requires `create_notification` and `create_document` to be static methods.

## Dataset

Ten tasks are used: two each for Factory, Observer, Strategy, Singleton, and Adapter patterns.

## Generation & Data Capture

For each task and condition:

- Use the same model for paired baseline/CAPS generations.
- Use a fresh conversation for each generation.
- Do not show one condition's output to the other generation.
- Do not repair, reformat, or manually correct generated code before evaluation.
- Preserve the raw response exactly without inserting or removing indentation, repairing syntax, or rewriting code.
- Markdown code fences are removed only for normalized execution/parsing copies where necessary; raw output files (`raw_code.txt`) retain exact captured response content.
- Record the visible model identifier/version when available; never fabricate unavailable settings.

### Research Reproducibility Statement
"Raw model outputs were preserved without repair. Markdown fences were removed only for parsing/execution where necessary. Structural and functional evaluation was performed on the resulting program without manually correcting generated code."

## Repeated runs

Five independent runs per task and condition are recommended for the formal study.

With one model this gives 10 tasks x 2 conditions x 5 runs = 100 generated programs. With two models it gives 200.

## Evaluation

Structural evaluation uses the AST-based CodeEvaluator and checks required
classes, inheritance, class kind, methods, abstract/static declarations when
required, parameter names/types, and return annotations. The local benchmark
loader validates all ten task contracts before prompts or evaluations are
created.

Functional evaluation executes deterministic task-specific harnesses in a subprocess with a timeout. Only markdown code-fence stripping is permitted for execution; raw output is never changed.

## Primary metrics

- **AST Parse Rate:** proportion of generated programs that parse successfully.
- **Interface Conformance:** passed structural checks divided by total structural checks for each parse-valid generation; parse-invalid generations are assigned 0 for the stored overall score and are also reported separately through AST Parse Rate.
- **Violation Count:** number of structural contract violations.

## Secondary metric

- **Functional Correctness:** test-pass rate and all-tests-passed indicator from the task-specific harness.

## Statistical analysis

Repeated runs are retained as replications. For inferential comparisons, run-level observations are first summarized within each `(model, task, condition)` and the resulting baseline/CAPS task summaries are paired by task. Wilcoxon signed-rank testing is then applied to the paired task summaries. This avoids treating repeated generations of the same benchmark task as fully independent tasks.

Models are analyzed separately. Any pooled model result is descriptive only.

## Pilot data

The original Factory 01 Run 01 baseline/CAPS pair was collected before this protocol was frozen. It is retained as pilot data and excluded from formal statistical aggregation.
