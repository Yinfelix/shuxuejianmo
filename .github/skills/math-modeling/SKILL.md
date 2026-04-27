---
name: math-modeling
description: 'School contest mathematical modeling workflow. Use for math modeling, school contest prep, CUMCM-style problem framing, assumption design, model selection, Excel or Python analysis planning, sensitivity analysis, and report outline generation.'
argument-hint: 'Describe the problem type, available data, deadline, and deliverables.'
user-invocable: true
---

# Math Modeling Workflow

## When to Use

- Build a contest plan from a fresh problem statement
- Choose between optimization, prediction, evaluation, simulation, or graph models
- Turn messy data into a notebook, script, and report workflow
- Review whether assumptions, validation, and sensitivity analysis are adequate

## Procedure

1. Reframe the problem.
   Extract the required outputs, the decision variables, the measurable inputs, and the real scoring target.
2. Narrow the model family.
   Use the [model selection guide](./references/model-selection.md) to shortlist one primary model and one fallback model.
3. Plan the implementation path.
   Decide which parts belong in Excel, which belong in Python, and which figures or tables must appear in the paper.
4. Lock the validation plan.
   Include baseline comparison, ablation or parameter variation, and at least one sensitivity or robustness check.
5. Draft the paper in parallel.
   Use the [report outline](./assets/report-outline.md) so the report grows together with the analysis rather than after it.

## Output Checklist

- Clear problem framing
- Primary and fallback model choices
- Data cleaning and feature plan
- Validation and sensitivity plan
- Report section checklist