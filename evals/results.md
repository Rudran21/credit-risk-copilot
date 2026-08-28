# Evaluation results

Run 2026-08-23 10:58 UTC on `gemini-3.5-flash-lite` via gemini. 15 golden questions, 120s.

Truth values are computed by calling the engine at eval time, not hardcoded, so the eval cannot drift out of date when a model parameter changes.

## Headline

| metric | result |
|---|---|
| Answer accuracy | 15/15 (100%) |
| Numeric grounding | 14/15 (93%) |
| Tool selection | 15/15 (100%) |
| Figures checked | 108 |
| Ungrounded figures | 1 |

## Per question

| id | type | correct | grounded | figures | tools called |
|---|---|---|---|---|---|
| g01 | numeric | yes | yes | 16 | compute_ecl |
| g02 | numeric | yes | yes | 1 | compute_ecl |
| g03 | numeric | yes | yes | 13 | compute_ecl |
| g04 | numeric | yes | yes | 2 | compute_ecl |
| g05 | numeric | yes | yes | 16 | compute_ecl |
| g06 | numeric | yes | yes | 17 | compute_ecl |
| g07 | numeric | yes | yes | 4 | get_portfolio_slice |
| g08 | numeric | yes | yes | 3 | compare_segments, get_portfolio_slice |
| g09 | numeric | yes | yes | 2 | get_portfolio_slice |
| g10 | numeric | yes | yes | 9 | run_stress |
| g11 | numeric | yes | yes | 8 | compare_segments |
| g12 | numeric | yes | NO | 4 | compare_segments |
| g13 | tool | yes | yes | 13 | get_portfolio_slice, explain_stage_assignment |
| g14 | refusal | yes | yes | 0 | - |
| g15 | refusal | yes | yes | 0 | - |

## Failures

### g12

**Question.** What is the coverage ratio on grade A under the base scenario, and how does it compare to grade G?

**Expected value.** `2.27`

**Ungrounded figure.** `21.9383` in: ... a coverage ratio of 24.2098% [c1]—Grade A's coverage is lower by 21.9383 percentage points [c1], re...

**Answer.**

> The coverage ratio on Grade A loans under the base scenario is 2.2715% [c1]. 

Compared to Grade G—which carries a coverage ratio of 24.2098% [c1]—Grade A's coverage is lower by 21.9383 percentage points [c1], resulting in a coverage ratio of 0.0938x [c1] relative to Grade G.


## What this eval does not test

The grounding check verifies **figures**, not **mechanisms**. During development the agent produced a fully cited answer that described exposure migrating between IFRS 9 stages under stress. Stage assignment in this model is computed once and is scenario-invariant, so no migration occurs. Every number was grounded; the causal claim was fabricated. A numeric guardrail cannot catch that class of error.

The check also allowlists structurally unavoidable integers (stage numbers, month horizons, "IFRS 9"). A fabricated figure equal to one of those would pass.
