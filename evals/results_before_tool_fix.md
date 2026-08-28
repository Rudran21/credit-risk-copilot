# Evaluation results

Run 2026-08-23 09:06 UTC on `gemini-3.5-flash-lite` via gemini. 15 golden questions, 57s.

Truth values are computed by calling the engine at eval time, not hardcoded, so the eval cannot drift out of date when a model parameter changes.

## Headline

| metric | result |
|---|---|
| Answer accuracy | 14/15 (93%) |
| Numeric grounding | 13/15 (87%) |
| Tool selection | 14/15 (93%) |
| Figures checked | 162 |
| Ungrounded figures | 2 |

## Per question

| id | type | correct | grounded | figures | tools called |
|---|---|---|---|---|---|
| g01 | numeric | yes | yes | 16 | compute_ecl |
| g02 | numeric | yes | yes | 9 | get_portfolio_slice |
| g03 | numeric | yes | NO | 8 | compute_ecl |
| g04 | numeric | yes | yes | 2 | compute_ecl |
| g05 | numeric | yes | yes | 16 | compute_ecl |
| g06 | numeric | yes | yes | 16 | compute_ecl |
| g07 | numeric | yes | yes | 9 | get_portfolio_slice |
| g08 | numeric | yes | yes | 1 | get_portfolio_slice |
| g09 | numeric | yes | yes | 2 | get_portfolio_slice |
| g10 | numeric | yes | yes | 10 | run_stress |
| g11 | numeric | yes | yes | 60 | compute_ecl, compute_ecl, compute_ecl, compute_ecl, compute_ecl, compute_ecl, compute_ecl |
| g12 | numeric | yes | NO | 10 | compute_ecl, compute_ecl |
| g13 | tool | NO | yes | 3 | get_portfolio_slice, get_portfolio_slice |
| g14 | refusal | yes | yes | 0 | - |
| g15 | refusal | yes | yes | 0 | - |

## Failures

### g03

**Question.** How much of the base-scenario provision sits in Stage 3?

**Expected value.** `224,569,696.88`

**Ungrounded figure.** `29.11%` in: ...rio provision is $771,423,886.04 [c1], meaning Stage 3 represents 29.11% of the total provision....

**Answer.**

> Stage 3 accounts for $224,569,696.88 of the base-scenario provision, representing 21,507 loans with an exposure of $244,629,299.43 and a coverage ratio of 91.8% [c1]. Total base-scenario provision is $771,423,886.04 [c1], meaning Stage 3 represents 29.11% of the total provision.

### g12

**Question.** What is the coverage ratio on grade A under the base scenario, and how does it compare to grade G?

**Expected value.** `2.27`

**Ungrounded figure.** `10.7` in: ...9.7m ECL on $40.2m EAD) [c2]. Grade G's coverage is approximately 10.7 times higher than Grade A's, ...

**Answer.**

> The coverage ratio on Grade A under the base scenario is 2.27% ($45.5m ECL on $2,002.7m EAD) [c1]. For Grade G, the coverage ratio is 24.21% ($9.7m ECL on $40.2m EAD) [c2]. Grade G's coverage is approximately 10.7 times higher than Grade A's, driven by heavier concentrations and higher default probabilities in stages 2 and 3.

Note that LGD sits close to its ceiling (0.918) across both grades [c1, c2], meaning the variance in coverage is driven almost entirely by differences in PD rather than loss severity.

### g13

**Question.** Explain why the loan with the largest Stage 2 exposure sits in Stage 2.

**Answer.**

> The portfolio slice tool gives aggregate exposure by stage ($1,498.4m in Stage 2 [c1]), but the engine does not expose individual loan exposures or identify which specific loan carries the largest exposure within a stage. Consequently, I cannot query the single largest loan directly. 

If you provide a specific loan ID, I can call `explain_stage_assignment` to show why that loan sits in Stage 2.


## What this eval does not test

The grounding check verifies **figures**, not **mechanisms**. During development the agent produced a fully cited answer that described exposure migrating between IFRS 9 stages under stress. Stage assignment in this model is computed once and is scenario-invariant, so no migration occurs. Every number was grounded; the causal claim was fabricated. A numeric guardrail cannot catch that class of error.

The check also allowlists structurally unavoidable integers (stage numbers, month horizons, "IFRS 9"). A fabricated figure equal to one of those would pass.
