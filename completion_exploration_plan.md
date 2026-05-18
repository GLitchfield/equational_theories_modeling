# Completion as a Unified Decision Procedure — Exploration Plan

## Goal

Extend the rewrite engine with Knuth-Bendix completion, turning it from a one-sided proof-search tool into a unified TRUE/FALSE decision procedure. The headline question: **how much of the entire 22M-pair implication problem — both directions — does pure symbolic rewriting decide?**

This is an exploration, not a single build. It proceeds in phases, each producing a decisive measurement. Early phases stand on their own; later phases extend coverage.

## Methodology resolutions (locked in 2026-05-17)

After scrutiny against the rewrite-chain study findings, these were
resolved before implementation. The plan body below has been updated
to reflect them; this section is the canonical record.

1. **Phase 0 added: variable-inclusion pre-check.** A rule
   `lhs → rhs` is only valid if `Vars(rhs) ⊆ Vars(lhs)` — otherwise
   applying it introduces fresh variables. This is a stricter and
   more common orientability blocker than the commutativity-like
   case the plan originally emphasized. Phase 0 classifies every
   ETP equation by how many directions (0, 1, 2) satisfy variable
   inclusion alone, before any reduction order is involved. This
   sets the ceiling for plain KB completion before a line of it is
   written, and identifies a structural hard core that **ordered
   completion does not rescue either**: when neither side's
   variables contain the other's (disjoint or incomparable variable
   sets, like `x ◇ y = z ◇ w`), ordered rewriting can't be applied
   in either direction — σ doesn't define the fresh-on-rhs
   variables, so the target term can't even be formed. These
   variable-disjoint equations are structurally outside all
   rewriting-based methods.
2. **Phase 3 pair-level coverage is the headline.** Phase 2's
   equation-level completion rate is the input; the
   decision-driving number is "what fraction of the 22M pairs does
   completion decide, TRUE and FALSE." Dense-implication equations
   that complete are worth more than sparse-implication ones.
3. **Reduction order: LPO** (lexicographic path order) with a fixed
   variable precedence. Simpler than KBO, no weight function, well-
   behaved for a single binary operator. Revisit only if Phase 2's
   orientation rate is disappointing.
4. **Completion budget: three independent per-equation caps** —
   max critical pairs processed, max system size, wall-clock
   seconds. Hitting any one classifies as "did not complete within
   budget." Per-equation, not a global pool.
5. **Soften the Vampire connection.** Ordered completion *shares
   the ordered-equational core with superposition*; it does not
   reach the Vampire-proof category as a whole. Superposition adds
   inference rules, selection functions, and redundancy elimination
   on top of ordered rewriting.
6. **Phase 5 integrated from the start, but as exploration not as a
   competition deliverable.** Phases 1-3 track ground normal forms
   and finite-model hooks as completion runs; Phase 5 reads from
   that data rather than re-running. Question: when completion
   decides E1 ⊭ E2, does the completed system yield a finite
   counter-model directly, or only an infinite one?
7. **Cross-method consistency: record it.** If Phase 0 shows that
   the variable-disjoint equations overlap heavily with the
   family_c "weak" laws (the TRUE-by-vacuity failure mode for the
   JEPA distance metric — see `outputs/analyses/failure_overlap/`),
   that is a finding in its own right: the same equations resist
   contrastive-embedding training AND rewriting-based decision
   procedures. Worth flagging in the Phase 0 output and again in
   the synthesis report.

The two numbers the exploration's bottom line turns on:
**Phase 3 pair-level coverage** (TRUE+FALSE decided by plain
completion) and **Phase 4 pair-level coverage** (with ordered
completion added). Phase 0's variable-inclusion fraction puts an
upper bound on what plain-KB Phase 3 can achieve.

## The idea

For a single magma law E1, completion attempts to produce a confluent, terminating term rewriting system R with the same equational theory as E1. If completion succeeds, R is a *decision procedure* for everything E1 implies:

- Normalize both sides of E2 under R.
- Equal normal forms → E1 implies E2 (TRUE).
- Different normal forms → E1 does not imply E2 (FALSE), and the differing normal forms certify a counterexample.

This restructures the problem. The ETP did 22M pairwise implication checks. Completion changes the unit of work: ~4,694 completions (one per equation), after which each completed equation's system decides its entire row — all implications from that equation — by fast normalization. And it decides FALSE, which the current engine cannot do at all.

Completion has real failure modes:

- **Variable-inclusion failure** (Phase 0): a rule `lhs → rhs` is only valid if `Vars(rhs) ⊆ Vars(lhs)`. ETP has many equations like `x = x ◇ y` (lhs vars are a *subset* of rhs vars) — orientable only as `x ◇ y → x`. And some like `x ◇ y = z ◇ w` (variable-disjoint or incomparable) — orientable in *neither* direction, structurally outside rewriting-based methods including ordered completion.
- **Non-orientability under the reduction order** (Phase 2): variable-inclusion holds in some direction, but neither side is greater than the other under the chosen order — commutativity is the canonical case.
- **Looping**: completion generates rules without end, even when the equation is in principle completable.

So the completion rate across the 4,694 equations is an open empirical question, not an assumption. Ordered (unfailing) completion is the fallback for the non-orientability case (Phase 4); it shares the ordered-equational core with superposition (Vampire) but does not itself reach the Vampire-proof category — superposition adds inference rules, selection functions, and redundancy elimination on top. The variable-disjoint case is outside both plain and ordered completion.

Embeddings play no role in this exploration. It is pure symbolic computation. They may return later (predicting completion success, guiding critical-pair selection) but are out of scope here.

## Phases

### Phase 0: Variable-inclusion pre-check

Before implementing anything, scan every ETP equation `lhs = rhs` and classify by how many of the two directions satisfy `Vars(rhs') ⊆ Vars(lhs')`:

- **Both directions valid** (`Vars(lhs) = Vars(rhs)`): plain KB completion can orient based on the reduction order; both options open.
- **Exactly one direction valid** (variables strictly contained one-way): orientation is forced to that direction; the reduction order's job is just to confirm.
- **Zero directions valid** (`Vars(lhs)` and `Vars(rhs)` are incomparable or disjoint): no rule. Even ordered completion can't rescue these — σ doesn't define the fresh-on-rhs variables, so the target term can't be formed. This is a structural hard core for rewriting-based methods.

Output:
- The three-way breakdown across 4,694 equations.
- Per-equation classification (parquet or JSON, indexed by equation ID), so Phases 2-6 can read it.
- **Cross-method consistency check**: does the variable-disjoint set overlap with the family_c "weak" laws — the TRUE-by-vacuity cluster identified in `outputs/analyses/failure_overlap/`? If yes, that's a finding in its own right: the same equations resist contrastive-embedding training and rewriting-based decision procedures.

This pre-check sets the ceiling for plain KB completion (Phases 1-3) and identifies the genuine residual that even ordered completion (Phase 4) won't reach.

### Phase 1: Implement basic Knuth-Bendix completion

Implement standard KB completion on top of the existing rewrite engine (`src/rewrite/`), which already has first-order matching, rewriting, and term utilities.

Components needed:
- A reduction order on magma terms. Start with the Knuth-Bendix order (KBO) or lexicographic path order (LPO) — pick one, justify the choice. For a single binary operator the order is mostly about term structure.
- Orientation: turn an equation into a rule using the reduction order (larger side rewrites to smaller).
- Critical-pair computation: find overlaps between rules (and a rule with itself) where two rewrites could diverge.
- The completion loop: orient, compute critical pairs, check joinability, add unjoinable pairs as new equations, repeat until all critical pairs join or a budget is hit.

Test on known examples — group axioms, simple magma laws with known completed forms — before running at scale.

### Phase 2: Completion rate across the 4,694 equations — the headline measurement

Run completion on every ETP equation, each with a budget (max rules, max steps, wall-clock cap). Classify each:

- **Completed**: produced a finite confluent terminating system.
- **Failed to orient**: hit an equation neither side of which can be oriented.
- **Did not complete within budget**: hit the budget (may or may not complete with more — report as budget-limited).

Report:
- The completion rate — what fraction of the 4,694 complete cleanly. This is the headline number.
- For completed equations, the completed-system size distribution (number of rules).
- Wall-clock and step-count distributions.

This number determines how far the easy version of the approach reaches, the same way the rewrite study's coverage curve determined the prover architecture.

### Phase 3: Completion as a decision procedure — validation and coverage

For every equation E1 that completed, use its system R to decide all pairs (E1, E2): normalize both sides of E2, compare normal forms.

Report:
- **Correctness**: agreement with the ETP ground-truth 22M-pair table. Theory says this should be 100% for completed equations — it is a decision procedure. Verify it. Any disagreement is a bug in completion or the reduction order, and must be run down.
- **Coverage**: what fraction of all 22M pairs have a completing E1, and are therefore decided — TRUE and FALSE — by normalization alone.
- The TRUE/FALSE split of that coverage. The FALSE side is the new capability — the current engine has no refutation primitive.

### Phase 4: Ordered (unfailing) completion for the non-completing equations

For equations that failed to orient under plain KB completion, implement ordered completion — ordered rewriting that applies an equation only in instances that decrease under the reduction order, so unorientable equations (commutativity-like) still yield a decision procedure.

Report:
- The additional equations that become decision-procedure-backed under ordered completion.
- The new total coverage of the 22M-pair problem.
- The residual — equations where even ordered completion does not terminate or decide.

### Phase 5: Finite counterexample extraction

When completion decides a pair FALSE, the counterexample is the term algebra modulo the completed system — potentially infinite. Stage 2 requires a finite `Fin n` magma.

Investigate:
- For completed systems with finitely many ground normal forms, the normal forms themselves form a finite magma — a directly submittable counterexample. Measure how often this happens.
- For systems with infinitely many normal forms, a finite quotient satisfying E1 and refuting E2 must be found. The completed system's structure should seed the finite-model search well. Measure how much the FALSE counterexample search is accelerated when seeded by completion versus run blind.

### Phase 6: Synthesis

The synthesis report answers the headline question: across the full 22M-pair problem, how much does completion-based symbolic rewriting decide — TRUE and FALSE — and what is the residual that needs something else (finite-model search, neural-guided proof search, or delegation)?

This report stands in the same relation to the project that the proof survey did — it maps how far one unified method reaches and where the genuine hard core remains.

## Decision points

- **Phase 2 completion rate** decides whether plain completion is a major method or a minor one. High → the project's center of gravity shifts to symbolic completion. Low → ordered completion (Phase 4) carries the weight, or the method is partial.
- **Phase 3 coverage** is the first real measurement of how much of the *whole* problem (not just TRUE, not just one category) a single uniform method handles.
- **Phase 5** decides whether completion is a competition asset for the FALSE side or only a theoretical decision procedure.

## Notes for Claude Code

- Build on `src/rewrite/` — the matching, rewriting, and term utilities exist. Completion adds the reduction order, orientation, critical pairs, and the completion loop.
- Pick ONE reduction order for Phase 1 (KBO or LPO) and justify it. The order can be revisited if Phase 2's completion rate is low — a different order orients different equations.
- Completion needs budgets (max rules, max steps, wall-clock) to detect looping. Make them CLI parameters. A non-completing equation is "did not complete within budget B," not "provably non-completing" — keep that distinction in the reporting.
- Phase 1 must be validated on known examples (group axioms complete to a known 10-rule system; simple magma laws have known completed forms) before any scale run. A bug in the reduction order or critical-pair computation produces a wrong "decision procedure," and Phase 3's correctness check is the safety net — but catch it early.
- Phase 3's correctness check against the 22M-pair ground truth is critical. A completed system that disagrees with the ground truth means a soundness bug. Expect 100%; investigate any deviation before trusting the coverage numbers.
- Report completion rate (Phase 2) and total decided coverage (Phase 3, then Phase 4) as the headline numbers. "Completion decides X% of the 22M-pair problem, TRUE and FALSE, by normalization" is the sentence this exploration is built to produce.
- This is exploration — the user has explicitly prioritized it over competition-scoped work and expects to follow investigations where they lead. Serendipitous findings (completion behavior correlating with equation structure, the completed-system sizes, the Vampire-category connection via ordered completion) are worth recording even when they aren't the headline.
