# Q12 — ETP Vampire / Egg recon

Recon of `~/git/equational_theories/` (lean-toolchain pinned at **`leanprover/lean4:v4.20.1`** — confirms Q3) for Vampire and Egg integration, plus equation predicates and closure. Findings verified by inspection.

## Vampire

| Item | Location |
|---|---|
| Runner / driver | `equational_theories/Generated/VampireProven/src/vampire_proofs.py` |
| Input encoder | same file, `encode_problem()` — emits TPTP FOF |
| **Lean translator** | same file, `leanify()` — Python → Lean tactic text |
| Lean elaborator backend | `equational_theories/Superposition.lean` — defines the `superpose` tactic that consumes leanify's output |
| Generated artifacts | `equational_theories/Generated/VampireProven/Proofs1.lean` … `Proofs13.lean` |
| Vampire binary expected at | `~/Downloads/vampire` (hardcoded in driver) |

**Pipeline:** TPTP file → `vampire` binary with `--proof_extra full` flag → parse Vampire's superposition trace with `leanify()` → emit Lean `have eqN ... := superpose eqA eqB` lines → wrap in a theorem and import `Superposition.lean`.

**Reuse story for the pilot:** clean. The translator exists, takes Vampire stdout, returns Lean text. We need to (a) install the Vampire binary at the expected path, (b) call `vampire_proofs.py` (or import its functions) for each of our 1k pairs, (c) ship the resulting Lean with `import equational_theories.Superposition`. The `leanify()` raises `NotImplementedError` on uncommon proof patterns — flagged as a possible per-pair failure mode in the pilot.

**Generated proof shape** (example from `Proofs1.lean`):
```lean
@[equational_result]
theorem EquationX_implies_EquationY (G : Type _) [Magma G] (h : EquationX G) : EquationY G := by
  have eq1 : ... := h x y
  have eq2 : ... := superpose eq1 eq1
  ...
```

Tactic-mode `by` block — gives LeanDojo the intermediate states we need for Q5.

## Egg (MagmaEgg)

| Item | Location |
|---|---|
| Runner / driver | **external Rust repo:** `github.com/lyphyser/equational_theories` branch `magmaegg-code` |
| Input format | not documented in ETP repo (lives in the Rust crate) |
| **Lean output** | tool emits Lean directly (no Python translator step) |
| Generated artifacts | `equational_theories/Generated/MagmaEgg/small/_000.lean` … `_018.lean` + `small.lean` bundles them |
| README | `equational_theories/Generated/MagmaEgg/README.md` (one paragraph, points at the Rust repo) |

**Pipeline:** Rust binary → reads equation set + target pair → emits a Lean proof using ETP-defined abbreviations T (Eq.trans), S (Eq.symm), R (Eq.refl), M (Magma.op), C (congr_op).

**Generated proof shape** (from `Generated/MagmaEgg/small/_000.lean`):
```lean
@[equational_result]
theorem Equation3102_implies_Equation3 (G : Type _) [Magma G] (h : Equation3102 G) : Equation3 G := fun x =>
  T (h x (M x x)) (C (S (h x x)) (R x))
```

Term-mode `fun x => ...` body, no `by` block. **Implication for Q5:** without a `by` block, LeanDojo has no per-tactic state transitions to step through. If we want intermediate states for Egg outputs we either (a) post-process them into tactic-mode form, or (b) accept that Egg proofs are atomic from LeanDojo's perspective and only get (initial, final) states.

**Reuse story for the pilot:** complicated. To re-run Egg on our 1k sample we need to clone and build the external Rust repo, figure out its CLI/input format (not documented in ETP), and integrate. That's meaningful Phase 1 setup. Alternative: skip Egg for Phase 1 and accept matcher + completion + Vampire as the 3-tool comparison. Or replace Egg with **lean-egg** (a different tool — see Gaps).

## Equation predicates

| Item | Location |
|---|---|
| Predicate definitions | `equational_theories/Equations/{Eqns1_999.lean, Eqns1000_1999.lean, ..., Eqns4000_4694.lean}` |
| Master import | `equational_theories/Equations/All.lean` |
| `equation N := ...` DSL | `equational_theories/Equations/Command.lean` |

**Usage in our pilot:**
```lean
import equational_theories.Equations.All

@[equational_result]
theorem EquationX_implies_EquationY (G : Type _) [Magma G] (h : EquationX G) : EquationY G := ...
```

Confirms Q6: reuse ETP's predicates directly.

## Closure

`equational_theories/Closure.lean` — 418 lines. Implements bitset-based transitive closure over the implication graph. Reads ~11K direct proofs + ~4,694 equations, derives ~8M+ transitive labels at Lean compile-time. **This is the architecture documented in `et_studies/etp_structure_analysis/`.** Not something we need to reuse for the pilot — its role is producing the labeled corpus we sample *from*, not part of our pipeline.

## Gaps / unknowns

1. **Vampire's `leanify()` may raise `NotImplementedError`** on uncommon proof patterns (superposition + backward demodulation + equality resolution are handled; others may not be). Phase 1 should track these as per-pair failures and characterize them.

2. **MagmaEgg input format is opaque from the ETP repo.** Requires reading the Rust crate. Cost-benefit: real engineering before we know whether Egg adds value on variable-disjoint pairs that matcher already handles trivially. Consider deferring.

3. **lean-egg ≠ MagmaEgg.** `Subgraph.lean` mentions 86 proofs from `lean-egg` (Rossel's Lean tactic). Different tool, different integration — lives Lean-side as a tactic, not a separate prover. If we want an "Egg-style" tool in Phase 1 and don't want to build the Rust pipeline, lean-egg is a viable substitute. We'd write proofs of the form `theorem ... := by egg [...]` and let the Lean tactic do the work.

4. **No streaming/incremental pipeline** for either tool. ETP ran them offline once; rerunning on 1k pairs is feasible (Vampire is fast per-pair given its `-t 1` second timeout) but no shared infrastructure exists.

5. **Vampire binary path is hardcoded** to `~/Downloads/vampire` in the driver. Trivial to patch but worth noting.

## Implications for Phase 1

- **Vampire: GREEN.** Clean reuse path; install binary, call existing driver, get Lean tactic output back. Estimated half-day to integrate.
- **MagmaEgg: YELLOW/RED.** Significant setup cost (Rust repo) for uncertain pilot value. **Recommend swapping to `lean-egg` (the Lean tactic) for Phase 1** — it produces Lean proofs natively, no separate runner needed, and gives us the "Egg-flavored" data point at a fraction of the integration cost. If after Phase 1 we want to compare to MagmaEgg specifically, we can add it later.
- **Equation predicates: GREEN.** `import equational_theories.Equations.All` and we have all 4,694 available.
- **LeanDojo readiness:** Vampire proofs are tactic-mode (good for state capture). MagmaEgg / lean-egg proofs are term-mode lambdas (atomic for LeanDojo; only initial + final states). Worth deciding whether to *force* tactic-mode wrappers around term-mode proofs to give LeanDojo something to step through, or accept the asymmetry.

Suggested plan adjustment: revise Q4 to **Vampire + lean-egg** (instead of MagmaEgg) unless there's a specific reason to want MagmaEgg.
