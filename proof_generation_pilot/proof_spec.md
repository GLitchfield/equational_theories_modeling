# Proof output specification (v0.1)

Defines the uniform Lean output language all proof-producing tools target. Goal: produce a training dataset where the model's vocabulary is bounded and inspectable; produce submissions Stage 2's judge accepts without ETP-side imports.

**Status:** v0.1, iterates with empirical findings from per-tool characterization. Each tool's *minimum* needed primitives are catalogued; the spec's allowed vocabulary is the union of those minimums.

## Why a spec

Three problems this addresses:

1. **Training tractability.** Raw tool output spans hundreds of distinct tactics (1,450 instances / 140 templates for Vampire in Phase 1). Restricting to a small uniform vocabulary gives a transformer a bounded target.
2. **Stage 2 portability.** Vampire's natural output uses ETP's `Superposition.lean` tactics (`superpose`, `mod_symm`, `subsumption`) the Stage 2 judge doesn't bundle. A uniform spec built on Lean core + minimal Mathlib makes proofs directly submittable.
3. **Inspectability.** A small vocabulary lets us read model outputs as plain Lean and reason about why a proof works or doesn't.

## Vocabulary (v0.1)

Iterates as we characterize each tool. The current allowed set:

### Tactics

| Tactic | Purpose | Used by |
|---|---|---|
| `intro <ids>` | Bind universals + hypotheses into local context | All tools (every proof opens with `intro G _ h <vars>`) |
| `exact <term>` | Close goal with a term-mode proof | Matcher |
| `rw [<eq>]` and `rw [← <eq>]` | Rewrite via an equality | Completion, Vampire (translated) |
| `rfl` | Close by definitional equality | Completion, Vampire (translated) |
| `have <name> : <type> := <term>` | Introduce a named intermediate equation | Vampire (translated) |
| `apply <fn>` | Goal-directed function application | (TBD — may not need) |
| `calc <term> = ... := ... = ... := ...` | Chained equational reasoning | Vampire (translated, optional alternative to `rw` chains) |

### Term primitives

| Primitive | Purpose |
|---|---|
| Function application `f x y z` | Standard |
| Lambda `fun x => ...` | Closing matcher-style proofs |
| `Eq.refl` | Reflexivity term |
| `Eq.symm` | Equality reversal |
| `Eq.trans` | Equality composition |
| `congrArg`, `congrArg₂` | Congruence-under-context rewriting |

### Explicitly excluded (v0.1)

| Excluded | Reason |
|---|---|
| `superpose`, `mod_symm`, `subsumption` | ETP-side; not in Stage 2 judge environment |
| `simp`, `simp_all` | Too broad; opens grammar significantly |
| `omega`, `decide`, `norm_num`, `polyrith` | Specialized; not needed for magma equations |
| `grind` | Black-box; can't decompose into uniform vocabulary (see "Grind" below) |
| `by_contra`, `obtain`, `rcases` | Refutation-style proofs (Vampire's native shape). Translation produces forward-style proofs instead. Re-evaluate during Vampire characterization if forward translation proves intractable. |

## Per-tool spec status

### Matcher

- **Natural output already conforms.** `intro G _ h <vars>; exact fun <ids> => h <σ-applied args>`.
- **Required:** `intro`, `exact`, function application, lambda.
- **Spec adjustment needed:** none beyond what Phase 1's adapter already produces.

### Completion

- **Status:** ready to characterize. Code in `proof_generation_pilot/methods/completion/`; not yet exercised on a sample where it produces proofs (Phase 1 used the var-disjoint slice where completion returns `NO_RULES_FROM_INPUT`).
- **Expected minimum:** `intro`, `rw` (forward and `←`), `rfl`. Each rewrite chain step from the completion's normalization becomes one `rw`.
- **Characterization task:** run completion on 10 non-variable-disjoint matcher-solvable pairs (from the 19,576 intersection), inspect what shape the proof takes. Update spec if anything else is needed (e.g., `Eq.symm` term inside the rule application).

### Vampire

- **Status:** needs translator. Phase 1's adapter calls ETP's `leanify()` which produces ETP-flavored tactic body using `superpose`/`mod_symm`/`subsumption` — out-of-spec.
- **Translation target:** convert Vampire's refutation-style trace into a forward-style proof in the uniform vocabulary.
- **Approach (to validate empirically):**
  - Parse Vampire's `--proof_extra full` output into a DAG of `(premise_eqs → conclusion_eq, direction)` superposition / demodulation steps
  - Topologically order so each step's premises are already established
  - Emit a `calc` block or chained `rw`s that proves `E2` forward from instances of `E1`
  - The translator's output is `intro G _ h <vars>; calc ... := h <args>; ...` or `intro ...; rw [...]; rfl`
- **Open during characterization:**
  - Are there proof patterns Vampire produces that don't translate cleanly to forward style? (Edge case: contradictions involving multiple Skolem witnesses with no obvious forward orientation.) If so, fallback may need refutation-style tactics — re-add `by_contra` to spec.
  - How to handle congruence-under-context — use `congrArg`/`congrArg₂` or `rw` with explicit position?

### Grind (coverage oracle, NOT in training set)

- **Status:** opaque tactic; emits a proof term but doesn't expose intermediate states.
- **Disposition:** record grind successes/failures as a per-pair coverage stat. The (pair, status) outcome is data. The proof term grind produces is **not** included in the training set's transitions (no per-step decomposition available).
- **Schema implication:** grind rows appear in `proofs.parquet` with `status` and timing, but contribute zero rows to `transitions.parquet`.

## Adapter contract

Every tool's adapter implements a uniform interface:

```python
def prove(eq1_id: int, eq2_id: int, **tool_specific_kwargs) -> ProofResult
```

Where `ProofResult` is:

```python
@dataclass
class ProofResult:
    status: Literal['proved', 'no_proof', 'timeout', 'error']

    # In-spec, validated Lean (None if status != 'proved')
    proof_lean: Optional[str]

    # Raw tool output before translation (always populated)
    # Vampire: stdout from the binary
    # Completion: rule set + per-pair derivation trace
    # Matcher: the σ tuple as a Python object
    # Grind: the raw tactic + Lean's success/failure response
    raw_output: str | dict

    # Timing
    wall_clock_ms: int

    # When status != 'proved', explains why
    validation_log: str
```

The `raw_output` field is the irreversible record of what the tool produced before translation. **Translation is deterministic from `raw_output`** — if we change the spec, we re-translate all `raw_output`s without re-running the underlying tool. This is the user's "regenerate later" requirement.

## Validator

The validator confirms a `proof_lean` string:

1. **Lexically conforms** to the vocabulary (parser-level check, fast)
2. **Type-checks against the equation predicates** in the Stage-2-style judge environment (Mathlib + Magma + Equation definitions, no ETP `Superposition` import)
3. **Goal is closed** at the end (state == `no goals`)

PyPantograph's `goal_start` + step-through covers (2) and (3). The lexical check (1) is a separate parser pass over the tactic body — flags any out-of-vocab tactic or term primitive before sending to Lean.

## Schema additions (vs Phase 1)

`proofs.parquet` gains:

```
raw_output       str | bytes  # the tool's pre-translation output
spec_version     str          # which spec version the translation targets ('v0.1', ...)
```

`transitions.parquet` is unchanged in shape — it remains `(pair_id, eq1_id, eq2_id, tool, step_idx, state_before, tactic, state_after, is_terminal)`. The `tactic` field is restricted to in-spec strings (validator enforces).

## Iteration policy

This is v0.1. Updates happen when:

- A tool's characterization run produces output that requires a primitive not yet in the vocab (e.g., Vampire's translator can't avoid `congrArg₂` → add it explicitly)
- The Stage 2 judge environment changes (e.g., adds a new Mathlib tactic we want to use)
- The trained model's output reveals a vocab gap (e.g., model wants to emit `Eq.mpr` for propositional rewrites — decide whether to allow)

Each update bumps `spec_version` and is logged at the bottom of this doc.

### Changelog

- **v0.1** (initial): tactics `intro`/`exact`/`rw`/`rfl`/`have`/`apply`/`calc`; term primitives function-app/lambda/`Eq.refl`/`Eq.symm`/`Eq.trans`/`congrArg`/`congrArg₂`. Matcher conforms natively; completion expected to need only `intro`/`rw`/`rfl`; Vampire requires translator from refutation-style; grind treated as coverage oracle (not in training set).

- **Empirical update (2026-05-22)** from 10-pair characterization:
  - **Completion's minimum** confirmed as **`intro` + `rw` (with `←`)**. All 10 non-variable-disjoint matcher-solvable pairs close with `intro G _ h <vars>; rw [← h]` — single 3-tactic sequence. `rfl` not needed (Lean's `rw` auto-closes when both sides match post-rewrite). For `rhs_to_lhs` orientability E1s, the `←` direction is what fires; `rw [h]` (forward) doesn't.
  - **Vampire's minimum on this slice is the SAME as the matcher's.** On all 3 Phase 1 sample pairs inspected, the only inference step in Vampire's trace is `subsumption resolution N,M` between a CNF axiom (line 9) and the Skolemized negated conjecture (line 10). Lines 1–10 are pure TPTP preprocessing (flattening, Skolemization, CNF conversion). The forward translation collapses to `intro G _ h; exact fun <σ> => h <permuted_args>` — identical to the matcher's natural output.
  - **Phase 1's 7-step Vampire transitions were preprocessing overhead, not reasoning.** The variable-disjoint matcher-solvable slice doesn't exercise multi-step inference from any tool. To get genuine multi-step proof grammar in the dataset, Phase 2's sample must include pairs where the underlying proof requires multiple inference steps — e.g., pairs not in matcher's solved set, or the `c_rewrite_chain` category from proof_survey, or Vampire-only pairs in ETP's corpus.
  - **No vocab additions yet.** The v0.1 vocabulary is sufficient for single-step substitution-style proofs; multi-step shapes (`have`, `calc`, `Eq.trans`) become *necessary* (not just allowed) when Phase 2's sample exercises genuine inference.
