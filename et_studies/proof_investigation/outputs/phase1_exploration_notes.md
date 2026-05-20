# Proof Investigation Phase 1 — Exploration Notes

## ETP repo structure

`~/git/equational_theories/equational_theories/` is the main Lean root. Generated proofs live in `Generated/`:

| dir | contents | relevance |
|---|---|---|
| `Generated/VampireProven/Proofs{1..N}.lean` | Vampire-derived proofs in tactic mode | source of `e_vampire_superposition` |
| `Generated/MagmaEgg/small/_{000..018}.lean` | Egg-derived proofs in term mode | source of `e_egg_combinator` |
| `Generated/SimpleRewrites/` | rewrite-chain proofs | source of `c_rewrite_chain` (not used here) |
| `ManuallyProved/Equation*.lean` | manually proved equations | NOT the source of `e_complex_construction` (these are model-construction proofs of single equations, not implications) |
| `Subgraph.lean` | named lemmas with custom proofs | 2 of the 11 eligible complex pairs |

## File-to-category mapping (proof-survey labels)

Verified against `data/processed/proof_corpus/true_corpus.parquet`:

- **`e_vampire_superposition`** → `Generated/VampireProven/Proofs*.lean` (1,507 pairs total, spread across 13 files at ~120 per file).
- **`e_egg_combinator`** → `Generated/MagmaEgg/small/_*.lean` (796 pairs total, concentrated in `_000` and `_001`).
- **`e_complex_construction`** → mixed: 9 of the 11 eligible pairs are in `MagmaEgg/small/_*` files (Egg-style proofs the survey classified as complex), 2 are in `Subgraph.lean` (structured `have`-based proofs).

The naming "complex_construction" is misleading. Of the 11 eligible pairs, none are model constructions in the `ManuallyProved/` sense. They're proofs in the standard Egg or `Subgraph.lean` formats that the survey's classifier flagged as more structurally complex than the typical Egg pattern.

## Format-by-format visibility verdict

For each of the three target categories, spot-checked one pair and characterized the format:

### `e_vampire_superposition` (392 eligible)

Example: `Equation102_implies_Equation1029` in `Proofs1.lean`.

```lean
theorem Equation102_implies_Equation1029 (G : Type*) [Magma G] (h : Equation102 G) : Equation1029 G := by
  by_contra nh
  simp only [not_forall] at nh
  obtain ⟨sK0, sK1, nh⟩ := nh
  have eq9 (X0 X1 : G) : (X0 ◇ ((X0 ◇ X1) ◇ X1)) = X0 := mod_symm (h ..)
  have eq10 : sK0 ≠ (sK0 ◇ ((sK0 ◇ (sK1 ◇ sK1)) ◇ sK1)) := mod_symm nh
  have eq11 (X0 X1 : G) : (X0 ◇ (X0 ◇ ((X0 ◇ X1) ◇ X1))) = X0 := superpose eq9 eq9 -- superposition 9,9, 9 into 9, unify on (0).2 in 9 and (0).1.2.1 in 9
  have eq12 (X0 : G) : (X0 ◇ X0) = X0 := superpose eq9 eq11 -- forward demodulation 11,9
  have eq13 : sK0 ≠ (sK0 ◇ ((sK0 ◇ sK1) ◇ sK1)) := superpose eq12 eq10 -- backward demodulation 10,12
  subsumption eq13 eq9
```

**Intermediates: 100% visible.** Each `have eqN (X0 ... : G) : LHS = RHS := superpose ...` line is an explicit intermediate clause. The format gives us:

- `term_lhs`, `term_rhs` of each intermediate as concrete Lean expressions over `◇`.
- Quantified variables (e.g., `X0 X1 : G`).
- Skolem constants (`sK0`, `sK1`) for the goal-side intermediates.
- Derivation source: `mod_symm (h ..)` (input), `superpose eqA eqB` (output of a superposition step), `subsumption` (terminal).
- Comments after `--` describe the inference rule (`superposition`, `forward demodulation`, `backward demodulation`).

Parsing: regex-extractable. The pattern is uniform across all `Proofs*.lean` files.

### `e_egg_combinator` (319 eligible)

Example: `Equation3102_implies_Equation3` in `MagmaEgg/small/_000.lean`.

```lean
theorem Equation3102_implies_Equation3 (G: Type _) [Magma G] (h: Equation3102 G) : Equation3 G := fun x =>
  T (h x (M x x)) (C (S (h x x)) (R x))
```

Combinators (defined at the top of each file): `T := Eq.trans`, `S := Eq.symm`, `R := Eq.refl`, `M := Magma.op`, `C := congr_op`.

**Intermediates: visible, embedded.** The proof term is a tree of combinator applications. The "intermediates" are the *terms appearing as arguments to `h ...`* (instantiations of the hypothesis at specific points in the chain) and the *terms built via `M ...`* (Magma op applications).

Some Egg proofs additionally use `let v0 := M ...` to name an intermediate term — that's the named-intermediate analog of Vampire's `have eqN`.

Parsing: requires walking the Lean expression tree. Each `M a b` is an Op term; each `h a b ...` instantiates the hypothesis at terms `a, b, ...`. The chain structure (which `T` is composed with which `C`) determines the semantic intermediate values, but a structural extraction (count of `M`-subterms, count of unique terms passed to `h`) is easy and informative.

Fresh-variable verdict for Egg: term-mode Lean cannot introduce universally-quantified fresh variables. The only variables in a `fun x y z => ...` proof are the lambda-bound goal variables. Egg's intermediates are *composite terms over those existing variables*, not fresh-variable equations. So the higher-variable hypothesis test on Egg is essentially "do intermediates use the goal's variables in deeper compositions" rather than "do intermediates introduce variables the goal doesn't have."

### `e_complex_construction` (11 eligible)

Of the 11 eligible: 9 are in `MagmaEgg/small/_*` files (Egg-style, parse with Egg parser), 2 are in `Subgraph.lean` (structured `have`-based).

Example of the Subgraph form: `Equation1571_implies_Equation40`:

```lean
theorem Equation1571_implies_Equation40 (G: Type _) [Magma G] (h: Equation1571 G) : Equation40 G := by
  have sqRotate : ∀ x y z : G, (x ◇ y) ◇ (x ◇ y) = (y ◇ z) ◇ (y ◇ z) :=
    fun x y z ↦ ...
  have sqConstInImage : ∀ x y z w : G, (x ◇ y) ◇ (x ◇ y) = (z ◇ w) ◇ (z ◇ w) :=
    fun x y z w ↦ (sqRotate x y z).trans (sqRotate y z w)
  exact fun x y ↦ h x x x ▸ h y y y ▸ sqConstInImage (x ◇ x) (x ◇ (x ◇ x)) (y ◇ y) (y ◇ (y ◇ y))
```

**Intermediates: visible but custom-shaped.** Named lemmas (`sqRotate`, `sqConstInImage`) play the role of intermediates. The `∀ x y z : G, ...` quantifier reintroduces fresh universals — this format CAN have intermediate equations with more variables than the goal. The two Subgraph pairs would need a custom-ish parser (the structure is regular but smaller-scale than Vampire's).

Given the small count (2 pairs in Subgraph format + 9 in Egg format), descriptive-only treatment of the complex category remains the right move. The 11-pair size is too small for entropy analysis regardless.

## Partition verdict

**Hidden-intermediate proofs are essentially 0% of the eligible pool.** All 722 eligible pairs have visible intermediate terms.

The partition by *parsing difficulty* (not visibility) splits cleanly:

| stratum | n | parsing approach |
|---|---|---|
| Vampire (`have eqN ... := <derivation>`) | 392 | regex per-line, straightforward |
| Egg term-mode (`T/S/R/C/M` combinator) + Egg-style "complex" | 319 + 9 = 328 | Lean expression-tree walk |
| Subgraph-style structured `have` | 2 | small-scale custom parser or descriptive-only |

The higher-variable test reaches all 720 of Vampire + Egg cleanly. Subgraph's 2 pairs report descriptively if extracted; if not, they're noted as a 0.3% sample shortfall.

## What "higher-variable" means per format

- **Vampire**: each `have eqN (X0 X1 ... : G) : LHS = RHS := ...` is an equation universally quantified over `X0, X1, ...`. The test is `max(N_intermediate quantifiers) − N_E2 quantifiers`. **Fresh universals are possible and the natural test target.**
- **Egg**: lambda-bound goal variables only; no fresh universals in term mode. The test reframes as "intermediate term depth / operator count" — a structural complexity proxy, not a literal variable count.
- **Subgraph complex** (2 pairs): named `have` lemmas use `∀ x y z : G`-style quantifiers; can have fresh universals.

This is a meaningful framing shift the synthesis should be explicit about. The hypothesis "intermediates use more variables than the goal" only directly applies to Vampire (and the 2 Subgraph pairs). For Egg, the equivalent question is "intermediates use deeper term structure than the goal" — measured via depth/operator count, not variable count.

## What's next

If this verdict is acceptable, Phase 2 (single-pair trace) proceeds:

- One Vampire pair: extract all `have eqN` lines with their LHS/RHS/quantifiers/derivation source.
- One Egg pair: walk the term tree, extract `M`-built subterms, `h`-argument terms, `let v0` named intermediates.
- One Subgraph pair: extract named `have` lemmas with their statement and proof.

The output is a manually-traced reference set used to validate the Phase 3 parsers.
