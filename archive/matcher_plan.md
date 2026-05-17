# First-Order Matcher — Implementation Plan

## Goal

Build a deterministic matcher that, given two equations E1 and E2, decides whether E2 is a variable-substitution instance of E1, and if so, produces the Lean proof. This solves the 62.5% "variable substitution" category from the proof survey with exact, verifiable correctness — no learned model, no embeddings.

The matcher has two roles:
1. **Competition track**: the first and cheapest pass in the solver cascade. Handles ~62% of TRUE proofs deterministically.
2. **Research track**: generates the ground-truth proof dataset (E1, E2, substitution, Lean proof) that the decoder model trains and is evaluated against.

The matcher uses zero learned components. It is pure symbolic computation on parse trees.

## Methodology resolutions (locked in 2026-05-17)

After scrutiny, the following were resolved before implementation.
The plan body has been updated to reflect them; this section is the
canonical record.

1. **The matcher REPLACES Track 1 from the survey's three-track
   architecture recommendation.** The survey posited a "small
   structured-prediction model on v1.0 embeddings" for the
   variable-substitution majority. This matcher does the same
   coverage deterministically, with exact correctness and zero
   learning. Updated architecture:
     1. Deterministic matcher (62.5% of TRUE)
     2. Tactic-sequence generator (15% of TRUE — rewrite chains)
     3. Upstream-tool delegation (22% of TRUE — Vampire/Egg)
     + FALSE counterexample search.

   v1.0 embeddings have no role in the variable-substitution
   category. They remain relevant for the tactic-sequence generator
   (Track 2) and any future research decoder.
2. **Reuse `src/data/parser.py`.** (Revised 2026-05-17.) The
   original intent was a standalone matcher-parser to avoid
   "JEPA-era variable canonicalization." On re-reading
   `src/data/parser.py`, that premise was inaccurate: the parser's
   `parse_equation(text)` returns raw `Equation(lhs, rhs)` with
   `Var(name)` dataclasses preserving the original variable names
   (`x, y, z, w, u, v`). The canonicalization the original
   resolution was concerned about happens in a separate downstream
   function (`canonical_var_ids`) that the GNN calls but the
   matcher won't. So the existing parser returns exactly what the
   matcher needs. Reuse it (`from src.data.parser import
   parse_equation, Var, Op`); skip writing a redundant standalone.
   The existing parser has 27 unit tests already.
3. **False-positive check runs on the FULL FALSE corpus** (all
   13,855,357 FALSE pairs in `data/processed/implication_pairs.parquet`),
   not a sample. Matching is O(tree size) ≈ 1µs per pair, so this
   completes in ~15s; sampling leaves 99% of the correctness gate
   uncovered for no compute reason. Zero false positives required.
4. **Substitution format: fixed-length-6 padded.** Max variable
   count in the ETP is 6 (per `data/processed/equation_invariants.parquet`
   distribution). The decoder downstream consumes a 6-element
   tuple with a sentinel (e.g., -1 or 255) in unused positions.
   Vectorizes cleanly; simplest output shape for the decoder.
5. **Stage 2 tactic format must be pinned BEFORE mass emission.**
   The plan's `intro h` example is likely wrong — if the judge
   binds `h` as a theorem parameter, the body is just
   `exact fun <e2_vars> => h <σ(e1_vars)>`. Check
   `examples/solo/demos/baseline/solver.py` and the JudgeMagma
   module, pin the exact format, submit 5-10 end-to-end test
   proofs through the judge to confirm acceptance.
6. **Commit to 2 orientations, not 4.** Flipping both equations is
   the same as flipping neither, so the distinct cases are
   `(lhs1→lhs2, rhs1→rhs2)` (direct) and `(lhs1→rhs2, rhs1→lhs2)`
   (flipped, proof needs `.symm`). The "try 4 and dedupe" hedge is
   2× cost for nothing — drop it.
7. **Read the pair count literally from the parquet.** 8,173,585
   TRUE / 13,855,357 FALSE / 22,028,942 total. Don't use approximate
   numbers in code.
8. **Eq1 (`x = x`) as hypothesis edge case.** If E1 = Eq1, then
   `lhs1 = rhs1 = <single variable>`, so the matcher succeeds only
   when E2 is also of the form `v = v` — i.e., E2 = Eq1. So Eq1→Eq1
   is the only Eq1-as-E1 pair the matcher catches, which is correct
   since Eq1 implies nothing else (every more specific equation
   makes Eq1 trivially true; only Eq1 itself is implied by Eq1 as
   a hypothesis in this matcher's restricted sense). One-line
   comment in the matcher for implementer confidence.

## The Matching Problem

An equation is `lhs = rhs`, where each side is a magma term: either a variable, or `t₁ ◇ t₂` for subterms t₁, t₂.

E1 is the hypothesis law: `∀ vars(E1), lhs₁ = rhs₁`.
E2 is the goal law: `∀ vars(E2), lhs₂ = rhs₂`.

E2 is a **variable-substitution instance** of E1 if there exists a substitution σ mapping each variable of E1 to a variable of E2 (not necessarily injective) such that:

```
σ(lhs₁) ≡ lhs₂   and   σ(rhs₁) ≡ rhs₂
```

where ≡ is syntactic term equality. When such a σ exists, E1 implies E2, and the proof is:

```
fun <vars(E2)> => h <σ(v) for v in vars(E1), in E1's argument order>
```

Example: E1 = `a ◇ (b ◇ c) = a`, E2 = `x ◇ (y ◇ x) = x`. The substitution σ = {a↦x, b↦y, c↦x} satisfies σ(lhs₁) = lhs₂ and σ(rhs₁) = rhs₂. Proof: `fun x y => h x y x`.

Because σ maps variables to variables (never to compound terms), σ can only RENAME variables, not change tree structure. So lhs₁ and lhs₂ must be structurally identical trees that differ only in variable labels — and likewise rhs₁ and rhs₂. This is what makes the matching deterministic and linear-time.

## Algorithm

### Core: structural match

```python
def structural_match(t1, t2, sigma):
    """
    Walk E1-term t1 and E2-term t2 in parallel, extending sigma.
    sigma: dict mapping E1 variable -> E2 variable.
    Returns True if t1 can be renamed to t2 consistently with sigma,
    False on any structural mismatch or inconsistency.
    Mutates sigma.
    """
    if t1.is_variable and t2.is_variable:
        if t1.var in sigma:
            return sigma[t1.var] == t2.var   # consistency check
        sigma[t1.var] = t2.var
        return True
    if t1.is_op and t2.is_op:
        return (structural_match(t1.left, t2.left, sigma)
                and structural_match(t1.right, t2.right, sigma))
    # variable vs op, or op vs variable -> structural mismatch
    return False
```

### Matching an equation pair (one orientation)

```python
def match_oriented(e1_lhs, e1_rhs, e2_lhs, e2_rhs):
    """Try to match E1 onto E2 in the given orientation."""
    sigma = {}
    if not structural_match(e1_lhs, e2_lhs, sigma):
        return None
    if not structural_match(e1_rhs, e2_rhs, sigma):
        return None
    return sigma   # consistent substitution found
```

The two `structural_match` calls share the same `sigma` dict — this enforces that the substitution is consistent across BOTH sides of the equation. A σ that works for the LHS but contradicts on the RHS is rejected.

### Both orientations (equation symmetry)

An equation `a = b` is logically identical to `b = a`. So E1 can be matched against E2 with either side of E2 as the target. Try both:

```python
def match_equations(E1, E2):
    # Orientation 1: direct
    sigma = match_oriented(E1.lhs, E1.rhs, E2.lhs, E2.rhs)
    if sigma is not None:
        return ("direct", sigma)
    # Orientation 2: E2 flipped (proof needs .symm)
    sigma = match_oriented(E1.lhs, E1.rhs, E2.rhs, E2.lhs)
    if sigma is not None:
        return ("flipped", sigma)
    return None   # not a variable-substitution instance
```

Per Methodology Resolution 6: **2 orientations are sufficient and complete.** Flipping both equations gives the same matching as flipping neither, so the distinct cases are exactly the direct match above and the flipped match (`lhs1→rhs2, rhs1→lhs2`).

### The matcher is self-classifying

`match_equations` returns either a substitution (the pair IS a variable-substitution instance, and you have the proof) or None (it is not — fall through to the next track). No separate classifier needed.

## Proof Emission

Given a successful match (orientation, σ), emit the Lean proof.

The proof body, in term mode:

```
fun <e2_var_1> <e2_var_2> ... => h <σ(e1_var_1)> <σ(e1_var_2)> ...
```

For the flipped orientation, the proof needs `.symm` somewhere — the hypothesis application proves `rhs₂ = lhs₂` and you need `lhs₂ = rhs₂`, so wrap with `.symm`:

```
fun <e2_vars> => (h <σ(e1_vars)>).symm
```

For the Stage 2 judge, the submission is a tactic body (the judge wraps it in `theorem ... := by`). Per Methodology Resolution 5, **pin the exact format against the Stage 2 demos BEFORE mass emission**. The expected form is probably (with `h` already in scope as a theorem parameter, NOT requiring `intro h`):

```
exact fun <e2_vars> => h <σ(e1_vars)>
```

But this must be confirmed by reading `examples/solo/demos/baseline/solver.py` and submitting 5-10 test proofs through the live judge before generating proofs at corpus scale. If the format is wrong, the matcher's output is unusable.

## Verification

Matching is correct by construction — if σ(E1) ≡ E2 structurally, then E2 is a substitution instance of E1 and the emitted proof is valid. Verification is a format check, not a correctness check.

So: verify a SAMPLE (say 200 emitted proofs across different equation shapes) against the actual Stage 2 judge to confirm the proof format is accepted. Once the format is confirmed, trust the algorithm for the full corpus — re-verifying millions of structurally-identical proofs adds nothing.

If the judge rejects a sampled proof, it's a format bug (wrong variable names, missing argument, wrong wrapper), not a logic bug. Fix the emission template and re-sample.

## Corpus Run and Coverage Measurement

Run the matcher on the full set of TRUE pairs to measure coverage.

### Measurements

1. **Coverage of the directly-proved set.** Run the matcher on the 10,676 directly-proved implications. The survey predicts ~62.5% (the variable-substitution category) should match. Confirm this — it validates the matcher against the survey's classification. If the matcher solves substantially more or fewer than 6,567, investigate the discrepancy.

2. **Coverage of the full TRUE set.** Run the matcher on all 8.17M TRUE pairs. This is the important number. The matcher can solve ANY pair where E2 is a substitution instance of E1, regardless of how the ETP originally proved it. Most of the 8.17M are derived by transitivity and won't be direct substitution instances — but some fraction will be. Report what fraction of all TRUE pairs the matcher closes.

3. **False-positive check on the FULL FALSE corpus** (Methodology Resolution 3). Run the matcher on all 13,855,357 FALSE pairs in `data/processed/implication_pairs.parquet`. It must return `None` on every one — if E2 were a variable-substitution instance of E1, the implication would be TRUE by construction. Any FALSE pair the matcher "solves" indicates a bug in the matching algorithm. Matching is microseconds per pair, so the full corpus completes in ~15 seconds; sampling would leave 99% of the correctness gate uncovered for no compute saving.

### Performance

Matching is O(tree size) per pair, and equations have ~10 nodes. 8.17M pairs × constant ≈ seconds to low minutes. No GPU, no embeddings. If it's slow, the bottleneck is parsing, not matching — parse all 4,694 equations once into term trees, cache them, then matching is pure tree comparison.

## Output: Ground-Truth Proof Dataset

For every pair the matcher solves, record:

```
eq1_id            : int32
eq2_id            : int32
orientation       : int8     # 0 = direct, 1 = flipped (proof needs .symm)
substitution      : int8[6]  # fixed-length-6 padded (Methodology Resolution 4);
                             # entry i is the E2-variable index that σ assigns
                             # to E1's i-th variable, or sentinel (-1) for unused
                             # slots when E1 has < 6 variables
lean_proof_term   : string   # e.g. "fun x y => h x y x"
lean_proof_tactic : string   # e.g. "exact fun x y => h x y x" (Stage 2 body)
```

Save as a parquet file at `data/processed/proof_corpus/matcher_proofs.parquet`. This serves two purposes:

- **Competition track**: a lookup table — for any solved pair, the proof is precomputed and can be submitted to Stage 2 directly.
- **Research track**: the training and evaluation set for the eventual decoder (if a learned variant ever revisits this category). The decoder would map `(z_e1, z_e2) → substitution`, and this dataset is the verified-correct ground truth.

Fixed-length-6 padded substitution per Methodology Resolution 4: max variable count in the ETP corpus is 6 (verified against `data/processed/equation_invariants.parquet`'s variable-count distribution). Decoder-side, the output is a 6-way tuple over E2-variable indices; unused slots are sentinel.

## Edge Cases

- **E1 and E2 have different numbers of variables.** Fine — σ maps E1's variables onto E2's, and need not be surjective (E2 may have variables E1's substitution doesn't reach... actually no — every variable in lhs₂/rhs₂ is the image of some E1 variable under the structural walk, so σ is surjective onto the variables that appear. An E2 variable appearing nowhere can't happen for a real equation). σ need not be injective (the `h x y x` case maps two E1 vars to x).
- **E1 = E2 (identical equations).** Matches trivially with σ = identity. The proof is `fun vars => h vars`. This is correct — every equation implies itself.
- **The trivial law `x = x` (eq 1) as E2.** `lhs₂ = rhs₂ = x`. For E1 to match, σ(lhs₁) and σ(rhs₁) must both equal the single variable x — meaning lhs₁ and rhs₁ are both single variables. Only equations of the form `a = b` (both bare variables) match onto `x = x`. Most equations won't, so the matcher will NOT solve most pairs implying eq 1 — those are the TRUE-by-vacuity cases that need a different argument. Expected; the matcher isn't supposed to catch vacuity.
- **Compound-term substitutions.** If E2 would require an E1 variable to map to a compound term (e.g., `h (x ◇ y) z`), the structural walk fails (variable in E1's tree vs `◇` in E2's tree → mismatch). The matcher correctly returns None and the pair falls through. These are not in the variable-substitution category by definition.
- **E1 = Eq1 (`x = x`)** (Methodology Resolution 8). When E1 is the trivial reflexive law, `lhs1 = rhs1 = <single variable>`, so the matcher succeeds only when E2 is also of the form `v = v` — i.e., E2 = Eq1. So **Eq1 → Eq1 is the only Eq1-as-hypothesis pair the matcher catches**, which is correct: Eq1 implies nothing else in the variable-substitution sense (any non-trivial E2 would require σ to map a variable to a compound term, which the matcher prohibits). Add a one-line comment in the matcher for implementer confidence — this is not a bug.

## Notes for Claude Code

- **Reuse `src/data/parser.py`** (Methodology Resolution 2 as revised 2026-05-17). Its `parse_equation(text)` returns `Equation(lhs, rhs)` with `Var(name)` preserving the original variable names. Skip writing a standalone parser.
- The matcher needs **NO embeddings, NO model, NO GPU**. It is pure symbolic code. Keep it that way — it's the deterministic baseline.
- **2 orientations, not 4** (Methodology Resolution 6). Direct: `(lhs1→lhs2, rhs1→rhs2)`. Flipped: `(lhs1→rhs2, rhs1→lhs2)` with `.symm` on the resulting term.
- **Pin the Stage 2 TRUE submission format BEFORE mass emission** (Methodology Resolution 5). Read `examples/solo/demos/baseline/solver.py` and the JudgeMagma module; submit 5-10 test proofs through the live judge to confirm. The plan's `intro h` example is probably wrong — `h` is likely a theorem parameter already in scope.
- **False-positive check on the FULL FALSE corpus** (13,855,357 pairs; Methodology Resolution 3), not a sample. Microseconds per pair = ~15 seconds total. Zero false positives required.
- Report the three coverage numbers clearly: directly-proved set (~expect 6,567 per the survey's 62.5% variable-substitution finding), full 8,173,585 TRUE set (unknown, measure it), false-positive count (must be 0).
- **Substitution format: fixed-length-6 padded** (Methodology Resolution 4). Decoder-side simplicity.
- Save the ground-truth dataset with both the substitution tuple and the formatted Lean proof. The substitution is the decoder's training target if it's ever revived; the Lean proof is the deliverable for the competition track.
- The `orientation` field matters for the decoder later — direct vs flipped proofs differ by a `.symm` wrap. Keep it in the output.
- Read the literal pair count from `data/processed/implication_pairs.parquet` (8,173,585 TRUE / 13,855,357 FALSE / 22,028,942 total). Don't use approximate numbers in code.
