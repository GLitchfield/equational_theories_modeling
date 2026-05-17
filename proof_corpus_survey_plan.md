# ETP Proof Corpus Survey

## Goal

Characterize the Lean proofs in the Equational Theories Project before designing the proof generation model. The survey answers the question that determines the architecture: is generating a proof a template-matching problem (most proofs follow a few patterns with substitutions) or a genuine sequence-generation problem (proofs are highly varied)?

The survey produces a characterization report. It does NOT build any model — it's pure data exploration.

## Why this comes first

The proof generator's architecture depends entirely on what the proofs actually look like:

- If 80%+ of TRUE proofs follow 3-5 tactic templates → the generator is a template selector + parameter predictor (small, simple)
- If proofs are highly varied → the generator is a decoder transformer over a proof-token vocabulary (larger, autoregressive)
- If FALSE proofs are uniformly `finOpTable` + `decideFin!` → the FALSE generator only needs to output a magma table (an MLP, not a sequence model)

You can't pick the architecture without knowing the distribution. The survey is the prerequisite.

## Methodology resolutions (locked in 2026-05-16)

After scrutiny, the following were resolved before implementation. The
plan body has been updated to reflect them; this section is the
canonical record.

1. **The "4,824 edges" number is from the McKenna Spine Isolation
   paper** — that's the transitive reduction of the *equivalence-class*
   graph (1,415 classes, 4,824 edges). It is NOT the same as "how many
   implications have explicit Lean proofs." The ETP may prove things
   at a different granularity. Treat the directly-proved-implication
   count as an open question for Task 1 to determine empirically;
   don't anchor sampling decisions on 4,824.
2. **Use the existing ETP repo at `/home/gavin/git/equational_theories/`**
   rather than re-cloning. The existing `data/raw/etp` symlink points
   only at its `data/` subdirectory; the full repo (Lean sources,
   proofs, lakefile) sits one level up. Only clone the Stage 2 repo
   fresh.
3. **Sample sizes:** Task 1 inventories the full corpus (no sampling
   — counting on-disk text is free). Task 2 uses 5,000 FALSE samples;
   Task 3 uses 5,000 TRUE samples stratified by E2's spine. Bumped
   from the original 500/1,000 to sharpen tail-behavior detection.
4. **Stratification axis for Task 3: E2's spine.** The proof's
   structure tracks what's being proved, not what's given.
5. **Tactic-counting rule (Task 3):** split on Lean tactic separators
   `;`, `<;>`, and newlines outside nested `by` blocks. Treat
   combinators (`;`, `<;>`, `<;>`, etc.) as part of the preceding
   tactic. Example: `simp [h1]; rfl` = 2 tactics. Document the rule
   in the report.
6. **Normalization for Task 4** expands beyond just variable / equation
   references. Apply these transformations in order, documenting each:
   variable names → `v1, v2, ...`; equation references → `E1, E2, ...`;
   local hypothesis names → `h1, h2, ...`; implicit-argument values
   normalized; whitespace collapsed. If cluster count varies more than
   ~2× across schemes, report the count under multiple schemes — the
   architecture decision shouldn't swing 10× on a normalization choice.
7. **Artifact locations:** extracted / normalized proofs go to
   `data/processed/proof_corpus/` (large, regenerable, gitignored).
   The survey report and statistics (CSV / JSON summary tables, the
   final README) go to `outputs/analyses/proof_survey/` (small,
   committed; reproducer scripts live in `scripts/`).
8. **Lean version:** verify from `lean-toolchain` / `lakefile.lean`
   rather than assuming. The version determines LeanDojo
   compatibility (Task 6).
9. **Tokenization (Task 5):** multi-character Unicode identifiers
   (`α`, `β`, `≃`, etc.) are single tokens — matches how Lean's parser
   handles them.
10. **Sampling seed: 42.** Matches the rest of the project's seeds for
    reproducibility.

## Data Sources

Two repositories:

1. **The ETP main repo**, already on disk at `/home/gavin/git/equational_theories/` (Methodology Resolution 2). Contains the full implication graph and all proofs/counterexamples. The existing `data/raw/etp` symlink points only at its `data/` subdir; for proof extraction, reference the full repo at the path above.
2. **The Stage 2 repo**: `github.com/SAIRcompetition/equational-theories-lean-stage2` — contains the judge, the submission format, and demo solvers. The submission format here is the TARGET format the generator must produce. Clone fresh into a scratch directory.

## Survey Tasks

### Task 1: Inventory (full corpus, no sampling)

Locate where proofs live in the ETP repo. The McKenna Spine Isolation paper reports the equivalence-class graph has 4,824 edges in the transitive reduction — but the ETP may prove things at a different granularity (e.g., per-pair, per-equivalence-class, or batched). Determine the directly-proved-implication count **empirically**, don't anchor on 4,824.

Count and report:
- How many TRUE implications have explicit Lean proofs (vs derived by transitivity)
- How many FALSE pairs have explicit counterexample certificates
- How the proofs are organized (one file per implication? batched? generated by a script?)
- Total size of the proof corpus (lines of Lean, number of files)
- Lean toolchain version (from `lean-toolchain` / `lakefile.lean`)

### Task 2: FALSE proof characterization

Extract a sample of FALSE counterexample certificates (target **5,000**, seed=42 per Methodology Resolution 3 & 10). For each:
- Is it the `finOpTable` + `decideFin!` pattern, or something else?
- What magma size (2, 3, 4, 5+ elements)?
- How is the operation table represented (JSON string, explicit matrix, other)?

Report:
- Distribution of magma sizes across the FALSE corpus
- Fraction that follow the standard `finOpTable` + `decideFin!` template
- Any FALSE proofs that DON'T fit the template (these are the hard cases)
- The exact template structure (so the generator's output format is pinned down)

**Expected outcome**: FALSE proofs are highly uniform. If 95%+ are `finOpTable` + `decideFin!` with only the table varying, the FALSE generator's job is just "output an n×n integer matrix" — no sequence modeling needed.

### Task 3: TRUE proof characterization

This is the important one. Extract a sample of TRUE proofs (target **5,000**, seed=42, **stratified by E2's spine** per Methodology Resolution 4 — the proof's structure tracks what's being proved, not what's given). Spine labels come from `data/processed/equation_invariants.parquet`'s `spine_type` column: `family_c`, `mixed`, `no_lhs_var_in_rhs`, `left`, `right`, `trivial`.

For each proof, classify by structure. **Tactic-counting rule** (Methodology Resolution 5): split on Lean tactic separators `;`, `<;>`, and newlines outside nested `by` blocks; combinators stay with the preceding tactic. So `simp [h1]; rfl` = 2 tactics, `· decide` = 1.

- **Single-tactic close**: the proof is one tactic (`simp`, `grind`, `aesop`, `decide`, etc.). No intermediate states.
- **Short tactic sequence**: 2-5 tactics, no explicit rewrite chain.
- **Rewrite chain**: an explicit sequence of `rw`/`calc`/substitution steps transforming one term to another. Genuine intermediate states.
- **Other**: anything that doesn't fit.

Report:
- The distribution across these four categories (overall and per E2 spine)
- For rewrite chains: distribution of chain length (number of steps)
- For single-tactic closes: which tactics, and how often each
- Proof length distribution (in tokens) across the whole TRUE corpus

### Task 4: Proof clustering

Take the extracted TRUE proofs, tokenize them (whitespace + Lean syntax splitting), and cluster by structural similarity via exact-match on a normalized form. Apply these transformations in order (Methodology Resolution 6), documenting each in the report so the cluster count is reproducible:

1. Object-level variable names → `v1, v2, ...` (in order of first appearance)
2. Equation references → `E1, E2, ...` (in order of first appearance)
3. Local hypothesis names → `h1, h2, ...` (in order of first appearance)
4. Implicit-argument values normalized (concrete values → `?`)
5. Whitespace collapsed (multiple spaces/newlines → single space)

Then cluster by exact string match on the normalized proofs.

Report:
- How many distinct proof "shapes" exist after full normalization
- What fraction of proofs the top 10 shapes cover
- What fraction the top 50 shapes cover
- **Sensitivity check:** also report cluster counts under partial normalization (e.g., variables-only; variables + equations; variables + equations + hypotheses). If the count changes by more than ~2× across schemes, that's important diagnostic information — report it.

**This is the architecture-deciding metric.** If the top 10-20 normalized shapes cover 80%+ of TRUE proofs robustly across normalization schemes, the generator is a template classifier. If coverage is spread across hundreds of shapes — or if cluster count varies wildly with the normalization scheme — it's a sequence generator.

### Task 5: Tokenization analysis

Build the proof-token vocabulary from the full extracted corpus. Tokens include: tactic names, structural keywords (`intro`, `have`, `calc`, `by`), operators, variable names, equation references, punctuation. Multi-character Unicode identifiers (`α`, `β`, `≃`, etc.) are **single tokens** (Methodology Resolution 9 — matches Lean's parser).

Report:
- Total vocabulary size (distinct tokens)
- Token frequency distribution (how many tokens cover 95% of usage)
- Whether variable names and equation references should be tokenized as literals or normalized to placeholder tokens
- Estimated vocabulary size after normalization (placeholders for variables/equations)

### Task 6: Intermediate state feasibility

For the rewrite-chain proofs (from Task 3), assess whether intermediate proof states can be extracted.

- Can the ETP proofs be replayed through Lean to capture goal states between tactics?
- Is LeanDojo (or a similar tool) compatible with the ETP repo's Lean version?
- For a sample of rewrite-chain proofs, what do the intermediate states look like? Are they terms/equations (encodable by the existing JEPA encoder) or richer goal states (hypotheses + goal + context)?

Report:
- Feasibility of intermediate-state extraction (yes/no/with-effort)
- What an intermediate state looks like for this domain
- Whether the existing JEPA encoder can represent intermediate states (it can if they're parse trees of terms; it can't if they're full Lean goal contexts)

## Output: Characterization Report

A single report answering:

1. **FALSE proofs**: uniform template or varied? → determines if FALSE generator is an MLP or a sequence model.
2. **TRUE proofs**: what's the breakdown of single-tactic / short-sequence / rewrite-chain / other?
3. **Proof shape coverage**: do the top N normalized templates cover most TRUE proofs? → determines template-selector vs sequence-generator.
4. **Vocabulary**: how many distinct tokens? → determines the output space size.
5. **Proof length**: distribution in tokens → determines the sequence length the generator must handle.
6. **Intermediate states**: extractable? what do they look like? → determines whether step-by-step generation is viable and whether the existing encoder can condition it.

The report ends with an architecture recommendation: given what the proofs actually look like, what should the proof generator be?

## Notes for Claude Code

- This is pure data exploration. No model training, no architecture building. Just extract, count, characterize.
- The ETP repo is already on disk at `/home/gavin/git/equational_theories/` (Methodology Resolution 2). Only the Stage 2 repo needs cloning fresh.
- For the proof extraction, you'll need to parse Lean files. The ETP proofs may be partly machine-generated, which means they're regular and easy to parse. Check whether there's a consistent structure.
- The Stage 2 repo's demo solvers and judge show the exact submission format — that's the TARGET the generator must hit. Document it precisely.
- For proof clustering (Task 4), don't over-engineer. Apply the 5 normalizations from Methodology Resolution 6 in order, then exact-match the normalized strings. A simple approach is enough to answer "is it 10 templates or 1,000."
- For Task 6, you don't need to actually run LeanDojo — just assess feasibility. Does the ETP repo build? What Lean version? Is the proof structure amenable to state extraction? A full LeanDojo extraction is a later task if the survey says it's worthwhile.
- Report numbers and distributions, not vibes. "73% of TRUE proofs are single-tactic `simp` or `grind` closes" is useful. "Most proofs are simple" is not.
- Extracted / normalized proofs go to `data/processed/proof_corpus/` (gitignored, regenerable). The survey report + statistics CSVs / JSONs go to `outputs/analyses/proof_survey/` (committed, with a `scripts/`-side reproducer per CLAUDE.md's reproducer rule).
