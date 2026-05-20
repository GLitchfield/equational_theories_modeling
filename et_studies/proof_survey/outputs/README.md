# ETP Proof Corpus Survey — Characterization Report

Output of `proof_corpus_survey_plan.md` (now archived). Pure data
exploration of the Equational Theories Project's Lean proofs to
inform the proof-generator architecture.

The survey ran six tasks. This README synthesizes their findings
and ends with an architecture recommendation. Each task has its
own artifacts (script + JSON + parquet); see "Files in this
directory" at the bottom.

## TL;DR

- **The TRUE corpus is dominated by a single simple pattern.** 62.5%
  of all 10,676 explicit-implication proofs are "apply one hypothesis
  to a permutation of the goal's input variables" (`λ x y => h x y x`
  and equivalents). The generator for that majority is a
  structured-prediction model that emits a short permutation tuple —
  much smaller than a sequence-generation transformer.
- **The remaining 38% splits cleanly into three buckets**, each with
  its own generator strategy: rewrite chains (~15%, sequence
  generation over a small step set), Vampire/Egg machine output
  (~22%, delegate to the upstream tools), and a tiny diverse
  remainder (~0.5%).
- **The FALSE corpus is more diverse at the source-theorem level
  than the plan assumed** (74% lifting-family proofs, only 6.3%
  Fin-n decide), but **the Stage 2 judge accepts only Fin-n matrix
  counterexamples** — so the generator's output format is uniform
  and simple regardless of how the source proofs are structured.
  FALSE generation reduces to "find a small counterexample magma"
  — search/brute-force is the natural baseline.
- **The v1.0 embeddings cover most of the picture.** Goal-entry
  states (variable substitution) and intermediate equations
  (rewrite chains) are exactly what the JEPA encoder represents.
  Only the Vampire/Egg long-tail needs a richer representation,
  and the recommendation there is to delegate to those tools
  rather than encode their state.

## Headline numbers

| Metric | Value |
|---|---|
| ETP Lean toolchain | `leanprover/lean4:v4.20.1` |
| Stage 2 Lean toolchain | `leanprover/lean4:v4.30.0-rc2` |
| Registry entries (`full_entries.json`) | 12,373 |
| ↳ implication proofs | 10,674 |
| ↳ facts entries (counterexamples) | 1,698 |
| Explicit TRUE proofs (extracted from source) | 10,676 |
| FALSE counterexample theorems (source-level) | 394 |
| Total .lean files in ETP | 1,298 (162K lines) |
| Total tokens in TRUE corpus | 1,092,548 |
| Vocabulary size (raw) | 3,155 |
| Vocabulary size (after normalization) | 2,059 |
| Tokens needed for 95% coverage | 119 raw / 102 normalized |
| Median proof length (tokens) | **10** |
| Mean proof length (tokens) | 102.3 |

## 1. FALSE proofs: uniform template or varied?

At the **source-theorem level (394 distinct counterexample
theorems)**: varied. Distribution:

  - lifting-family (`LiftingMagmaFamily.establishNonimplication`): 74.6%
  - tactic block (`by ...`): 14.2%
  - angle-tuple `⟨Fin n, ⟨op⟩, by decide⟩`: 6.3%
  - angle-tuple `⟨ℕ, ⟨op⟩, ...⟩` (infinite carrier): 3.6%
  - angle-tuple Bool / other: ~1.3%

At the **registry-entry level (1,698 entries)**: the 394 source
theorems expand to 1,698 entries because each `LiftingMagmaFamily`
theorem registers one entry per refuted equation. So the corpus
covers 58,963 FALSE pairs via 394 source theorems plus what the
missing `Generated/VampireProven/Conjectures.lean` adds (216
entries the file isn't on disk for).

**But the Stage 2 judge wants only Fin-n matrix counterexamples**
(`finOpTable` parses `"[[0,1],[1,0]]"` into `Magma (Fin n)`). So
the generator's target format is uniform — just an n×n integer
matrix — regardless of how the source proofs are structured.

**Implication for the FALSE generator:** it's a SEARCH problem,
not a sequence problem. Find an n×n magma where the hypothesis
equation holds but the goal equation fails. Brute force over small
`Fin n` magmas is the natural baseline; the v1.0 embeddings could
help pick a good starting magma but aren't strictly needed.

(Infinite-carrier counterexamples in the corpus — the ~3.6%
angle-tuple-Nat cases — can't be submitted to Stage 2 directly.
For equations whose only known counterexample uses an infinite
carrier, the generator may have to give up and submit `verdict:
true` or skip the problem.)

## 2. TRUE proofs: structural breakdown

Out of all 10,676 implication proofs, classified by **semantic
structure** (the syntactic term vs tactic split is mostly an
artifact of which ETP automation produced the proof, not a
meaningful structural distinction — see `orientation_notes.md`):

| Category | % | What the generator must predict |
|---|---:|---|
| **a — variable substitution** | **61.5%** | A short tuple: which hypothesis, applied to which permutation of input vars |
| a — dot fill (`h ..`) | 0.5% | Same as above, with Lean's underscore-fill |
| a — trivial rfl | 0.5% | Constant: `rfl` |
| b — hypothesis composition | 0.1% | Term with `.symm` / `.trans` / `▸` |
| c — rewrite chain | 15.3% | Short tactic sequence: `rw [...]`, `calc`, `simp_rw` |
| d — automation close | 0.0% | (no `by simp` / `by decide` single-tactic closes survive once the `a` patterns are filtered) |
| e — Vampire/superposition | 14.1% | Uniform template, hundreds of `have eqN := ...` steps |
| e — Egg/combinator | 7.5% | Uniform template, `let v0 := M ...; T (T (...) ...)` |
| e — genuinely diverse complex | 0.4% | Varied, case-by-case |

**62.5% of the corpus reduces to "predict an argument permutation."**
This is the architectural headline — see §4 (architecture
recommendation).

The 22% combined Vampire + Egg fraction is high-volume but very
uniform within itself — these are machine output from a specific
upstream tool, not human-written proofs with a wide structural
range. Generator architecture for these: don't try to learn them;
delegate to the upstream tool.

## 3. Proof-shape coverage (template selector vs sequence generator)

Exact-match clustering on the normalized proof bodies (5 levels:
whitespace, variable renaming, equation refs, hypothesis names,
implicit-arg numbers — all five give the same cluster count
because the corpus is already variable-name-consistent):

| Coverage | Templates needed |
|---|---:|
| 50% | ~12 |
| 80% | 601 |
| 95% | 2,203 |
| 99% | ~3,000 |

Total distinct clusters: 2,736.

**The plan's "top 10-20 templates cover 80%+" criterion is not met.**
Top 10 covers 48.4%, top 20 covers 63.1%.

But the top 10 clusters are all **permutation variants** of the same
shape (`λ v1 v2 [v3] => h1 <permutation of vars>`):

```
705   λ v1 v2 => h1 v1 v2 v1
662   λ v1 v2 => h1 v1 v2 v2
646   λ v1 v2 => h1 v1 v1 v2
618   λ v1 => h1 v1 v1
589   λ v1 v2 v3 => h1 v1 v2 v3 v3
490   λ v1 v2 v3 => h1 v1 v2 v3 v2
430   λ v1 v2 v3 => h1 v1 v2 v2 v3
418   λ v1 v2 v3 => h1 v1 v2 v1 v3
324   λ v1 v2 v3 => h1 v1 v2 v3 v1
284   by repeat intro repeat rw [<-h1]
```

The first nine clusters are nine distinct permutation tuples in the
same "apply h to a permutation of n vars" shape. A model that
emits `(n_vars, permutation_tuple)` covers all of them — and any
unseen permutation — with O(1) output size.

So the architecture is **not** a template classifier with 600+
classes. It's a structured-prediction model whose output is a
short integer tuple (a permutation), trainable on the
6,567-instance (a) sub-corpus. The "templates" emerge from the
permutation space rather than being enumerated.

## 4. Vocabulary

|  | Vocab size | Tokens for 95% | Tokens for 99% |
|---|---:|---:|---:|
| Raw bodies | 3,155 | 119 | 418 |
| After N4 normalization | 2,059 | 102 | 210 |

Top tokens (raw): `(`, `)`, `◇`, `X0`, `,`, `h`, `x`, `:=`, `X1`,
`-`, `:`, `.`, `y`, `have`, `9`. Parens dominate (~24K each),
then the magma operator, then variable names.

For a sequence-generation transformer trained on this corpus, a
500-token vocab covers 99%+ of usage with room for new tokens.
For the structured-prediction model recommended in §3, the output
space is `(hypothesis_id × permutation_tuple)` and the vocab
question is moot — the structured output uses no vocab at all
beyond the input-vars-indexed permutation.

## 5. Proof length

Token-level lengths across 10,676 proofs:

| Percentile | Tokens |
|---|---:|
| Min | 1 |
| p25 | 5 |
| **median** | **10** |
| Mean | 102.3 |
| p75 | 12 |
| p90 | 33 |
| p95 | 488 |
| p99 | 1,222 |
| Max | 12,209 |

**Bimodal.** Most proofs (median 10) are tiny. The long tail
(p95 = 488, p99 = 1,222) is Vampire/Egg machine output. Half the
corpus is under ~10 tokens and structurally simple; the top 5%
exceeds 488 tokens and is auto-generated.

For the generator architecture: a short-output fast path (capacity
for ~50 tokens) handles the bulk; the long tail is delegated and
needn't be modeled.

## 6. Intermediate-state extraction

Full assessment in `intermediate_state_feasibility.md`. Headline
table:

| Proof category | Corpus % | Intermediate states needed? | v1.0 fit |
|---|---:|---|---|
| variable_substitution + trivial_rfl + dot_fill | 63.5% | No (single-term construction) | trivial |
| rewrite_chain | 15.3% | Yes — they're equations | direct |
| vampire_superposition | 14.1% | Yes — but with Skolems + multi-hyp contexts | poor |
| egg_combinator | 7.5% | Yes — term-construction sites | poor |
| hypothesis_composition | 0.1% | Borderline | borderline |
| complex (genuine diverse) | 0.4% | Varies | varies |

For the 78% of the corpus where intermediate states are nonexistent
or are themselves equations, the v1.0 embeddings (4694 × 768
parse-tree embeddings) handle it natively. The 22% with richer
intermediate states (Vampire + Egg) doesn't fit v1.0; recommendation
is to delegate that 22% to upstream tools rather than extend the
embedding scheme to cover their goal contexts.

LeanDojo compatibility with the ETP Lean 4.20.1 toolchain should be
verified with a small extraction (~2-4 hours of effort) before
committing to it as the corpus-replay tool. A custom Lean 4
extractor via `Lean.Elab.tactic.getGoals` is a viable fallback.

## Architecture recommendation

Given the survey findings, the proof generator should be a
**three-track ensemble**, sized by corpus coverage:

### Track 1 — Permutation predictor (62.5%+ of TRUE corpus)

A small structured-prediction model on the v1.0 embeddings.
Inputs:
- v1.0 embedding of E1 (the hypothesis equation)
- v1.0 embedding of E2 (the goal equation)
- Number of input variables in E2 (1-4 in this corpus)

Output: a permutation tuple `(p_1, ..., p_k)` where `k` is the
number of arguments E1's hypothesis takes, and each `p_i ∈ {1..n}`
selects an input variable. Wrapped in `by exact fun v_1 .. v_n =>
h <permutation>` for Stage 2 submission.

Scale: tiny. Even with a generous embedding head (a few hundred
hidden units → softmax over `n^k` possible permutations for small
`n`, `k`), the model fits on the v1.0 setup easily. Training data:
the 6,000+ variable-substitution proofs in the corpus.

### Track 2 — Short tactic-sequence generator (15% rewrite chains)

A small transformer over a Lean-tactic vocab (~100 tokens cover
95%) emitting `rw [...]` / `calc` / `simp_rw [...]` sequences.
Conditioned on:
- v1.0 embeddings of E1, E2
- Optionally, intermediate-equation embeddings (predict the next
  rewrite step given the current intermediate equation)

Scale: medium. The output sequence is short (median <20 tokens for
this category; few hundred tokens worst case). Training data:
~1,600 rewrite-chain proofs.

### Track 3 — Upstream-tool delegation (22% Vampire + Egg)

For problems Tracks 1 and 2 don't close, hand off to Vampire or
Egg directly. The ETP demonstrates that ~22% of proofs in its own
corpus are output by these tools — so a generator that calls them
inherits that capability rather than learning to mimic their
output. Practically: a wrapper that invokes Vampire on the
problem, takes its proof script, and either submits it or wraps
it into a Stage-2-compatible certificate.

### FALSE side

A search-based brute force over small `Fin n` magmas (`n` up to
some budget, e.g., 4-8). v1.0 embeddings could prioritize search
order — e.g., for hard cases, prefer magmas similar to ones that
worked on related equations — but the core algorithm is search,
not generation.

### What this looks like in scope-of-work terms

- **Track 1 + FALSE search** is the MVP: covers ~62% TRUE + most
  FALSE. Implementable in days, not weeks. Probably matches or
  beats any prior solver on the Stage 2 normal-difficulty set.
- **Track 2** is the second wave: rewrite chains, another ~15% of
  TRUE coverage. A few weeks of work.
- **Track 3** is glue + tool wrappers, not a learned model.
  Whether to invest there depends on whether the Stage 2 hard
  set requires Vampire-class proofs.

The plan's framing of "template-selector vs sequence-generator" is
unnecessarily binary — the corpus supports both, in different
proportions, and the best architecture is a hybrid.

## Files in this directory

| File | Source task | What's in it |
|---|---|---|
| `README.md` | this synthesis | the report you're reading |
| `orientation_notes.md` | Setup | repo layout, version skew, initial findings |
| `inventory.json` | Task 1 | full-corpus counts (10,674 implications, 1,698 facts) |
| `false_characterization.json` | Task 2 | FALSE template distribution |
| `true_characterization.json` | Task 3 | TRUE structural categories + permutation-shape distribution |
| `clustering_results.json` | Task 4 | normalized-template cluster counts and coverage curve |
| `tokenization_results.json` | Task 5 | vocab size, token-frequency distribution, proof-length percentiles |
| `intermediate_state_feasibility.md` | Task 6 | per-category assessment of intermediate-state extraction + v1.0 fit |

Reproducers in `scripts/`:

- `scripts/survey_inventory.py` (Task 1)
- `scripts/survey_false_proofs.py` (Task 2)
- `scripts/survey_true_proofs.py` (Task 3)
- `scripts/survey_proof_clustering.py` (Task 4)
- `scripts/survey_tokenization.py` (Task 5)

Per-proof corpora (gitignored, regenerable):

- `data/processed/proof_corpus/facts_corpus.parquet` (394 FALSE proofs)
- `data/processed/proof_corpus/true_corpus.parquet` (10,676 TRUE proofs)
- `data/processed/proof_corpus/normalized_proofs.parquet` (TRUE proofs at 5 normalization levels)
