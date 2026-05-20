# ETP proof structure: sparse direct proofs + closure layer

## Question

How are the proofs structured in the ETP project? Is there an implication graph backed by a relatively small list of direct proofs, or one direct proof per implication pair?

## Answer

**Sparse direct proofs plus a Lean-computed closure.** Only a small fraction of the 22M off-diagonal implication pairs has an explicit `theorem Equation_X_implies_Equation_Y` declaration in the repo; the rest are derived by transitive closure of those explicit edges (and by bulk refutations from finite-magma `Facts` records on the FALSE side).

## Direct theorem declarations

Counted across the ETP repo (`~/git/equational_theories/`) via `grep -rh '^theorem Equation.*implies_Equation' equational_theories --include='*.lean' | wc -l`:

| source | count | format |
|---|---|---|
| `Generated/SimpleRewrites/theorems` | 6,305 | `rw [h, ← h]` rewrite chains |
| `Generated/VampireProven` | 1,507 | Vampire superposition (`have eqN ... := superpose ...`) |
| `Generated/TrivialBruteforce/theorems` | 1,497 | trivial bruteforce |
| `Generated/MagmaEgg` | 832 | term-mode Egg combinators (`T/S/R/C/M`) |
| `Subgraph.lean` | 171 | manual structured proofs |
| Various other manually-proved files | ~780 | mix |
| **Total** | **~11,094** | |

This is the direct-proof set: ~11K `theorem` declarations across the whole repo, each one explicitly proving a single `Equation_X_implies_Equation_Y` statement.

The proof-corpus survey (`outputs/analyses/proof_survey/`) extracted a curated subset of 10,676 of these for its categorization — close to the total, the small gap reflecting the survey's specific inclusion criteria.

## Refutation declarations

Counted from `FinSearch`, `FinitePoly`, `All4x4Tables`, `Confluence`, etc.: each file declares a `Facts G [<satisfied equations>] [<unsatisfied equations>]` for a single finite magma `G`. Each such Facts record certifies a structured set of non-implications all at once: if `G` satisfies `E_a` but not `E_b`, then `E_a` does not imply `E_b`, and `G` is the witness.

A handful of well-chosen finite magmas refutes a large share of the 13.86M GT FALSE pairs. The refutation side is heavy-headed (few magmas, many witnessed non-implications) — the inverse pattern of the TRUE side.

## The closure machinery

`equational_theories/Closure.lean` defines:

```lean
inductive Outcome
  | explicit_proof_true       -- has a written-down theorem
  | implicit_proof_true        -- derivable from proven theorems (transitive)
  | explicit_conjecture_true   -- claimed but not Lean-verified
  | implicit_conjecture_true   -- derivable from theorems + conjectures
  | unknown
  | implicit_conjecture_false
  | explicit_conjecture_false
  | implicit_proof_false       -- derivable from proven refutations
  | explicit_proof_false       -- has a direct disproof
```

The distinction `explicit` vs `implicit` is exactly the closure distinction: `explicit` means a written `theorem` line; `implicit` means "follows from the explicit set by transitivity / refutation propagation."

The repo's dashboard and the `export_raw_implications_5_9_2026.csv` ground truth are the resolved-to-`±3/±4` form of running this closure over the entire ~11K explicit-edge set plus the refutation Facts records. By the time a pair shows up in our ground truth as code `+3` or `+4`, the closure has been computed.

## The 22M-pair picture

| stratum | count | how it's certified |
|---|---|---|
| TRUE explicit (has a `theorem` line) | ~11,094 | direct proof — what Proof Investigation extracts from |
| TRUE implicit (transitive closure) | ~8,162,491 | derived from explicit edges; no written proof in the repo |
| FALSE (refuted by a finite magma) | ~13,855,357 | witnessed by `Facts` records, not direct disproofs |

So **~99.87% of the GT TRUE pairs have no explicit proof in the ETP repo.** They're certified by the closure module having derived them from the explicit 11K. Conceptually they're "the equation holds because of `theorem A_implies_B; theorem B_implies_C` and Lean can chain them," but the chain is implicit — Lean's closure code knows the derivation exists; the chain isn't written down.

## Implication for Proof Investigation

This sharpens the scope materially. Within-ceiling residual breaks down by stratum (within-ceiling sample, n=19,982):

| stratum | n in sample | direct proof in repo? |
|---|---|---|
| `e_vampire_superposition` | 350 | yes |
| `e_egg_combinator` | 186 | yes |
| `e_complex_construction` | 7 | yes |
| `c_rewrite_chain` | 123 | yes |
| `a_variable_substitution` + `a_trivial_rfl` | 805 | yes |
| **`uncategorized_transitive`** | **18,511 (92.6%)** | **no — only the closure derivation exists** |

Proof Investigation as currently scoped (722 Vampire + Egg + complex eligible pairs) studies the **direct-proof subset** of the within-ceiling residual: 3.6% of within-ceiling pairs by count, but the only segment where an extracted "intermediate terms" set actually exists as a proof artifact in the repo.

For the 92.6% transitive segment, there is no intermediate-term extraction to do — the proof "is" "by closure", and the closure machinery's certification doesn't expose chain structure. To study those pairs we'd ask a different question entirely: **what's the shortest chain through the explicit-edge graph, and how often is the chain accessible to a successor method?** That's graph-traversal, not term-extraction.

The framing change is worth being explicit about in the eventual synthesis: "what does a successor method need to produce" splits into two distinct sub-questions depending on stratum.

| pair stratum | "what would a successor method need to produce" |
|---|---|
| direct-proof (722 eligible) | a proof artifact like the existing Lean proof — intermediate equations, term chains, combinator expressions |
| transitive (~600K residual after density filter) | a chain through the existing explicit-edge graph, OR a direct proof shorter than the chain |

## Where embeddings could land — refined

This re-reads the coverage-measurement-synthesis's "embeddings-as-heuristic" framings:

- **Triage layer** (predict method-class per pair): unchanged — still feasible, still useful regardless of stratum.
- **Direct-proof generator** (the "Lean output mimicking Vampire / Egg" angle): operates on the 722-pair direct-proof subset of the residual. Coverage upper bound is bounded by the absolute size of that subset, not by the within-ceiling residual at large.
- **Chain finder over the explicit-edge graph** (new role surfaced by this note): operates on the 18,511 transitive pairs by traversing the ~11K explicit edges. This is a graph-learning problem, not a term-generation problem. May be the larger opportunity if the chains are short enough to learn.
- **Search heuristic over safe rewriting**: still ruled out — depth/wall plateaus settled this.

## Open question for after Proof Investigation

Whether the 18,511 transitive within-ceiling pairs have *short* chains through the explicit-edge graph (in which case a chain-finder is the natural successor method) or *long* chains (in which case the closure derivation isn't easily mimicked and direct-proof generation has to carry more weight).

Cheaply answerable with a graph-shortest-path computation over the 11K explicit edges. Not part of Proof Investigation's current scope; flag for separate investigation.

## Inputs verified

- `~/git/equational_theories/equational_theories/` — ETP Lean root.
- `equational_theories/Closure.lean` — the Outcome enum + closure logic.
- `Generated/{VampireProven, MagmaEgg, SimpleRewrites, TrivialBruteforce}` — direct-proof files.
- `Generated/{FinSearch, FinitePoly, All4x4Tables}` — refutation `Facts` records.
- `data/processed/proof_corpus/true_corpus.parquet` — survey's per-pair category labels.
- `data/raw/etp_export/export_raw_implications_5_9_2026.csv` — ground truth, the closure-resolved 4694×4694 outcome table.
