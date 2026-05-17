# ETP + Stage 2 — Orientation Notes

Setup pass for `proof_corpus_survey_plan.md`. Records where things
live, the Lean versions in play, and the initial high-level
findings that justify the survey tasks. Not the final report — just
enough scaffolding for Tasks 1-6 to start from.

## Repos

| Repo | Path | Lean toolchain |
|---|---|---|
| **ETP main** | `/home/gavin/git/equational_theories/` (already on disk; the existing `data/raw/etp` symlink points only at its `data/` subdir) | `leanprover/lean4:v4.20.1`, mathlib `v4.20.0` |
| **Stage 2** | `/home/gavin/git/equational-theories-lean-stage2/` (cloned shallow, depth 1) | `leanprover/lean4:v4.30.0-rc2` |

**Lean version mismatch** between the two repos is the first
nontrivial finding. The ETP proofs were written against Lean
4.20.1; the Stage 2 judge runs Lean 4.30.0-rc2. Any generated proof
must compile against the Stage 2 toolchain, not the ETP one. Most
ETP proofs likely port forward without changes (term-mode λ-proofs
don't usually break across Lean point versions), but the version
gap is worth flagging for Task 6 (intermediate-state feasibility)
and for the eventual generator (the corpus is in 4.20.1; the target
is 4.30.0-rc2).

## ETP proof registry

`/home/gavin/git/equational_theories/full_entries.json` is a single
JSON file (4.1 MB) indexing every named theorem in the project. **12,373
total entries**:

| Variant | Count | What it is |
|---|---:|---|
| `implication` | 10,674 | Explicit `Equation_lhs → Equation_rhs` Lean proof |
| `facts` | 1,698 | A counterexample magma with `{satisfied: [E_a], refuted: [E_b, E_c, ...]}` |
| `unconditional` | 1 | `Equation1` (reflexive `x = x` over any magma) |

Each entry records `name`, `filename`, and `line`. So the registry
plus the source files give the full corpus addressable by
(implication_pair → file:line). Used by Tasks 1, 2, 3.

Per-status: 12,119 proven, 254 unproven.

### Counts that fall out

- **Directly proved implications: 10,674.** (NOT the 4,824 from the
  McKenna Spine Isolation paper — that number was the *equivalence-
  class* transitive reduction, while the ETP proves at the per-pair
  granularity. Methodology Resolution 1 in the plan called this out
  in advance.)
- **FALSE pairs covered by explicit `facts` entries: 58,963.** This
  is `Σ len(refuted) for fact in facts`. Each `facts` entry can refute
  many equations with the same counterexample magma.
- **Corpus organization: 80 source files contain implication proofs**,
  heavily concentrated in `equational_theories/Generated/SimpleRewrites/`
  (top file `Rewrite_zx.lean` has 705 proofs). The `Generated/` tree
  is machine-generated; `Subgraph.lean` and `ManuallyProved/` contain
  human-curated proofs.

### Implication vs full-corpus TRUE/FALSE

Worth noting alongside the inventory: the full corpus has **8.17M
TRUE pairs** and **13.86M FALSE pairs** (per the pair-table parquet
the JEPA work was built on). So:

- TRUE: 10,674 explicit ÷ 8.17M total = 0.13% directly proved; the
  rest are derived by transitivity.
- FALSE: 58,963 covered by `facts` entries; the rest (~13.8M) are
  derived from the same counterexamples via "if E_a refutes E_b
  and E_c → E_a, then E_c refutes E_b" closure.

## Proof shape: term-mode vs tactic-mode

Quick scan over all 1,298 `.lean` files under `equational_theories/`
counting theorems matching `theorem \w+_implies_\w+ ... := <body>`:

| Mode | Count | Fraction |
|---|---:|---:|
| Term mode (`:= λ ...` / `:= fun ...` / `:= rfl` / ...) | 7,694 | 69% |
| Tactic mode (`:= by ...`) | 3,395 | 31% |

(Total 11,089 — slightly more than the 10,674 official implications;
the extra ~400 are helper lemmas in the same naming scheme. Task 1
will reconcile against `full_entries.json` for precise counts.)

**This is a significant survey finding before we even start.**
Plan's Task 3 expects to classify proofs as "single-tactic close /
short tactic sequence / rewrite chain / other" — but ~69% of the
corpus is **term-mode**, which has no tactics at all. It's a
function expression, often a single λ. The classification needs a
fifth category: **term-mode proof** (or term-mode proofs need to
be analyzed by their λ-body shape rather than tactic structure).

Sample term-mode proofs from `Generated/SimpleRewrites/Rewrite_zx.lean`:

```lean
theorem Equation12_implies_Equation10 (G : Type*) [Magma G] (h : Equation12 G) : Equation10 G :=
  λ x y => h x y x

theorem Equation19_implies_Equation13 (G : Type*) [Magma G] (h : Equation19 G) : Equation13 G :=
  λ x y => h x y x
```

These are extremely regular: a λ-abstraction of the goal's variables
that applies the hypothesis with some permutation/reuse of arguments.
For the proof generator, this means the term-mode 69% reduces to
"predict an argument permutation given the type signature" — much
simpler than sequence generation.

Sample tactic-mode proofs from `Subgraph.lean`:

```lean
theorem Equation387_implies_Equation43 ...
theorem Equation1571_implies_Equation40 ...
```

(Bodies not yet inspected — Task 3 will categorize.)

## Stage 2 submission format

The judge expects, per problem, one of:

```json
{"verdict": "true",  "proof": "<tactic body, no imports, no theorem statement>"}
{"verdict": "false", "counterexample_table": [[0,1],[1,0]]}
```

Notable:

- **TRUE submissions are a tactic body**, NOT term mode. The judge
  wraps the body in a `theorem ... := by <body>` shell. So even
  though 69% of the ETP corpus is term mode, the generator must
  emit tactic-mode proofs. Either convert term-mode proofs at
  training time, or train the generator to produce tactic-mode
  equivalents (e.g., `exact λ x y => h x y x` is the tactic-mode
  one-liner that wraps a term).
- **FALSE submissions are a Fin-N operation table** — a list of
  lists of integers, where entry `[i][j]` is the result of `i ◇ j`.
  Implicitly an n×n integer matrix; n is determined by the matrix
  size. So the FALSE generator's output space is `(n, n × n int)`
  — pick a magma size, pick the table.
- The Stage 2 problem set uses a **larger law list** (~62K order-5
  laws in `examples/problems/eq_size5.txt`) than the ETP corpus
  (4,694 order-≤4 laws). The v1.0 embeddings only cover the ETP
  4,694. Generating proofs for the order-5 laws will require
  embedding equations the JEPA never saw.

Judge implementations live in `judge/`:
`JudgeDecide`, `JudgeFinOp`, `JudgeMagma`, `JudgeSupport`, plus
`verify.py`. Demo solvers are in `examples/solo/demos/`
(`baseline`, `opnorm`, `twophase`).

## Things to flag for later tasks

- **Term-mode dominance** changes the framing of Task 3 (proof
  classification). Add a 5th category "term-mode" or split the
  analysis: term-mode proofs analyzed by λ-body shape; tactic-mode
  proofs analyzed by tactic sequence. Probably want both.
- **Lean version skew** is something Task 6 needs to confirm: can
  ETP proofs replay against Stage 2's toolchain at all, or do they
  need porting?
- **Tactic-vs-term gap for the generator.** If the corpus is 69%
  term-mode but the judge wants tactic-mode, we either need a
  term→tactic converter as a preprocessing step OR a generator that
  natively emits tactics. The corpus then becomes a smaller pool
  of tactic-mode examples (3,395) for direct training, or a larger
  pool (10,674) if we use a conversion step.
- **Out-of-corpus generalization.** Stage 2's law list extends to
  order 5 (~62K laws). The JEPA was trained on order-≤4 (4,694
  laws). Embedding the order-5 equations requires either re-running
  the JEPA on the larger corpus or accepting that the embeddings
  may be less discriminative for the larger law family.

Next: Task 1 (full-corpus inventory — formalize the counts above,
write the inventory script, save inventory.json).
