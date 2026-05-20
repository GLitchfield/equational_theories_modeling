# Task 6 — Intermediate-State Extraction Feasibility

Assessment of whether ETP proofs can be replayed through Lean to
capture goal states between tactics, and whether those intermediate
states are compatible with the v1.0 embeddings. Feasibility-only — no
actual replay is run here.

## Toolchain status

| Component | Version | Source |
|---|---|---|
| ETP Lean toolchain | `leanprover/lean4:v4.20.1` | `/home/gavin/git/equational_theories/lean-toolchain` |
| ETP mathlib | `v4.20.0` | `/home/gavin/git/equational_theories/lakefile.toml` |
| Stage 2 Lean toolchain | `leanprover/lean4:v4.30.0-rc2` | `/home/gavin/git/equational-theories-lean-stage2/lean-toolchain` |

**There's a 10-point version gap between the ETP corpus and the Stage
2 judge.** This is unlikely to break most term-mode proofs (the bulk
of the corpus) — Lean's term elaboration is fairly stable across
minor versions — but tactic-mode proofs may need porting. Any
intermediate-state extraction would need to run against whichever
toolchain is available; the ETP toolchain is the natural choice for
extracting from the existing corpus.

The ETP repo has a working `lakefile.toml` declaring mathlib as a
git dependency. A standard `lake build` should compile the corpus,
though it's a large build (1,298 .lean files, 162K lines).

## LeanDojo compatibility

LeanDojo (the standard tool for goal-state extraction from Lean
proofs) is compatible with Lean 4.x in principle, but compatibility
with specific versions varies:

- LeanDojo's mainline ([github.com/lean-dojo/LeanDojo](https://github.com/lean-dojo/LeanDojo))
  targets Lean 4 and gets updated for recent mathlib releases.
- Lean 4.20.x is supported as of LeanDojo's recent releases — but
  this should be verified by attempting a small extraction on the
  ETP repo before committing to it as the corpus-extraction tool.
- Alternatives: `Lean.Elab` directly (write a Lean 4 program that
  loads the corpus and extracts goals via `tactic.getGoals`).
  More work to set up but no third-party dependency.

**Recommendation:** before any serious work on intermediate states,
spend ~2-4 hours getting LeanDojo to extract one proof from the ETP
corpus end-to-end. If it works cleanly, use it. If it doesn't,
write a custom extractor in Lean.

## What intermediate states look like, by proof category

Drawing on Task 3's structural classification of the 10,676 TRUE
implication proofs:

### Category (a) — variable substitution (62.5% of corpus)

These proofs are single-term constructions like `λ x y => h x y x`.
There are **no intermediate states** — the proof is a function
expression that's checked by elaboration in one step. LeanDojo
(or any tactic-state replay tool) would yield a single state:
the goal at entry. The proof "step" is the whole term.

For generator training, these proofs supervise on `(goal_at_entry,
hypothesis_application_term)` pairs — no sequence modeling, no
intermediate-state-conditioned prediction.

**v1.0 embedding fit: trivial.** The goal at entry IS an equation
(`Equation_Y G`); v1.0 already encodes equations.

### Category (c) — rewrite chain (15.3% of corpus)

These use `rw`, `rwa`, `calc`, `conv`, `simp_rw`, `nth_rw`. Each
step rewrites the goal using an equation, producing a new equation
as the goal. So **intermediate states ARE equations**, just like
the corpus's input/output equations.

A `calc` proof like:

```lean
calc x ◇ y = (h ▸ ...) = ... = z ◇ w
```

produces intermediate states `x ◇ y = e1`, `e1 = e2`, ..., `eN = z ◇ w`,
each of which is an equation in the same algebraic language as the
4,694 ETP equations.

**v1.0 embedding fit: direct.** Intermediate-state equations can be
encoded by the same context encoder that produced the v1.0
embeddings. The generator can condition on the embedding of the
current intermediate equation when predicting the next rewrite step.

### Categories (e_vampire / e_egg) — auto-generated long proofs (21.6% combined)

Vampire-superposition proofs introduce Skolem variables (`sK0`,
`sK1`, ...) and chained `have eqN : T = T'` hypotheses. Intermediate
states are **proof goals with auxiliary hypotheses** — not just
equations. A typical state:

```
sK0 sK1 : G
nh : ¬ Equation_R G
eq9 (X0 X1 : G) : (X0 ◇ X1) ◇ X1 = X0
eq10 : sK0 ≠ (sK0 ◇ (sK1 ◇ sK1)) ◇ sK1
eq11 (X0 X1 : G) : ...
⊢ False
```

The state includes Skolemized terms, multiple hypotheses with
universally-quantified equations, and the residual goal.

**v1.0 embedding fit: poor.** The v1.0 encoder takes an equation's
parse tree as input. It can't represent a full Lean goal context
(multiple hypotheses, Skolem variables, residual goal). Extending
the embeddings to handle these would need either a richer encoder
(treating goal contexts as graphs of multiple equations + Skolem
typing) or — more pragmatically — using a different model entirely
for this 22% of the corpus.

Egg-combinator proofs use `let v0 := M x y; ...; T (T (h ..)
(C ...))` constructions where intermediate states are
**term-construction sites**, each annotated by which combinator
applies. These are similarly outside the v1.0 representation.

### Category (b) — hypothesis composition (0.1%)

Tiny. Term proofs with `.symm`, `.trans`, `▸`. Intermediate states
are partial term reconstructions. v1.0 fit is borderline but the
category is too small to matter.

### Category (e_complex) — diverse complex (0.4%)

Tiny. Genuinely diverse cases (`have` chains, named-lemma
applications). Wouldn't try to handle generically.

## Summary

| Category | Corpus % | Intermediate states needed? | What they are | v1.0 fit |
|---|---:|---|---|---|
| Variable substitution | 62.5% | No | (just the entry goal) | trivial |
| Rewrite chain | 15.3% | Yes | equations | direct |
| Vampire superposition | 14.1% | Yes | goal contexts with Skolems + many hypotheses | poor |
| Egg combinator | 7.5% | Yes | term-construction sites | poor |
| Hypothesis composition | 0.1% | Borderline | partial terms | borderline |
| Diverse complex | 0.4% | Yes (varies) | varies | varies |
| Dot-fill / trivial rfl | 1.0% | No | trivial | trivial |

**Combined:** ~78% of the corpus has either no intermediate states or
intermediate states that are equations, both of which the v1.0
embeddings already handle. ~22% (Vampire + Egg) have intermediate
states that the v1.0 embeddings can't represent; for those, the
generator either needs a richer encoder OR should delegate to the
upstream automation tools (Vampire, Egg) directly.

**Conclusion:** intermediate-state extraction is **feasible without
extending the v1.0 embeddings**, provided we accept the corpus
partition implied by Task 3 — handle 78% of proofs with a generator
that uses v1.0 embeddings (over entry-goal + intermediate-equation
states for rewrite chains), and delegate the remaining 22% to
external tools rather than learning to emit their proof shapes.

**Recommendation:**
- Don't bake intermediate-state extraction into the first
  generator pass. The 62.5% variable-substitution majority needs
  no intermediate states; build that first.
- Add intermediate-state-conditioned prediction in a second pass,
  for the 15% rewrite-chain category. v1.0 embeddings handle this
  directly.
- Punt on intermediate-state modeling for the Vampire/Egg 22%;
  delegate to the upstream tools.
- Verify LeanDojo compatibility with ETP's Lean 4.20.1 before
  committing to it as the extraction backbone; have the custom-
  Lean-extractor option as a fallback.
