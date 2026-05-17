# Changelog

Deviations from the active plan and other notable project decisions, in reverse-chronological order. The plan itself is updated to reflect the new reality; entries here explain *why* a deviation was made and on what evidence, so a fresh session can understand the plan's current state without digging through git history. The active plan is currently `rewrite_chain_study_plan.md`; prior plans are archived.

---

## 2026-05-17 — Proof decoder phase 1 wrapped; rewrite_chain_study_plan active

Phase 1 of the proof decoder is complete. The research question
("do the JEPA embeddings encode the proof witness, not just
TRUE/FALSE?") was answered cleanly:

  Both-novel σ-exact:
    Majority baseline         : 0.000
    Random-embedding control  : 0.056
    Linear probe (68K params) : 0.576
    Transformer decoder       : 0.963   (+90.7pp vs control)

Mechanism localized via the linear probe + per-position
decomposition: σ positions are approximately linearly independent
(per-position accuracy ≈ joint^(1/k) within ±1pp for arities 3-5),
so the decoder's 40pp lift over linear is specifically about
modeling inter-position consistency — the structural-walk
invariants the matcher exploits, not per-position discriminability.

Mechanism reframing (saved as a feedback memory:
preservation_vs_discovery): σ is a deterministic function of the
two parse trees (the matcher computes it structurally), so the
finding is preservation, not discovery. The contrastive loss
preserved enough of the equation structure that a downstream
computation on that structure (σ-extraction) remains recoverable
from `(z_e1, z_e2)`. This isn't "the model learned proofs" — it's
"the embedding didn't destroy the structure that lets you compute
proofs." The reframing matters because phase 2 / rewrite chains are
genuinely different: those proofs are NOT a function of the input
equations' structure (search-requiring, multiple valid chains),
so phase 2 is a real second question, not a curriculum extension.

`proof_decoder_plan.md` archived. `rewrite_chain_study_plan.md` is
now active.

The rewrite-chain plan parallels the matcher's: ask whether the
category is deterministically search-solvable (as the matcher
turned out to be for variable substitution) before assuming it
needs a learned model. Ten methodology resolutions locked in
before implementation. Most consequential:

  - **First-order matching is a NEW function**, not a reuse of the
    matcher's structural-match. Rewriting binds rule variables to
    arbitrary subterms (compound included), which the matcher
    explicitly prohibits. Implementation: a recursive walk identical
    in shape to the matcher's, with one rule changed (pattern-var-
    vs-anything binds instead of requiring var-vs-var). New module
    `src/rewrite/matching.py`.
  - **Source full bodies from normalized_proofs.parquet**'s
    `n1_whitespace` column, NOT from true_corpus.parquet (which
    only has `first_80` truncated).
  - **Bidirectional iterative-deepening BFS** with per-side
    `d_max=6` and term-size cap `max(50, 5*max(|lhs2|, |rhs2|))`,
    both CLI-tunable for sensitivity sweeps. Bidirectional saves
    ~10^3x for short chains.
  - **Task 3 (intermediate-state characterization) reads from
    Task 4's engine output** — without LeanDojo, simulating
    rewrites IS the only way to get intermediate states in Python,
    and the engine does that anyway.
  - **Rule-set detection** classifies each `rw [...]` arg into
    `{h, ← h, other_name}`. The fraction that stays within
    `{h, ← h}` is the strongest predictor of search tractability
    alongside the coverage curve — the two decision-driving
    numbers.
  - Search cost reported as **distributions** (median, p95,
    per-depth), not means — the "90% easy at depth 2, hard 10%
    needs depth 6+" shape is the informative picture.

CLAUDE.md and README.md updated to point at the new active plan and
add a proof-decoder recap section (Rule C).

---

## 2026-05-17 — Matcher wrapped; proof_decoder_plan is the new active plan

The deterministic matcher is complete. Coverage: 59.8% of the 10,658
directly-proved implications (6,374), 0.607% of all 8,173,585 TRUE
pairs (49,586), and **0 / 13,855,357 false positives** — the
correctness gate held at full scale. Wall clock: 58.8 s on a single
core in pure Python. Outputs at `outputs/analyses/matcher/` and the
49,586-row `data/processed/proof_corpus/matcher_proofs.parquet`.

`matcher_plan.md` archived. `proof_decoder_plan.md` is now active —
phase 1 of the research decoder, targeting exactly the matcher's
49,586 verified proofs. The research question: do the JEPA
embeddings encode the proof witness (σ), or just TRUE/FALSE?

Seven methodology resolutions locked into the decoder plan before
implementation. Most consequential:

  - 16-token phase 1 vocab (13 proof tokens + 3 special). Drops
    `intro`, `G`, `_` — those are judge-preamble tokens the decoder
    never emits.
  - Pure embeddings-only conditioning. No explicit arity input, even
    as a "safety belt" — probing already showed arity is linearly
    decodable from the embedding at ~99%.
  - Nested metric stack: parse rate -> arity-correct rate ->
    σ-exact-match. Does the failure-attribution work that an
    explicit arity input would do, without compromising the
    embeddings-only claim.
  - Equation-level (not pair-level) train/test split, seed 42. With
    only 4,694 equations and 49,586 pairs, a random pair split would
    leak nearly every test equation into training and let the
    decoder memorize per-equation features. The research question
    requires testing on equations the decoder didn't train on.
  - Random-embedding control with seed 42 — frozen random vectors
    replace the JEPA embeddings, identical decoder. The control
    bounds how much of accuracy is dataset-regularity vs embedding-
    geometry.
  - Phase 2 vocab pre-tokenization done as part of phase 1 setup.

The decoder is the RESEARCH track, parallel to the competition
cascade (matcher → tactic-sequence generator → tool delegation +
FALSE search). The matcher already solves this category exactly;
the decoder's value is the scientific claim and the eventual
extension (phase 2) to rewrite chains where no deterministic
shortcut exists.

CLAUDE.md and README.md updated to reflect the new active plan
(Rule C).

---

## 2026-05-17 — matcher plan: reuse src/data/parser.py instead of writing a standalone parser

The original Methodology Resolution 2 in matcher_plan.md called for a standalone matcher-specific parser, on the premise that the JEPA-era parser canonicalizes variables to `v0/v1/...` and would lose the original names the matcher needs for λ-binding.

On re-reading `src/data/parser.py`, that premise was wrong:
`parse_equation(text)` returns `Equation(lhs, rhs)` with raw
`Var(name)` dataclasses preserving the original variable names.
Variable canonicalization happens later in a separate function
(`canonical_var_ids`) that the GNN calls; the matcher won't call
it. So the existing parser returns exactly what the matcher needs.

Plan updated to reuse `src/data/parser.py` directly. Task 17 (the
standalone parser task) is consequently a no-op — collapsing it
into Phase 2's matcher-algorithm import.

---

## 2026-05-17 — Architecture update: matcher replaces Track 1 of the survey's three-track recommendation

The proof-corpus survey concluded with a three-track architecture
recommendation: (1) learned permutation predictor on v1.0 embeddings
for 62.5% of TRUE, (2) tactic-sequence generator for ~15% (rewrite
chains), (3) upstream-tool delegation for ~22% (Vampire/Egg). Plus
FALSE counterexample search.

`matcher_plan.md` (now active) replaces Track 1 with a
**deterministic first-order matcher**. Same coverage (62.5%), exact
correctness, zero embeddings, zero learning, zero GPU. It's strictly
stronger than the survey's proposed learned model — pure symbolic
computation on parse trees, O(tree size) per pair.

Why this is a strict improvement:

  - The variable-substitution category is defined by an exact
    algorithmic test (does there exist a σ: vars(E1) → vars(E2)
    such that σ(lhs1) ≡ lhs2 ∧ σ(rhs1) ≡ rhs2?). The matcher just
    runs that test. No learning needed.
  - The matcher's output is verifiable by construction — the emitted
    proof is `fun <e2_vars> => h <σ(e1_vars)>` (or its `.symm`
    flip), which type-checks iff σ is correct, which it is by the
    structural-walk invariant.
  - The matcher can be applied to the full 22M-pair corpus in
    seconds. No GPU, no embeddings, no inference loop.

Updated three-track architecture, going forward:

  1. **Deterministic matcher** (62.5% TRUE) — matcher_plan.md
  2. Short tactic-sequence generator (15% TRUE — rewrite chains)
  3. Upstream-tool delegation (22% TRUE — Vampire / Egg)
  + Brute-force FALSE search.

The v1.0 embeddings are not needed for Track 1 anymore. They remain
relevant for Track 2 (where intermediate states are equations the
JEPA encoder handles natively) and as inputs to any future research
decoder. CLAUDE.md and README.md updated to reflect the architecture
change.

Methodology resolutions locked into the plan before implementation
(8 of them, per the standard pattern). Most consequential ones:

  - Standalone matcher-parser (don't reuse the JEPA's
    variable-canonicalizing parser, which would lose the original
    variable names needed for λ-binding).
  - False-positive check on the FULL 13,855,357-pair FALSE corpus
    (not a sample — matching is microseconds per pair).
  - Fixed-length-6 padded substitution format for the ground-truth
    proof dataset (max var count in ETP is 6).
  - 2 orientations, not 4 (flipping both equations = flipping
    neither; provably 2 is complete).
  - Pin the Stage 2 tactic submission format BEFORE mass emission
    by reading the demo solvers and submitting 5-10 test proofs
    end-to-end.

---

## 2026-05-17 — Proof-corpus survey wrapped; plan archived

The proof-corpus survey ran the six tasks specified in
`proof_corpus_survey_plan.md` and produced a characterization report
at `outputs/analyses/proof_survey/README.md`. The plan is archived
(`archive/proof_corpus_survey_plan.md`).

Headline findings, all derived from the 10,676 explicit implication
proofs and 394 source-level counterexample theorems in the ETP:

  - **62.5% of TRUE proofs reduce to "predict an argument permutation"**
    (single hypothesis applied to a permutation of input variables).
    The architecture-deciding finding.
  - **15.3% are rewrite chains.** Intermediate states are equations,
    which v1.0 embeddings already represent.
  - **21.6% combined are uniform machine output** (14.1% Vampire-style,
    7.5% Egg-combinator). Delegation candidates rather than learning
    targets.
  - **FALSE corpus is more diverse at the source level than the plan
    assumed** (74.6% lifting-family proofs, only 6.3% Fin-n decide),
    but **Stage 2's judge accepts only Fin-n matrix counterexamples**,
    so the FALSE generator's target output is uniform. FALSE
    generation reduces to brute-force search over small Fin n magmas.
  - **Plan's "top 10-20 templates cover 80%+" criterion is NOT met**
    (top-20 covers 63%). But the top clusters are all permutation
    variants of the same shape, so a structured-prediction model that
    emits the permutation tuple covers them generatively — no fixed
    template vocabulary needed.
  - **Vocab is small** (119 tokens cover 95% of TRUE-proof usage).
    Proof length is bimodal: median 10 tokens (variable-substitution),
    Vampire/Egg long tail goes to 12K tokens.

Architecture recommendation (three-track ensemble plus FALSE search):

  Track 1 — Permutation predictor for 62.5% of TRUE. Small structured-
            prediction model on v1.0 embeddings. Output: (hypothesis_id,
            permutation_tuple), wrapped as
            `by exact fun v1..vn => h <perm>`.
  Track 2 — Tactic-sequence generator for 15% (rewrite chains). Small
            transformer over a ~100-token Lean tactic vocab.
  Track 3 — Upstream-tool delegation for 22% (Vampire / Egg). Wrap
            those tools rather than learning to mimic their output.
  FALSE   — Brute force over small Fin n magmas, v1.0 embeddings
            optionally guiding search order.

  Track 1 + FALSE search is the MVP. Tracks 2 and 3 are second-wave
  additions.

Methodological note worth keeping: the plan initially framed Task 3
as "single-tactic / short-sequence / rewrite-chain / other" — all
tactic-mode categories. Orientation surfaced that 69% of ETP proofs
are term-mode (no tactics), with the term/tactic split determined
by which ETP automation generated the proof, not by proof
difficulty. The plan was reframed mid-survey (Methodology Resolution
11) to classify by SEMANTIC structure rather than surface mode.
This reframing is the principal reason the survey produced a
clearly architecture-deciding answer instead of a mass of
"unclassified" complex proofs.

Cleanup actions:

  - `proof_corpus_survey_plan.md` moved from the repo root to
    `archive/`.
  - CLAUDE.md updated: added a "Proof-corpus survey recap" section
    with the headlines; removed the "active plan" pointer (no
    active plan right now).
  - README.md updated to match (Rule C): leads with both
    deliverables (v1.0 embeddings + proof survey) side by side.
  - The next plan, when written, will design the Track-1 generator
    based on the survey's recommendation.

---

## 2026-05-16 — JEPA phase wrapped; embeddings promoted to v1.0

The JEPA training phase is **complete**. The deliverable is a stable
per-equation embedding artifact at `outputs/embeddings/v1.0/`,
intended as a frozen input for downstream work. We may revisit if a
downstream task surfaces a specific deficiency, but the v1.0
embeddings are considered good enough to build on.

What's in the v1.0 artifact:

  - `embeddings.npy` — `(4694, 768) float32`, L2-normalized. Gitignored
    (large, regenerable from the checkpoint).
  - `eq_ids.npy` — `(4694,) int32`, ascending eq IDs (1..4694).
    Gitignored.
  - `source_checkpoint.pt` — copy of
    `outputs/checkpoints/skip_h128_seed42_diag/epoch_009.pt` so the
    artifact is fully self-contained. **Committed** (~3 MB).
  - `MODEL_CARD.md` — spec, source, validation numbers, known
    limitations, reproduction recipe. Committed.

Headline numbers (matched 10% pair holdout, seed 42):

  - JEPA distance bal_acc:   0.9841 (0.9840 on the full 22M corpus)
  - Linear probe bal_acc:    0.9530
  - MLP-head bal_acc:        0.9996
  - MLP-head accuracy on always-wrong-trained ∩ holdout: 91.37%
    (JEPA = 0% by definition)

Cleanup actions taken alongside the promotion:

  - `skip_connection_plan.md` and `mlp_classifier_plan.md` moved
    from the repo root to `archive/`. Both are complete; the active
    plan is now `proof_corpus_survey_plan.md` (the next phase).
  - `.gitignore` updated to track `outputs/embeddings/**/*.pt` and
    `MODEL_CARD.md` while still ignoring `.npy` binaries.
  - CLAUDE.md and README.md rewritten to lead with the v1.0
    embedding artifact rather than the active-training framing.
    The JEPA-phase recap is preserved in CLAUDE.md as historical
    context for if we ever return to retraining.

---

## 2026-05-16 — MLP-classifier Phase 2: torch+GPU linear probe instead of sklearn

The plan's Phase 2 called for `sklearn.linear_model.LogisticRegression` on `concat(z_e1, z_e2)`. First-pass implementation OOM'd in WSL — sklearn is CPU-only and lbfgs needs to materialize ~6 GB of train features + ~13.5 GB of holdout features in RAM, plus several internal copies for the optimizer state (~25+ GB working set). WSL caps at ~half host RAM by default.

Switching to a torch `Linear(1536, 1)` trained with `BCEWithLogitsLoss(pos_weight=n_FALSE/n_TRUE)` on GPU. Methodologically equivalent to sklearn's `LogisticRegression(class_weight='balanced')` — both minimize logistic loss on the same data; lbfgs vs SGD converge to the same global optimum on a convex objective. Side benefits: reuses the same training infrastructure as the MLP variants (Phase 3.2), runs in seconds on the 4080 instead of minutes on CPU, no awkward feature subsampling.

The plan's Ablation #2 (verify torch single-layer matches sklearn) is now redundant — the linear probe IS the torch single-layer. Dropping that sub-experiment.

The failed sklearn script is preserved at `trash/mlp_classifier_sklearn_attempt/run_linear_probe_pair.py` (no committed code path, just for posterity if a CPU fallback is ever needed).

---

## 2026-05-16 — MLP-classifier plan: reuse `level1_holdout_seed42.parquet` instead of creating a new split file

The `mlp_classifier_plan.md` originally specified creating `data/splits/mlp_pair_split_seed42.parquet`. At Phase 1.2 (build the split) we found `data/splits/level1_holdout_seed42.parquet` already exists and is bit-for-bit equivalent — built by `scripts/build_splits.py` with seed 42, 10% holdout, sourced from the current `implication_pairs.parquet` (22,028,942 rows; 2,202,894 holdout). Reusing it instead of creating a redundant file. Plan updated to reference the existing path. No new code or data; just a name change.

---

## 2026-05-12 — SIGReg shelved; pivot to auxiliary prediction heads

The `testing_plan_phase2.md` failure analysis ran to completion and surfaced a clear pattern at h=256: VICReg-trained encoders peak at ep0 (best `stable_h256_m1p5_vic0p05_diag` at 0.9469) then degrade into TRUE-bias drift (TRUE recall climbs to 0.99, FALSE recall falls to 0.87). The h=128 baseline (`stable_m1p5_vic0p1_diag` at 0.9492 ep6) doesn't show this drift; the wider encoder finds a shortcut.

We tested SIGReg (LeJEPA, Balestriero & LeCun 2025, `arXiv:2511.08544`) as a more principled anti-collapse regularizer. Three variants run at h=256, all failing the same way:

  - `stable_h256_m1p5_sigreg_diag` (λ=0.05, L2-norm on): bal_acc 0.6095 ep1; cl_TRUE pinned at 1.97 = random unit-sphere distance²
  - `stable_h256_m1p5_sigreg_lam0p005_diag` (λ=0.005, L2-norm on): bal_acc 0.6891 ep1; same cl_TRUE pin, stability fixed
  - `stable_h256_lejepa_3ep_diag` (λ=0.05, L2-norm off, margin=25): bal_acc 0.5545 ep2; cl_TRUE = K = R^512 Gaussian random-pair distance, clip fires 100% of steps

Diagnosis: SIGReg's target distribution (isotropic Gaussian) and our contrastive loss's geometric requirement (TRUE pairs clustered) are incompatible at h=256 in our small-batch / random-init regime. With L2-norm, the conflict is structural (sphere-uniform vs sphere-clustered). Without L2-norm, the conflict is dynamical (clip rate-limits contrastive while SIGReg's globally-coherent direction wins). The pattern is consistent enough across variants that further hyperparameter sweeps are unlikely to help. SIGReg is the right idea for image-SSL setups where the prediction loss is MSE between same-instance views; ours is contrastive-with-margin between paired instances with TRUE/FALSE labels — a different signal that doesn't balance against SIGReg the same way. The src/training/sigreg.py implementation stays in the codebase (9 unit tests pass; correct against Algorithm 1 from the paper), available if the task setup ever changes. The `model.l2_normalize_outputs` flag also stays — defaults to `true` so VICReg recipes are unchanged.

The new active plan is `aux_heads_implementation_plan.md`. Auxiliary prediction heads (single-layer linear probes from the pooled encoder representation) predict per-equation structural invariants (spine type, balance, singleton-forcing, variable count). The hypothesis: anchor known-useful features so the encoder can't suppress them during the late-training drift. Unlike SIGReg, the aux heads operate per-equation in the feature space the encoder ALREADY produces — no global distributional pressure that fights the per-pair contrastive force. They serve as a floor (must-keep features) without imposing a ceiling (other dims still free for the contrastive signal). Test 2 in the plan is the headline: does h=256 sustain bal_acc above 0.94 through 10 epochs instead of degrading after ep0?

`testing_plan_phase2.md` is moved to `trash/` since its findings are now subsumed by the run record (`outputs/runs/stable_*/diagnostics_report.md`) and CLAUDE.md's pointer chain. The `SIGReg/` paper-and-plan folder is also in `trash/`. Neither is committed (per `.gitignore`). CLAUDE.md's "Regularization options" section now documents which regularizers are in the codebase and which are shelved.

---

## 2026-05-09 — Original Phase 2+ archived; new plan is failure analysis

`jepa_engineering_plan.md` was moved to `archive/jepa_engineering_plan.md`. Its Phase 1 (architecture sweep + loss form) is complete and produced the current best model (`vicreg_encoder_v0p25c0p25_tw1p0_v2`, bal_acc 0.9112, near-balanced recall, Austin holdout recall 0.94). The original Phases 2–4 (level-2/3 holdouts, architecture refinement, separate probing phase) are on hold pending a different investigation.

The new active plan is `testing_plan_phase2.md` — a **failure analysis** on the current best model. Train on ALL ~22M pairs (no holdout) for 20 epochs; evaluate on every pair; study what the model gets wrong despite training on it. Eight analyses planned, covering: cross-reference with Austin pairs (does the GNN learn an "Austin-like" syntactic signature?), structural-invariant distribution of failures, magma-signature separability of confidently-wrong FALSE pairs, Vampire-solve-time correlation, distance distribution, equation-level failure concentration, embedding-space UMAP visualization, and linear probing of frozen embeddings against the invariants.

This requires new infrastructure: a `--no-holdout` training mode, a `--save-predictions` flag, a 22M-pair full-eval pass that writes per-pair (eq1_id, eq2_id, ground_truth, predicted, distance) parquet, and several analysis / figure scripts. The CHANGELOG will track those as they're built.

CLAUDE.md and README.md were updated to point at the new active plan and note the archive location.

---

## 2026-05-09 — ETP CSV is now the ground-truth source for implication labels

The pair table at `data/processed/implication_pairs.parquet` was previously built from `data/raw/etp/2025-08-11-vampire.json.gz`. Vampire could not decide 1,062 of the 22M pairs (its fmb mode found no finite counterexample because none exists; its casc mode could not close the proof because the implication is actually FALSE in general); those were marked `value=2` and excluded from training and the L1 holdout. The Austin-implications diagnostic exposed this: of the 820 Austin pairs, 610 were among the unresolved ones, even though all 820 are unambiguously FALSE per the ETP's Lean-verified infinite-counterexample constructions.

The user supplied `data/raw/etp_export/export_raw_implications_5_9_2026.csv` — a 4694×4694 dense int8 matrix with **every off-diagonal pair labeled**, derived from the ETP's authoritative Lean results. Codes:

- `+3`: TRUE (regular)
- `+4`: TRUE (special / structural; ~10K entries)
- `-3`: FALSE (regular counterexample)
- `-4`: FALSE (special; ~590K entries — broader than the 820 curated Austin pairs)

`scripts/build_pair_table.py` was rewritten to read the CSV instead of the Vampire JSON. The output schema gains a `code: int8` column carrying the full ±3/±4 label alongside the binary `value: int8`. The 1,062 previously-unknown pairs are all FALSE under this CSV — confirming the user's clarification that Vampire timeouts are not a separate epistemic state, just an artifact of the prover.

The L1 split was rebuilt at `seed=42`. With no unknowns to exclude, the candidate pool is the full 22,028,942; the random 10% subsample changes:

- holdout: 2,202,894 (was 2,202,788; +106)
- train  : 19,826,048 (was 19,825,092; +956)

This is bit-for-bit incompatible with the previous split; results from runs trained against the old split remain locked in their existing JSONs and continue to be valid as historical artifacts, but new runs use the new data. CLAUDE.md's "Data layout" section was updated to point at `data/raw/etp_export/` and document the cutover. `tests/test_dataset.py::test_make_loaders_smoke` was updated to assert the new sizes.

---

## 2026-05-08 — Phase 1 baseline locked

`outputs/results/baseline_phase1_l2norm.json` is the project's reference baseline. It captures `epoch_006.pt` from the `phase1_l2norm_10ep` wandb run, evaluated on all 2,202,788 L1 val pairs (no batch cap):

  - balanced_accuracy = 0.8121
  - true_recall = 0.7535
  - false_recall = 0.8708
  - threshold = 1.0613 (L2-normalized embeddings, distances ∈ [0, 2])
  - epoch 6 of 10 (per-epoch peak; epoch 9 is statistically identical at 0.8119)
  - encoder 153,920 params, predictor 16,576 params
  - default config: hidden 128, embed 64, 4 layers, 4 heads, dropout 0.1, batch 2048, lr 1e-3, weight_decay 0.05, margin 1.0, true_weight 1.0
  - git commit: `f601c00`
  - wandb run: `fxwe85fy`

Subsequent variants will be recorded the same way under `outputs/results/<variant_name>.json` via `scripts/record_run.py`. Comparison table: `python scripts/record_run.py --summary`.

---

## 2026-05-07 — L2-normalize encoder + predictor outputs (collapse fix)

The first end-to-end Phase 1 sanity training (1 epoch on 19.8M train pairs, run `phase1_sanity_1ep` on wandb) showed a classic JEPA failure mode: distances grew unboundedly during training, exceeding the contrastive loss's `margin = 1.0` after the LR warmup, at which point the FALSE term `relu(margin - d)² = 0` went silent. With only the TRUE term active, no force kept the embedding scale bounded; train loss saturated near 42 and val balanced accuracy plateaued at 0.635 (barely above the 0.63 majority-class baseline). Calibrated val threshold settled at ~7.99 — about 8× the loss's margin.

**Fix**: L2-normalize the output of the encoder *and* the predictor (`F.normalize(..., dim=-1)`). Both encoder outputs (which the EMA target encoder also produces by construction) and predictor outputs now lie on the unit sphere. Pairwise distances are bounded to `[0, 2]`, so:

- The `margin = 1.0` sits at the midpoint of the achievable range (rather than being trivially exceeded).
- The FALSE term is always active when distances are below 1.
- There is no scale dimension for the optimizer to escape into.

This is the standard practice in modern contrastive learning (SimCLR, BYOL, DINO). User direction was specifically: try L2-normalization first; reach for VICReg only if dimensional collapse appears *after* this fixes the scale issue.

**Code changes**:
- `src/model/encoder.py`: `forward(batch)` returns `F.normalize(self.proj(pooled), dim=-1)`.
- `src/model/predictor.py`: `forward(z)` returns `F.normalize(self.net(z), dim=-1)`.
- This supersedes a 2026-05-07 design note saying the predictor should *not* L2-normalize. Predictor module docstring updated to reflect the new state.

**New tests** (3, all in addition to the existing suite):
- `test_encoder_output_is_unit_norm`
- `test_predictor_output_is_unit_norm`
- `test_forward_outputs_are_unit_norm_and_distance_is_bounded` (JEPA level — verifies distances ≤ 2)

---

## 2026-05-07 — Predictor design clarifications

The plan's §"Predictor Architecture" is brief: "Small MLP, 2 layers, hidden_dim=128, smaller than encoder." Several details needed clarification at implementation time. User direction:

- **"2 layers" = two weight matrices** (BYOL/SimSiam convention): `Linear(64→128) → ReLU → Linear(128→64)`. *Not* three weight matrices. Three would add expressivity, pushing against the design principle of keeping the predictor bottlenecked.
- **No dropout** (the predictor's `dropout=0`, in contrast to the encoder's `0.1`). The predictor must be a clean, deterministic mapping from embedding space to embedding space; dropout would inject noise into `z_predicted` and degrade the contrastive loss signal. BYOL and SimSiam predictors have no dropout for exactly this reason — the precise-bottleneck role is the design principle.
- **No layer normalization, no output L2-normalization**. If embedding collapse occurs at training time, try VICReg regularization first; only L2-normalize the predictor output if VICReg doesn't help.
- **Activation = ReLU**, **bias = on**.

Final structure (`src/model/predictor.py`): 16,576 parameters. Encoder is 152,896, so predictor is ~9× smaller — comfortably within the "smaller than encoder" hard constraint.

The plan stays as-is (still vague on these specifics); this entry is the canonical record of the design choices.

---

## 2026-05-07 — Bidirectional edges with 4-dim edge features (plan amendment)

The plan previously specified parent→child unidirectional edges with a 2-dim `[is_left, is_right]` edge feature, plus a single directional root-to-root edge for Family-C with `edge_attr = [0, 0]`. With unidirectional edges, the root never receives any message from its descendants in any number of message-passing layers — all structural integration was forced into the global pooling step.

User authorized a plan amendment: edges are now **bidirectional** (`parent→child` *and* `child→parent` for every tree link), and the edge feature is 4-dim **`[is_left, is_right, is_downward, is_upward]`**. Family-C cross-link is also bidirectional, with both edges carrying `[0, 0, 0, 0]` (neither left/right nor up/down — an all-zeros pattern that attention can learn to treat as a distinct cross-link edge type).

Plan §"Parse Tree Representation" updated and a new §"Edge Representation" sub-section added (verbatim from the user's spec).

Effects:
- `EDGE_FEATURE_DIM`: 2 → 4 in `src/data/parser.py`.
- Parser emits two edges per tree link (down and up) with the four bits set per the spec.
- Family-C: bidirectional cross-link replacing the previous single directional root-to-root edge.
- Encoder default `edge_features`: 2 → 4 (`GATv2Conv(..., edge_dim=4)`).
- `configs/default.yaml` `encoder.edge_features`: 2 → 4.
- `data/processed/equation_graphs.pt` rebuilt: max edges per equation went 9 → 18, file 6.25 MB → 7.67 MB. Family-C count unchanged at 1,555.
- New test `test_bidirectional_edges_propagate_root_from_leaf_in_one_layer` in `tests/test_encoder.py` validates the entire reason for the change: a single GATv2Conv layer over a 3-node tree, perturbing only a leaf's input features, and asserting the root's output changes.

---

## 2026-05-07 — Encoder parameter count: ~150K for Phase 1 sanity

Implementing the plan's encoder config literally (`hidden_dim=128`, `embed_dim=64`, `num_layers=4`, `num_heads=4`, `dropout=0.1`; GATv2Conv with `concat=True` and `per_head_dim = hidden_dim / num_heads = 32`) yields **152,896 parameters** — an order of magnitude below the plan's "Estimated ~1-2M parameters."

User confirmed the plan's 1-2M estimate was off and that we should proceed with the ~150K config for Phase 1 ("let's just see if this works at all"). Larger models are flagged for later experimentation, presumably under the Phase 3 architecture sweep.

The plan still reads "~1-2M" — that wording is the user's prerogative; this changelog entry exists so future sessions reading the plan don't misinterpret the ~150K implementation as accidental drift.

Per-GATv2-layer breakdown (in=128, out=32, heads=4, edge_dim=2): `lin_l` 16,512 + `lin_r` 16,512 + `lin_edge` 256 + `att` 128 + `bias` 128 = 33,536. × 4 layers = 134,144. Plus input proj (1,280) + final proj 256→64 (16,448) + four LayerNorms (1,024 total) ≈ 152,896.

Bonus consequence: GPU forward on a batch of 2,048 graphs is ~4.6 ms, putting an epoch over 22M pairs at ~3 minutes — well below the plan's "30-60 minutes/epoch" estimate (which presumably assumed the larger encoder).

---

## 2026-05-06 — Parser deviations from the engineering plan

When implementing `src/data/parser.py` (Phase 1, first task), three encoding deviations and one implementation choice came up:

### 1. Variable slots: 4 → 6 (node features 7-dim → 9-dim)

The plan specified `[is_op, is_var, var_id_0..3, is_lhs_var]` (7 dimensions). Inspection of `data/raw/etp/equations.txt` shows the corpus alphabet is `{u, v, w, x, y, z}` — six distinct variables — and 22 equations actually use all six. With only 4 slots, 347 equations (5- and 6-variable cases combined) would lose information.

Updated to `[is_op, is_var, var_id_0..5, is_lhs_var]` = 9 dimensions. The constant `NUM_VAR_SLOTS = 6` is exposed at the top of `src/data/parser.py`. `configs/default.yaml` updated to `encoder.node_features: 9` accordingly.

Evidence: distinct-variable histogram across the 4,694 equations is `{1: 31, 2: 779, 3: 2090, 4: 1447, 5: 325, 6: 22}`.

### 2. Family-C root-to-root edge label: `edge_attr = [0, 0]`

The plan calls for a 2-dim edge feature `[is_left, is_right]` (one-hot) and a "special root-to-root edge" between LHS and RHS subtrees in Family-C equations. The plan does not specify how that edge should be labelled. We use `[0, 0]` — neither left nor right — as a third implicit class encoded via all-zeros. This keeps the 2-dim edge feature unchanged while still distinguishing root-links from tree edges.

Alternative considered: extend `edge_attr` to 3-dim with an explicit `is_root_link` bit. Rejected for now to preserve the plan's edge dimensionality; revisit if attention struggles to distinguish the `[0, 0]` case from masked-out tree edges.

### 3. `is_lhs_var = 0` everywhere in Family-C inputs

The plan's `is_lhs_var` flag marks "the distinguished LHS variable." For Family-C (LHS is a product, not a single variable), no single variable is distinguished. We set `is_lhs_var = 0` throughout for these equations; the LHS-vs-RHS distinction is carried by the LHS subtree's presence plus the directional root-link.

### 4. Hand-rolled parser instead of `lark`

The plan's parser sketch uses a `lark` grammar. Inspection of the corpus shows every sub-product is fully parenthesized and there is no operator-precedence ambiguity; a 50-line recursive-descent parser is faster, dependency-free, and equally correct. `lark` remains in `requirements.txt` for now in case it's useful elsewhere; can be dropped if no other use emerges.

### 5. Canonical variable ordering — traversal explicitly defined

The plan says variables are assigned "by order of first appearance (DeBruijn-style): first variable seen = 0, second = 1, etc." but doesn't specify the traversal order, which matters when LHS and RHS share variables in different positions. The parser uses LHS pre-order, then RHS pre-order. Plan updated to specify this explicitly.

### 6. Dropped `torch-scatter` and `torch-sparse` from `requirements.txt`

The plan's dependency list included `torch-scatter` and `torch-sparse`. Verified empirically that `torch_geometric` 2.7 + `torch` 2.6 handle scatter and sparse ops via PyTorch's built-in primitives — `Data`, `Batch`, `GATv2Conv` (with `edge_dim`), and `global_mean_pool` / `global_max_pool` all work on CPU and GPU without those packages. Only an informational warning that "max" scatter could be accelerated by `torch-scatter`. The packages also need a special wheel index from the PyG project that has to match the exact `torch + cuda` build, making them awkward to reproduce. Dropped from `requirements.txt`; can be added back if profiling shows scatter ops are a bottleneck during training.

### Verification

`tests/test_parser.py` covers the parser deviations (27 tests, full-corpus parse, Family-C count = 1,555). The full pipeline (parser → PyG `Data` → `Batch` → `GATv2Conv` → pooling) was smoke-tested on CPU and CUDA. Cache built via `scripts/parse_equations.py` (6.25 MB, 4,694 entries, max 10 nodes / 9 edges per equation).
