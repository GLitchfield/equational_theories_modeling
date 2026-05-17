# CLAUDE.md — Equational Implication

The active plan is `proof_decoder_plan.md` — phase 1 of the research decoder: train an autoregressive decoder on the matcher's 49,586 verified proofs to test whether the JEPA embeddings encode the proof witness (σ), not just TRUE/FALSE. Pure embeddings-only conditioning; equation-level train/test split; nested metric stack (parse rate → arity-correct → σ-exact-match); random-embedding control. The previous matcher plan is complete and archived.

The two completed phases left two durable artifacts:

1. **v1.0 embeddings** at `outputs/embeddings/v1.0/` — frozen 768-dim equation embeddings, the JEPA-phase deliverable.
2. **Proof-corpus characterization** at `outputs/analyses/proof_survey/README.md` — survey of the ETP Lean proofs with an architecture recommendation for the proof generator.

The JEPA training phase is **complete and wrapped**. From here on, downstream work consumes the v1.0 embeddings as a frozen input; we don't expect to retrain the JEPA unless a downstream task surfaces a specific deficiency.

The proof-corpus survey is **complete and wrapped**. Its three-track architecture recommendation (permutation predictor → tactic-sequence generator → upstream-tool delegation, plus a search-based FALSE generator) is the starting point for the next plan.

## v1.0 embeddings (current deliverable)

- **Spec**: 4694 × 768 float32, L2-normalized; produced by the context encoder of `skip_h128_seed42_diag` epoch 9.
- **Files**: `outputs/embeddings/v1.0/embeddings.npy` (gitignored, regenerable), `outputs/embeddings/v1.0/eq_ids.npy` (gitignored), `outputs/embeddings/v1.0/source_checkpoint.pt` (committed), `outputs/embeddings/v1.0/MODEL_CARD.md` (committed).
- **JEPA bal_acc (matched 10% holdout)**: 0.9841 (essentially identical to the 0.9840 full-corpus number).
- **Frozen-embedding MLP head bal_acc (matched holdout)**: 0.9996. The MLP recovers 91.37% of the "always-wrong-trained" subset (TRUE-by-vacuity cases the JEPA distance metric structurally couldn't represent). Downstream classifiers operating on these embeddings can pick that signal up.
- **Full model card** with sha256s, reproduction recipe, and known limitations: `outputs/embeddings/v1.0/MODEL_CARD.md`.

## JEPA-phase recap (historical)

All JEPA plans are in `archive/`. In rough order:

- `archive/jepa_engineering_plan.md` — original Phase 1–4 plan; Phase 1 (architecture sweep + loss) is complete.
- `trash/testing_plan_phase2.md` — failure-analysis plan that produced the diagnostic infrastructure and the `outputs/runs/stable_*` series.
- `archive/aux_heads_implementation_plan.md` — delivered `aux_h128_seed1_diag` at 0.9619 (prior best before skip).
- `archive/probing_experiments_plan.md` — showed cohomology features are linearly accessible in the embeddings to 80–88%.
- `archive/dual_encoder_plan.md` — deferred pending the skip-connection result; in-progress code in `archive/dual_encoder/`. Skip ended up not preserving cohom features either, so this remains relevant if a future task needs cohom in the embedding.
- `archive/skip_connection_plan.md` — delivered the current v1.0 embeddings (`skip_h128_seed42_diag`, 0.9840 full corpus / 0.9841 matched holdout).
- `archive/mlp_classifier_plan.md` — frozen-embedding MLP head; validated the v1.0 embeddings (0.9996 holdout, 91.37% vacuity).
- `archive/proof_corpus_survey_plan.md` — surveyed the ETP Lean proofs to inform the generator architecture. Output: `outputs/analyses/proof_survey/README.md` with the three-track recommendation.
- `archive/matcher_plan.md` — deterministic first-order matcher; solves Track 1 with 59.8% coverage of the directly-proved set, 0 false positives. Output: `outputs/analyses/matcher/README.md`, 49,586-row proof dataset.

Cross-run failure-overlap analysis (8 trained models, partitioned per Rule B): `outputs/analyses/failure_overlap/`. Reproducer: `scripts/analyze_failure_overlap.py`. Headline: a stable set of 32,496 always-wrong-trained pairs (99.99% TRUE), dominated by the "TRUE-by-vacuity" pattern; an MLP on the v1.0 embeddings recovers 91% of them.

## Matcher recap (Track 1 — complete)

`archive/matcher_plan.md` (now archived). Deterministic first-order matcher for the variable-substitution category. Outputs:

- **Coverage** (`outputs/analyses/matcher/`):
  - Directly-proved set: 6,374 / 10,658 = 59.8%
  - Full TRUE corpus: 49,586 / 8,173,585 = 0.607%
  - **FALSE positives: 0 / 13,855,357 ✓** (the correctness gate)
- Wall clock: 58.8 s on a single core, pure Python (~375K pairs/sec).
- Ground-truth proof dataset at `data/processed/proof_corpus/matcher_proofs.parquet` (49,586 rows, gitignored, regenerable).
- Reproducer: `scripts/run_matcher.py`; module: `src/matcher/matcher.py`; 27 unit tests.
- Methodology note: 59.8% vs the survey's 62.5% prediction is a methodology difference (survey counted proofs whose source body LOOKS like arg-permutation by regex; matcher counts equations where E2 IS a substitution instance of E1). Worth ~2.7pp; not a bug.
- Stage 2 submission format pinned against `tests/fixtures/answers/accepted_true_basic.answer.json` in the Stage 2 repo: `import JudgeProblem\n\ndef submission : Goal := by\n  intro G _ h\n  <body>`. Live-judge verification deferred to when a built Stage 2 env (Lean 4.30.0-rc2) is available.

## Proof-corpus survey recap

Survey of ETP's Lean proofs at `outputs/analyses/proof_survey/` (READ THIS BEFORE designing the proof generator — it captures the recommended architecture). Reproducers in `scripts/survey_*.py`. Key findings:

- **62.5% of TRUE proofs are arg-permutation closes** (`λ x y => h x y x`-style). The generator for that majority is a structured-prediction model emitting a `(hypothesis_id, permutation_tuple)`. Training data: 6,567 examples already in the corpus.
- **15.3% are rewrite chains.** Intermediate states are equations, which v1.0 embeddings already represent natively. Generator: small transformer over a ~100-token Lean tactic vocab, conditioned on v1.0 embeddings.
- **22% are machine output from Vampire (14.1%) or Egg (7.5%).** Don't try to learn these — delegate by invoking the upstream tool directly.
- **0.5% genuinely diverse complex.** Skip / corner-case.
- **FALSE side: Stage 2 accepts only Fin-n matrix counterexamples**, so the FALSE generator is a brute-force search over small `Fin n` magmas regardless of how the ETP corpus's proof shapes look (which are 74.6% lifting-family proofs at the source level).
- Vocab is small: 119 tokens cover 95% of TRUE proof usage. Proof length is bimodal — median 10 tokens, mean 102 (Vampire/Egg long tail).
- Lean version skew: ETP uses 4.20.1, Stage 2 judge uses 4.30.0-rc2. Most term-mode proofs port; tactic-mode may need adjustment. LeanDojo compatibility unverified; assess with a small extraction before committing to it.

**Updated architecture (matcher replaces Track 1 for the competition track):**
  1. **Deterministic matcher** (62.5% TRUE) — `archive/matcher_plan.md`, complete. Pure symbolic computation on parse trees; no embeddings, no learned model. Exact correctness by construction. Outputs at `outputs/analyses/matcher/` + the 49,586-row `data/processed/proof_corpus/matcher_proofs.parquet`.
  2. Short tactic-sequence generator (15% TRUE — rewrite chains) — v1.0 embeddings condition this; still a learned model.
  3. Upstream-tool delegation (22% TRUE — Vampire / Egg) — wrap, don't learn.
  + Brute-force FALSE counterexample search over small Fin n magmas.

**Research track (parallel to the competition cascade):** `proof_decoder_plan.md`. Train an autoregressive decoder on the matcher's 49,586 verified proofs to test the research question — do the JEPA embeddings encode the proof witness even though they were trained only on TRUE/FALSE? Pure embeddings-only conditioning; equation-level train/test split; nested metric stack; random-embedding control. The decoder is not the competition path (the matcher already solves this category exactly) — its value is the scientific claim and the eventual extension to rewrite chains where no deterministic shortcut exists.

The v1.0 embeddings have **no competition-track role in the variable-substitution category** after the matcher subsumes it. They are central to the research-track decoder (phase 1) and to Track 2 of the competition cascade (rewrite chains, where intermediate states are equations the JEPA encoder handles natively).

Numbers convention: all `bal_acc` figures in this document are full-corpus re-evaluations from `outputs/runs/<name>/failure_summary.json` (`scripts/save_predictions.py` on all 22M pairs, threshold calibrated on a 1M-sample). The per-epoch monitor values in `per_epoch_eval.csv` differ by ~0.0002–0.0004 from val-batch subsampling — use the former for cross-run comparison, the latter for per-epoch trajectory analysis.

A few JEPA-internal results worth knowing if we ever return to training:

- **h=128 is the reliable recipe.** Two-seed mean 0.9810 (seed=1 → 0.9779, seed=42 → 0.9840), seed sensitivity +0.61pp.
- **h=256 is unreliable** — `aux_h256_diag` hit 0.9636 via a regime-B phase transition that did NOT reproduce under seed=1 (reverted to 0.9488). The h=256 simplicity-bias drift problem (TRUE-recall → 0.99, FALSE-recall → 0.87) is unresolved on the aux-head path.
- **20-epoch extension underperformed** the 10-epoch run (`skip_h128_seed42_20ep_diag` peaked at 0.9774); see that run's diagnostics_report for the cosine-LR-scaling explanation (untested directly, called out in its Limitations section).

`CHANGELOG.md` records deviations from the plans and explains *why* each archived plan reads the way it does — read it whenever a plan or commit references a CHANGELOG entry.

When you deviate from the active plan, update it to match reality AND add a CHANGELOG.md entry explaining why, in the same commit as the deviating code.

**Always read planning markdown files critically before implementing.** When the user adds a plan or experiment-design `.md`, scrutinize it like a code reviewer scrutinizes a PR: look for broken methodology (missing baselines, class-imbalance metrics, ill-defined CV), inconsistencies with existing code (wrong constants, paths, function signatures), and ambiguities a reasonable implementer could get wrong. Raise concerns ranked by severity BEFORE starting implementation; wait for the user's resolution on criticals at minimum. Don't be performatively skeptical — only flag issues that would actually change the implementation or interpretation.

## Operational rules

- Do NOT modify files in `data/raw/`. All processing goes to `data/processed/` or `data/splits/`.
- Use `wandb` for ALL experiment tracking. Every training run must be logged.
- Always report metrics per holdout level separately: L1, L2 one-unknown, L2 both-unknown, L3. Never average across levels — the gaps BETWEEN levels are the headline finding.
- Save holdout split indices to `data/splits/` so experiments are reproducible. Every experiment log must reference which split file was used.
- Run baselines (1–4 in `archive/jepa_engineering_plan.md`) before training the JEPA. Without baselines you can't interpret the JEPA results.
- The equation parser is the critical path — if parsing is wrong, everything downstream is wrong. Test it thoroughly against known equations before any training.
- Hardware: RTX 4080, 16 GB VRAM. Monitor and reduce batch size if OOM.
- Any committed analysis output (parquet, CSV, summary markdown, figure) must be reproducible from committed code in the same commit — a script in `scripts/` or a notebook in `notebooks/`. README prose describing the methodology doesn't count as a reproducer. If the analysis was ad-hoc, write the script first, then commit script + outputs together.
- README.md and CLAUDE.md must stay in sync. When CLAUDE.md changes the active plan, the current stable best, or other project-state facts, update README.md in the same commit. The two should never disagree on what the active plan is or which run holds the current best.
- Keep checkpoints small — encoder weights + full JEPA state, skip optimizer state unless resuming from a crash.
- Every completed training run gets recorded artifacts (`outputs/runs/<name>/` with `diagnostics_report.md`, `diagnostic_extract.json`, `per_step_trace.csv`, `per_epoch_eval.csv`) AND a commit with the message pattern `Phase N: record <run_name> — <one-line takeaway>`. Run records are part of the run's deliverable, not a separate request.

## JEPA implementation reference (historical, for if we ever return to training)

VICReg (`src/training/losses.py::vicreg_terms`) was the production regularizer. Active via `loss.vicreg_var_weight` and `loss.vicreg_cov_weight` in any config. Works robustly at h=128 (0.9619 bal_acc with structural aux heads, 0.9582 stable from prior seed); degrades into TRUE-bias drift at h=256.

SIGReg (`src/training/sigreg.py`) is implemented and verified (9 unit tests pass) but **shelved as not-a-fit for our task**. Three variants tested at h=256 — see commits `4fec7a3` → `cf5e00a` and `outputs/runs/stable_h256_*sigreg*/`, `outputs/runs/stable_h256_lejepa_3ep_diag/`. Failure modes: with L2-norm, SIGReg's pre-norm Gaussian target → uniform on sphere, conflicts with contrastive's TRUE-clustering need (cl_TRUE pinned at 1.97 = random unit-sphere distance²); without L2-norm, SIGReg's globally-coherent spread direction beats contrastive in clip-saturated steps (cl_TRUE = K = orthogonal-pair distance). Code stays in repo as a known-correct implementation for future setups where the prediction loss isn't margin-based contrastive. Don't re-attempt without revisiting the linked diagnostics_report.md files.

The `model.l2_normalize_outputs` config flag (default `true`) routes through `JEPA.forward(normalize_outputs=...)` to drop the L2-norm on encoder/predictor outputs — added for the SIGReg LeJEPA-style attempt; safe to leave at default for any contrastive-with-margin recipe.

Auxiliary heads (`src/model/aux_heads.py`) attach single-layer linear probes to the encoder's pooled rep. Active when `loss.aux_weight > 0` and `data.aux_labels_path` is set. `HEAD_SPECS` defines all 18 heads (10 structural + 8 cohomology); `loss.aux_heads` selects the active subset — accepts `"structural"`, `"cohom"`, `"all"` (named subsets in `HEAD_SUBSETS`), or an explicit list of head names. Defaults to all heads when omitted. Reproducing pre-cohom runs requires `loss.aux_heads: structural`.

## Data layout

- `data/raw/etp/` → symlink to the Equational Theories Project's `data/` directory. Source of `equations.txt`, the `Austin_implications.txt` curated subset, and other ETP artifacts.
- `data/raw/etp_export/` → exports from the ETP project (not symlinked). Currently holds `export_raw_implications_5_9_2026.csv`, the **authoritative ground truth** for every off-diagonal implication pair (see CHANGELOG 2026-05-09). The pair table is built from this, not from the older `2025-08-11-vampire.json.gz` (which left 1,062 pairs unresolved).
- `data/raw/invariants/` → symlink to the sibling invariants project's `assembly/output/` directory. Currently empty until that project is run.
- `data/processed/` → derived caches (parsed-tree PyG `Data` cache, pair-table parquet). Not committed.
- `data/splits/` → holdout split index files. Committed, for reproducibility.
