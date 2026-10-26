# CLAUDE.md — Equational Implication

**Active phase:** **Proof generation pilot** — plan at `proof_generation_pilot/pilot_plan.md`, proof-output spec at `proof_generation_pilot/proof_spec.md`. Phase 1 is complete (1k-pair matcher/Vampire/grind transition dataset, synthesis at `proof_generation_pilot/outputs/phase1_synthesis.md`); Phase 2 is in design (open decisions P2Q1–P2Q8 in the plan).

**Recently completed:** public-release prep (2026-10-26; see CHANGELOG) and the project refactor (Phases 1–8) that reorganized the top level into `lib/`, `embeddings/`, `et_studies/`, `proof_generation_pilot/` (inventory archived at `archive/project_refactor_inventory.md`).

## Repository layout

```
data/                       # raw + processed (shared, stays at root)
  ground_truth/             # committed ETP raw-implications export (.csv.gz) — the label source
  raw/                      # symlinks to external projects; never modified
  processed/                # generated PyG cache + invariants + pair table
  splits/                   # holdout split indices, committed for reproducibility
lib/                        # shared primitives (parser, matching, unification, LPO)
  data/                     # parser, dataset, splits modules
  rewrite/                  # matching, unification, lpo modules
  scripts/                  # parse_equations.py, build_pair_table.py, build_equation_invariants.py
embeddings/                 # JEPA work, frozen v1.0 embeddings, MLP classifier
  src/                      # model/, training/, probing/, baselines/, evaluation/
  scripts/                  # train, evaluate, save_embeddings, save_predictions, analyze_*
  configs/                  # default.yaml + experiments/*.yaml
  outputs/                  # checkpoints, runs, embeddings/v1.0, probing, results, figures, logs
et_studies/                 # frozen studies — see et_studies/README.md for the index
  proof_survey/             # ETP proof characterization
  matcher_deliverable/      # first-order matcher coverage
  proof_decoder/            # Phase 1 research decoder (incl. code/)
  rewrite_chain_study/      # 1,636 c_rewrite_chain proof characterization (incl. code/)
  completion_exploration/   # plain + ordered KB completion
  coverage_measurement/     # deterministic-stack TRUE coverage
  failure_overlap/          # cross-run JEPA failure overlap
  etp_structure_analysis/   # ETP closure architecture note
  proof_investigation/      # PAUSED — Phase 8 intermediate-term analysis
proof_generation_pilot/     # active phase — pilot_plan.md, proof_spec.md
  methods/{matcher, completion, safe_engine, vampire, grind}/  # provers + Lean adapters
  lean_project/                                 # PilotShim Lake project (Lean 4.29.1, requires sibling ETP)
  scripts/, data/processed/, outputs/           # Phase 1 sampling + orchestration, samples, dataset
tests/                      # flat layout, repo-root
archive/                    # archived plans (every completed plan ends up here)
wandb/, trash/              # local-only state and reversible-removal staging (both gitignored)
```

`lib/` has no upward dependencies. Everything else can import from `lib/`. Cross-area imports (e.g., embeddings/ importing from pilot/) are not expected.

## Five durable deliverables

1. **v1.0 embeddings** at `embeddings/outputs/embeddings/v1.0/` — frozen 768-dim equation embeddings; `MODEL_CARD.md` ships with the .pt source checkpoint.
2. **Proof-corpus characterization** at `et_studies/proof_survey/outputs/README.md` — three-track recommendation (matcher / rewrite-chain decoder / upstream-tool delegation) + brute-force `Fin n` FALSE search.
3. **Rewrite-chain decision** at `et_studies/rewrite_chain_study/outputs/synthesis_report.md` — 83.13% deterministic-search coverage at d_max=6; recommend neural-guided search on the 17% hard tail.
4. **Completion-based decision procedure** at `et_studies/completion_exploration/outputs/synthesis_report.md` — 13.46% of 22M decided soundly (TRUE from ordered, FALSE from plain); 3.28% of GT FALSE come with `Fin n ≤ 6` counter-models for free.
5. **Coverage measurement** at `et_studies/coverage_measurement/outputs/coverage_measurement_synthesis.md` — combined sound TRUE coverage 6.54% of GT TRUE; structural ceiling 46.07%; engine plateau 14.20% of ceiling; `frontier_died` at 85.74% of within-ceiling residual.

## v1.0 embeddings (current deliverable)

- **Spec:** 4694 × 768 float32, L2-normalized; produced by the context encoder of `skip_h128_seed42_diag` epoch 9.
- **Files:** `embeddings/outputs/embeddings/v1.0/embeddings.npy` (gitignored, regenerable), `eq_ids.npy` (gitignored), `source_checkpoint.pt` (committed), `MODEL_CARD.md` (committed).
- **JEPA bal_acc (matched 10% holdout):** 0.9841 (essentially identical to the 0.9840 full-corpus number).
- **Frozen-embedding MLP head bal_acc (matched holdout):** 0.9996. The MLP recovers 91.37% of the "always-wrong-trained" subset (TRUE-by-vacuity cases the JEPA distance metric structurally couldn't represent).
- **Full model card** with sha256s, reproduction recipe, and known limitations: `embeddings/outputs/embeddings/v1.0/MODEL_CARD.md`.

## JEPA-phase recap (historical)

All JEPA plans live in `archive/`: `jepa_engineering_plan.md`, `aux_heads_implementation_plan.md`, `probing_experiments_plan.md`, `dual_encoder_plan.md` (deferred), `skip_connection_plan.md` (delivered v1.0), `mlp_classifier_plan.md` (validated v1.0).

JEPA-internal results worth remembering if we ever return to training:

- **h=128 is the reliable recipe.** Two-seed mean 0.9810 (seed=1 → 0.9779, seed=42 → 0.9840), seed sensitivity +0.61pp.
- **h=256 is unreliable** — `aux_h256_diag` hit 0.9636 via a regime-B phase transition that did NOT reproduce under seed=1 (reverted to 0.9488). The simplicity-bias drift problem (TRUE-recall → 0.99, FALSE-recall → 0.87) at h=256 is unresolved on the aux-head path.
- **20-epoch extension underperformed** the 10-epoch run (`skip_h128_seed42_20ep_diag` peaked at 0.9774); cosine-LR-scaling explanation called out in its Limitations section, not retested directly.

## Studies — one-line each

See `et_studies/README.md` for the full index. Quick summary:

- **proof_survey** — 62.5% TRUE proofs are arg-permutation; 15.3% rewrite chains; 22% Vampire/Egg. Three-track architecture recommendation.
- **matcher_deliverable** — 59.8% of directly-proved set, 0 false positives across 13.8M GT FALSE. The matcher subsumes Track 1.
- **proof_decoder** — both-novel σ-exact 96.32% (JEPA) vs 5.6% (random ctl). Established JEPA embeddings preserve the proof witness; reframed as preservation, not discovery (σ is a deterministic function of the parse trees).
- **rewrite_chain_study** — 83.13% at d_max=6, 99.7% rules within {h, ← h}, 87% unimodal. Recommend neural-guided search on the hard tail.
- **completion_exploration** — 13.46% combined sound TRUE/FALSE; ordered KB lifts TRUE 3× over plain KB.
- **coverage_measurement** — 6.54% combined TRUE / 46.07% structural ceiling / 14.20% engine plateau; `frontier_died` dominates residual.
- **failure_overlap** — 32,496 always-wrong-trained pairs (99.99% TRUE), TRUE-by-vacuity dominant; MLP head recovers 91.37%.
- **etp_structure_analysis** — ~11K direct proofs + Closure.lean derives 8M+ transitive labels from ~4 MB source evidence.
- **proof_investigation** — PAUSED at Phase 1; tests three hypotheses on the dense-row residual; may resume.

## Operational rules

- Do NOT modify files in `data/raw/`. All processing goes to `data/processed/` or `data/splits/` or per-study `et_studies/X/data/`.
- Never `rm` files — move to `trash/` so removals are reversible.
- Use `wandb` for ALL experiment tracking. Every training run must be logged.
- Always report metrics per holdout level separately: L1, L2 one-unknown, L2 both-unknown, L3. Never average across levels — the gaps BETWEEN levels are the headline finding.
- Save holdout split indices to `data/splits/` so experiments are reproducible. Every experiment log must reference which split file was used.
- The equation parser (`lib/data/parser.py`) is the critical path — if parsing is wrong, everything downstream is wrong.
- The repo is public-facing: never hardcode absolute machine paths (`/home/...`, `~/...`) in code or configs. External checkouts resolve as sibling directories of this repo, overridable by `ETP_ROOT` (teorth/equational_theories), `INVARIANTS_ROOT` (`equational_theories_data/invariants`), and `VAMPIRE_BIN`; follow that pattern for any new external dependency.
- Hardware: RTX 4080, 16 GB VRAM. Monitor and reduce batch size if OOM.
- Any committed analysis output (parquet, CSV, summary markdown, figure) must be reproducible from committed code in the same commit — a script in the relevant area's `scripts/`. README prose describing the methodology doesn't count as a reproducer. If the analysis was ad-hoc, write the script first, then commit script + outputs together.
- README.md and CLAUDE.md must stay in sync. When CLAUDE.md changes the active plan, the current stable best, or other project-state facts, update README.md in the same commit. The two should never disagree.
- Commit at granular logical breakpoints, not in giant batches.
- Never edit an active plan markdown unilaterally — raise problems as questions and let the user decide.
- When you deviate from an active plan, update the plan to match reality AND add a CHANGELOG.md entry explaining why, in the same commit as the deviating code.
- Keep checkpoints small — encoder weights + full JEPA state, skip optimizer state unless resuming from a crash.
- Every completed training run gets recorded artifacts (`embeddings/outputs/runs/<name>/` with `diagnostics_report.md`, `diagnostic_extract.json`, `per_step_trace.csv`, `per_epoch_eval.csv`) AND a commit with message pattern `Phase N: record <run_name> — <one-line takeaway>`. Run records are part of the run's deliverable, not a separate request.
- **Always read planning markdown files critically before implementing.** When the user adds a plan or experiment-design `.md`, scrutinize it like a code reviewer scrutinizes a PR: look for broken methodology (missing baselines, class-imbalance metrics, ill-defined CV), inconsistencies with existing code (wrong constants, paths, function signatures), and ambiguities a reasonable implementer could get wrong. Raise concerns ranked by severity BEFORE starting implementation; wait for the user's resolution on criticals at minimum. Don't be performatively skeptical — only flag issues that would actually change the implementation or interpretation.

## JEPA implementation reference (historical, for if we ever return to training)

Most modules live at `embeddings/src/`. References below use that prefix.

VICReg (`embeddings/src/training/losses.py::vicreg_terms`) was the production regularizer. Active via `loss.vicreg_var_weight` and `loss.vicreg_cov_weight` in any config. Works robustly at h=128 (0.9619 bal_acc with structural aux heads, 0.9582 stable from prior seed); degrades into TRUE-bias drift at h=256.

SIGReg (`embeddings/src/training/sigreg.py`) is implemented and verified (9 unit tests pass) but **shelved as not-a-fit for our task**. Three variants tested at h=256 — see commits `4fec7a3` → `cf5e00a` and run records under `embeddings/outputs/runs/stable_h256_*sigreg*/`, `embeddings/outputs/runs/stable_h256_lejepa_3ep_diag/`. Failure modes: with L2-norm, SIGReg's pre-norm Gaussian target → uniform on sphere, conflicts with contrastive's TRUE-clustering need (cl_TRUE pinned at 1.97 = random unit-sphere distance²); without L2-norm, SIGReg's globally-coherent spread direction beats contrastive in clip-saturated steps (cl_TRUE = K = orthogonal-pair distance). Code stays in repo as a known-correct implementation for future setups where the prediction loss isn't margin-based contrastive. Don't re-attempt without revisiting the linked diagnostics_report.md files.

The `model.l2_normalize_outputs` config flag (default `true`) routes through `JEPA.forward(normalize_outputs=...)` to drop the L2-norm on encoder/predictor outputs — added for the SIGReg LeJEPA-style attempt; safe to leave at default for any contrastive-with-margin recipe.

Auxiliary heads (`embeddings/src/model/aux_heads.py`) attach single-layer linear probes to the encoder's pooled rep. Active when `loss.aux_weight > 0` and `data.aux_labels_path` is set. `HEAD_SPECS` defines all 18 heads (10 structural + 8 cohomology); `loss.aux_heads` selects the active subset — accepts `"structural"`, `"cohom"`, `"all"` (named subsets in `HEAD_SUBSETS`), or an explicit list of head names. Defaults to all heads when omitted. Reproducing pre-cohom runs requires `loss.aux_heads: structural`.

## Numbers convention

All `bal_acc` figures are full-corpus re-evaluations from `embeddings/outputs/runs/<name>/failure_summary.json` (`embeddings/scripts/save_predictions.py` on all 22M pairs, threshold calibrated on a 1M-sample). The per-epoch monitor values in `per_epoch_eval.csv` differ by ~0.0002–0.0004 from val-batch subsampling — use the former for cross-run comparison, the latter for per-epoch trajectory analysis.

## Data layout (post-refactor)

- `data/raw/etp/` → symlink to the Equational Theories Project's `data/` directory. Source of `equations.txt`, the `Austin_implications.txt` curated subset, and other ETP artifacts.
- `data/ground_truth/export_raw_implications_5_9_2026.csv.gz` → **authoritative ground truth** for every off-diagonal implication pair: the ETP explorer's raw-implications export, committed gzip-compressed (provenance + sha256 in `data/ground_truth/README.md`; see CHANGELOG 2026-05-09 and 2026-10-26). Codes ±4 = explicitly proven/refuted, ±3 = implied by transitive closure. `lib/scripts/build_pair_table.py` reads it by default.
- Per-equation invariants come from the sibling `equational_theories_data` checkout (`$INVARIANTS_ROOT`, default `../equational_theories_data/invariants`), read by `lib/scripts/build_equation_invariants.py` and `embeddings/scripts/run_probes.py`. That project is not yet public.
- `data/processed/` → shared derived caches (parsed-tree PyG `Data` cache, pair-table parquet, equation-invariants parquet). Not committed.
- `data/splits/` → holdout split index files. Committed, for reproducibility.
- `et_studies/X/data/` → per-study processed parquets (matcher_proofs, normalized_proofs, true_corpus, etc.). Gitignored.
- `embeddings/outputs/X/` → JEPA-era outputs (checkpoints, runs, embeddings, results, figures, logs, probing, failure_analysis). Most binaries gitignored; small JSONs + MD reports committed.

`CHANGELOG.md` records deviations from the plans and explains *why* each archived plan reads the way it does — read it whenever a plan or commit references a CHANGELOG entry.
