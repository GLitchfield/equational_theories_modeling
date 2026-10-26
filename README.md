# Equational Theories Modeling

Machine-learning and symbolic-reasoning experiments on the implication graph of the [Equational Theories Project](https://github.com/teorth/equational_theories) (ETP). The graph records, for every ordered pair of the 4,694 magma laws with at most four operations, whether one law implies the other: 22,028,942 pairs, each settled in Lean.

The project has three threads:

- **Equation embeddings.** A JEPA over equation parse trees, trained on the full implication graph, gives frozen 768-dim per-equation embeddings (v1.0). Embedding distance alone classifies implications at 0.9841 balanced accuracy on a 10% pair holdout; an MLP head on the frozen embeddings reaches 0.9996.
- **What deterministic methods can prove.** A first-order matcher, a safe rewrite engine, and Knuth–Bendix completion (plain and ordered) are each measured against ETP's ground truth, alongside a characterization of ETP's own proof corpus. These studies are finished and frozen.
- **Proof generation (active).** A pilot that runs these methods, Vampire, and Lean's `grind`, checks their Lean output, and records per-tactic proof states as training data for a learned prover. It targets the ETP Stage 2 competition format.

## Status

**Active phase: proof generation pilot.** Phase 1 is complete: a 1,000-pair dataset of `(state, tactic, next_state)` transitions from matcher, Vampire, and `grind` on variable-disjoint TRUE implications, with all proofs checked in Lean 4.29.1 through PyPantograph ([synthesis](proof_generation_pilot/outputs/phase1_synthesis.md)). Phase 2 (a broader, structurally varied sample plus a uniform proof-output spec) is in design. The plan is at [`proof_generation_pilot/pilot_plan.md`](proof_generation_pilot/pilot_plan.md) and the spec at [`proof_generation_pilot/proof_spec.md`](proof_generation_pilot/proof_spec.md).

## Repository layout

```
data/                       # ground_truth/ (committed ETP labels), splits/ (committed), raw/ + processed/ (local)
lib/                        # shared primitives (parser, matching, unification, LPO)
embeddings/                 # JEPA work, frozen v1.0 embeddings, MLP classifier
et_studies/                 # frozen studies (each with its own outputs/, scripts/, data/, code/)
proof_generation_pilot/     # active phase: methods/, Lean project, Phase 1 dataset + outputs
tests/                      # flat test layout, repo-root
archive/                    # archived plans (every completed plan ends up here)
```

Each top-level area has its own README. See [`et_studies/README.md`](et_studies/README.md) for an index of the nine completed studies with one-line headlines.

## Five durable deliverables

1. **v1.0 embeddings** at [`embeddings/outputs/embeddings/v1.0/`](embeddings/outputs/embeddings/v1.0/): frozen 768-dim equation embeddings; see [`MODEL_CARD.md`](embeddings/outputs/embeddings/v1.0/MODEL_CARD.md).
2. **Proof-corpus characterization** at [`et_studies/proof_survey/outputs/README.md`](et_studies/proof_survey/outputs/README.md): surveyed ETP's 10,676 explicit-implication proofs and 1,698 facts entries; ends with the three-track architecture recommendation for proof generation.
3. **Rewrite-chain decision** at [`et_studies/rewrite_chain_study/outputs/synthesis_report.md`](et_studies/rewrite_chain_study/outputs/synthesis_report.md): 83.13% deterministic-search coverage at d_max=6, 99.7% rule set within {h, ← h}, 87% unimodal. Recommendation: neural-guided search targeting the 17% hard tail.
4. **Completion-based decision procedure** at [`et_studies/completion_exploration/outputs/synthesis_report.md`](et_studies/completion_exploration/outputs/synthesis_report.md): KB completion (plain + ordered) as a unified TRUE/FALSE decision procedure. Headline: 13.46% of 22M pairs decided soundly. Ordered completion lifts TRUE coverage 3× (2.13% → 6.06% of GT TRUE); plain KB FALSE coverage is 17.82% of GT FALSE.
5. **Deterministic-stack coverage measurement** at [`et_studies/coverage_measurement/outputs/coverage_measurement_synthesis.md`](et_studies/coverage_measurement/outputs/coverage_measurement_synthesis.md): combined matcher + rewrite engine + completion soundly covers 6.54% of GT TRUE. Structural ceiling 46.07%; the deterministic stack captures 14.20% of that ceiling. The dominant within-ceiling failure mode is `frontier_died` (85.74%), meaning the limit is structural reachability of safe first-order rewriting, not search budget.

## v1.0 embeddings at a glance

- **Spec:** 4694 × 768 float32, L2-normalized.
- **Source:** `skip_h128_seed42_diag` ep9 (h=128 GATv2 encoder with parameter-free skip connection from the pooled rep). The source checkpoint ships with the artifact at `embeddings/outputs/embeddings/v1.0/source_checkpoint.pt`.
- **JEPA distance bal_acc on the matched 10% pair holdout:** 0.9841 (0.9840 on the full 22M corpus).
- **Frozen-embedding MLP-head bal_acc on the same holdout:** 0.9996, recovering 91.37% of the "always-wrong-trained" subset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest   # passes on a fresh clone; one corpus test skips until data/raw/etp exists
```

Training was done on a single RTX 4080 (16 GB). Experiment tracking uses [Weights & Biases](https://wandb.ai); set `WANDB_MODE=offline` or `wandb.enabled: false` in the config to train without an account.

### External checkouts

Code that reads other repositories expects them as sibling directories next to this one. An environment variable overrides each location.

| Repository | Default location | Override | Needed for |
|---|---|---|---|
| [teorth/equational_theories](https://github.com/teorth/equational_theories) | `../equational_theories` | `ETP_ROOT` | `equations.txt` (through `data/raw/etp`), the proof-survey scripts, the pilot's Lean project and Vampire adapter |
| [equational_theories_data](TODO_INVARIANTS_REPO_URL) | `../equational_theories_data` | `INVARIANTS_ROOT` (its `invariants/` directory) | Per-equation invariants (spine type, LHS structure, variable count, magma signature, cohomology), used by aux-head training, the probing experiments, and analyses that read `data/processed/equation_invariants.parquet` |

```bash
git clone https://github.com/teorth/equational_theories ../equational_theories
mkdir -p data/raw && ln -s ../../../equational_theories/data data/raw/etp
```

The pilot was built against ETP commit `a1e7f690` (2026-05-08, Lean 4.29.1).

### Proof-generation pilot tooling

Only `proof_generation_pilot/` needs these:

- **Lean 4.29.1** via [elan](https://github.com/leanprover/elan). Build ETP first (`cd ../equational_theories && lake exe cache get && lake build`), then the pilot shim (`cd proof_generation_pilot/lean_project && lake build`).
- **[PyPantograph](https://github.com/stanford-centaur/PyPantograph) v0.3.15**, built from source against Lean 4.29.1 and installed with `pip install -e`.
- **[Vampire](https://github.com/vprover/vampire) 4.9** on `PATH`, or set `VAMPIRE_BIN` to the binary.

## Data

- `data/raw/etp/equations.txt`: the 4,694 equations (line N = equation ID N), from the ETP checkout.
- [`data/ground_truth/export_raw_implications_5_9_2026.csv.gz`](data/ground_truth/README.md): **committed** ground-truth label for every off-diagonal pair. It is the ETP explorer's raw-implications export (4694×4694 int8 matrix; codes ±4 = explicitly proven/refuted, ±3 = implied by transitive closure). Provenance and sha256 are in the linked README.
- `data/raw/etp/2025-08-11-vampire.json.gz`: *legacy* Vampire-derived labels. Older configs still reference it, but it is no longer the source of truth (it left 1,062 pairs unresolved).
- `data/splits/`: committed holdout split indices, for reproducibility.

Do not write to `data/raw/`. All processing goes to `data/processed/` (shared caches), `data/splits/` (committed split indices), or per-study `et_studies/X/data/` (gitignored).

## Build the shared caches

```bash
python lib/scripts/parse_equations.py             # data/processed/equation_graphs.pt   (~6 MB)
python lib/scripts/build_pair_table.py            # data/processed/implication_pairs.parquet  (~35 MB)
python lib/scripts/build_equation_invariants.py   # data/processed/equation_invariants.parquet  (needs the invariants checkout)
```

All three are deterministic and re-runnable; their outputs are gitignored.

## Regenerate v1.0 embeddings

```bash
python embeddings/scripts/save_embeddings.py \
    embeddings/outputs/embeddings/v1.0/source_checkpoint.pt \
    --output-dir embeddings/outputs/embeddings/v1.0 --force
```

The full reproduction recipe and validation numbers are in [`embeddings/outputs/embeddings/v1.0/MODEL_CARD.md`](embeddings/outputs/embeddings/v1.0/MODEL_CARD.md).

## License and acknowledgments

Licensed under the [Apache License 2.0](LICENSE).

This project builds on the [Equational Theories Project](https://github.com/teorth/equational_theories), led by Terence Tao with many contributors. Its equation list and Lean-verified implication results (Apache-2.0) are the data everything here is trained and measured against; the committed ground-truth export is redistributed under that license. If you use the data, please cite the ETP (see its [`CITATION.cff`](https://github.com/teorth/equational_theories/blob/main/CITATION.cff)).

## See also

- [`CHANGELOG.md`](CHANGELOG.md): deviations from each plan and why.
- [`archive/`](archive/): every completed plan, archived in chronological order.
- [`CLAUDE.md`](CLAUDE.md): the project-state guide used for AI-assisted development (operational rules, current state, JEPA internals worth knowing).
