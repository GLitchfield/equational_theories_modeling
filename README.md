# Equational Theories Modeling

Machine learning and automated reasoning experiments on the implication graph of the [Equational Theories Project](https://github.com/teorth/equational_theories) (ETP).

## The problem

A magma is a set with one binary operation, written `◇`, and no other rules. An equational law is an identity such as `x = y ◇ (x ◇ z)`, which holds in some magmas and not others. Law A implies law B if every magma that satisfies A also satisfies B.

The ETP, led by Terence Tao, took the 4,694 laws with at most four operations and settled every question of the form "does A imply B?" with proofs or counterexamples checked in Lean. That gives 22,028,942 labeled pairs, about 37% of them true.

This repository asks two questions about that graph. Can a neural network learn its structure from the laws' syntax alone? And how much of it can standard automated reasoning methods prove?

## Equation embeddings

The main result is a set of 768-dimensional embeddings, one per law, where the distance between two laws predicts whether one implies the other.

**The model.**

- Each law is parsed into a small tree, and a graph neural network (GATv2) encodes the tree.
- The setup is JEPA-style. One copy of the encoder reads the first law, and a slowly updated copy (an exponential moving average of the first) reads the second. A small predictor maps the first law's embedding toward where the second's should be. The predictor makes the relation one-directional, as implication is.
- A margin contrastive loss pulls true pairs together and pushes false pairs apart. The labels come from the ETP, so this is supervised metric learning that borrows the JEPA architecture.

**Training problems and fixes.**

- Embedding distances grew without bound until the loss stopped penalizing false pairs. Normalizing the embeddings to the unit sphere fixed it.
- The embeddings collapsed onto about 9 effective dimensions. VICReg variance and covariance terms on the encoder output raised that to 31.
- Late in training, the model drifted toward predicting "true." Small linear heads that predict structural properties of each law, such as its number of variables, stopped the drift.
- A parameter-free skip connection from the pooled graph features to the output added about 2 points of balanced accuracy.
- A SIGReg regularizer (from LeJEPA) failed in all three variants tried. The run records under `embeddings/outputs/runs/` explain why.

**Results.**

| Model | Evaluated on | Balanced accuracy |
|---|---|---:|
| Earlier recipe | A random 10% of pairs held out from training | 0.911 |
| v1.0, embedding distance | All 22M pairs, which it trained on | 0.984 |
| v1.0 embeddings with a 2-layer MLP head | A random 10% of pairs held out from the head (the encoder trained on all pairs) | 0.9996 |

The v1.0 model was trained on every pair so that a failure analysis could study what it gets wrong on pairs it has already seen. Its numbers measure how well the embeddings represent the whole graph. The earlier recipe's 0.911 is the result on pairs the encoder never saw.

The failure analysis found about 32,500 pairs that every trained model gets wrong. Nearly all are true because the second law is so weak that almost every magma satisfies it, so it looks nothing like the first law. A single distance threshold can't represent that, but an MLP on the same embeddings gets 91% of these pairs right, so the information is in the embeddings.

The v1.0 embeddings, the checkpoint that produced them and a model card are in [`embeddings/outputs/embeddings/v1.0/`](embeddings/outputs/embeddings/v1.0/MODEL_CARD.md).

## What automated reasoning can prove

The studies in [`et_studies/`](et_studies/README.md) measure how much of the graph standard symbolic methods settle, checked against the ETP's labels:

- A first-order matcher proves 59.8% of the implications the ETP proved directly, with no false positives.
- Knuth–Bendix completion, plain and ordered, soundly decides 13.46% of all 22M pairs.
- The matcher, a safe rewriting engine and completion together prove 6.54% of all true implications. Most of the rest are out of reach of safe rewriting, rather than limited by search time.

## Proof generation

A pilot in [`proof_generation_pilot/`](proof_generation_pilot/) runs these methods, the Vampire prover and Lean's `grind` tactic on true implications, checks every proof in Lean, and records each proof step as training data for a learned prover. Its first phase produced proof steps for 1,000 implications ([summary](proof_generation_pilot/outputs/phase1_synthesis.md)).

## Repository layout

```
data/                       # ground_truth/ (committed ETP labels), splits/ (committed), raw/ + processed/ (local)
lib/                        # shared code: parser, matching, unification, term ordering
embeddings/                 # the JEPA model, v1.0 embeddings, MLP classifier
et_studies/                 # the automated reasoning studies, each with its own outputs and scripts
proof_generation_pilot/     # the proof generation pilot and its Lean project
tests/                      # test suite
archive/                    # completed project plans
```

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

## How this was built

Most of the code was written by Claude Code agents under my direction. [`CLAUDE.md`](CLAUDE.md) holds the rules they worked under, [`CHANGELOG.md`](CHANGELOG.md) records why the plans changed, and [`archive/`](archive/) holds every completed plan.
