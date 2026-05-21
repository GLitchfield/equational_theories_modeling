# proof_decoder

Phase 1 research decoder — trained an autoregressive decoder on the matcher's 49,586 verified proofs to test whether the JEPA v1.0 embeddings encode the proof witness, even though they were trained only on TRUE/FALSE labels.

## Headline

| Setup | both-novel σ-exact match | Δ vs random |
|---|---|---|
| **JEPA decoder** | **96.32%** | +90.7pp |
| Random-embedding control | 5.6% | — |
| Majority baseline | 0.0% | — |

Decisively answers the research question: yes, the JEPA embeddings preserve the proof witness. Mechanism reframing (per the preservation-vs-discovery framing): σ is a deterministic function of the two parse trees, so the finding is "contrastive loss preserved enough structure that σ remains recoverable from (z_e1, z_e2)", not "contrastive learning mystically learned theorem-proving."

Linear probe localizes the work: per-position σ accuracy is 88–93% linearly accessible (positions approximately independent in linear space). The decoder's 40pp lift over linear is specifically about modeling inter-position consistency constraints — the structural-walk invariants the matcher exploits.

## Contents

| Path | What |
|---|---|
| `code/decoder.py`, `eval_helpers.py`, `tokenizer.py` | Decoder model + evaluation helpers + Lean tactic tokenizer |
| `scripts/train_decoder.py` | Training script |
| `scripts/build_decoder_dataset.py` | Dataset builder from matcher proofs |
| `scripts/probe_decoder_linear.py` | Linear probe localization |
| `scripts/eval_decoder_majority_baseline.py` | Majority baseline |
| `data/decoder_phase1_dataset.parquet` | 49,586-row dataset (gitignored) |
| `outputs/dataset_summary.json`, `phase2_vocab.json` | Dataset and vocab stats |

Run records (gitignored locations of the actual training artifacts):
- `embeddings/outputs/runs/proof_decoder_jepa/diagnostics_report.md` — full writeup, JEPA decoder
- `embeddings/outputs/runs/proof_decoder_random_ctl/` — random-embedding control
- `embeddings/outputs/runs/proof_decoder_linear_probe/` — linear probe
- `embeddings/outputs/runs/proof_decoder_majority/` — baseline

## Archived plan

`archive/proof_decoder_plan.md`.
