# Proof Decoder — Implementation Plan (Phase 1: Variable Substitution)

## Goal and Research Question

Build an autoregressive decoder that generates Lean proofs from the frozen JEPA equation embeddings. Phase 1 targets the variable-substitution category — the 49,586 pairs the matcher solved.

The research question is sharp and falsifiable: **the JEPA embeddings were trained on a contrastive TRUE/FALSE objective and never saw a proof. Does the contrastive geometry nonetheless encode the proof witness?** If a decoder conditioned only on (z_e1, z_e2) reconstructs the exact substitution proof, the embeddings encoded not just *that* E1 implies E2 but *how*. If it can't, we learn precisely where the embedding is lossy.

This is the research track. It is not the competition track — the matcher already solves this category deterministically and exactly. The decoder's value is the scientific claim, and its eventual extension (phase 2) to rewrite chains, where no deterministic shortcut exists.

## Methodology resolutions (locked in 2026-05-17)

After scrutiny, the following were resolved before implementation.
The plan body has been updated to reflect them; this section is the
canonical record.

1. **Phase 1 vocabulary: 13 tokens, not 20.** The decoder generates
   only the proof body (`exact fun <vars> => h <args>` with optional
   `.symm`). `intro G _ h` is the judge's preamble, prepended outside
   the decoder. Don't carry preamble tokens the decoder never emits.
   Vocab: `exact`, `fun`, `=>`, `h`, `(`, `)`, `.symm`,
   `v1, v2, v3, v4, v5, v6`, `<BOS>`, `<EOS>`, `<PAD>`. (Phase 2 will
   define its own vocab so no future-proofing is lost.)
2. **Pure embeddings-only conditioning. No explicit arity input.**
   Probing already established that variable count is linearly
   decodable from the JEPA embedding at ~99%, so "can the decoder
   read arity" is close to a settled question. Adding an explicit
   arity one-hot would be belt-and-suspenders for a signal we've
   already shown is in the embeddings. The failure-attribution
   concern is handled by Resolution 3 (nested metric decomposition)
   instead of by changing the input.
3. **Nested metric decomposition.** Instead of a single σ-exact-match
   number, report three nested rates:
     - **parse rate**: fraction of predictions that are well-formed
       token sequences (σ-extractable).
     - **arity-correct rate** (of those that parse): fraction with the
       correct number of λ-binders and `h`-arguments.
     - **σ-exact-match** (of those with correct arity): fraction with
       the exact right substitution.
   This decomposition does the failure-attribution work an explicit
   arity input would do — it tells you whether a failure is malformed
   output, wrong arity, or wrong witness — without compromising the
   embeddings-only claim. If arity-correct rate is unexpectedly low,
   that itself is a finding about the embedding that motivates an
   explicit-arity follow-up.
4. **Random-embedding control: seed 42.** Same RNG seed as the rest
   of the project. Otherwise control variance across runs is
   uncontrolled.
5. **Phase 2 vocab pre-tokenization in phase 1 setup.** Run one
   tokenization pass on the matcher-solved + rewrite-chain proofs
   (excluding the 22% Vampire/Egg), bucket the tokens into
   instruction / structural / content, report counts. Doesn't touch
   the phase 1 decoder; pinning the number now means it doesn't slip.
6. **K=4 prefix tokens to start.** Bump to 8 or 16 only if accuracy
   is borderline.
7. **Flag small-N noise for arity 5-6 in the final report** — per
   the survey's shape data, those are a thin slice of the corpus.

## Scope

Phase 1 covers exactly the matcher's output: 49,586 pairs with verified ground-truth substitutions in `data/processed/proof_corpus/matcher_proofs.parquet`. Each row has (eq1_id, eq2_id, orientation, substitution_0..5, lean_proof_term, lean_proof_tactic).

The decoder generates the proof; evaluation checks the substitution against this ground truth. Because the proof is fully determined by (σ, orientation), σ-exact-match is a complete correctness measure for phase 1.

## Architecture

```
E1 → JEPA context encoder (frozen) → z_e1 (768)
E2 → JEPA context encoder (frozen) → z_e2 (768)
                                       │
                  concat [z_e1; z_e2] (1536)
                                       │
                       conditioning projection
                       Linear(1536 → K × d_model)
                                       │
                      K prefix tokens (d_model each)
                                       │
              ┌────────────────────────┴───────────┐
              │   autoregressive transformer decoder │
              │   d_model=256, 4 layers, 4 heads     │
              │   causal self-attention              │
              └────────────────────────┬───────────┘
                                       │
                          Linear(d_model → vocab)
                                       │
                    proof tokens, generated left to right
```

- **Conditioning**: concat the two embeddings (order matters — implication is asymmetric, z_e1 is the hypothesis), project to K=4 prefix tokens, prepend to the sequence. The decoder's self-attention reads them as context. Prefix conditioning is sufficient for two fixed vectors; no cross-attention needed.
- **Decoder**: d_model 256, 4 layers, 4 heads, learned positional encodings, causal masking. A few million parameters. Trains on the 4080 in well under an hour for phase 1.
- **Context length**: 64 for phase 1 (proofs are ~15-25 tokens). Phase 2 will need 512.
- **Embeddings frozen**: only the conditioning projection and the decoder train. The encoder is locked (skip_h128_seed42 epoch 9).

**Critical constraint — embeddings-only input.** The decoder sees ONLY z_e1 and z_e2. It never sees the parse trees. If it saw the parse trees it would learn the matching algorithm and the embeddings would be irrelevant. The research claim is only tested if the embeddings are the sole input.

## Token Vocabulary

The decoder generates in normalized variable space (v1…v6); a deterministic post-processing step maps normalized variables back to the equation's actual variable names. Same first-appearance normalization the proof survey used.

### Phase 1 vocabulary (16 total: 13 proof tokens + 3 special)

Per Methodology Resolution 1, the decoder generates only the proof body
(`exact fun <vars> => h <args>` with optional `.symm` wrap). The
judge's preamble `intro G _ h` is outside the decoder's output.

| Category | Tokens | Count |
|---|---|---|
| Instruction | `exact` | 1 |
| Structural | `fun`, `=>`, `(`, `)`, `.symm` | 5 |
| Content | `h`, `v1`, `v2`, `v3`, `v4`, `v5`, `v6` | 7 |
| Special | `<BOS>`, `<EOS>`, `<PAD>` | 3 |
| **Vocab size** | | **16** |

The "13 tokens" framing in Methodology Resolution 1 counts only the
proof-emitting tokens (excluding BOS/EOS/PAD). Both numbers refer to
the same vocabulary.

Phase 1 generation is almost entirely content — the witness is the
argument list after `h`. The instruction token `exact` and the
structural tokens are constant across every variable-substitution
proof.

### Phase 2 vocabulary (preparation task)

Before phase 2, re-run tokenization restricted to the matcher-solved pairs PLUS the rewrite-chain category, EXCLUDING the 22% Vampire/Egg proofs. Bucket each token:

- **Instruction** (tactic verbs): `rw`, `simp_rw`, `nth_rw`, `rwa`, `calc`, `conv`, `intro`, `exact`, `apply`, `have`, `show` — a small closed set.
- **Structural** (syntax): `[`, `]`, `(`, `)`, `,`, `:=`, `:`, `=>`, `<-`, `_`, `fun`.
- **Content** (proof-state items): `◇`, `v1`–`v6`, `h`/`h1`…`hN`, `G`, equation-reference tokens, `=`, numbers.

Report the exact count in each bucket. This gives the decoder's true vocabulary size for phase 2 and the instruction-to-content ratio. Expectation: ~10-15 instruction tokens, ~12-15 structural, the remainder content — well under 100 total, since Vampire/Egg (the source of the survey's 3,155-token count) is excluded.

## Data and Train/Test Split

49,586 pairs over 4,694 equations. The split must be at the **equation level**, not the pair level.

Rationale: with only 4,694 equations and 49,586 pairs, a random pair split leaves nearly every test equation present in some training pair. The decoder could then memorize per-equation features instead of learning the embedding→witness map. The research question — does the embedding encode the witness — requires testing on equations the decoder did not train on.

### Split procedure

1. Hold out ~15% of the 4,694 equations (seed 42). Call them held-out equations.
2. **Train set**: pairs where BOTH equations are in the training set.
3. **Test set**: pairs where AT LEAST ONE equation is held out.
4. Within the test set, report two subsets separately:
   - **one-novel**: exactly one of the pair's equations is held out.
   - **both-novel**: both equations are held out (the hardest test).

Expected rough split: holding out 15% of equations puts ~28% of pairs in the test set (~14K test, ~35K train). Report the actual numbers.

The JEPA embeddings themselves were trained on all 4,694 equations (the locked skip_h128_seed42 model used no holdout). So the held-out equations' embeddings are in-distribution for the encoder — the test isolates *decoder* generalization given fixed embeddings, which is the right scope for the phase 1 research question. End-to-end new-equation generalization (a JEPA trained with equation holdout) is a possible later study, not phase 1.

## Training

- **Loss**: token-level cross-entropy with teacher forcing over the proof sequence.
- **Optimizer**: AdamW, lr 3e-4, cosine decay.
- **Batch size**: 256 (proofs are short; batches are cheap).
- **Epochs**: train to convergence on a validation slice of the training set; expect fast convergence given the small vocabulary and short sequences.
- **Regularization**: dropout 0.1. The training set is only ~35K examples for a few-million-parameter model — overfitting is a real risk. The equation-holdout test set is the overfitting detector: if train σ-accuracy is high and test σ-accuracy lags badly, the model is memorizing.
- **Frozen encoder**: no gradient to the JEPA. Only the conditioning projection and decoder update.

## Evaluation

### Primary metric stack (nested, per Methodology Resolution 3)

Three nested rates, each computed on the test set:

1. **Parse rate**: fraction of predictions that are well-formed token
   sequences from which a substitution can be extracted at all.
   Detects malformed output (bad token sequence, missing args, etc.).
2. **Arity-correct rate** (conditional on parse): of those that parse,
   the fraction with the correct number of λ-binders and the correct
   number of `h`-arguments. Detects "right shape, wrong content"
   failures.
3. **σ-exact-match** (conditional on arity-correct): of those with the
   correct arity, the fraction where the substitution and orientation
   exactly match the matcher's ground truth.

This decomposition does the failure-attribution work that an explicit
arity input would do — it tells you whether a failure is malformed
output, wrong arity, or wrong witness — without compromising the
embeddings-only claim. If arity-correct rate is unexpectedly low
(it shouldn't be, given probing showed arity is linearly decodable
from the embedding at ~99%), that's a finding about the embedding
that motivates an explicit-arity follow-up.

Report all three rates on:
- the full test set,
- the one-novel subset,
- the both-novel subset.

### Secondary

- **Lean judge acceptance**: deferred until the Stage 2 Lean environment is built. σ-exact-match is a strong proxy — the proof is structurally determined by σ — but eventually verify a sample against the real judge.
- **Per-arity breakdown**: each metric split by E1's variable count (2, 3, 4, 5, 6). Does the decoder degrade as the substitution gets longer? Per Methodology Resolution 7, **flag small-N noise for arity 5-6 in the report** — the survey showed those are a thin slice of the corpus.
- **Per-orientation breakdown**: direct vs flipped.

### Controls

The research claim ("the embedding encodes the witness") requires showing the signal is in the embeddings, not in the decoder's capacity or in dataset regularities. Two controls:

1. **Random-embedding control**: train the identical decoder with the JEPA embeddings replaced by random vectors (frozen, same dimension, **seed 42** per Methodology Resolution 4). If the real-embedding decoder substantially beats the random-embedding decoder on the test set, the witness signal is in the embeddings. If they perform similarly, the decoder is exploiting dataset regularities (e.g., arity priors), not the embedding geometry.
2. **Majority-baseline**: for each E1 arity, predict the most common substitution shape from the training set. This measures how much of the test accuracy is just "the most frequent permutation for this arity." The decoder must beat this clearly.

### Interpretation

- Decoder ≫ both controls, high on both-novel: the embeddings encode the witness, and the decoder learned a general embedding→σ map. The research claim holds.
- Decoder ≈ random-embedding control: the embeddings don't carry the witness in a decoder-accessible way. A finding — it bounds what the contrastive objective captured.
- Decoder high on one-novel, poor on both-novel: the decoder needs at least one anchor equation it has seen; partial generalization. Also a finding.

## Phase 2 Forward-Look

Phase 2 extends the SAME decoder to rewrite chains. What changes:
- Training data grows to include the rewrite-chain proofs (~1,600 in the directly-proved set).
- Vocabulary grows to the phase 2 set (the re-tokenization task above).
- Optionally, intermediate-equation embeddings get added to the conditioning as generation proceeds — rewrite-chain intermediate states are equations, encodable by the same frozen JEPA encoder.

Architecture, conditioning mechanism, training setup: unchanged. Phase 2 is a curriculum extension, not a redesign — one model, easy-to-hard.

## Notes for Claude Code

- Embeddings come from the locked `skip_h128_seed42_diag` epoch 9 checkpoint, context encoder, the same `equation_embeddings.npy` the MLP classifier used (`outputs/embeddings/v1.0/embeddings.npy`).
- The decoder input is ONLY (z_e1, z_e2). Do not pass parse trees, equation IDs, invariants, or explicit arity. Embeddings only (Methodology Resolution 2) — this is the experimental control, not an implementation convenience. Failure attribution is via the nested metric stack (Resolution 3), not via extra inputs.
- Normalize the matcher's ground-truth proofs to v1…v6 (first-appearance order) for the training targets. Keep the denormalization map for deployment.
- **Phase 1 vocab is 16 tokens** (13 proof + 3 special). Don't include `intro`, `G`, `_` — those are judge-preamble tokens the decoder never emits.
- **Phase 1 metric stack: parse rate → arity-correct rate → σ-exact-match**, nested (each conditional on the previous). Report on full / one-novel / both-novel subsets, plus per-arity (flag small-N noise for arity 5-6), plus per-orientation.
- Equation-level split, seed 42. Report train/test/one-novel/both-novel counts.
- **Phase 2 vocab pre-tokenization is part of phase 1 setup** (Methodology Resolution 5). One tokenization pass on matcher-solved + rewrite-chain proofs (excluding Vampire/Egg), bucketed into instruction/structural/content. Don't defer — it's small and pins the phase 2 vocab now.
- The random-embedding control is not optional — it's how the research claim is established. Build it into the experiment from the start, same architecture, **seed-42 frozen random vectors**.
- Log to wandb under group "proof_decoder".
- Save the best checkpoint by both-novel σ-exact-match.
- Phase 1 is small and fast. If it converges to high both-novel σ-exact-match in the first run, the next step is the phase 2 rewrite-chain curriculum. If it doesn't, the nested-metric decomposition and control breakdowns diagnose why before any architecture change.
