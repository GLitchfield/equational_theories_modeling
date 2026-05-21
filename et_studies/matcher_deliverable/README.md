# matcher_deliverable

Deterministic first-order matcher for the variable-substitution category of ETP proofs (the 62.5% identified by the proof survey). Pure symbolic computation on parse trees — no embeddings, no learned model. Exact correctness by construction.

## Headline

| Set | Coverage | False positives |
|---|---|---|
| Directly-proved (10,658 pairs) | 6,374 / 10,658 = **59.8%** | 0 / 13,855,357 GT FALSE |
| Full TRUE corpus (8.17M pairs) | 49,586 / 8.17M = 0.607% (transitive closure not exploited) | — |

Wall clock: 58.8 s on a single core, pure Python (~375K pairs/sec).

Methodology note: the survey predicted 62.5% based on regex on proof bodies; the matcher catches 59.8% based on "is E2 a substitution instance of E1?". The 2.7pp gap is methodological, not a bug.

## Contents

| Path | What |
|---|---|
| `outputs/README.md` | Matcher writeup + coverage breakdown |
| `outputs/coverage.json` | Coverage stats by holdout level |
| `data/matcher_proofs.parquet` | 49,586-row ground-truth proof dataset (gitignored) |
| `scripts/run_matcher.py` | Reproducer |

The matcher *module* (the algorithm) was promoted into the pilot at `proof_generation_pilot/methods/matcher/`. This study directory keeps the historical reproducer + the coverage artifact.

## Stage 2 submission format

Pinned against `tests/fixtures/answers/accepted_true_basic.answer.json` in the Stage 2 repo:

```
import JudgeProblem

def submission : Goal := by
  intro G _ h
  <body>
```

Live-judge verification deferred to when a built Stage 2 env (Lean 4.30.0-rc2) is available.

## Archived plan

`archive/matcher_plan.md`.
