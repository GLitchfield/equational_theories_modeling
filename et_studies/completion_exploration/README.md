# completion_exploration

Knuth-Bendix completion (plain + ordered/unfailing) on the 22M-pair ETP problem. Tests whether completion can serve as a unified TRUE/FALSE decision procedure.

## Headline

| Side | Engine | Coverage of GT |
|---|---|---|
| TRUE  | Ordered (unfailing) KB | 6.06% of GT TRUE |
| TRUE  | Plain KB | 2.13% of GT TRUE |
| FALSE | Plain KB (sound) | 17.82% of GT FALSE |
| FALSE | Ordered KB | unsound (precedence-dependent rewriting) — not counted |
| **Combined sound** | **TRUE from ordered + FALSE from plain** | **13.46% of 22M** |

Notable findings:
- **Ordered completion adds a sound TRUE-coverage capability** that plain KB structurally lacks (3× lift, 2.13% → 6.06%).
- **3.28% of GT FALSE pairs come with a Stage-2-ready finite counter-model** (`Fin n ≤ 6`) for free — directly usable for the FALSE side of the pilot.
- 1,133 newly-completed equations beyond the plain-KB baseline.

## Contents

| Path | What |
|---|---|
| `outputs/synthesis_report.md` | Full study writeup |
| `outputs/phase0_variable_inclusion.json` | Pre-check on variable-inclusion ceiling |
| `outputs/phase2_completion_outcomes.json`, `phase2_cap_sweep_60.json` | Plain KB outcomes + size-cap sweep |
| `outputs/phase3_decision_coverage.json`, `phase3_cap_sweep_60.json` | Decision procedure + cap sweep |
| `outputs/phase4_decision_coverage.json`, `phase4_ordered_outcomes.json` | Ordered KB outcomes |
| `outputs/phase4a_failed_cp_classification.json` | Classification of failed critical-pair convergences |
| `outputs/phase4b_strong_vs_weak.json` | Strong vs weak ordering breakdown |
| `outputs/phase5_finite_models.json` | Finite-model extraction results |
| `scripts/phase{0,2,3,4,4a,4b,5}_*.py` | Per-phase reproducers |

The completion engines themselves were promoted to the pilot at `proof_generation_pilot/methods/completion/` (`kb.py` for plain, `ordered.py` for unfailing). 99 unit tests cover them.

## Archived plan

`archive/completion_exploration_plan.md`.
