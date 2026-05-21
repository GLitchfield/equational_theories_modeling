# etp_structure_analysis

Carved out from `proof_investigation` Phase 1. Documents the ETP repository's two-tier proof architecture.

## Headline

- **~11K direct proofs in source** — explicit human/tool-written proofs.
- **8M+ transitive labels derived by `Closure.lean`** from those ~11K seeds plus the FALSE side.
- Total ETP source evidence: **~4 MB** producing the entire 22M off-diagonal pair table.

The implication is that ground truth for the 22M-pair problem is *not* 22M independently-asserted facts — it's an explicit seed set + a closure pass. Any work characterizing what proof generation needs to produce has to keep this distinction front of mind: only the seed set has explicit proof bodies; the closure-derived pairs only have labels.

This finding is what motivated:
- The proof-corpus survey targeting only the ~11K seeds (where proof bodies exist).
- The matcher's distinction between "directly-proved pairs" (the seeds) and "closure-derived pairs" (the rest).
- Coverage measurements being reported separately for the seed set vs the full 22M.

## Contents

| Path | What |
|---|---|
| `outputs/etp_proof_structure_note.md` | The structural note |

No `scripts/` or `data/` — this study is a documentation deliverable, not a computational one. The reproducer would be re-reading the ETP repository tree and verifying file sizes / proof counts.

## Related

Most studies in `et_studies/` reference the ~11K seed count vs 22M total distinction. This note is the canonical explanation.
