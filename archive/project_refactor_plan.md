# Project Refactor

## Goal

Restructure the project into four top-level areas before the proof generation pilot begins.

## Target structure

```
~/git/equational_theories_data/
├── data/
│   └── raw/                                 # ground truth, shared
├── embeddings/                              # JEPA, MLP classifier, decoder probe
│   ├── data/processed/
│   ├── outputs/
│   ├── scripts/
│   ├── notebooks/
│   └── README.md
├── et_studies/                              # frozen analyses
│   ├── coverage_measurement/
│   ├── completion_exploration/
│   ├── proof_investigation/
│   ├── proof_corpus_survey/
│   ├── etp_structure_analysis/
│   └── README.md                            # index of studies
├── proof_generation_pilot/                  # active phase
│   ├── data/processed/
│   ├── outputs/
│   ├── methods/
│   │   ├── matcher/
│   │   ├── completion/
│   │   └── safe_engine/
│   ├── scripts/
│   ├── notebooks/
│   └── README.md
└── README.md                                # project overview
```

## Phases

### Phase 1 — Survey current state

Goal: understand the existing layout before moving anything.

Steps:
1. Map current directory structure: `find ~/git/equational_theories_data -maxdepth 3 -type d > current_layout.txt`
2. Identify where each component currently lives:
   - Embedding code and data (JEPA, MLP, decoder probe)
   - Symbolic method code (matcher, completion, safe rewriting engine)
   - Coverage measurement outputs and scripts
   - Completion exploration outputs
   - Proof investigation outputs (Phase 1 exploration notes)
   - Proof corpus survey outputs
   - ETP structure analysis outputs (if persistent files exist)
3. Identify cross-references (imports, hard-coded paths, config files referencing existing paths).

Output: `refactor_inventory.md` listing current → target mappings for each component.

Checkpoint: review the inventory before moving anything.

### Phase 2 — Create new directory structure

Create empty target directories. Don't move files yet.

### Phase 3 — Move studies into et_studies/

For each frozen analysis:
1. `git mv` data, outputs, scripts, notebooks into the corresponding et_studies/ subdirectory.
2. Verify the files moved with history preserved.

These are frozen — they don't need to keep running, just stay readable.

### Phase 4 — Move symbolic methods into pilot

The matcher, completion (plain and ordered KB), and safe rewriting engine move into `proof_generation_pilot/methods/`:
1. `git mv` the relevant source files.
2. Update imports that reference the old paths.
3. Verify the methods still run from their new location.

### Phase 5 — Move embedding work into embeddings/

JEPA model, MLP classifier, decoder probe code and data move into `embeddings/`:
1. `git mv` everything to the new location.
2. Update import paths.
3. Verify the embedding pipeline still loads.

### Phase 6 — Update path references

Audit and update:
- Hard-coded paths in scripts
- Config files
- Notebook paths
- README files referencing old paths
- `/mnt/user-data/outputs/proof_generation_pilot_plan.md` — currently uses old structure

### Phase 7 — Write READMEs

One README per top-level directory plus a project root README:

- `README.md` (root): project overview, links to each area, current phase
- `embeddings/README.md`: what's in here, key results (JEPA 98.4% balanced accuracy, MLP 99.96%, decoder probe 96.3% σ-exact-match)
- `et_studies/README.md`: index of completed studies with brief outcome for each
- `et_studies/coverage_measurement/README.md`: Phase 1-7 summary including 6.54% combined TRUE coverage, 14.19% engine plateau, frontier_died domination at 85.74%, row-density anti-correlation
- `et_studies/completion_exploration/README.md`: 6.06% TRUE / 17.82% FALSE / 13.46% combined sound coverage, 1,133 newly-completed equations finding
- `et_studies/proof_investigation/README.md`: Phase 1 exploration findings — repo structure, format verification, Vampire/Egg archetype distinction
- `et_studies/proof_corpus_survey/README.md`: category counts (1,507 Vampire, 796 Egg, 45 complex, etc.) and survey methodology
- `et_studies/etp_structure_analysis/README.md`: closure architecture findings — ~11K direct proofs + Closure.lean derives 8M+ transitive labels from ~4 MB source evidence
- `proof_generation_pilot/README.md`: current phase status, link to pilot plan

### Phase 8 — Verify

1. Sanity check: can the pilot methods (matcher, completion, engine) be imported and called from their new location?
2. Embedding pipeline loads from `embeddings/`.
3. Studies' outputs readable (parquet files load, markdown renders).
4. Git status clean, history preserved.
5. Reference the pilot plan to confirm path updates went through.

## Decisions to surface during execution

1. **Naming**: `et_studies/` is the default. Switch to `analyses/` if preferred.
2. **Phase 7 synthesis status**: if not finished, flag it in the coverage_measurement README. If in progress, leave the working file in place.
3. **Methods organization**: matcher/completion/safe_engine is my proposed division. If the actual code groups differently (e.g., completion is split between plain and ordered KB across files), adapt.
4. **Path-reference dependencies in archived studies**: if updating paths in studies breaks their notebooks or scripts, decide whether to fix them or leave them as historical records. Leaving them broken-but-readable is acceptable since the studies are frozen.

## Success criteria

- New directory structure in place.
- All files moved with git history preserved.
- Symbolic methods still run from their new location (verified by a small smoke test).
- Embedding pipeline still loads from its new location.
- READMEs at each top-level directory and each study subdirectory.
- Pilot plan updated with new paths.
- Git tree clean.
