# data/ground_truth/

Ground-truth implication labels for all 4,694 × 4,694 equation pairs, redistributed from the [Equational Theories Project](https://github.com/teorth/equational_theories) (ETP) under its Apache-2.0 license.

| File | What |
|---|---|
| `export_raw_implications_5_9_2026.csv.gz` | The ETP raw-implications export, gzip-compressed byte-for-byte |

## Provenance

Downloaded on 2026-05-09 from the "export raw implications" button in the ETP [implications explorer](https://teorth.github.io/equational_theories/implications/) (`downloadRawImplicationsCSV` in the ETP repo's `home_page/implications/script.js`). The labels come from the ETP's Lean-verified results.

- Original CSV: 57,922,628 bytes, sha256 `fde7f1b56477cb6768d6e5b1573a8622108034ea11c16ab244797eb9a8ee2f62`
- Compressed with `gzip -9n -c export_raw_implications_5_9_2026.csv > export_raw_implications_5_9_2026.csv.gz`

Check the decompressed file with `gunzip -c export_raw_implications_5_9_2026.csv.gz | sha256sum`.

## Format

A dense 4694 × 4694 comma-separated int8 matrix. `M[i, j]` is the status of "equation `i+1` implies equation `j+1`" (1-indexed into ETP's `data/equations.txt`). Only four codes occur:

| Code | Meaning |
|---|---|
| `4` | TRUE, explicitly proven (a direct proof exists for this pair; 10,657 entries) |
| `3` | TRUE, implicitly proven (follows by transitive closure of explicit results) |
| `-4` | FALSE, explicitly refuted (a direct counterexample exists for this pair) |
| `-3` | FALSE, implicitly refuted (follows by transitive closure of explicit results) |

The explorer's export also defines codes for conjectured (±1, ±2) and unknown (0) statuses; none occur in this snapshot, so every pair is settled in Lean.

`lib/scripts/build_pair_table.py` reads this file directly (`np.loadtxt` handles `.gz`) and writes the off-diagonal pair table to `data/processed/implication_pairs.parquet`. See the CHANGELOG entry for 2026-05-09 for why this export replaced the earlier Vampire-derived labels.

## Attribution

The Equational Theories Project, led by Terence Tao and contributors. If you use these labels, cite the ETP (see its [`CITATION.cff`](https://github.com/teorth/equational_theories/blob/main/CITATION.cff)).
