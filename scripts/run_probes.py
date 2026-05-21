"""Run all six embedding probing experiments.

Loops over every (checkpoint, representation_type) embedding matrix in
``embeddings/outputs/probing/embeddings/`` and applies the probes specified by
``probing_experiments_plan.md``.

Output layout:
    embeddings/outputs/probing/results/
        exp1_structural.json
        exp2_cohom_summary.json
        exp3_cohom_cell.json
        exp4_non_anchored.json
        exp5_pairwise.json
        exp6_equiv_class_nn.json

Usage:
    python scripts/run_probes.py [--experiments 1 2 3 4 5 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embeddings.src.probing.probes import (  # noqa: E402
    majority_class_baseline,
    probe_classification,
    probe_classification_with_mask,
    probe_regression,
    probe_regression_with_mask,
    top_k_nn_accuracy,
)
from scripts.build_equation_invariants import (  # noqa: E402
    COHOM_PRIMES, COHOM_OFFSETS, COHOM_COUNT_BINS, COHOM_MEAN_BINS,
    _bin_count, _bin_mean,
)

DATA_PROJECT = "/home/gavin/git/equational_theories_data/invariants"
EMBED_DIR = _REPO_ROOT / "embeddings/outputs/probing/embeddings"
RESULTS_DIR = _REPO_ROOT / "embeddings/outputs/probing/results"
PAIR_TABLE = _REPO_ROOT / "data/processed/implication_pairs.parquet"


# --- embedding inventory --------------------------------------------------

def load_embeddings() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (eq_ids, {tag: embedding_matrix}). tag = "<run>_ep<N>_<repr>" or "random_init_<repr>"."""
    eq_ids = np.load(EMBED_DIR / "eq_ids.npy")
    embeds: dict[str, np.ndarray] = {}
    for f in sorted(EMBED_DIR.glob("*.npy")):
        if f.name == "eq_ids.npy":
            continue
        tag = f.stem  # filename without .npy
        embeds[tag] = np.load(f)
    return eq_ids, embeds


# --- target loaders -------------------------------------------------------

def load_invariants(eq_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Load all target-providing parquets aligned to eq_ids order."""
    base = Path(DATA_PROJECT)
    out: dict[str, np.ndarray] = {}

    # Structural (Experiment 1) — also the canonical aux head targets.
    spine = pd.read_parquet(base / "03_spine/output/spine.parquet")
    SPINE_LABELS = ["family_c", "mixed", "no_lhs_var_in_rhs", "left", "right", "trivial"]
    out["spine_type"] = spine.loc[eq_ids, "spine_type"].map(
        {l: i for i, l in enumerate(SPINE_LABELS)}
    ).to_numpy(dtype=np.int64)

    lhs = pd.read_parquet(base / "01_lhs_structure/output/lhs_structure.parquet")
    LHS_LABELS = ["bare_variable", "product"]
    out["lhs_structure"] = lhs.loc[eq_ids, "lhs_structure"].map(
        {l: i for i, l in enumerate(LHS_LABELS)}
    ).to_numpy(dtype=np.int64)

    vars_ = pd.read_parquet(base / "04_variables/output/variables.parquet")
    out["num_variables"] = (vars_.loc[eq_ids, "num_variables"] - 1).to_numpy(dtype=np.int64)
    out["is_balanced"] = vars_.loc[eq_ids, "is_balanced"].astype(int).to_numpy(dtype=np.int64) \
        if "is_balanced" in vars_.columns else None

    magma = pd.read_parquet(base / "08_magma_signature/output/magma_signature.parquet")
    sig = np.stack(magma.loc[eq_ids, "magma_signature"].values).astype(np.int64)
    for bit in [0, 1, 2, 3, 4, 6, 7]:
        out[f"magma_bit_{bit}"] = sig[:, bit]

    # Non-anchored (Experiment 4).
    out["is_trivial"] = pd.read_parquet(base / "02_trivial/output/trivial.parquet").loc[
        eq_ids, "is_trivial"
    ].astype(int).to_numpy(dtype=np.int64)
    out["is_singleton_forcing"] = pd.read_parquet(
        base / "05_singleton_forcing/output/singleton_forcing.parquet"
    ).loc[eq_ids, "is_singleton_forcing"].astype(int).to_numpy(dtype=np.int64)

    leaf = pd.read_parquet(base / "06_leaf_variables/output/leaf_variables.parquet")
    # leftmost_match and rightmost_match are binary; leaf identity is categorical.
    out["leftmost_match"] = leaf.loc[eq_ids, "leftmost_match"].astype(int).to_numpy(dtype=np.int64)
    out["rightmost_match"] = leaf.loc[eq_ids, "rightmost_match"].astype(int).to_numpy(dtype=np.int64)

    tree = pd.read_parquet(base / "07_tree_metrics/output/tree_metrics.parquet")
    out["op_count_asymmetry"] = tree.loc[eq_ids, "op_count_asymmetry"].to_numpy(dtype=np.float64)

    # Spine depth (regression, where defined). spine.parquet may have a depth column.
    if "spine_depth" in spine.columns:
        out["spine_depth"] = spine.loc[eq_ids, "spine_depth"].to_numpy(dtype=np.float64)

    return out


def load_cohom_profiles(eq_ids: np.ndarray) -> np.ndarray:
    """Returns a [N_eq, 87] float array with NaN for non-satisfying cells."""
    cohom = pd.read_parquet(Path(DATA_PROJECT) / "11_cohomology/output/cohomology.parquet").set_index("eq_id")
    return np.stack(cohom.loc[eq_ids, "cohom_profile"].values).astype(np.float64)


def load_cohom_summary_targets(profiles: np.ndarray) -> dict[str, np.ndarray]:
    """count_p / mean_h2_p binned per the canonical scheme."""
    out: dict[str, np.ndarray] = {}
    for p, lo, hi in zip(COHOM_PRIMES, COHOM_OFFSETS[:-1], COHOM_OFFSETS[1:]):
        block = profiles[:, lo:hi]
        cnt = (~np.isnan(block)).sum(axis=1).astype(np.int32)
        with np.errstate(invalid="ignore"):
            mh = np.nanmean(block, axis=1)
        out[f"count_p{p}"] = _bin_count(cnt, COHOM_COUNT_BINS[p]).astype(np.int64)
        out[f"mean_h2_p{p}"] = _bin_mean(mh, COHOM_MEAN_BINS[p]).astype(np.int64)
    return out


def derive_equivalence_classes(eq_ids: np.ndarray) -> np.ndarray:
    """Compute eq_id -> SCC id from the implication pair table.

    Two equations are equivalent iff E1 -> E2 AND E2 -> E1 (mutual implication).
    Returns a [N_eq] int array of class IDs.
    """
    import networkx as nx

    df = pd.read_parquet(PAIR_TABLE)
    g = nx.DiGraph()
    g.add_nodes_from(eq_ids.tolist())
    true_pairs = df[df["value"] == 1]
    g.add_edges_from(zip(true_pairs["eq1_id"].to_numpy(), true_pairs["eq2_id"].to_numpy()))
    sccs = list(nx.strongly_connected_components(g))
    eq_to_class = {eq: cid for cid, members in enumerate(sccs) for eq in members}
    return np.array([eq_to_class[int(eid)] for eid in eq_ids], dtype=np.int64)


# --- experiment runners ---------------------------------------------------

def run_classification_targets(
    embeds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    target_names: list[str],
    label: str,
) -> dict:
    """Probe each named target across every (epoch, repr) embedding."""
    results: dict[str, dict[str, dict]] = {}
    for tag, X in embeds.items():
        per_target: dict[str, dict] = {}
        for tname in target_names:
            y = targets.get(tname)
            if y is None:
                continue
            per_target[tname] = probe_classification(X, y)
        results[tag] = per_target
    return {"label": label, "by_embedding": results}


def run_experiment_1(embeds, targets) -> dict:
    """Structural sanity check: 10 aux-head targets."""
    names = ["spine_type", "lhs_structure", "num_variables"] + \
            [f"magma_bit_{b}" for b in [0, 1, 2, 3, 4, 6, 7]]
    return run_classification_targets(embeds, targets, names, "exp1_structural")


def run_experiment_2(embeds, profiles) -> dict:
    """Per-prime cohom summaries (binned classification)."""
    targets = load_cohom_summary_targets(profiles)
    names = list(targets.keys())
    return run_classification_targets(embeds, targets, names, "exp2_cohom_summary")


def run_experiment_3(embeds, profiles) -> dict:
    """Cell-level cohom: 87 binary 'satisfies' + 87 dim H² regressions."""
    n = profiles.shape[0]
    n_cells = profiles.shape[1]
    out: dict = {"label": "exp3_cohom_cell", "by_embedding": {}, "by_cell": {}}
    # Compute cell -> (a, b, p) mapping.
    triples = [(a, b, p) for p in COHOM_PRIMES for a in range(p) for b in range(p)]
    assert len(triples) == n_cells

    # Per-cell metadata: positive rate + variance of dim H² among satisfying.
    cell_meta = []
    for i in range(n_cells):
        mask = ~np.isnan(profiles[:, i])
        cell_meta.append({
            "triple": triples[i],
            "n_satisfying": int(mask.sum()),
            "satisfies_frac": float(mask.mean()),
            "h2_std": float(np.std(profiles[mask, i])) if mask.any() else 0.0,
        })
    out["cell_metadata"] = cell_meta

    for tag, X in embeds.items():
        per_cell: dict[str, dict] = {}
        for i in range(n_cells):
            triple = triples[i]
            y_bin = (~np.isnan(profiles[:, i])).astype(np.int64)
            frac = y_bin.mean()
            cell_key = f"({triple[0]},{triple[1]},p{triple[2]})"
            per_cell_entry: dict = {"triple": triple, "n_satisfying": int(y_bin.sum())}
            if 0.01 < frac < 0.99 and y_bin.sum() >= 50:
                per_cell_entry["binary"] = probe_classification(X, y_bin)
            mask = y_bin.astype(bool)
            if mask.sum() >= 50:
                y_h2 = profiles[:, i].copy()
                reg_result = probe_regression_with_mask(X, y_h2, mask)
                if reg_result is not None:
                    per_cell_entry["regression_h2"] = reg_result
            per_cell[cell_key] = per_cell_entry
        out["by_embedding"][tag] = per_cell
    return out


def run_experiment_4(embeds, targets) -> dict:
    """Non-anchored invariants: features the model wasn't trained on."""
    classification_names = ["is_trivial", "is_singleton_forcing", "is_balanced",
                            "leftmost_match", "rightmost_match"]
    out: dict = {"label": "exp4_non_anchored", "by_embedding": {}}
    for tag, X in embeds.items():
        per_target: dict = {}
        for tname in classification_names:
            y = targets.get(tname)
            if y is None:
                continue
            per_target[tname] = probe_classification(X, y)
        # Regression targets: op_count_asymmetry, spine_depth (if available).
        for tname in ["op_count_asymmetry", "spine_depth"]:
            y = targets.get(tname)
            if y is None:
                continue
            per_target[tname] = probe_regression(X, y)
        out["by_embedding"][tag] = per_target
    return out


def run_experiment_5(embeds, profiles, eq_ids, n_pairs: int = 100_000, seed: int = 0) -> dict:
    """Pairwise cohom: has_obstruction, compatibility, ground-truth baseline."""
    df = pd.read_parquet(PAIR_TABLE)
    rng = np.random.default_rng(seed)

    # Stratified sample: half TRUE, half FALSE.
    true_df = df[df["value"] == 1]
    false_df = df[df["value"] == 0]
    n_true = min(n_pairs // 2, len(true_df))
    n_false = min(n_pairs - n_true, len(false_df))
    pairs_true = true_df.sample(n=n_true, random_state=seed)
    pairs_false = false_df.sample(n=n_false, random_state=seed + 1)
    pairs_df = pd.concat([pairs_true, pairs_false], ignore_index=True)
    pairs_df = pairs_df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    print(f"  exp5 sampled {len(pairs_df):,} pairs ({n_true:,} TRUE, {n_false:,} FALSE)")

    # Map eq_id -> row index in eq_ids/profiles.
    id_to_idx = {int(eid): i for i, eid in enumerate(eq_ids)}
    idx1 = np.array([id_to_idx[int(e)] for e in pairs_df["eq1_id"].to_numpy()])
    idx2 = np.array([id_to_idx[int(e)] for e in pairs_df["eq2_id"].to_numpy()])
    y_truth = pairs_df["value"].to_numpy(dtype=np.int64)

    # Compute pair-level cohom features.
    p1 = profiles[idx1]  # [N, 87]
    p2 = profiles[idx2]
    both_have = (~np.isnan(p1)) & (~np.isnan(p2))  # shared satisfying cells
    p1_only = (~np.isnan(p1)) & np.isnan(p2)
    p2_only = np.isnan(p1) & (~np.isnan(p2))

    # has_obstruction: any shared cell where dim_h2(eq1) > dim_h2(eq2).
    # Use np.where to safely compare ignoring NaNs.
    obstruction_per_cell = np.zeros_like(p1, dtype=bool)
    obstruction_per_cell[both_have] = (p1[both_have] > p2[both_have])
    has_obstruction = obstruction_per_cell.any(axis=1).astype(np.int64)

    # compatibility = fraction of shared cells where dim H² agrees.
    n_shared = both_have.sum(axis=1)
    n_agree = (both_have & (p1 == p2)).sum(axis=1)
    has_shared = n_shared > 0
    compatibility = np.full(len(idx1), np.nan, dtype=np.float64)
    compatibility[has_shared] = n_agree[has_shared] / n_shared[has_shared]

    # Subset masks for has_obstruction.
    both_have_any = (~np.isnan(p1)).any(axis=1) & (~np.isnan(p2)).any(axis=1)
    one_or_both_null = ~both_have_any

    out: dict = {
        "label": "exp5_pairwise",
        "n_pairs": int(len(pairs_df)),
        "n_true": int(n_true),
        "n_false": int(n_false),
        "n_clean_subset": int(both_have_any.sum()),
        "n_confounded_subset": int(one_or_both_null.sum()),
        "n_with_shared_triples": int(has_shared.sum()),
        "by_embedding": {},
    }

    pair_reps = ("concat", "hadamard", "absdiff")

    for tag, X in embeds.items():
        z1 = X[idx1]
        z2 = X[idx2]
        per_repr: dict = {}
        for pair_rep in pair_reps:
            if pair_rep == "concat":
                Xpair = np.concatenate([z1, z2], axis=1)
            elif pair_rep == "hadamard":
                Xpair = z1 * z2
            else:  # absdiff
                Xpair = np.abs(z1 - z2)

            # 5a: has_obstruction on three subsets.
            sub_results: dict = {}
            for sub_name, sub_mask in [
                ("clean", both_have_any),
                ("confounded", one_or_both_null),
                ("all", np.ones_like(both_have_any, dtype=bool)),
            ]:
                if sub_mask.sum() < 100:
                    continue
                Xsub = Xpair[sub_mask]
                ysub = has_obstruction[sub_mask]
                if len(np.unique(ysub)) < 2:
                    sub_results[sub_name] = {"skipped": "constant target", "n": int(sub_mask.sum())}
                    continue
                sub_results[sub_name] = probe_classification(Xsub, ysub)
            # 5b: compatibility regression (drop unshared).
            comp_result = None
            if has_shared.sum() >= 100:
                Xc = Xpair[has_shared]
                yc = compatibility[has_shared]
                comp_result = probe_regression(Xc, yc)
                comp_result["n_dropped_unshared"] = int((~has_shared).sum())
            # 5c: ground-truth baseline (linear classifier on the same pair rep).
            truth_result = probe_classification(Xpair, y_truth)

            per_repr[pair_rep] = {
                "has_obstruction": sub_results,
                "compatibility": comp_result,
                "ground_truth": truth_result,
            }
        out["by_embedding"][tag] = per_repr
    return out


def run_experiment_6(embeds, eq_ids) -> dict:
    """Equivalence class via nearest-neighbor accuracy."""
    print("  exp6 deriving equivalence classes from implication graph ...")
    classes = derive_equivalence_classes(eq_ids)
    n_classes = len(np.unique(classes))
    class_sizes = np.bincount(classes)
    out: dict = {
        "label": "exp6_equiv_class_nn",
        "n_equations": int(len(eq_ids)),
        "n_classes": int(n_classes),
        "class_size_p50": int(np.median(class_sizes)),
        "class_size_p95": int(np.percentile(class_sizes, 95)),
        "class_size_max": int(class_sizes.max()),
        "n_singletons": int((class_sizes == 1).sum()),
        "by_embedding": {},
    }
    print(f"  exp6 {n_classes} classes; size p50={out['class_size_p50']}, "
          f"p95={out['class_size_p95']}, max={out['class_size_max']}, "
          f"singletons={out['n_singletons']}")

    for tag, X in embeds.items():
        out["by_embedding"][tag] = top_k_nn_accuracy(X, classes)
    return out


# --- main -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
        help="Experiment numbers to run (1..6).",
    )
    args = parser.parse_args(argv)
    requested = set(args.experiments)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading embeddings ...")
    eq_ids, embeds = load_embeddings()
    print(f"  {len(eq_ids):,} equations, {len(embeds)} embedding sets:")
    for tag, X in embeds.items():
        print(f"    {tag}: {X.shape}")

    print("loading invariant targets ...")
    targets = load_invariants(eq_ids)
    print(f"  {len(targets)} target arrays loaded")

    if any(e in requested for e in (2, 3, 5)):
        print("loading cohomology profiles ...")
        profiles = load_cohom_profiles(eq_ids)
    else:
        profiles = None

    def _save(name: str, payload: dict) -> None:
        path = RESULTS_DIR / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"  wrote {path}")

    if 1 in requested:
        print("\n== Experiment 1: structural sanity check ==")
        t0 = time.perf_counter()
        _save("exp1_structural.json", run_experiment_1(embeds, targets))
        print(f"  exp1 {time.perf_counter()-t0:.1f}s")
    if 2 in requested:
        print("\n== Experiment 2: per-prime cohom summaries ==")
        t0 = time.perf_counter()
        _save("exp2_cohom_summary.json", run_experiment_2(embeds, profiles))
        print(f"  exp2 {time.perf_counter()-t0:.1f}s")
    if 3 in requested:
        print("\n== Experiment 3: cell-level cohom (87 cells) ==")
        t0 = time.perf_counter()
        _save("exp3_cohom_cell.json", run_experiment_3(embeds, profiles))
        print(f"  exp3 {time.perf_counter()-t0:.1f}s")
    if 4 in requested:
        print("\n== Experiment 4: non-anchored invariants ==")
        t0 = time.perf_counter()
        _save("exp4_non_anchored.json", run_experiment_4(embeds, targets))
        print(f"  exp4 {time.perf_counter()-t0:.1f}s")
    if 5 in requested:
        print("\n== Experiment 5: pairwise cohom ==")
        t0 = time.perf_counter()
        _save("exp5_pairwise.json", run_experiment_5(embeds, profiles, eq_ids))
        print(f"  exp5 {time.perf_counter()-t0:.1f}s")
    if 6 in requested:
        print("\n== Experiment 6: equivalence-class NN ==")
        t0 = time.perf_counter()
        _save("exp6_equiv_class_nn.json", run_experiment_6(embeds, eq_ids))
        print(f"  exp6 {time.perf_counter()-t0:.1f}s")

    print("\nall requested experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
