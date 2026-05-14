"""Extract embeddings from one or more JEPA checkpoints for linear probing.

For each (checkpoint, representation_type) we save a [N_eq, dim] numpy
matrix of embeddings, indexed by sorted eq_id.

Two representation types per checkpoint:
  - ``pooled``  : 2 * hidden_dim, pre-projection (mean+max pool concat).
                  This is the rep the auxiliary heads were trained on.
  - ``output``  : embed_dim, post-projection, L2-normalized. This is the
                  rep the contrastive loss operates on.

Plus a ``random_init`` set: a freshly-instantiated JEPA (same architecture
as the trained run, no weights loaded) — the no-information floor for
probing baselines.

Usage:
    python scripts/extract_probe_embeddings.py \
        --run-name aux_h128_seed1_diag \
        --epochs 0 3 6 \
        --include-random-init

Outputs to ``outputs/probing/embeddings/``:
    eq_ids.npy                                   # [N_eq] int64
    aux_h128_seed1_diag_ep0_pooled.npy           # [N_eq, 256]
    aux_h128_seed1_diag_ep0_output.npy           # [N_eq, 512]
    aux_h128_seed1_diag_ep3_pooled.npy
    aux_h128_seed1_diag_ep3_output.npy
    aux_h128_seed1_diag_ep6_pooled.npy
    aux_h128_seed1_diag_ep6_output.npy
    random_init_pooled.npy                       # [N_eq, 256]
    random_init_output.npy                       # [N_eq, 512]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import load_graph_cache  # noqa: E402
from src.model.encoder import EquationEncoder  # noqa: E402


def build_encoder(config: dict, seed: int | None = None) -> EquationEncoder:
    """Construct an EquationEncoder from the saved config dict."""
    if seed is not None:
        torch.manual_seed(seed)
    enc_cfg = config["encoder"]
    return EquationEncoder(
        node_features=enc_cfg["node_features"],
        edge_features=enc_cfg["edge_features"],
        hidden_dim=enc_cfg["hidden_dim"],
        embed_dim=enc_cfg["embed_dim"],
        num_layers=enc_cfg["num_layers"],
        num_heads=enc_cfg["num_heads"],
        dropout=enc_cfg["dropout"],
        skip_pooled=enc_cfg.get("skip_pooled", False),
        proj_dim=enc_cfg.get("proj_dim", None),
    )


def load_encoder_from_ckpt(ckpt_path: Path, device: str) -> tuple[EquationEncoder, dict]:
    """Load JEPA checkpoint, extract context_encoder weights, return encoder + config."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    config = ckpt["config"]
    encoder = build_encoder(config)
    # Strip the 'context_encoder.' prefix from JEPA state_dict keys.
    enc_state = {
        k[len("context_encoder."):]: v
        for k, v in ckpt["jepa_state_dict"].items()
        if k.startswith("context_encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {ckpt_path}: missing={missing}, unexpected={unexpected}"
        )
    return encoder.to(device).eval(), config


@torch.no_grad()
def extract_embeddings(
    encoder: EquationEncoder,
    graphs: dict[int, "Data"],
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run every equation through the encoder. Return (eq_ids, pooled, output)."""
    eq_ids = sorted(graphs.keys())
    pooled_chunks: list[np.ndarray] = []
    output_chunks: list[np.ndarray] = []
    for i in range(0, len(eq_ids), batch_size):
        chunk_ids = eq_ids[i:i + batch_size]
        batch = Batch.from_data_list([graphs[eid] for eid in chunk_ids]).to(device)
        out, pooled = encoder(batch, normalize=True, return_pooled=True)
        pooled_chunks.append(pooled.cpu().numpy())
        output_chunks.append(out.cpu().numpy())
    return (
        np.array(eq_ids, dtype=np.int64),
        np.concatenate(pooled_chunks, axis=0),
        np.concatenate(output_chunks, axis=0),
    )


def save_pair(name: str, pooled: np.ndarray, output: np.ndarray, out_dir: Path) -> None:
    np.save(out_dir / f"{name}_pooled.npy", pooled)
    np.save(out_dir / f"{name}_output.npy", output)
    print(f"  wrote {name}_pooled.npy [{pooled.shape}] and {name}_output.npy [{output.shape}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="Run name (matches outputs/checkpoints/<name>/)")
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=[0, 3, 6],
        help="Epoch numbers to probe (matches epoch_NNN.pt files)",
    )
    parser.add_argument("--include-random-init", action="store_true",
                        help="Also save a random-init baseline (instantiates encoder fresh, no checkpoint).")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="Seed for random-init weights (default 0).")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--graph-cache", default="data/processed/equation_graphs.pt")
    parser.add_argument("--output-dir", default="outputs/probing/embeddings")
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading graph cache ...")
    graphs = load_graph_cache(args.graph_cache)
    print(f"  {len(graphs):,} equations")

    eq_ids_path = out_dir / "eq_ids.npy"
    eq_ids_saved = False

    saved_config: dict | None = None
    for ep in args.epochs:
        ckpt_path = Path(f"outputs/checkpoints/{args.run_name}/epoch_{ep:03d}.pt")
        if not ckpt_path.exists():
            print(f"WARN: {ckpt_path} not found, skipping")
            continue
        print(f"loading {ckpt_path} ...")
        encoder, cfg = load_encoder_from_ckpt(ckpt_path, device)
        if saved_config is None:
            saved_config = cfg
        eq_ids, pooled, output = extract_embeddings(encoder, graphs, device)
        if not eq_ids_saved:
            np.save(eq_ids_path, eq_ids)
            print(f"  wrote eq_ids.npy [{eq_ids.shape}]")
            eq_ids_saved = True
        save_pair(f"{args.run_name}_ep{ep}", pooled, output, out_dir)
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.include_random_init:
        if saved_config is None:
            raise SystemExit("--include-random-init needs at least one valid epoch ckpt for the config")
        print(f"building random-init encoder (seed={args.random_seed}) ...")
        encoder = build_encoder(saved_config, seed=args.random_seed).to(device).eval()
        eq_ids, pooled, output = extract_embeddings(encoder, graphs, device)
        if not eq_ids_saved:
            np.save(eq_ids_path, eq_ids)
            eq_ids_saved = True
        save_pair("random_init", pooled, output, out_dir)

    print(f"\nall embeddings written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
