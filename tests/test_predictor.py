"""Tests for src/model/predictor.py."""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.model.predictor import ImplicationPredictor  # noqa: E402


def test_predictor_forward_shape():
    pred = ImplicationPredictor()
    z = torch.randn(8, 64)
    out = pred(z)
    assert out.shape == (8, 64)


def test_predictor_output_is_unit_norm():
    """Predictor output is L2-normalized; bounds JEPA distances to [0, 2]."""
    torch.manual_seed(0)
    pred = ImplicationPredictor()
    pred.eval()
    z = torch.randn(16, 64)
    out = pred(z)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_predictor_normalize_false_returns_raw_mlp():
    """``normalize=False`` skips the L2 step. Raw output should preserve
    direction with the normalized version (raw / ||raw|| == normed) but
    have non-unit norms — used by VICReg, which needs the unconstrained
    space."""
    torch.manual_seed(0)
    pred = ImplicationPredictor()
    pred.eval()
    z = torch.randn(16, 64)
    normed = pred(z, normalize=True)
    raw = pred(z, normalize=False)
    raw_dir = torch.nn.functional.normalize(raw, dim=-1)
    assert torch.allclose(raw_dir, normed, atol=1e-5)
    raw_norms = raw.norm(dim=-1)
    assert not torch.allclose(raw_norms, torch.ones_like(raw_norms), atol=1e-3)


def test_predictor_default_dims():
    pred = ImplicationPredictor()
    assert pred.embed_dim == 64
    assert pred.hidden_dim == 128


def test_predictor_param_count():
    """Linear(64→128) + Linear(128→64) = 8320 + 8256 = 16,576 params."""
    pred = ImplicationPredictor()
    assert pred.num_parameters() == 16_576


def test_predictor_smaller_than_encoder():
    """Plan's hard constraint: predictor must be smaller than the encoder."""
    from src.model.encoder import EquationEncoder
    enc = EquationEncoder()
    pred = ImplicationPredictor()
    assert pred.num_parameters() < enc.num_parameters()


def test_predictor_is_deterministic_in_train_mode():
    """No dropout — predictor must give identical outputs in train and eval
    mode given identical inputs. This validates the design choice to keep
    the predictor as a clean, deterministic mapping."""
    torch.manual_seed(0)
    pred = ImplicationPredictor()
    z = torch.randn(4, 64)

    pred.train()
    out_train_1 = pred(z)
    out_train_2 = pred(z)
    assert torch.allclose(out_train_1, out_train_2)

    pred.eval()
    out_eval = pred(z)
    assert torch.allclose(out_train_1, out_eval)


def test_predictor_backward_yields_gradients():
    pred = ImplicationPredictor()
    z = torch.randn(4, 64)
    loss = pred(z).pow(2).sum()
    loss.backward()
    assert all(p.grad is not None for p in pred.parameters())
    assert any(p.grad.abs().sum().item() > 0 for p in pred.parameters())


def test_predictor_configurable_dims():
    pred = ImplicationPredictor(embed_dim=32, hidden_dim=64)
    z = torch.randn(5, 32)
    out = pred(z)
    assert out.shape == (5, 32)


def test_predictor_cuda_forward():
    if not torch.cuda.is_available():
        return
    pred = ImplicationPredictor().cuda()
    z = torch.randn(4, 64, device='cuda')
    out = pred(z)
    assert out.shape == (4, 64)
    assert out.device.type == 'cuda'
    assert torch.isfinite(out).all()


# --- driver -----------------------------------------------------------------

if __name__ == '__main__':
    import inspect
    import traceback

    test_funcs = [
        v for k, v in sorted(globals().items())
        if k.startswith('test_') and inspect.isfunction(v)
    ]
    fails = []
    for fn in test_funcs:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except Exception as e:
            fails.append(fn.__name__)
            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print()
    print(f'{len(test_funcs) - len(fails)}/{len(test_funcs)} passed')
    sys.exit(1 if fails else 0)
