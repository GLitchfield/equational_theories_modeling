"""Tests for src/training/train.py — calibrate_threshold and evaluate."""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.dataset import collate_pairs  # noqa: E402
from src.data.parser import equation_to_pyg  # noqa: E402
from src.model.encoder import EquationEncoder  # noqa: E402
from src.model.jepa import JEPA  # noqa: E402
from src.model.predictor import ImplicationPredictor  # noqa: E402
from src.training.train import calibrate_threshold, evaluate  # noqa: E402


# --- calibrate_threshold ----------------------------------------------------

def test_calibrate_perfect_separation():
    distances = torch.tensor([0.1, 0.2, 0.3, 1.5, 2.0, 2.5])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    thr = calibrate_threshold(distances, labels)
    assert 0.3 < thr < 1.5


def test_calibrate_threshold_predicts_correctly_at_optimum():
    """Apply the calibrated threshold and verify perfect separation cases give 100% bal_acc."""
    distances = torch.tensor([0.1, 0.2, 0.3, 1.5, 2.0, 2.5])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    thr = calibrate_threshold(distances, labels)
    preds = (distances < thr).float()
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    sens = tp / n_pos
    spec = tn / n_neg
    bal = 0.5 * (sens + spec)
    assert bal == 1.0


def test_calibrate_threshold_random_returns_finite():
    torch.manual_seed(0)
    distances = torch.rand(1000) * 2.0
    labels = (torch.rand(1000) > 0.5).float()
    thr = calibrate_threshold(distances, labels)
    assert isinstance(thr, float)
    assert 0.0 <= thr < 5.0


def test_calibrate_only_one_label_class_returns_finite():
    distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    labels = torch.ones(5)
    thr = calibrate_threshold(distances, labels)
    assert isinstance(thr, float)


def test_calibrate_invalid_shapes_raise():
    distances = torch.randn(10)
    labels = torch.randn(8)
    try:
        calibrate_threshold(distances, labels)
    except ValueError:
        return
    raise AssertionError('expected ValueError on shape mismatch')


# --- evaluate ---------------------------------------------------------------

class _TinyPairsDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_evaluate_returns_expected_keys():
    g_a = equation_to_pyg('x = x ◇ y')
    g_b = equation_to_pyg('x ◇ x = x ◇ y')
    items = [
        (g_a, g_b, 1),
        (g_a, g_b, 0),
        (g_b, g_a, 1),
        (g_b, g_a, 0),
    ]
    loader = DataLoader(_TinyPairsDataset(items), batch_size=2, collate_fn=collate_pairs)
    jepa = JEPA(EquationEncoder(), ImplicationPredictor())

    metrics = evaluate(jepa, loader, device='cpu', margin=1.0, true_weight=1.0)

    expected_keys = {
        'loss', 'balanced_accuracy', 'true_recall', 'false_recall',
        'threshold', 'n_evaluated',
    }
    assert set(metrics) == expected_keys
    assert metrics['n_evaluated'] == 4
    assert 0.0 <= metrics['balanced_accuracy'] <= 1.0
    assert 0.0 <= metrics['true_recall'] <= 1.0
    assert 0.0 <= metrics['false_recall'] <= 1.0


def test_evaluate_max_batches_caps_iteration():
    g = equation_to_pyg('x = x ◇ y')
    items = [(g, g, i % 2) for i in range(20)]
    loader = DataLoader(_TinyPairsDataset(items), batch_size=2, collate_fn=collate_pairs)
    jepa = JEPA(EquationEncoder(), ImplicationPredictor())

    metrics = evaluate(
        jepa, loader, device='cpu', margin=1.0, true_weight=1.0, max_batches=3,
    )
    # 3 batches * 2 = 6 samples evaluated
    assert metrics['n_evaluated'] == 6


def test_evaluate_does_not_change_train_mode_state():
    """evaluate() should restore train/eval mode to whatever it was on entry."""
    loader = DataLoader(
        _TinyPairsDataset([(equation_to_pyg('x = x'), equation_to_pyg('x = x'), 1)]),
        batch_size=1, collate_fn=collate_pairs,
    )
    jepa = JEPA(EquationEncoder(), ImplicationPredictor())

    jepa.train()
    evaluate(jepa, loader, device='cpu')
    assert jepa.training

    jepa.eval()
    evaluate(jepa, loader, device='cpu')
    assert not jepa.training


# --- driver ----------------------------------------------------------------

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
