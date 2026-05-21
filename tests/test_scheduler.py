"""Tests for src/training/scheduler.py."""

from __future__ import annotations

import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings.src.training.scheduler import (  # noqa: E402
    cosine_tau,
    cosine_warmup_lr_multiplier,
)


# --- LR ---------------------------------------------------------------------

def test_lr_step_zero_with_warmup_is_zero():
    assert cosine_warmup_lr_multiplier(0, total_steps=1000, warmup_steps=100) == 0.0


def test_lr_at_warmup_end_is_one():
    assert cosine_warmup_lr_multiplier(100, total_steps=1000, warmup_steps=100) == 1.0


def test_lr_at_total_steps_is_zero():
    assert cosine_warmup_lr_multiplier(1000, total_steps=1000, warmup_steps=100) == 0.0


def test_lr_beyond_total_steps_is_zero():
    assert cosine_warmup_lr_multiplier(2000, total_steps=1000, warmup_steps=100) == 0.0


def test_lr_warmup_is_linear_at_midpoint():
    assert cosine_warmup_lr_multiplier(50, total_steps=1000, warmup_steps=100) == 0.5


def test_lr_cosine_midpoint_is_half():
    """Halfway through cosine decay: progress=0.5 → 0.5 * (1 + cos(pi/2)) = 0.5."""
    m = cosine_warmup_lr_multiplier(550, total_steps=1000, warmup_steps=100)
    assert abs(m - 0.5) < 1e-9


def test_lr_no_warmup_starts_at_one():
    """warmup_steps=0 → first step is the start of cosine decay = 1.0."""
    assert cosine_warmup_lr_multiplier(0, total_steps=1000, warmup_steps=0) == 1.0


def test_lr_monotonically_decreases_after_warmup():
    """LR after warmup should monotonically decrease toward 0."""
    values = [
        cosine_warmup_lr_multiplier(s, total_steps=1000, warmup_steps=100)
        for s in [100, 200, 400, 600, 800, 1000]
    ]
    for a, b in zip(values, values[1:]):
        assert a >= b, f'expected non-increasing LR, got {values}'


def test_lr_invalid_args_raise():
    for args in [
        (-1, 1000, 100),       # negative step
        (0, 0, 0),             # zero total_steps
        (0, 1000, -10),        # negative warmup
        (0, 1000, 1500),       # warmup > total
    ]:
        try:
            cosine_warmup_lr_multiplier(*args)
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for {args}')


# --- tau --------------------------------------------------------------------

def test_tau_step_zero_is_tau_start():
    tau = cosine_tau(0, total_steps=1000, tau_start=0.996, tau_final=0.999)
    assert abs(tau - 0.996) < 1e-9


def test_tau_at_total_steps_is_tau_final():
    tau = cosine_tau(1000, total_steps=1000, tau_start=0.996, tau_final=0.999)
    assert abs(tau - 0.999) < 1e-9


def test_tau_beyond_total_steps_clipped_to_final():
    tau = cosine_tau(2000, total_steps=1000, tau_start=0.996, tau_final=0.999)
    assert abs(tau - 0.999) < 1e-9


def test_tau_midpoint():
    """At step=500/1000 with tau_start=0.996, tau_final=0.999:
    progress=0.5, cos(pi*0.5)=0, so tau = 0.999 - 0.003 * 0.5 = 0.9975."""
    tau = cosine_tau(500, total_steps=1000, tau_start=0.996, tau_final=0.999)
    assert abs(tau - 0.9975) < 1e-9


def test_tau_monotonically_increases():
    """Tau grows from tau_start to tau_final over training."""
    taus = [
        cosine_tau(s, total_steps=1000, tau_start=0.996, tau_final=0.999)
        for s in [0, 100, 250, 500, 750, 1000]
    ]
    for a, b in zip(taus, taus[1:]):
        assert a <= b, f'expected non-decreasing tau, got {taus}'


def test_tau_default_values():
    """Default tau_start=0.996, tau_final=0.999 from the plan."""
    assert abs(cosine_tau(0, total_steps=1000) - 0.996) < 1e-9
    assert abs(cosine_tau(1000, total_steps=1000) - 0.999) < 1e-9


def test_tau_custom_values():
    tau = cosine_tau(0, total_steps=1000, tau_start=0.9, tau_final=0.99)
    assert abs(tau - 0.9) < 1e-9
    tau = cosine_tau(1000, total_steps=1000, tau_start=0.9, tau_final=0.99)
    assert abs(tau - 0.99) < 1e-9


def test_tau_invalid_args_raise():
    for args, kwargs in [
        ((-1, 1000), {}),
        ((0, 0), {}),
        ((0, 1000), {'tau_start': -0.1}),
        ((0, 1000), {'tau_final': 1.1}),
    ]:
        try:
            cosine_tau(*args, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for {args} {kwargs}')


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
