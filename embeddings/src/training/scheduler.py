"""LR and EMA tau schedules for JEPA training.

Both are pure functions of the current step. The training loop calls them
each step and applies the result. Pure-function form keeps testing trivial
and avoids coupling to PyTorch's lr_scheduler API.
"""

from __future__ import annotations

import math


def cosine_warmup_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Return the LR multiplier in [0, 1] for the given step.

    - Linear warmup from 0 → 1 over ``[0, warmup_steps]``.
    - Cosine decay from 1 → 0 over ``[warmup_steps, total_steps]``.
    - Returns 0 for any step ≥ ``total_steps``.

    The training loop applies this as ``lr_at_step = base_lr * multiplier``.
    """
    if step < 0:
        raise ValueError(f'step must be non-negative; got {step}')
    if total_steps <= 0:
        raise ValueError(f'total_steps must be positive; got {total_steps}')
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(
            f'warmup_steps must be in [0, total_steps]; '
            f'got {warmup_steps} of {total_steps}'
        )

    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    if step >= total_steps:
        return 0.0
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(decay_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_tau(
    step: int,
    total_steps: int,
    tau_start: float = 0.996,
    tau_final: float = 0.999,
) -> float:
    """Cosine anneal of the EMA tau from ``tau_start`` to ``tau_final``.

    - At ``step = 0``: returns ``tau_start`` (faster target tracking early).
    - At ``step ≥ total_steps``: returns ``tau_final`` (slower late on).
    - Smooth cosine interpolation between.

    For the plan's defaults (0.996 → 0.999), tau is small and growing — the
    target encoder lags the context encoder more as training progresses.
    """
    if step < 0:
        raise ValueError(f'step must be non-negative; got {step}')
    if total_steps <= 0:
        raise ValueError(f'total_steps must be positive; got {total_steps}')
    if not 0.0 <= tau_start <= 1.0 or not 0.0 <= tau_final <= 1.0:
        raise ValueError(
            f'tau_start and tau_final must be in [0, 1]; '
            f'got {tau_start}, {tau_final}'
        )

    progress = min(step / total_steps, 1.0)
    return tau_final - (tau_final - tau_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
