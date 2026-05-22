"""grind → Lean adapter for the Phase 1 pilot.

`grind` is Lean's built-in e-graph-based equational reasoning tactic.
For each pair, we emit `def submission := by intro G _ h; grind` and let
Lean's `grind` find the proof from the local hypothesis `h`. In Lean
4.29 `grind` picks up local hypotheses automatically — passing `[h]`
explicitly is flagged as redundant.

Yields 2 transitions per pair (initial → post-intro → closed).

Public API:
    make_submission(eq1_id, eq2_id) -> str
"""

from __future__ import annotations


def make_submission(eq1_id: int, eq2_id: int) -> str:
    return (
        f"def submission : ∀ (G : Type) [Magma G], "
        f"Equation{eq1_id} G → Equation{eq2_id} G := by\n"
        f"  intro G _ h\n"
        f"  grind\n"
    )


def expected_tactics() -> list[str]:
    return ['intro G _ h', 'grind']
