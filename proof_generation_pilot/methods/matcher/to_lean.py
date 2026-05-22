"""Matcher → Lean adapter for the Phase 1 pilot.

Wraps the existing `lean_proof_tactic` column from matcher_proofs.parquet
in `def submission := by intro G _ h; <tactic>`. The output is tactic-mode
Lean PyPantograph can step through (yields ~2 transitions per pair:
initial → post-intro → closed).

Public API:
    make_submission(eq1_id, eq2_id, lean_proof_tactic) -> str
"""

from __future__ import annotations


def make_submission(eq1_id: int, eq2_id: int, lean_proof_tactic: str) -> str:
    """Build the full `def submission` block for the matcher proof of
    Equation<eq1_id> G → Equation<eq2_id> G.

    The `lean_proof_tactic` is the matcher's pre-computed body (e.g.
    `exact fun x y z w => h x y y z w`). We prepend the outer `intro`
    that strips the universally-quantified G, the Magma instance, and
    the hypothesis h.
    """
    return (
        f"def submission : ∀ (G : Type) [Magma G], "
        f"Equation{eq1_id} G → Equation{eq2_id} G := by\n"
        f"  intro G _ h\n"
        f"  {lean_proof_tactic}\n"
    )


def expected_tactics(lean_proof_tactic: str) -> list[str]:
    """The ordered tactics that get applied (one per PyPantograph step).
    Used by the orchestration pipeline to drive `goal_tactic` calls."""
    return ['intro G _ h', lean_proof_tactic]
