import equational_theories.Equations.All
import equational_theories.Superposition

-- Smoke test: prove Equation 2 (singleton law: x = y) ⇒ Equation 1 (reflexivity: x = x).
-- This pair (e1=2, e2=1) is in matcher_proofs.parquet row 0; the matcher's
-- term-mode proof is `fun x => h x x`. Tactic-mode equivalent below — the form
-- Phase 1 actually captures and feeds to PyPantograph for state stepping.

example (G : Type _) [Magma G] (h : Equation2 G) : Equation1 G := by
  intro x
  exact h x x
