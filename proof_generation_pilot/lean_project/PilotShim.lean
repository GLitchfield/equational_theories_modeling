-- Pilot Phase 1 Lean shim root. All ETP + Mathlib imports needed by the
-- three tool adapters (matcher, Vampire, grind) live here so the PyPantograph
-- Server (which loads this lib by name) has them available for tactic
-- invocation during proof stepping.

import equational_theories.Equations.All
import equational_theories.Superposition
import Mathlib.Tactic.TypeStar
import Mathlib.Tactic.ByContra
import Mathlib.Tactic
