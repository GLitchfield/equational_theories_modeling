"""Vampire → Lean adapter for the Phase 1 pilot.

Thin wrapper around ETP's Vampire pipeline:

  1. Encode the (E1 → E2) pair as TPTP first-order logic (`encode_problem`)
  2. Run Vampire 4.9 subprocess with `--proof_extra full` (-t 1 sec timeout)
  3. Parse the superposition trace and emit Lean tactic body (`leanify`)
  4. Wrap in `def submission := by ...`

The encoder + leanify logic is adapted from
ETP's `equational_theories/Generated/VampireProven/src/vampire_proofs.py`. We re-implement the small functions here rather
than importing that script (it has module-load side effects). We do
import the `eqs` table + `count_vars` from ETP's `scripts/generate_eqs_list.py`
since that's a clean pure-computation module.

Public API:
    make_submission(eq1_id, eq2_id, vampire_bin=$VAMPIRE_BIN or 'vampire',
                    timeout_s=1) -> dict
        Returns {'status': 'proved' | 'no_proof' | 'timeout' | 'error',
                 'lean': str | None, 'vampire_output': str | None,
                 'wall_clock_ms': int, 'error': str | None}
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

# ETP's pure-computation eqs generator (no side effects on import).
# Sibling ETP checkout by default; override with $ETP_ROOT.
_ETP_ROOT = os.environ.get('ETP_ROOT', os.path.normpath(os.path.join(_REPO_ROOT, os.pardir, 'equational_theories')))
_ETP_SCRIPTS = os.path.join(_ETP_ROOT, 'scripts')
if _ETP_SCRIPTS not in sys.path:
    sys.path.insert(0, _ETP_SCRIPTS)
from generate_eqs_list import eqs, count_vars  # type: ignore  # noqa: E402

# TPTP encoding uses single-letter binders X, Y, Z, W, U, V.
_BVARS = "XYZWUV"


def _format_expr_tptp(expr) -> str:
    """Format an ETP equation expression as a TPTP term using `mul(...)`."""
    if isinstance(expr, int):
        return _BVARS[expr]
    return f"mul({_format_expr_tptp(expr[0])}, {_format_expr_tptp(expr[1])})"


def _format_eq_tptp(eq) -> str:
    """Format an equation `(lhs, rhs)` as a TPTP universally-quantified equality."""
    n = count_vars(eq)
    binders = "".join(f"! [{c}] : " for c in _BVARS[:n])
    return f"{binders}{_format_expr_tptp(eq[0])} = {_format_expr_tptp(eq[1])}"


def _encode_problem(eq1_id: int, eq2_id: int) -> str:
    """Build the TPTP input for Vampire: E1 as axiom, E2 as conjecture."""
    assumption = eqs[eq1_id - 1]
    goal = eqs[eq2_id - 1]
    return (
        f"fof(lhs, axiom, {_format_eq_tptp(assumption)}).\n"
        f"fof(rhs, conjecture, {_format_eq_tptp(goal)}).\n"
    )


def _natural_sort(items) -> list:
    def alphanum_key(key):
        return [int(c) if c.isdigit() else c.lower() for c in re.split("([0-9]+)", key)]
    return sorted(items, key=alphanum_key)


def _leanify_statement(statement: str) -> str:
    """Translate a TPTP-style statement (e.g. `mul(X, Y) = X`) into a Lean
    type-ascription string (e.g. `(X Y : G) : X ◇ Y = X`)."""
    s = re.sub("mul", "", statement)
    s = re.sub(",", " ◇ ", s)
    s = re.sub("!=", "≠", s)
    variables = _natural_sort({x for x in re.findall(r"\w+", s) if "sK" not in x})
    var_block = f'({" ".join(variables)} : G) ' if variables else ""
    return f"{var_block}: {s}"


def _leanify_proof_step(proof: str) -> str:
    """Translate a single Vampire proof annotation (e.g. `superposition 3,5`)
    into a Lean term invoking ETP's `superpose` tactic. Raises
    `NotImplementedError` for proof patterns ETP's translator doesn't handle."""
    sr = re.match(r"superposition (\d+),(\d+)", proof)
    if sr:
        return f"superpose eq{sr[2]} eq{sr[1]}"
    sr = re.match(r"backward demodulation (\d+),(\d+)", proof)
    if sr:
        return f"superpose eq{sr[2]} eq{sr[1]}"
    sr = re.match(r"forward demodulation (\d+),(\d+)", proof)
    if sr:
        return f"superpose eq{sr[2]} eq{sr[1]}"
    raise NotImplementedError(proof)


def _leanify(vampire_output: str, eq1_id: int, eq2_id: int) -> str:
    """Parse Vampire's `--proof_extra full` output and emit a Lean tactic body
    using ETP's superpose tactic. The body assumes `intro G _ h` has already
    been applied — i.e., it starts with `by_contra nh` over the bound goal.

    Returns just the tactic body (no `theorem` / `def` wrapper).
    """
    goal = eqs[eq2_id - 1]
    sk_vars = ", ".join(f"sK{i}" for i in range(count_vars(goal)))
    body = ""
    body += "  by_contra nh\n"
    body += "  simp only [not_forall] at nh\n"
    body += f"  obtain ⟨{sk_vars}, nh⟩ := nh\n"
    for eqnum, statement, proof in re.findall(r"(\d+)\. ([^[]+) \[([^\]]+)\]", vampire_output):
        if proof.startswith("X") or proof.startswith("skolemisation"):
            continue
        if proof.startswith("cnf transformation"):
            if "!=" in statement:
                body += f"  have eq{eqnum} {_leanify_statement(statement)} := mod_symm nh\n"
            else:
                body += f"  have eq{eqnum} {_leanify_statement(statement)} := mod_symm (h ..)\n"
            continue
        if statement.strip() == "$false":
            sr = re.match(r"subsumption resolution (\d+),(\d+)", proof)
            if sr:
                body += f"  subsumption eq{sr[1]} eq{sr[2]}\n"
                continue
            sr = re.match(r"trivial inequality removal (\d+)", proof)
            if sr:
                body += f"  subsumption eq{sr[1]} rfl\n"
                continue
            sr = re.match(r"equality resolution (\d+)", proof)
            if sr:
                body += f"  subsumption eq{sr[1]} rfl\n"
                continue
            raise NotImplementedError(f"$false step: {proof}")
        body += f"  have eq{eqnum} {_leanify_statement(statement)} := {_leanify_proof_step(proof)}\n"
    return body


def make_submission(
    eq1_id: int,
    eq2_id: int,
    vampire_bin: str = os.environ.get("VAMPIRE_BIN", "vampire"),
    timeout_s: int = 1,
) -> dict:
    """Run Vampire on the (E1, E2) pair and translate the proof to a Lean
    `def submission := by ...` block.

    Returns a dict with status, lean source, raw vampire output (for
    debugging), and wall-clock timing.
    """
    bin_path = os.path.expanduser(vampire_bin)
    tptp = _encode_problem(eq1_id, eq2_id)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [bin_path, "--proof_extra", "full", "/proc/self/fd/0", "-t", str(timeout_s)],
            input=tptp.encode(),
            capture_output=True,
            timeout=max(timeout_s + 5, 10),
        )
        out = result.stdout.decode(errors="replace")
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout', 'lean': None, 'vampire_output': None,
            'wall_clock_ms': int((time.perf_counter() - t0) * 1000),
            'error': f'vampire wall-clock exceeded {timeout_s + 5}s',
        }
    except Exception as e:
        return {
            'status': 'error', 'lean': None, 'vampire_output': None,
            'wall_clock_ms': int((time.perf_counter() - t0) * 1000),
            'error': f'subprocess failure: {e}',
        }
    wall_ms = int((time.perf_counter() - t0) * 1000)

    # Vampire signals refutation via "Refutation found" line in stdout.
    if "Refutation" not in out:
        return {
            'status': 'no_proof', 'lean': None, 'vampire_output': out,
            'wall_clock_ms': wall_ms, 'error': None,
        }

    try:
        body = _leanify(out, eq1_id, eq2_id)
    except NotImplementedError as e:
        return {
            'status': 'error', 'lean': None, 'vampire_output': out,
            'wall_clock_ms': wall_ms,
            'error': f'leanify NotImplementedError: {e}',
        }

    lean = (
        f"def submission : ∀ (G : Type) [Magma G], "
        f"Equation{eq1_id} G → Equation{eq2_id} G := by\n"
        f"  intro G _ h\n"
        f"{body}"
    )
    return {
        'status': 'proved', 'lean': lean, 'vampire_output': out,
        'wall_clock_ms': wall_ms, 'error': None,
    }
