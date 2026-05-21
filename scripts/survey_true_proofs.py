"""Task 3 of `proof_corpus_survey_plan.md` — TRUE proof structural characterization.

Walks every .lean file under the ETP repo, finds every implication
theorem (declarations of the form `theorem Equation_X_implies_Equation_Y
(G : Type*) [Magma G] (h : Equation_X G) : Equation_Y G := <body>`),
parses out E1 and E2, and classifies the proof body by **semantic
structure** (Methodology Resolution 11 — not surface mode):

  (a) **variable_substitution** — single hypothesis applied to a
      permutation/repetition of input variables. Term `λ x y => h x y x`,
      tactic `by exact h x y x`, or `by intro x y; exact h x y x`
      all fall here. The generator's job for this category is to
      predict a short integer tuple (which input variable goes in
      which argument position of the hypothesis).
  (b) **hypothesis_composition** — multiple hypothesis identifiers,
      dot-accessors (`h.symm`), or the rewrite operator (`▸`).
      Compositional but still a structured term.
  (c) **rewrite_chain** — contains `rw`, `rewrite`, `calc`, or
      `simp_rw`. Step-by-step transformation.
  (d) **complex_construction** — `have` chains, intermediate terms,
      anything that doesn't fit a-c.

For (a) the script also extracts the **argument-permutation shape**
(a tuple like `(x, y, x)` for `h x y x` against goal vars `x y`),
which Task 4 reuses for clustering.

Cross-references with `data/processed/equation_invariants.parquet`
to stratify by E2's spine type (Methodology Resolution 4).

Output:
  - `outputs/analyses/proof_survey/true_characterization.json` (committed)
  - `et_studies/proof_survey/data/true_corpus.parquet` (gitignored,
    per-proof records for Task 4's clustering)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

SPINE_LABELS = ['family_c', 'mixed', 'no_lhs_var_in_rhs', 'left', 'right', 'trivial']


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Implication theorem: theorem Equation_X_implies_Equation_Y (G : Type*) [Magma G] (h : Equation_X G) : Equation_Y G := <body>
IMPL_THEOREM_RE = re.compile(
    r'(?:^|\n)theorem\s+Equation(\d+)_implies_Equation(\d+)\s*'
    r'\([^)]*\)\s*\[[^]]*\]\s*\([^)]*\)\s*:\s*Equation\d+\s*G\s*'
    r':=\s*([\s\S]*?)'
    r'(?=\n(?:@\[|theorem |lemma |def |instance |end |namespace )|\Z)',
    re.MULTILINE,
)


def normalize_body(body: str) -> str:
    """Strip leading/trailing whitespace + comments + extra whitespace.

    Removes:
      - Leading line comments (`-- ...`)
      - Trailing block comments (`/- ... -/` and `/-- ... -/`)
      - Trailing line comments (`-- ...` at end)
      - Collapses runs of whitespace to single spaces.
    """
    b = body.strip()
    while b.startswith('--'):
        nl = b.find('\n')
        if nl == -1:
            return ''
        b = b[nl + 1:].lstrip()
    # Strip trailing block comments
    b = re.sub(r'/-[\s\S]*?-/\s*$', '', b).strip()
    # Strip trailing line comments
    b = re.sub(r'--[^\n]*$', '', b).strip()
    # Collapse whitespace
    b = re.sub(r'\s+', ' ', b)
    return b


# A pre-stripped body classifier. All patterns operate on the
# whitespace-collapsed body.

# Term-mode variable substitution: `fun X Y => h X Y X` / `λ X ↦ h X Y X`
VAR_SUB_TERM_RE = re.compile(
    r'^(?:fun|λ)\s+([\w\s_]+?)\s*(?:=>|↦)\s*(\w+)((?:\s+\w+)*)\s*$',
)

# Lean's `..` syntax = "fill all remaining args with underscores"
VAR_SUB_TERM_DOTS_RE = re.compile(
    r'^(?:fun|λ)\s+([\w\s_]+?)\s*(?:=>|↦)\s*(\w+)\s*\.\.\s*$',
)

# Just `fun X Y => rfl`
TRIVIAL_RFL_TERM_RE = re.compile(
    r'^(?:fun|λ)\s+([\w\s_]+?)\s*(?:=>|↦)\s*rfl\s*$',
)

# Tactic-mode: `by exact <term>` wrappings
BY_EXACT_TERM_RE = re.compile(
    r'^by\s+exact\s+(?:fun|λ)\s+([\w\s_]+?)\s*(?:=>|↦)\s*(\w+)((?:\s+\w+)*)\s*$',
)
BY_EXACT_PLAIN_RE = re.compile(
    r'^by\s+exact\s+(\w+)((?:\s+\w+)*)\s*$',
)
BY_EXACT_RFL_RE = re.compile(r'^by\s+(?:exact\s+)?rfl\s*$')

# `by intro X Y ; exact h X Y X` — handles `;` OR space-separated multi-line form.
BY_INTRO_EXACT_RE = re.compile(
    r'^by\s+intro\s+([\w\s_]+?)\s*(?:;|\s)\s*exact\s+(\w+)((?:\s+\w+)*)\s*$',
)
# `by intro X (Y); rfl` and the multi-line `by intro X rfl`
BY_INTRO_RFL_RE = re.compile(r'^by\s+intro\s+[\w\s_]+?\s+rfl\s*$')
# `by repeat intro; apply h` (and the space-only variant `by repeat intro apply h`)
BY_REPEAT_INTRO_APPLY_RE = re.compile(
    r'^by\s+repeat\s+intro\s*(?:;|\s)\s*apply\s+(\w+)\s*$'
)
# `by apply h` — minimal apply close
BY_APPLY_PLAIN_RE = re.compile(r'^by\s+apply\s+(\w+)\s*$')

# Single-tactic automation closes
SINGLE_TAC_RE = re.compile(
    r'^by\s+(simp(?:_all)?|decide|aesop|grind|trivial|omega|tauto|positivity)\s*$',
)
# Single tactic with simple args, e.g. `by simp [h]` or `by exact?`
SINGLE_TAC_ARGS_RE = re.compile(
    r'^by\s+(simp(?:_all)?|decide|aesop|grind|trivial|omega|tauto|positivity|exact|apply)\s*[\[\?][\s\S]*$',
)

# Rewrite chain detection (any occurrence of one of these tokens)
REWRITE_RE = re.compile(
    r'\b(?:rw|rewrite|rwa|simp_rw|nth_rw|nth_rewrite|calc|conv)\b',
)

# Hypothesis composition: dot accessors anywhere in the body (after \w or \))
HYP_COMP_RE = re.compile(r'[)\w]\.\s*(symm|trans|comp|mp|mpr|left|right|symm_iff)\b')
# Or the rewrite operator `▸` (Lean: substitute equal terms)
HYP_RWOP_RE = re.compile(r'▸')

# `have` / `suffices` indicates intermediate constructions
HAVE_RE = re.compile(r'\b(?:have|suffices|let|show)\b')

# `funext`, `ext`, `subst` etc. indicate proof-engineering tactics → complex
PROOF_ENG_RE = re.compile(r'\b(?:funext|ext|subst|cases|rcases|obtain|use|refine|apply\?|exact\?)\b')

# Vampire/superposition output: starts with `by by_contra` and uses `simp only
# [not_forall]` to extract Skolem variables, then chained `have eqN ... := mod_symm`
# / `superpose` lemmas. Very uniform across ~1,500 proofs in the corpus.
VAMPIRE_STYLE_RE = re.compile(
    r'^by\s+by_contra\s+\w+\s+simp\s+only\s+\[not_forall\]\s+at\s+\w+\s+obtain\b'
)

# Egg/combinator-style term: `fun <vars> => let v0 := M ...; ...; T (T (...))`
# with the `M, T, C, R, S` combinator vocabulary characteristic of e-graph
# rewrite output.
EGG_STYLE_RE = re.compile(
    r'^(?:fun|λ)\s+[\w\s_]+(?:=>|↦)\s+(?:let\s+v\d+\s*:=\s*M\b|have\s+h\d+\s*:=\s*[RST]\b|T\s*\(|S\s*\(|C\s*\()'
)


def classify_proof(body: str, e1: int, e2: int) -> tuple[str, dict]:
    """Return (category, extra_fields) for a proof body.

    Categories:
      'a_variable_substitution'  — single hypothesis with arg permutation
      'a_trivial_rfl'            — body is `rfl` (or `intro;rfl`)
      'a_dot_fill'               — `fun _ => h ..` (Lean's `..` fill syntax)
      'b_hypothesis_composition' — `.symm`, `.trans`, `▸`, ...
      'c_rewrite_chain'          — `rw`/`rwa`/`calc`/`conv`/`simp_rw`
      'd_automation_close'       — `by simp`, `by decide`, `by aesop`, `by grind`
      'e_complex_construction'   — `have`/`suffices`/multi-step / unmatched
    """
    b = normalize_body(body)
    if not b:
        return 'e_complex_construction', {'reason': 'empty body'}

    syn_mode = 'tactic' if b.startswith('by ') or b == 'by' else 'term'

    # ---------- (d) Automation closes (single-tactic) ----------
    m = SINGLE_TAC_RE.match(b)
    if m:
        return 'd_automation_close', {'syn_mode': syn_mode, 'tactic': m.group(1)}
    m = SINGLE_TAC_ARGS_RE.match(b)
    if m:
        # Single tactic with args / question mark (e.g., `by simp [h]`).
        # Still architecturally "pick a tactic" rather than sequence-gen.
        return 'd_automation_close', {'syn_mode': syn_mode, 'tactic': m.group(1) + '*'}

    # ---------- (a) Trivial reflexive ----------
    if TRIVIAL_RFL_TERM_RE.match(b) or BY_INTRO_RFL_RE.match(b) or BY_EXACT_RFL_RE.match(b):
        return 'a_trivial_rfl', {'syn_mode': syn_mode}

    # ---------- (a) "repeat intro; apply h" — semantically a variable subst ----------
    m = BY_REPEAT_INTRO_APPLY_RE.match(b)
    if m:
        return 'a_variable_substitution', {
            'syn_mode': syn_mode,
            'hyp': m.group(1),
            'goal_vars_text': '<repeat-intro>',
            'applied_args': ['<all>'],  # apply h fills with goal vars
            'n_applied': -2,  # special marker for "repeat intro; apply"
        }

    # ---------- (a) bare `by apply h` ----------
    m = BY_APPLY_PLAIN_RE.match(b)
    if m:
        return 'a_variable_substitution', {
            'syn_mode': syn_mode,
            'hyp': m.group(1),
            'goal_vars_text': '',
            'applied_args': [],
            'n_applied': 0,
        }

    # ---------- (a) Dot-fill (`h ..`) ----------
    m = VAR_SUB_TERM_DOTS_RE.match(b)
    if m:
        return 'a_dot_fill', {'syn_mode': syn_mode, 'hyp': m.group(2)}

    # ---------- (c) Rewrite chain ----------
    if REWRITE_RE.search(b):
        return 'c_rewrite_chain', {'syn_mode': syn_mode}

    # ---------- (e1) Vampire/superposition output ----------
    if VAMPIRE_STYLE_RE.match(b):
        return 'e_vampire_superposition', {'syn_mode': syn_mode}

    # ---------- (e2) Egg/combinator-style term ----------
    if EGG_STYLE_RE.match(b):
        return 'e_egg_combinator', {'syn_mode': syn_mode}

    # ---------- (e) Complex (have/suffices/funext/cases) — check before (b) ----------
    if HAVE_RE.search(b) or PROOF_ENG_RE.search(b):
        return 'e_complex_construction', {'syn_mode': syn_mode, 'reason': 'have/proof-eng'}

    # ---------- (b) Hypothesis composition ----------
    if HYP_RWOP_RE.search(b) or HYP_COMP_RE.search(b):
        return 'b_hypothesis_composition', {'syn_mode': syn_mode}

    # ---------- (a) Variable substitution (term-mode) ----------
    m = VAR_SUB_TERM_RE.match(b)
    if m:
        goal_vars = m.group(1).strip()
        hyp = m.group(2)
        applied = m.group(3).strip().split() if m.group(3).strip() else []
        return 'a_variable_substitution', {
            'syn_mode': syn_mode,
            'hyp': hyp,
            'goal_vars_text': goal_vars,
            'applied_args': applied,
            'n_applied': len(applied),
        }

    # ---------- (a) `by exact <term>` ----------
    m = BY_EXACT_TERM_RE.match(b)
    if m:
        return 'a_variable_substitution', {
            'syn_mode': syn_mode,
            'hyp': m.group(2),
            'goal_vars_text': m.group(1).strip(),
            'applied_args': m.group(3).strip().split() if m.group(3).strip() else [],
            'n_applied': len(m.group(3).strip().split()) if m.group(3).strip() else 0,
        }

    # ---------- (a) `by intro X Y; exact h ...` ----------
    m = BY_INTRO_EXACT_RE.match(b)
    if m:
        return 'a_variable_substitution', {
            'syn_mode': syn_mode,
            'hyp': m.group(2),
            'goal_vars_text': m.group(1).strip(),
            'applied_args': m.group(3).strip().split() if m.group(3).strip() else [],
            'n_applied': len(m.group(3).strip().split()) if m.group(3).strip() else 0,
        }

    # ---------- (a) bare `by exact h ...` ----------
    m = BY_EXACT_PLAIN_RE.match(b)
    if m:
        return 'a_variable_substitution', {
            'syn_mode': syn_mode,
            'hyp': m.group(1),
            'goal_vars_text': '',
            'applied_args': m.group(2).strip().split() if m.group(2).strip() else [],
            'n_applied': len(m.group(2).strip().split()) if m.group(2).strip() else 0,
        }

    # ---------- Fallback: complex ----------
    return 'e_complex_construction', {
        'syn_mode': syn_mode,
        'reason': 'unmatched',
        'first_80': b[:80],
    }


def survey_files(lean_root: str, invariants_path: str) -> tuple[dict, pd.DataFrame]:
    print(f'walking {lean_root}')
    files: list[str] = []
    for r, _d, fs in os.walk(lean_root):
        for f in fs:
            if f.endswith('.lean'):
                files.append(os.path.join(r, f))
    print(f'  {len(files):,} Lean files')

    inv = None
    if os.path.exists(invariants_path):
        inv = pd.read_parquet(invariants_path)
        print(f'  invariants: {invariants_path} ({len(inv):,} equations)')

    records: list[dict] = []
    for path in files:
        try:
            text = open(path, errors='ignore').read()
        except OSError:
            continue
        for m in IMPL_THEOREM_RE.finditer(text):
            e1, e2 = int(m.group(1)), int(m.group(2))
            body = m.group(3).strip()
            cat, extra = classify_proof(body, e1, e2)
            rec = {
                'e1': e1,
                'e2': e2,
                'file': path.split('/equational_theories/equational_theories/')[-1]
                        if '/equational_theories/equational_theories/' in path else path,
                'body_len_chars': len(body),
                'category': cat,
                **extra,
            }
            # E2 spine (1-indexed eq IDs)
            if inv is not None and 1 <= e2 <= len(inv):
                rec['e2_spine'] = SPINE_LABELS[int(inv['spine_type'].iloc[e2 - 1])]
            else:
                rec['e2_spine'] = None
            records.append(rec)
    df = pd.DataFrame(records)
    print(f'  extracted {len(df):,} implication theorems')

    # --- Aggregate stats -------------------------------------------------
    category_dist = df['category'].value_counts().to_dict()
    mode_dist = df['syn_mode'].value_counts().to_dict() if 'syn_mode' in df.columns else {}
    # Cross-tab: category x syn_mode
    if 'syn_mode' in df.columns:
        cat_mode = df.groupby(['category', 'syn_mode']).size().reset_index(name='n')
        cat_mode_dict = {f'{r.category}|{r.syn_mode}': int(r.n) for r in cat_mode.itertuples()}
    else:
        cat_mode_dict = {}
    # Per-spine breakdown for E2
    by_e2_spine = (
        df.groupby(['e2_spine', 'category']).size().unstack(fill_value=0).to_dict()
        if df['e2_spine'].notna().any() else {}
    )
    # For category (a) variants: distribution of (hyp, applied_args) shapes
    a_df = df[df['category'] == 'a_variable_substitution']
    a_shapes: Counter[tuple] = Counter()
    for _, row in a_df.iterrows():
        if isinstance(row.get('applied_args'), list):
            a_shapes[tuple(row['applied_args'])] += 1
    top_shapes = [(' '.join(k) if k else '<no_args>', v) for k, v in a_shapes.most_common(20)]
    # For category (a): how many distinct hypothesis identifiers? (just `h` usually)
    a_hyps = a_df['hyp'].value_counts().head(10).to_dict() if not a_df.empty else {}

    summary = {
        'lean_root': lean_root,
        'n_implication_theorems_extracted': int(len(df)),
        'category_distribution': category_dist,
        'syn_mode_distribution': mode_dist,
        'category_x_syn_mode': cat_mode_dict,
        'by_e2_spine': {
            spine: {cat: int(cnt) for cat, cnt in cats.items()}
            for spine, cats in by_e2_spine.items()
        },
        'top_20_variable_substitution_shapes': top_shapes,
        'top_hypothesis_identifiers_in_var_subst': a_hyps,
        'headline': {
            'fraction_variable_substitution': float(
                (df['category'] == 'a_variable_substitution').mean()
            ),
            'fraction_trivial_rfl': float(
                (df['category'] == 'a_trivial_rfl').mean()
            ),
            'fraction_dot_fill': float(
                (df['category'] == 'a_dot_fill').mean()
            ),
            'fraction_a_combined': float(
                df['category'].isin(['a_variable_substitution', 'a_trivial_rfl', 'a_dot_fill']).mean()
            ),
            'fraction_hypothesis_composition': float(
                (df['category'] == 'b_hypothesis_composition').mean()
            ),
            'fraction_rewrite_chain': float(
                (df['category'] == 'c_rewrite_chain').mean()
            ),
            'fraction_automation_close': float(
                (df['category'] == 'd_automation_close').mean()
            ),
            'fraction_complex_construction': float(
                (df['category'] == 'e_complex_construction').mean()
            ),
            'fraction_vampire_superposition': float(
                (df['category'] == 'e_vampire_superposition').mean()
            ),
            'fraction_egg_combinator': float(
                (df['category'] == 'e_egg_combinator').mean()
            ),
            'fraction_e_combined': float(
                df['category'].isin([
                    'e_complex_construction', 'e_vampire_superposition', 'e_egg_combinator'
                ]).mean()
            ),
        },
    }
    return summary, df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--etp-root', default='/home/gavin/git/equational_theories/equational_theories')
    parser.add_argument('--invariants', default='data/processed/equation_invariants.parquet')
    parser.add_argument('--output-summary',
                        default='outputs/analyses/proof_survey/true_characterization.json')
    parser.add_argument('--output-corpus',
                        default='et_studies/proof_survey/data/true_corpus.parquet')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    summary_path = _abs(args.output_summary)
    corpus_path = _abs(args.output_corpus)
    for p in [summary_path, corpus_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    summary, df = survey_files(args.etp_root, _abs(args.invariants))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote summary: {summary_path}')

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    # Some object-type columns (lists) need to be stored as Python objects in parquet;
    # convert lists -> strings for portability.
    df_save = df.copy()
    if 'applied_args' in df_save.columns:
        df_save['applied_args'] = df_save['applied_args'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else x
        )
    df_save.to_parquet(corpus_path, compression='zstd')
    print(f'wrote per-proof corpus: {corpus_path} ({len(df_save):,} rows)')

    print()
    print('=== TRUE-proof structural characterization ===')
    h = summary['headline']
    print(f"  fraction in (a) variable_substitution : {h['fraction_variable_substitution']*100:5.1f}%")
    print(f"  fraction in (a) trivial_rfl            : {h['fraction_trivial_rfl']*100:5.1f}%")
    print(f"  fraction in (a) dot_fill              : {h['fraction_dot_fill']*100:5.1f}%")
    print(f"  fraction in (a) all combined          : {h['fraction_a_combined']*100:5.1f}%")
    print(f"  fraction in (b) hypothesis_composition : {h['fraction_hypothesis_composition']*100:5.1f}%")
    print(f"  fraction in (c) rewrite_chain          : {h['fraction_rewrite_chain']*100:5.1f}%")
    print(f"  fraction in (d) automation_close       : {h['fraction_automation_close']*100:5.1f}%")
    print(f"  fraction in (e) vampire_superposition  : {h['fraction_vampire_superposition']*100:5.1f}%")
    print(f"  fraction in (e) egg_combinator         : {h['fraction_egg_combinator']*100:5.1f}%")
    print(f"  fraction in (e) genuine complex        : {h['fraction_complex_construction']*100:5.1f}%")
    print(f"  fraction in (e) all combined           : {h['fraction_e_combined']*100:5.1f}%")
    print()
    print(f'  Category x syntactic mode:')
    for k, v in sorted(summary['category_x_syn_mode'].items(), key=lambda kv: -kv[1]):
        print(f'    {k:<55} {v:>6,}')
    print()
    print(f'  Top variable-substitution shapes (applied args after the hypothesis):')
    for shape, n in summary['top_20_variable_substitution_shapes'][:10]:
        print(f'    "{shape}" {n:>6,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
