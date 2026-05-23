"""
spec-025 Unit E — migration parity.

Proves the Unit E migration (moving the 246 "Locked wk…" pure-pin entries out of
FORCED_GAMES into LOCKED_PAIRINGS in config/season_2026.py) is BEHAVIOUR-PRESERVING:
a full 2026 ``generate_X`` build with the migrated (split) config produces the SAME
set of ``sum == 1`` pinning constraints (same scopes, same matched var-sets) as a
pre-migration FORCED-only build.

Method (no mocks, hand oracle):

* The pre-migration baseline is reconstructed in-test by concatenating the migrated
  ``FORCED_GAMES + LOCKED_PAIRINGS`` back into a single FORCED list (this is exactly
  the list that existed before Unit E) and building generate_X with that as
  FORCED-only.
* The migrated build keeps them split (FORCED_GAMES has the 18 count/marquee rules;
  LOCKED_PAIRINGS has the 246 pins).
* For each build we capture the ``sum == 1`` pinning structure by replaying the SAME
  scope-registration primitives generate_X uses internally
  (``_build_scope_count_rules`` + ``_get_matching_forced_scopes``) over the produced
  variable key-space. We then represent each pinning constraint as the frozenset of
  variable KEYS it sums, and compare the multiset of those frozensets.

Behaviour preservation := the combined matched-var sets and the set of ``== 1``
scopes are IDENTICAL between (baseline FORCED-only) and (migrated FORCED + LOCKED).

Hand oracle on the count: the number of ``== 1`` pinning scopes contributed by the
pins equals the number of moved entries (asserted against the ACTUAL measured count).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_season_data
import config.season_2026 as s2026
from utils import (
    generate_X,
    _build_scope_count_rules,
    _get_matching_forced_scopes,
)

# The qualification rule applied by Unit E: an entry is a pure pin (moved to
# LOCKED_PAIRINGS) iff it has a pairing + grade + date, its description marks it
# "Locked", and it carries NONE of the forbidden scope fields. Everything else
# (count rules, marquee/anniversary, fixed-venue Friday nights, derbies) stays in
# FORCED_GAMES.
FORBIDDEN_FIELDS = {
    'time', 'day_slot', 'field_name', 'field_location',
    'day', 'week', 'round_no', 'count', 'constraint',
}
ALLOWED_FIELDS = {'teams', 'team1', 'team2', 'grade', 'date', 'description'}


def _qualifies_as_pin(entry: dict) -> bool:
    has_pair = ('teams' in entry) or ('team1' in entry and 'team2' in entry)
    has_grade = 'grade' in entry
    has_date = 'date' in entry
    has_locked = 'Locked' in entry.get('description', '')
    no_forbidden = not (set(entry.keys()) & FORBIDDEN_FIELDS)
    only_allowed = set(entry.keys()) <= ALLOWED_FIELDS
    return has_pair and has_grade and has_date and has_locked and no_forbidden and only_allowed


def _pinning_var_keysets(X, forced_games, locked_pairings, teams):
    """Reconstruct the set of `sum == 1` pinning constraints generate_X would build.

    Returns a dict mapping a stable scope-identity (the matched frozenset of var
    keys) -> that frozenset, but we collect them as a list of frozensets so that
    duplicate-scope coincidences (two pins covering an identical var-set) are
    preserved as a multiset.

    Only `== 1` (equal/count==1) FORCED scopes are pins; the count rules
    (lesse/equal-N) are excluded so we compare like-for-like with the pins. The
    LOCKED_PAIRINGS scopes are all `== 1` by construction.

    This replays the exact registration generate_X performs (no locked_weeks in
    these builds, so no in_locked_week skipping).
    """
    f_groups, f_ctypes, f_counts, _fw = _build_scope_count_rules(
        forced_games, teams, label='FORCED_GAMES', unique_per_entry=False)
    lp_groups, _lc, _lcounts, _lw = _build_scope_count_rules(
        locked_pairings, teams, label='LOCKED_PAIRINGS', unique_per_entry=True)

    forced_scope_vars = defaultdict(list)
    locked_scope_vars = defaultdict(list)
    for key in X.keys():
        if len(key) < 11 or not key[3]:
            continue  # skip dummy/short keys (mirror generate_X)
        for sk in _get_matching_forced_scopes(key, f_groups):
            forced_scope_vars[sk].append(key)
        for sk in _get_matching_forced_scopes(key, lp_groups):
            locked_scope_vars[sk].append(key)

    pins = []  # list of frozenset(var keys)
    # FORCED `== 1` scopes only (a pin expressed via FORCED is constraint 'equal'
    # with count defaulting to 1 — `constraint_counts` is ONLY populated when the
    # entry carries an explicit `count`, so an absent count means the default 1).
    # Count rules like sum<=2 / ==8 carry explicit counts/types and are excluded.
    for sk, keys in forced_scope_vars.items():
        ctype = f_ctypes.get(sk)
        cnt = f_counts.get(sk)
        if ctype == 'equal' and (cnt is None or cnt == 1):
            pins.append(frozenset(keys))
    # All LOCKED_PAIRINGS scopes are hard `== 1`.
    for sk, keys in locked_scope_vars.items():
        pins.append(frozenset(keys))
    return pins


# ============== Hand oracle: the partition ==============

class TestPartition:
    def test_migration_counts(self):
        """The migrated season config has exactly the partition Unit E produced."""
        moved = [e for e in (s2026.FORCED_GAMES + s2026.LOCKED_PAIRINGS)
                 if _qualifies_as_pin(e)]
        # 246 pins moved; FORCED retains 18 count/marquee/fixed entries.
        assert len(s2026.LOCKED_PAIRINGS) == 246
        assert len(s2026.FORCED_GAMES) == 18
        # Every LOCKED_PAIRINGS entry qualifies as a pure pin.
        assert all(_qualifies_as_pin(e) for e in s2026.LOCKED_PAIRINGS)
        # No surviving FORCED entry is a pure pin (none qualify).
        assert not any(_qualifies_as_pin(e) for e in s2026.FORCED_GAMES)
        # The combined (pre-migration) list had 246 pins.
        assert len(moved) == 246


# ============== Behaviour-preservation parity ==============

class TestMigrationParity:
    def test_split_build_matches_recombined_forced_only_build(self):
        """A full 2026 generate_X with the migrated split config produces the SAME
        set of sum==1 pinning constraints as the pre-migration FORCED-only build."""
        # --- Migrated (split) build ---
        data_split = load_season_data(2026)
        teams = data_split['teams']
        forced_split = list(data_split['forced_games'])
        locked_split = list(data_split['locked_pairings'])
        assert len(locked_split) == 246
        assert len(forced_split) == 18

        model_split = cp_model.CpModel()
        X_split, _ = generate_X(model_split, data_split)

        # --- Recombined FORCED-only baseline (pre-migration) ---
        data_base = load_season_data(2026)
        # Reconstruct the pre-migration single FORCED list: count/marquee rules +
        # the 246 pins that used to live inside FORCED_GAMES.
        data_base['forced_games'] = forced_split + locked_split
        data_base['locked_pairings'] = []

        model_base = cp_model.CpModel()
        X_base, _ = generate_X(model_base, data_base)

        # Variable key-spaces must be identical (config content doesn't change
        # which vars exist here — neither list blocks/eliminates vars).
        assert set(X_split.keys()) == set(X_base.keys()), \
            "variable key-space diverged between split and recombined builds"

        # --- Capture the sum==1 pinning structure from each ---
        pins_split = _pinning_var_keysets(
            X_split, forced_split, locked_split, teams)
        pins_base = _pinning_var_keysets(
            X_base, forced_split + locked_split, [], teams)

        # Compare as multisets of frozensets of var keys: identical pinning
        # constraints, scope-for-scope (a scope IS its matched var-set).
        from collections import Counter
        cnt_split = Counter(pins_split)
        cnt_base = Counter(pins_base)
        assert cnt_split == cnt_base, (
            "pinning constraints diverged: "
            f"split-only={list((cnt_split - cnt_base).elements())[:3]} "
            f"base-only={list((cnt_base - cnt_split).elements())[:3]}"
        )

        # --- Hand-verify the pin count ---
        # The migrated LOCKED_PAIRINGS contributes exactly one == 1 scope per
        # moved entry. Every pin has placeable vars (no empty scope), so all 246
        # appear. (If any were empty generate_X would have exited FATAL above.)
        lp_groups, _a, _b, _c = _build_scope_count_rules(
            locked_split, teams, label='LOCKED_PAIRINGS', unique_per_entry=True)
        assert len(lp_groups) == 246, \
            f"expected 246 pin scopes, got {len(lp_groups)}"

        # The number of pin-contributed == 1 constraints in the split build
        # (total pins minus FORCED count/marquee == 1 scopes) equals 246 + the
        # FORCED == 1 scopes. Cross-check: pins_base total == pins_split total.
        assert len(pins_split) == len(pins_base)
