"""Shared spacing helpers (spec-008).

`ideal_gap` is the single source of truth for "S played rounds between
repeat meetings" — the convenor-facing spacing number. Both the solver
constraint (`unified.py::_matchup_spacing_hard`) and the tester
(`analytics/tester.py::_check_equal_matchup_spacing`) call this so they
agree.

Semantics (spec-008 Part A):

- `S = effective_spacing(T, base_slack=0, config_slack=0)` is the
  threshold the solver enforces. The hard rule forbids any pair of
  meetings whose ``gap = r2 - r1`` satisfies ``gap <= S`` (equivalently:
  there must be strictly more than ``S`` calendar rounds between
  meetings, i.e. at least ``S`` *free* rounds in between).

- ``ideal_gap(T)`` is the default S when no slack is applied. It is the
  number of free rounds between repeat meetings that's ideal for a
  balanced round-robin of ``T`` teams. It is equal to the previous
  ``min_gap - 1`` (off-by-one fix) so the **physical schedule a healthy
  solver produces is unchanged** at default slack.

- ``base_slack`` (from ``CONSTRAINT_DEFAULTS['spacing_base_slack']``) and
  ``config_slack`` (from ``data['constraint_slack']['EqualMatchUpSpacingConstraint']``,
  surfaced as ``--slack N`` on the CLI) are convenor-facing knobs. Each
  unit of slack reduces the enforced ``S`` by one — i.e. allows the gap
  to shrink by one round from ideal. Slack is clamped so ``S`` never goes
  below 0.

Historical formula being preserved (T = number of teams in grade):

    ideal_distance = T - 1            # round-robin meeting distance
    legacy_hardcoded_slack = max(1, ideal_distance - 2 * ideal_distance // 3)
    legacy_min_gap = max(1, ideal_distance - legacy_hardcoded_slack)
    # Old: forbid `gap < legacy_min_gap`.
    # New: forbid `gap <= ideal_gap(T)` where ideal_gap = legacy_min_gap - 1.

For ``T < 3`` no meaningful spacing applies (≤ 2 teams) so the helpers
return 0 and the caller should skip the grade.
"""
from __future__ import annotations


def _legacy_min_gap(T: int) -> int:
    """Return the legacy `min_gap` (the old hard threshold) for grade size T.

    Kept private — callers use `ideal_gap(T)`. The minus-one shift between
    `_legacy_min_gap(T)` and `ideal_gap(T)` is the spec-008 off-by-one fix.
    """
    if T < 3:
        return 0
    ideal_distance = T - 1
    hardcoded_slack = max(1, ideal_distance - 2 * ideal_distance // 3)
    return max(1, ideal_distance - hardcoded_slack)


def ideal_gap(T: int) -> int:
    """Default S (number of free rounds between repeat meetings) for grade size T.

    Equivalent to ``_legacy_min_gap(T) - 1`` — preserves the physical
    schedule a healthy solver produces while shifting the *meaning* of
    the input number to the intuitive "rounds between meetings."

    Example: for T=10, ideal_gap(10) == 5. The hard rule will forbid any
    repeat meeting whose ``gap = r2 - r1 <= 5`` (matching legacy ``gap < 6``).
    """
    if T < 3:
        return 0
    return max(0, _legacy_min_gap(T) - 1)


def ideal_bye_gap(R: int, byes_per_team: int) -> int:
    """Default S for bye spacing — minimum free rounds between byes.

    spec-008 Part B. With ``byes`` byes distributed across ``R`` playable
    rounds, the perfectly-spread bye distance is ``avg = R // byes``;
    we accept one round of slack so the constraint isn't unnecessarily
    tight (mirrors the ``-1`` shift in `ideal_gap(T)` for matchups).

    Returns 0 when ``byes < 2`` — only one bye exists, no pairwise check
    is meaningful — so callers can short-circuit.

    Examples:
      R=18, byes=2  -> avg=9, S=8 (the two byes must sit at least 9 rounds apart)
      R=18, byes=3  -> avg=6, S=5
      R=20, byes=4  -> avg=5, S=4
      R=22, byes=1  -> S=0 (no constraint — only one bye)
    """
    if R <= 0 or byes_per_team < 2:
        return 0
    avg = R // byes_per_team
    return max(0, avg - 1)


def effective_spacing(T: int, base_slack: int = 0, config_slack: int = 0) -> int:
    """Return the enforced S after applying both slack sources.

    Each unit of slack reduces S by 1 (allowing the gap to shrink by one
    round from ideal). The result is clamped at 0 — slack can fully
    disable the constraint for a grade but never make it negative.

    Used by both the solver (`unified.py::_matchup_spacing_hard`) and the
    tester (`analytics/tester.py::_check_equal_matchup_spacing`) so they
    agree on every grade size and slack value.
    """
    base = ideal_gap(T)
    if base <= 0:
        return 0
    return max(0, base - int(base_slack) - int(config_slack))
