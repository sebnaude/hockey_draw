"""Shared helpers for the spec-005 `ClubVsClubStackedAlignment` atoms.

Splits the legacy `ClubVsClubAlignment` cluster into two atoms (replacing
the four Phase-3c atoms `ClubVsClubCoincidence`, `ClubVsClubFieldLimit`,
`ClubVsClubDeficitPenalty`, `PHLAnd2ndBackToBackSameField`):

- `ClubVsClubStackedWeekends` — pin the **per-grade meeting counts** to a
  strict nested-superset structure across weeks.
- `ClubVsClubStackedCoLocation` — on each stacked weekend (≥ 2 grades for the
  same pair on the same Sunday) all involved games must be on the same field
  with contiguous day_slots.

Shared building blocks (this module):

- `per_pair_grade_matchup_counts(data, club_pair)` — distinct matchups for
  the pair in each grade. Handles multi-team-per-club-per-grade correctly:
  Club-A with 2 teams in 3rd-grade vs Club-B with 1 team in 3rd-grade =
  2 distinct matchups (not 1, not 6).

- `pair_grade_sunday_meetings(data, club_pair, grade)` — Sunday meeting
  count budget for stacking. For PHL: `total_phl_meetings(A,B) -
  phl_forced_friday_meetings(A,B)`; for any other grade: full matchup count
  (those grades only play Sundays so all meetings are Sunday-available).

  NOTE: Use `pair_grade_sunday_aligned_weekends` for the stacking budget in
  spec-038 onward; `pair_grade_sunday_meetings` returns game-counts (which
  differs from weekend-counts when multi-team-per-club-per-grade).

- `team_pair_counts(data, club_pair, grade)` — `(a, b)` where `a` = number
  of teams club_A fields in `grade`, `b` = number for club_B. Order matches
  `club_pair` tuple order — NOT sorted by magnitude.

- `enumerate_team_pairs_in_pair_grade(data, club_pair, grade)` — every
  unordered cross-club team-pair `(t1, t2)` with `{t1.club, t2.club} ==
  set(club_pair)` and both teams in `grade`. Sorted deterministically.

- `per_pair_grade_aligned_weekends(data, club_pair, grade)` — number of
  distinct aligned weekends for this `(pair, grade)`: `max(a, b) * per_matchup`.
  This is the correct stacking budget for multi-team-per-club-per-grade.

- `pair_grade_sunday_aligned_weekends(data, club_pair, grade)` — like
  `pair_grade_sunday_meetings` but uses the aligned-weekend formula (spec-038).
  For non-PHL: returns `per_pair_grade_aligned_weekends`. For PHL: subtracts
  forced Friday meetings. Returns 0 if budget is exhausted or either club
  fields 0 teams in the grade.

- `enumerate_club_pairs(data)` — every unordered (A,B) where at least one
  grade has matchups between them.

- `collect_pair_grade_week_vars(X, data, grade, club_pair)` — Sunday-only
  decision vars indexed by week, scoped to the pair+grade.

The "stacking" math: given grades with Sunday meeting counts
`c[g]` (non-zero), sort grades by descending count. Force:

  sum_w play[g, w] == c[g]                            (hard, per grade)
  play[g_lower_count, w] <= play[g_higher_count, w]   (hard, per consec pair)

The implication chain forces nested supersets: every weekend on which a
lower-count grade plays for this pair, every higher-count grade also plays.
Across all weeks the count differences `c[g_k] - c[g_{k+1}]` are exactly
the "peel-off" buckets the convenor wants.

## PHL Sunday budget — worked example

(Maitland, Norths) PHL = 4 meetings total, of which 2 are FORCED Friday
(via `FORCED_GAMES`). The Sunday budget for stacking PHL with this pair
is `4 - 2 = 2`. The other grade counts feed in directly.

If the resulting Sunday budgets are `{PHL: 2, 2nd: 3, 3rd: 2, 4th: 2, 5th: 1}`
then after sorting desc:

  2nd=3, PHL=2, 3rd=2, 4th=2, 5th=1

The stacking atom forces:
  - 2nd plays on exactly 3 weekends, PHL/3rd/4th on exactly 2, 5th on exactly 1.
  - Whenever PHL plays, 2nd plays (PHL count <= 2nd count).
  - Whenever 3rd plays, PHL plays (3rd count <= PHL count) AND 2nd plays
    (transitive via consec-pair implication: 3rd <= PHL <= 2nd).
  - Same for 4th and 5th.

The peel-off layout: 1 weekend (5th + all higher), 1 weekend (4th + higher),
0 weekends pure-{2nd,PHL,3rd,4th} (since c[3rd]=c[4th]=c[PHL]=2 → no gap),
plus the 5th-only weekend uses the same superset, so net 2 weekends with
{2nd,PHL,3rd,4th,5th}-style superset, etc. (The atom doesn't enumerate
"layers" — it just forces the implication chain; the layout emerges.)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from constraints.atoms._phl_forced_friday_helper import (
    phl_forced_friday_meetings,
)


# Helper-var pool key prefixes. Distinct from `cvc_coincide` / `cvc_phl_btb_*`
# so spec-005 atoms don't collide with the obsolete Phase-3c atoms when both
# exist in the same registry (which they do — old atoms kept as parity
# reference per the plan).
STACK_PLAY_PREFIX = 'cvc_stack_play'           # (pair, grade, week)
STACK_FIELD_USED_PREFIX = 'cvc_stack_field_used'  # (pair, week, field_name)
STACK_SLOT_USED_PREFIX = 'cvc_stack_slot_used'    # (pair, week, day_slot)
STACK_ACTIVE_PREFIX = 'cvc_stack_active'        # (pair, week)
# spec-038: per-team-pair play indicator key prefix. Key shape: (team_pair, week)
# where team_pair is an unordered (t1, t2) tuple (internally sorted alphabetically).
STACK_TEAM_PAIR_PLAY_PREFIX = 'cvc_stack_team_pair_play'  # (team_pair, week)


def per_pair_grade_matchup_counts(
    data: Dict, club_pair: Tuple[str, str],
) -> Dict[str, int]:
    """Return `{grade: distinct_matchup_count}` for `club_pair`.

    A matchup is one entry in `data['games']` whose two teams belong one to
    each club in the pair. So Maitland (1 PHL team) vs Norths (1 PHL team)
    counts 1 matchup in PHL; Maitland (1 PHL team) vs Norths (2 PHL teams)
    counts 2 matchups (each Maitland-vs-Norths team pair is one matchup).

    Each matchup is played `num_rounds[grade] / num_matchups_in_grade`
    times across the season — same as how `EqualGamesAndBalanceMatchUps`
    interprets `games`. So the per-pair Sunday meeting count = number of
    matchups in `games` for this pair-in-this-grade × that frequency.

    For the typical case (round-robin where every matchup plays once or
    twice), counting matchup ENTRIES yields the per-season meeting count
    directly. The atom uses `meetings_for(pair, grade) = matchups *
    meetings_per_matchup` (see `per_pair_grade_meeting_counts`).
    """
    club_a, club_b = club_pair
    teams = data.get('teams', []) or []
    team_club = {t.name: t.club.name for t in teams}

    counts: Dict[str, int] = defaultdict(int)
    for (t1, t2, grade) in data.get('games', []) or []:
        c1 = team_club.get(t1)
        c2 = team_club.get(t2)
        if not c1 or not c2:
            continue
        # Cross-club between this specific pair.
        if {c1, c2} == {club_a, club_b}:
            counts[grade] += 1
    return dict(counts)


def per_pair_grade_meeting_counts(
    data: Dict, club_pair: Tuple[str, str],
) -> Dict[str, int]:
    """Return `{grade: per_season_meeting_count}` for `club_pair`.

    `meetings = matchups * meetings_per_matchup` where
    `meetings_per_matchup = R_grade // (T-1)` (even T) or `R_grade // T`
    (odd T). `R_grade = num_rounds[grade]` (the per-team game count for that
    grade — three-tier override system, see `utils.max_games_per_grade`).
    Falls back to `num_rounds['max']` if the per-grade entry is missing.

    For a single matchup pair (1 team vs 1 team) this equals the per-season
    meeting count for the pair. For multi-team-per-club-per-grade, every
    matchup plays that many times so the per-pair total scales linearly:
    Maitland (1 PHL team) vs Norths (2 PHL teams) = 2 matchups × per_matchup
    meetings per matchup.

    The legacy `ClubVsClubAlignment` used `num_rounds['max']` (the aggregate
    across all grades) for ALL grades because the legacy lower-grade block
    only enforced "at least N coincidences" loosely. spec-005 needs per-pair
    precision so we use per-grade `R`.

    NOTE (spec-038): use `per_pair_grade_aligned_weekends` for the stacking
    budget; this function returns the game-count which is a different quantity
    when multi-team-per-club-per-grade (a×b matchups × per_matchup games each
    ≠ max(a,b)×per_matchup aligned weekends). This function is correct for
    grade enumeration (checking which grades have cross-club matchups) and for
    per-season game-count totals, but NOT for the weekend stacking budget.
    """
    matchups = per_pair_grade_matchup_counts(data, club_pair)
    num_rounds = data.get('num_rounds', {}) or {}
    R_fallback = num_rounds.get('max', 0)

    grade_num_teams = {
        g.name: g.num_teams for g in data.get('grades', []) or []
    }

    out: Dict[str, int] = {}
    for grade, matchup_count in matchups.items():
        T = grade_num_teams.get(grade, 0)
        if T <= 1 or matchup_count == 0:
            continue
        R = num_rounds.get(grade, R_fallback)
        per_matchup = R // (T - 1) if T % 2 == 0 else R // T
        if per_matchup <= 0:
            continue
        out[grade] = matchup_count * per_matchup
    return out


def pair_grade_sunday_meetings(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> int:
    """Return the Sunday-available meeting count for `(pair, grade)`.

    For PHL: subtract FORCED-Friday meetings for the pair (FORCED-aware via
    `phl_forced_friday_meetings`), so the stacking is over only the Sundays
    PHL can actually use.

    For any other grade: full `per_pair_grade_meeting_counts(grade)` — those
    grades don't play Friday so every meeting is Sunday-available.

    Returns 0 if the grade has no per-pair meetings or the budget is
    exhausted by FORCED Fridays.
    """
    meetings = per_pair_grade_meeting_counts(data, club_pair)
    total = meetings.get(grade, 0)
    if total == 0:
        return 0
    if grade != 'PHL':
        return total
    club_a, club_b = club_pair
    forced_fri = phl_forced_friday_meetings(data, club_a, club_b)
    return max(0, total - forced_fri)


def team_pair_counts(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> Tuple[int, int]:
    """Return `(a, b)` = (# teams of club_A in `grade`, # teams of club_B).

    Order is preserved from `club_pair`: `a` counts teams belonging to
    `club_pair[0]`, `b` counts teams belonging to `club_pair[1]`. The
    result is NOT sorted by magnitude — `(a, b)` reflects `club_pair` order
    so callers can determine which club contributes more teams.

    Returns `(0, n)` or `(n, 0)` if one club fields no teams in the grade.

    Example (season_test, PHL, ('Maitland', 'Gosford')):
        Maitland fields 1 PHL team, Gosford fields 1 PHL team → (1, 1).
    Example (season_test, 6th, ('Tigers', 'University')):
        Tigers fields 2 6th teams, University fields 2 6th teams → (2, 2).
    """
    club_a, club_b = club_pair
    teams = data.get('teams', []) or []

    a = sum(1 for t in teams if t.club.name == club_a and t.grade == grade)
    b = sum(1 for t in teams if t.club.name == club_b and t.grade == grade)
    return (a, b)


def enumerate_team_pairs_in_pair_grade(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> List[Tuple[str, str]]:
    """Return every unordered cross-club team-pair `(t1, t2)` in `grade`.

    Conditions:
      - `{t1.club, t2.club} == set(club_pair)` (one team from each club).
      - Both teams play in `grade`.

    Each tuple is internally sorted so `t1 < t2` alphabetically (matching the
    convention in `data['games']` where team1 < team2). The result list is
    sorted by `(t1, t2)` string for deterministic iteration.

    Returns an empty list if either club fields 0 teams in the grade.

    Example (season_test, 6th, ('Tigers', 'University')):
        Returns the 4 cross-club pairs (Tigers Black × Uni Gentlemen,
        Tigers Black × Uni Seapigs, Tigers Yellow × Uni Gentlemen,
        Tigers Yellow × Uni Seapigs) — exactly 2×2=4 pairs.
    """
    club_a, club_b = club_pair
    teams = data.get('teams', []) or []

    teams_a = sorted(t.name for t in teams if t.club.name == club_a and t.grade == grade)
    teams_b = sorted(t.name for t in teams if t.club.name == club_b and t.grade == grade)

    result: List[Tuple[str, str]] = []
    for ta in teams_a:
        for tb in teams_b:
            # Internally sort alphabetically so t1 < t2.
            t1, t2 = (ta, tb) if ta < tb else (tb, ta)
            result.append((t1, t2))

    return sorted(result)


def _per_matchup_for_grade(data: Dict, grade: str) -> int:
    """Return `per_matchup` for a grade: R//(T-1) if T even else R//T.

    This is the same formula used in `per_pair_grade_meeting_counts`. Extracted
    here so `per_pair_grade_aligned_weekends` can reuse it without re-deriving.
    Returns 0 if the grade is not found or T <= 1.
    """
    num_rounds = data.get('num_rounds', {}) or {}
    grade_objs = data.get('grades', []) or []
    R_fallback = num_rounds.get('max', 0)

    T = 0
    for g in grade_objs:
        if g.name == grade:
            T = g.num_teams
            break
    if T <= 1:
        return 0

    R = num_rounds.get(grade, R_fallback)
    return R // (T - 1) if T % 2 == 0 else R // T


def per_pair_grade_aligned_weekends(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> int:
    """Return the number of distinct aligned weekends for `(club_pair, grade)`.

    Formula: `max(a, b) * per_matchup` where `a, b = team_pair_counts(...)`.

    This is the CORRECT stacking budget for spec-038. It differs from
    `per_pair_grade_meeting_counts()[grade]` (which returns `a*b*per_matchup`,
    a game count) when `a != b` or `min(a,b) > 1`.

    Case table:
      1×1, per_matchup=1 → max(1,1)*1 = 1 weekend, 1 game/weekend.
      1×1, per_matchup=2 → max(1,1)*2 = 2 weekends, 1 game/weekend.
      2×2, per_matchup=1 → max(2,2)*1 = 2 weekends, 2 games/weekend.
      2×2, per_matchup=2 → max(2,2)*2 = 4 weekends, 2 games/weekend.
      1×2, per_matchup=1 → max(1,2)*1 = 2 weekends, 1 game/weekend.
      1×2, per_matchup=2 → max(1,2)*2 = 4 weekends, 1 game/weekend.

    Returns 0 if either `a` or `b` is 0, or if `per_matchup` is 0.
    """
    a, b = team_pair_counts(data, club_pair, grade)
    if a == 0 or b == 0:
        return 0
    per_matchup = _per_matchup_for_grade(data, grade)
    if per_matchup == 0:
        return 0
    return max(a, b) * per_matchup


def pair_grade_sunday_aligned_weekends(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> int:
    """Return the Sunday-available aligned-weekend count for `(pair, grade)`.

    Replaces `pair_grade_sunday_meetings` as the stacking budget helper for
    spec-038. Uses the aligned-weekend formula instead of the game-count
    formula, which is correct when multi-team-per-club-per-grade.

    For non-PHL grades: returns `per_pair_grade_aligned_weekends(data, pair, grade)`
    directly (those grades only play Sundays, so all aligned weekends are
    Sunday-available).

    For PHL: subtracts FORCED-Friday meetings for the pair
    (`phl_forced_friday_meetings(data, A, B)`), since those consume the pair's
    matchup budget but cannot satisfy Sunday stacking.

    Returns 0 if:
      - Either club fields 0 teams in the grade (→ per_pair_grade_aligned_weekends == 0).
      - per_matchup is 0 (→ per_pair_grade_aligned_weekends == 0).
      - PHL budget is exhausted by FORCED Fridays (clamped to 0, not negative).
    """
    weekends = per_pair_grade_aligned_weekends(data, club_pair, grade)
    if weekends == 0:
        return 0
    if grade != 'PHL':
        return weekends
    club_a, club_b = club_pair
    forced_fri = phl_forced_friday_meetings(data, club_a, club_b)
    return max(0, weekends - forced_fri)


def pair_grade_sunday_aligned_weekend_range(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> Tuple[int, int]:
    """Return `(min_aligned_weekends, max_aligned_weekends)` for `(pair, grade)`.

    spec-038 fix: `EnsureEqualGamesAndBalanceMatchUps` enforces each per-team-pair
    meeting count in `[base, base+1]` where `base = per_matchup` (R // (T-1) for
    even T, R // T for odd T). When `extras = R - base * T_eff > 0` some pairs
    MUST meet `base+1` times to satisfy the per-team total `sum_vars == R`. The
    earlier exact `== per_matchup` budget under-counts those pairs and made the
    atom INFEASIBLE on `season_test` for 5th (T=9 R=16 base=1, all pairs forced to
    2 meetings) and similar for 3rd/4th.

    The atom must therefore allow `sum_w play_pg ∈ [max(a,b)*base, max(a,b)*(base+1)]`
    (mirroring BalancedMatchups), and per-tp `sum_w tp_play ∈ [base, base+1]`
    (mirroring the per-tp BalancedMatchups bound).

    For PHL: subtract `phl_forced_friday_meetings(A, B)` from BOTH ends, clamped
    to ≥ 0 (forced Friday meetings consume matchup budget without contributing to
    Sunday stacking).

    For non-PHL: no Friday subtraction (those grades only play Sundays).

    Returns `(0, 0)` if either club fields 0 teams in the grade, or if
    per_matchup == 0, or if PHL forced Fridays consume the entire range.
    """
    a, b = team_pair_counts(data, club_pair, grade)
    if a == 0 or b == 0:
        return (0, 0)
    base = _per_matchup_for_grade(data, grade)
    if base == 0:
        return (0, 0)

    max_ab = max(a, b)
    min_budget = max_ab * base
    max_budget = max_ab * (base + 1)

    if grade == 'PHL':
        forced_fri = phl_forced_friday_meetings(data, club_pair[0], club_pair[1])
        min_budget = max(0, min_budget - forced_fri)
        max_budget = max(0, max_budget - forced_fri)

    return (min_budget, max_budget)


def team_pair_sunday_meetings_range(
    data: Dict, club_pair: Tuple[str, str], grade: str,
) -> Tuple[int, int]:
    """Return `(min, max)` Sunday meetings PER team-pair for `(pair, grade)`.

    Mirrors `EnsureEqualGamesAndBalanceMatchUps`'s `[base, base+1]` per-team-pair
    bound, restricted to Sundays only (since non-PHL grades are Sunday-only and
    PHL Friday meetings are forced via FORCED_GAMES, not chosen freely).

    For non-PHL: `(base, base+1)`. For PHL: subtract `phl_forced_friday_meetings`
    from both ends, clamped to ≥ 0 (PHL is always 1×1 in the current league so
    per-tp and per-pair are the same).

    Returns `(0, 0)` if base == 0 or PHL forced consumes the entire range.
    """
    a, b = team_pair_counts(data, club_pair, grade)
    if a == 0 or b == 0:
        return (0, 0)
    base = _per_matchup_for_grade(data, grade)
    if base == 0:
        return (0, 0)

    min_tp = base
    max_tp = base + 1

    if grade == 'PHL':
        forced_fri = phl_forced_friday_meetings(data, club_pair[0], club_pair[1])
        min_tp = max(0, min_tp - forced_fri)
        max_tp = max(0, max_tp - forced_fri)

    return (min_tp, max_tp)


def enumerate_club_pairs(data: Dict) -> List[Tuple[str, str]]:
    """Return every unordered (club_a, club_b) appearing in `data['games']`
    as a cross-club matchup in at least one grade. Sorted for determinism."""
    teams = data.get('teams', []) or []
    team_club = {t.name: t.club.name for t in teams}

    pairs: Set[Tuple[str, str]] = set()
    for (t1, t2, _grade) in data.get('games', []) or []:
        c1 = team_club.get(t1)
        c2 = team_club.get(t2)
        if not c1 or not c2 or c1 == c2:
            continue
        pairs.add(tuple(sorted((c1, c2))))
    return sorted(pairs)


def collect_pair_grade_week_vars(
    X: Dict, data: Dict, grade: str, club_pair: Tuple[str, str],
) -> Dict[int, List]:
    """Return `{week: [Sunday vars for this pair+grade]}`.

    Skips locked weeks, dummy keys, and Friday vars (the stacking model is
    Sunday-only — PHL Fridays are subtracted from the budget upstream).
    """
    locked_weeks = set(data.get('locked_weeks', set()) or set())
    teams = data.get('teams', []) or []
    team_club = {t.name: t.club.name for t in teams}
    club_a, club_b = club_pair

    out: Dict[int, List] = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11 or not key[3]:
            continue
        if key[2] != grade:
            continue
        if key[3] != 'Sunday':
            continue
        if key[6] in locked_weeks:
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2:
            continue
        if {c1, c2} != {club_a, club_b}:
            continue
        out[key[6]].append(var)
    return dict(out)


def collect_pair_week_sunday_vars(
    X: Dict, data: Dict, club_pair: Tuple[str, str],
) -> Dict[int, List[Tuple]]:
    """Return `{week: [(var, key) for every Sunday cross-club var for the pair]}`.

    Used by the co-location atom — needs (var, field, day_slot) per (pair,
    week) regardless of grade. Skips locked weeks, dummy keys, Fridays.
    """
    locked_weeks = set(data.get('locked_weeks', set()) or set())
    teams = data.get('teams', []) or []
    team_club = {t.name: t.club.name for t in teams}
    club_a, club_b = club_pair

    out: Dict[int, List[Tuple]] = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11 or not key[3]:
            continue
        if key[3] != 'Sunday':
            continue
        if key[6] in locked_weeks:
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2:
            continue
        if {c1, c2} != {club_a, club_b}:
            continue
        out[key[6]].append((var, key))
    return dict(out)


def sorted_grades_by_desc_count(
    sunday_counts: Dict[str, int],
) -> List[Tuple[str, int]]:
    """Sort grades by Sunday count descending; tie-break alphabetically for
    deterministic stage ordering. Zero-count grades are excluded.

    The stacking implication chain runs consecutive-pair in this order, so
    determinism matters — a stable ordering across runs gives reproducible
    helper var labels.
    """
    items = [(g, c) for g, c in sunday_counts.items() if c > 0]
    return sorted(items, key=lambda gc: (-gc[1], gc[0]))


__all__ = [
    'STACK_PLAY_PREFIX',
    'STACK_FIELD_USED_PREFIX',
    'STACK_SLOT_USED_PREFIX',
    'STACK_ACTIVE_PREFIX',
    'STACK_TEAM_PAIR_PLAY_PREFIX',
    'per_pair_grade_matchup_counts',
    'per_pair_grade_meeting_counts',
    'pair_grade_sunday_meetings',
    'team_pair_counts',
    'enumerate_team_pairs_in_pair_grade',
    'per_pair_grade_aligned_weekends',
    'pair_grade_sunday_aligned_weekends',
    'pair_grade_sunday_aligned_weekend_range',
    'team_pair_sunday_meetings_range',
    'enumerate_club_pairs',
    'collect_pair_grade_week_vars',
    'collect_pair_week_sunday_vars',
    'sorted_grades_by_desc_count',
]
