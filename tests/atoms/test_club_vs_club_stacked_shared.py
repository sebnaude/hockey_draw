# spec-038 Unit A: hand-computed oracle tests for new shared helpers.
"""Tests for the spec-038 new helpers in `_club_vs_club_stacked_shared`.

Tests cover:
  - `team_pair_counts` — returns (a, b) in club_pair order.
  - `enumerate_team_pairs_in_pair_grade` — deterministic ordering + completeness.
  - `per_pair_grade_aligned_weekends` — case table from spec-038 "Why" section.
  - `pair_grade_sunday_aligned_weekends` — PHL subtraction + budget exhaustion.

Every oracle is hand-computed inline in each test or docstring.
No mocks, no monkeypatching.

Season_test team layout (relevant grades):
  PHL:  every club has exactly 1 team → all PHL pairs are 1×1, per_matchup=4.
  2nd:  Norths(1), Souths(1), Tigers(1), Wests(1) → per_matchup=6.
  4th:  Maitland(1), University(2: Redhogs + Seapigs), Wests(2: Green + Red), ..., T=11, per_matchup=1.
  5th:  Tigers(1), University(1), Colts(2: Gold + Green), Wests(2: Green + Red), T=9, per_matchup=1.
  6th:  Tigers(2: Yellow + Black), University(2: Gentlemen + Seapigs), T=10, per_matchup=2.

NOTE: spec-038 review note mentions "Tigers-University 5th (2×2)" but in season_test
Tigers has only 1 team in 5th and University has 1 team in 5th. The actual 2×2 pair is
Tigers-University 6th. All tests below use the actual season_test data.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import load_season_data
from constraints.atoms._club_vs_club_stacked_shared import (
    _per_matchup_for_grade,
    enumerate_team_pairs_in_pair_grade,
    pair_grade_sunday_aligned_weekend_range,
    pair_grade_sunday_aligned_weekends,
    per_pair_grade_aligned_weekends,
    team_pair_counts,
    team_pair_sunday_meetings_range,
)
from constraints.atoms._phl_forced_friday_helper import (
    club_umbrella_forced_friday_meetings,
    phl_forced_friday_meetings,
)
from constraints.atoms.base import MAITLAND
from tests.atoms.club_vs_club_stacked_fixture import build_stacked_fixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def season_test_data():
    """Load season_test data once for all tests in this module."""
    return load_season_data('test')


# ---------------------------------------------------------------------------
# team_pair_counts
# ---------------------------------------------------------------------------


class TestTeamPairCounts:
    """Tests for `team_pair_counts(data, club_pair, grade) -> (a, b)`.

    Returns (count_of_club_pair[0], count_of_club_pair[1]) — in club_pair order.
    """

    def test_phl_1x1_maitland_gosford(self, season_test_data):
        """Given (Maitland, Gosford) in PHL, both have exactly 1 PHL team.
        Expected: (1, 1).
        Hand oracle: Maitland PHL = 1 team; Gosford PHL = 1 team."""
        a, b = team_pair_counts(season_test_data, ('Maitland', 'Gosford'), 'PHL')
        assert a == 1
        assert b == 1

    def test_phl_1x1_norths_wests(self, season_test_data):
        """Given (Norths, Wests) in PHL, both have exactly 1 PHL team.
        Expected: (1, 1)."""
        a, b = team_pair_counts(season_test_data, ('Norths', 'Wests'), 'PHL')
        assert a == 1
        assert b == 1

    def test_6th_2x2_tigers_university(self, season_test_data):
        """Given (Tigers, University) in 6th: Tigers has 2 (Yellow + Black),
        University has 2 (Gentlemen + Seapigs).
        Expected: (2, 2).
        Hand oracle: season_test 6th grade Tigers = [Tigers Yellow 6th, Tigers Black 6th];
        University = [University Gentlemen 6th, University Seapigs 6th]."""
        a, b = team_pair_counts(season_test_data, ('Tigers', 'University'), '6th')
        assert a == 2
        assert b == 2

    def test_4th_1x2_maitland_university(self, season_test_data):
        """Given (Maitland, University) in 4th: Maitland has 1, University has 2.
        Expected: (1, 2).
        Hand oracle: Maitland 4th = [Maitland 4th]; University 4th = [Redhogs, Seapigs]."""
        a, b = team_pair_counts(season_test_data, ('Maitland', 'University'), '4th')
        assert a == 1
        assert b == 2

    def test_4th_order_preserved_flipped_pair(self, season_test_data):
        """Flipping the club_pair tuple gives the swapped (b, a) result.
        Given (University, Maitland) in 4th → (2, 1) not (1, 2).
        The function must NOT sort by magnitude."""
        a_orig, b_orig = team_pair_counts(season_test_data, ('Maitland', 'University'), '4th')
        a_flip, b_flip = team_pair_counts(season_test_data, ('University', 'Maitland'), '4th')
        # Original order (Maitland first) → (1, 2)
        assert a_orig == 1 and b_orig == 2
        # Flipped order (University first) → (2, 1)
        assert a_flip == 2 and b_flip == 1

    def test_edge_club_with_zero_teams_in_grade(self, season_test_data):
        """Gosford fields no teams in 2nd grade → a=1 (Norths), b=0 (Gosford).
        Hand oracle: Norths 2nd has 1 team; Gosford has 0 teams in 2nd.
        Expected: (1, 0)."""
        a, b = team_pair_counts(season_test_data, ('Norths', 'Gosford'), '2nd')
        assert a == 1
        assert b == 0

    def test_edge_both_clubs_zero_in_grade(self, season_test_data):
        """Gosford and Port Stephens both field 0 teams in 2nd grade.
        Expected: (0, 0)."""
        a, b = team_pair_counts(season_test_data, ('Gosford', 'Port Stephens'), '2nd')
        assert a == 0
        assert b == 0

    def test_5th_1x1_tigers_university(self, season_test_data):
        """In 5th grade: Tigers has 1 team (Tigers 5th), University has 1 team
        (University 5th). NOTE: 5th is NOT 2×2 in season_test despite spec review
        note saying otherwise — the actual 2×2 pair is in 6th grade.
        Expected: (1, 1)."""
        a, b = team_pair_counts(season_test_data, ('Tigers', 'University'), '5th')
        assert a == 1
        assert b == 1


# ---------------------------------------------------------------------------
# enumerate_team_pairs_in_pair_grade
# ---------------------------------------------------------------------------


class TestEnumerateTeamPairsInPairGrade:
    """Tests for `enumerate_team_pairs_in_pair_grade(data, club_pair, grade)`."""

    def test_deterministic_ordering_same_result_twice(self, season_test_data):
        """Calling the function twice with the same args returns identical lists.
        No non-determinism from set iteration."""
        first = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Tigers', 'University'), '6th'
        )
        second = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Tigers', 'University'), '6th'
        )
        assert first == second

    def test_2x2_completeness_tigers_university_6th(self, season_test_data):
        """Given (Tigers, University) in 6th with 2 teams each: expect exactly
        4 cross-club team-pairs (2×2=4).
        Hand oracle:
          Tigers: [Tigers Black 6th, Tigers Yellow 6th] (sorted alpha)
          University: [University Gentlemen 6th, University Seapigs 6th]
          Cross pairs (sorted, each internally sorted alpha):
            (Tigers Black 6th, University Gentlemen 6th)    ← T < U ✓
            (Tigers Black 6th, University Seapigs 6th)      ← T < U ✓
            (Tigers Yellow 6th, University Gentlemen 6th)   ← T < U ✓
            (Tigers Yellow 6th, University Seapigs 6th)     ← T < U ✓
        """
        pairs = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Tigers', 'University'), '6th'
        )
        assert len(pairs) == 4  # 2 × 2 = 4 cross-club pairs
        expected = [
            ('Tigers Black 6th', 'University Gentlemen 6th'),
            ('Tigers Black 6th', 'University Seapigs 6th'),
            ('Tigers Yellow 6th', 'University Gentlemen 6th'),
            ('Tigers Yellow 6th', 'University Seapigs 6th'),
        ]
        assert pairs == expected

    def test_each_tuple_internally_sorted_alphabetically(self, season_test_data):
        """Every tuple in the result has t1 < t2 alphabetically (matching the
        game key convention where team1 < team2)."""
        pairs = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Tigers', 'University'), '6th'
        )
        for t1, t2 in pairs:
            assert t1 < t2, f'Expected t1 < t2 alphabetically, got ({t1!r}, {t2!r})'

    def test_1x1_single_pair_phl(self, season_test_data):
        """Given (Maitland, Gosford) in PHL (1 team each): exactly 1 pair.
        Hand oracle: 1 × 1 = 1 pair. Internally sorted: Gosford < Maitland
        alpha → tuple is ('Gosford PHL', 'Maitland PHL')."""
        pairs = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Maitland', 'Gosford'), 'PHL'
        )
        assert len(pairs) == 1
        assert pairs[0] == ('Gosford PHL', 'Maitland PHL')

    def test_1x2_asymmetric_4th_maitland_university(self, season_test_data):
        """Given (Maitland, University) in 4th: Maitland=1, University=2.
        Expected: 2 pairs (1×2=2).
        Hand oracle:
          Maitland teams: [Maitland 4th]
          University teams: [University Redhogs 4th, University Seapigs 4th]
          Cross pairs (each internally sorted alpha):
            (Maitland 4th, University Redhogs 4th)   ← M < U ✓
            (Maitland 4th, University Seapigs 4th)   ← M < U ✓
        """
        pairs = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Maitland', 'University'), '4th'
        )
        assert len(pairs) == 2
        assert pairs == [
            ('Maitland 4th', 'University Redhogs 4th'),
            ('Maitland 4th', 'University Seapigs 4th'),
        ]

    def test_empty_when_one_club_has_zero_teams(self, season_test_data):
        """Given (Norths, Gosford) in 2nd: Gosford has 0 teams in 2nd.
        Expected: empty list."""
        pairs = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Norths', 'Gosford'), '2nd'
        )
        assert pairs == []

    def test_empty_when_both_clubs_have_zero_teams(self, season_test_data):
        """Given (Gosford, Port Stephens) in 2nd: both have 0 teams.
        Expected: empty list."""
        pairs = enumerate_team_pairs_in_pair_grade(
            season_test_data, ('Gosford', 'Port Stephens'), '2nd'
        )
        assert pairs == []


# ---------------------------------------------------------------------------
# per_pair_grade_aligned_weekends — case table from spec-038 "Why" section
# ---------------------------------------------------------------------------


# Parametrised case table covering every row in spec-038 "Why" table:
#
#  Layout              | a | b | per_matchup | weekends = max(a,b)*pm | games/wknd = min(a,b)
#  1×1, per_matchup=1  | 1 | 1 |     1       |          1             |          1
#  1×1, per_matchup=2  | 1 | 1 |     2       |          2             |          1
#  2×2, per_matchup=1  | 2 | 2 |     1       |          2             |          2
#  2×2, per_matchup=2  | 2 | 2 |     2       |          4             |          2
#  1×2, per_matchup=1  | 1 | 2 |     1       |          2             |          1
#  1×2, per_matchup=2  | 1 | 2 |     2       |          4             |          1
#
# Arithmetic verification for each row:
#   total games = a × b × per_matchup = weekends × games/weekend ✓
#   1×1, pm=1: 1×1×1=1 = 1×1 ✓
#   1×1, pm=2: 1×1×2=2 = 2×1 ✓
#   2×2, pm=1: 2×2×1=4 = 2×2 ✓
#   2×2, pm=2: 2×2×2=8 = 4×2 ✓
#   1×2, pm=1: 1×2×1=2 = 2×1 ✓
#   1×2, pm=2: 1×2×2=4 = 4×1 ✓


def _build_case_table_fixture(
    grade_counts,
    extra,
    per_matchup_override=None,
    num_weeks=6,
):
    """Helper: build a stacked fixture and optionally override per_matchup.

    `build_stacked_fixture` always sets `num_rounds[grade] = 1*(T-1)` for
    multi-team cases (per_matchup=1). For the pm=2 parametrized cases we need
    to double the `num_rounds` entry so that R//(T-1) or R//T == 2.

    Args:
      per_matchup_override: if not None, a dict {grade: desired_per_matchup}.
        For each grade listed, `num_rounds[grade]` is recomputed so that the
        per_matchup formula yields the desired value.
    """
    data = build_stacked_fixture(
        grade_counts,
        extra_teams_in_grade=extra,
        num_weeks=num_weeks,
    )
    if per_matchup_override:
        for grade, desired_pm in per_matchup_override.items():
            grade_objs = data.get('grades', [])
            T = next((g.num_teams for g in grade_objs if g.name == grade), 0)
            if T <= 1:
                continue
            # R // (T-1) = desired_pm (even T) or R // T = desired_pm (odd T)
            if T % 2 == 0:
                data['num_rounds'][grade] = desired_pm * (T - 1)
            else:
                data['num_rounds'][grade] = desired_pm * T
            # Keep 'max' consistent
            data['num_rounds']['max'] = max(
                v for k, v in data['num_rounds'].items() if k != 'max'
            )
    return data


@pytest.mark.parametrize(
    'grade_counts,extra,per_matchup_override,club_pair,grade,expected_weekends',
    [
        # 1×1, per_matchup=1 (fixture T=2, R=1 → pm=1//1=1)
        # max(1,1)*1 = 1 aligned weekend; 1 game/weekend; total 1×1×1=1 game.
        (
            {'PHL': 1}, None, None, ('Maitland', 'Norths'), 'PHL', 1,
        ),
        # 1×1, per_matchup=2 (fixture T=2, R=2 → pm=2//1=2)
        # max(1,1)*2 = 2 aligned weekends; 1 game/weekend; total 1×1×2=2 games.
        (
            {'PHL': 2}, None, None, ('Maitland', 'Norths'), 'PHL', 2,
        ),
        # 2×2, per_matchup=1 (T=4 even, pm=1; extra adds 1 Maitland+1 Norths in 3rd)
        # max(2,2)*1 = 2 aligned weekends; 2 games/weekend; total 2×2×1=4 games.
        (
            {'3rd': 1}, {'3rd': {'Maitland': 1, 'Norths': 1}}, None,
            ('Maitland', 'Norths'), '3rd', 2,
        ),
        # 2×2, per_matchup=2 (T=4 even, override pm to 2: R=6 → 6//3=2)
        # max(2,2)*2 = 4 aligned weekends; 2 games/weekend; total 2×2×2=8 games.
        (
            {'3rd': 1}, {'3rd': {'Maitland': 1, 'Norths': 1}}, {'3rd': 2},
            ('Maitland', 'Norths'), '3rd', 4,
        ),
        # 1×2, per_matchup=1 (T=3 odd, pm=1; extra adds 1 Norths only → a=1, b=2)
        # max(1,2)*1 = 2 aligned weekends; 1 game/weekend; total 1×2×1=2 games.
        (
            {'3rd': 1}, {'3rd': {'Norths': 1}}, None,
            ('Maitland', 'Norths'), '3rd', 2,
        ),
        # 1×2, per_matchup=2 (T=3 odd, override pm to 2: R=6 → 6//3=2)
        # max(1,2)*2 = 4 aligned weekends; 1 game/weekend; total 1×2×2=4 games.
        (
            {'3rd': 1}, {'3rd': {'Norths': 1}}, {'3rd': 2},
            ('Maitland', 'Norths'), '3rd', 4,
        ),
    ],
    ids=['1x1_pm1', '1x1_pm2', '2x2_pm1', '2x2_pm2', '1x2_pm1', '1x2_pm2'],
)
def test_per_pair_grade_aligned_weekends_case_table(
    grade_counts, extra, per_matchup_override, club_pair, grade, expected_weekends,
):
    """Case table from spec-038 'Why' section — parametrised.

    Constructs exact (a, b, per_matchup) combinations and asserts
    `per_pair_grade_aligned_weekends == max(a, b) * per_matchup`.

    Arithmetic (all rows):
      1×1 pm=1: max(1,1)*1=1.  Total games=1*1*1=1=1*1. ✓
      1×1 pm=2: max(1,1)*2=2.  Total games=1*1*2=2=2*1. ✓
      2×2 pm=1: max(2,2)*1=2.  Total games=2*2*1=4=2*2. ✓
      2×2 pm=2: max(2,2)*2=4.  Total games=2*2*2=8=4*2. ✓
      1×2 pm=1: max(1,2)*1=2.  Total games=1*2*1=2=2*1. ✓
      1×2 pm=2: max(1,2)*2=4.  Total games=1*2*2=4=4*1. ✓
    """
    data = _build_case_table_fixture(
        grade_counts, extra, per_matchup_override,
        num_weeks=max(expected_weekends + 1, 6),
    )
    result = per_pair_grade_aligned_weekends(data, club_pair, grade)
    assert result == expected_weekends, (
        f'per_pair_grade_aligned_weekends({club_pair}, {grade!r}) = {result}, '
        f'expected {expected_weekends} (max(a,b)*per_matchup)'
    )


class TestPerPairGradeAlignedWeekendsSeasonTest:
    """Real season_test data scenarios for `per_pair_grade_aligned_weekends`."""

    def test_phl_1x1_maitland_gosford(self, season_test_data):
        """PHL (1×1), per_matchup=4. Expected: max(1,1)*4 = 4."""
        result = per_pair_grade_aligned_weekends(season_test_data, ('Maitland', 'Gosford'), 'PHL')
        assert result == 4  # max(1,1) * 4

    def test_6th_2x2_tigers_university(self, season_test_data):
        """6th grade (2×2), per_matchup=2. Expected: max(2,2)*2 = 4.
        Hand oracle: T=10 (even), R=18, pm=18//9=2. a=2, b=2 → max=2. 2*2=4."""
        result = per_pair_grade_aligned_weekends(season_test_data, ('Tigers', 'University'), '6th')
        assert result == 4  # max(2,2) * 2

    def test_4th_1x2_maitland_university(self, season_test_data):
        """4th grade (1×2), per_matchup=1. Expected: max(1,2)*1 = 2.
        Hand oracle: T=11 (odd), R=18, pm=18//11=1. a=1, b=2 → max=2. 2*1=2."""
        result = per_pair_grade_aligned_weekends(season_test_data, ('Maitland', 'University'), '4th')
        assert result == 2  # max(1,2) * 1

    def test_zero_when_club_has_no_teams_in_grade(self, season_test_data):
        """Gosford has 0 teams in 2nd grade. Expected: 0 (guard condition)."""
        result = per_pair_grade_aligned_weekends(season_test_data, ('Norths', 'Gosford'), '2nd')
        assert result == 0

    def test_5th_1x1_tigers_university(self, season_test_data):
        """5th grade (1×1 for Tigers-University), per_matchup=1.
        Expected: max(1,1)*1 = 1.
        Hand oracle: T=9 (odd), R=16, pm=16//9=1. a=1, b=1 → 1."""
        result = per_pair_grade_aligned_weekends(season_test_data, ('Tigers', 'University'), '5th')
        assert result == 1  # max(1,1) * 1


# ---------------------------------------------------------------------------
# pair_grade_sunday_aligned_weekends
# ---------------------------------------------------------------------------


class TestPairGradeSundayAlignedWeekends:
    """Tests for `pair_grade_sunday_aligned_weekends(data, club_pair, grade)`.

    For non-PHL: identical to `per_pair_grade_aligned_weekends`.
    For PHL: subtracts `phl_forced_friday_meetings(data, A, B)`.
    Clamped to 0 — never negative.
    """

    def test_non_phl_equals_aligned_weekends(self, season_test_data):
        """Non-PHL grade: returned value equals `per_pair_grade_aligned_weekends`.
        Uses 6th grade (Tigers-University, 2×2, pm=2) → 4."""
        sunday = pair_grade_sunday_aligned_weekends(
            season_test_data, ('Tigers', 'University'), '6th'
        )
        aligned = per_pair_grade_aligned_weekends(
            season_test_data, ('Tigers', 'University'), '6th'
        )
        assert sunday == aligned
        assert sunday == 4

    def test_phl_zero_forced_fridays_season_test(self, season_test_data):
        """season_test has no FORCED_GAMES, so phl_forced_friday_meetings == 0.
        PHL (Maitland, Gosford): sunday_aligned == per_aligned == 4."""
        sunday = pair_grade_sunday_aligned_weekends(
            season_test_data, ('Maitland', 'Gosford'), 'PHL'
        )
        assert sunday == 4  # max(1,1)*4 - 0 = 4

    def test_phl_with_one_forced_friday_subtracts_one(self):
        """Synthetic: PHL pair with per_matchup=3 and 1 forced Friday.
        Expected Sunday budget: 3 - 1 = 2.
        Hand oracle:
          T=2, R=3 → per_matchup = 3//1 = 3. max(1,1)*3 = 3 aligned weekends.
          1 forced Friday → 3 - 1 = 2."""
        data = build_stacked_fixture({'PHL': 3})
        data['forced_games'] = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland PHL', 'Norths PHL'],
            'field_location': MAITLAND,
            'count': 1, 'constraint': 'equal',
            'description': 'One Maitland-vs-Norths PHL Friday',
        }]
        result = pair_grade_sunday_aligned_weekends(data, ('Maitland', 'Norths'), 'PHL')
        assert result == 2  # 3 - 1

    def test_phl_with_two_forced_fridays_subtracts_two(self):
        """Synthetic: PHL pair with per_matchup=4 and 2 forced Fridays.
        Expected Sunday budget: 4 - 2 = 2.
        Hand oracle: T=2, R=4 → per_matchup=4. 2 forced → 4-2=2."""
        data = build_stacked_fixture({'PHL': 4})
        data['forced_games'] = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland PHL', 'Norths PHL'],
            'field_location': MAITLAND,
            'count': 2, 'constraint': 'equal',
            'description': 'Two Maitland-vs-Norths PHL Fridays',
        }]
        result = pair_grade_sunday_aligned_weekends(data, ('Maitland', 'Norths'), 'PHL')
        assert result == 2  # 4 - 2

    def test_phl_budget_exhausted_clamped_to_zero(self):
        """Budget exhaustion: forced Fridays > aligned weekends → 0 (not negative).
        Synthetic: PHL per_matchup=2, 3 forced Fridays → max(0, 2-3) = 0.
        Hand oracle: T=2, R=2 → pm=2. 3 forced Fridays → max(0,2-3) = 0."""
        data = build_stacked_fixture({'PHL': 2})
        data['forced_games'] = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland PHL', 'Norths PHL'],
            'field_location': MAITLAND,
            'count': 3, 'constraint': 'equal',
            'description': 'Three Maitland-vs-Norths PHL Fridays (over budget)',
        }]
        result = pair_grade_sunday_aligned_weekends(data, ('Maitland', 'Norths'), 'PHL')
        assert result == 0  # clamped, not negative

    def test_zero_when_club_has_no_teams_in_grade(self, season_test_data):
        """Club with 0 teams in grade → 0 (early guard in per_pair_grade_aligned_weekends)."""
        result = pair_grade_sunday_aligned_weekends(
            season_test_data, ('Norths', 'Gosford'), '2nd'
        )
        assert result == 0

    def test_non_phl_forced_friday_does_not_affect_budget(self):
        """FORCED Friday entries for non-PHL grades are irrelevant to the budget.
        Only PHL subtracts forced Fridays; non-PHL returns full aligned weekends.
        Synthetic: 3rd grade pair with 1 forced Friday entry for 3rd → budget unchanged.
        Hand oracle: T=2, R=2 → per_matchup=2. max(1,1)*2=2. No subtraction for 3rd. → 2."""
        data = build_stacked_fixture({'3rd': 2})
        # Add a forced Friday for 3rd (which the helper ignores for non-PHL)
        data['forced_games'] = [{
            'grade': '3rd', 'day': 'Friday',
            'teams': ['Maitland 3rd', 'Norths 3rd'],
            'count': 1, 'constraint': 'equal',
            'description': 'Irrelevant forced Friday for non-PHL grade',
        }]
        result = pair_grade_sunday_aligned_weekends(data, ('Maitland', 'Norths'), '3rd')
        assert result == 2  # max(1,1)*2 unchanged — non-PHL never subtracts


# ---------------------------------------------------------------------------
# spec-044 Unit A: umbrella-forced-Friday-aware PHL Sunday FLOOR
# ---------------------------------------------------------------------------
#
# All oracles below use the REAL 2026 config (the authoritative source).
# PHL has T=6, R=20 → per_matchup base = 20 // (6-1) = 4 → range [4, 5].
# Live-config FORCED_GAMES PHL Friday entries are all venue-wide umbrellas:
#   - Central Coast Hockey Park (Gosford home venue): count==8 umbrella +
#     7 date-specific count==1 entries (dominated by the 8) → U(Gosford) = 8.
#   - Maitland Park (Maitland home venue): count==2 umbrella → U(Maitland) = 2.
# No pair-named PHL Friday entries exist, so phl_forced_friday_meetings == 0
# for every PHL pair.


@pytest.fixture(scope='module')
def real_data():
    """Load the real 2026 config once — the authoritative oracle source."""
    return load_season_data(2026)


class TestClubUmbrellaForcedFridayMeetings:
    """Tests for `club_umbrella_forced_friday_meetings(data, club)` (spec-044)."""

    def test_gosford_is_eight(self, real_data):
        # CCHP count==8 umbrella dominates the 7 date-specific count==1 entries
        # (max-per-away-venue de-dup). Oracle: max(8, 1, 1, 1, 1, 1, 1, 1) = 8.
        assert club_umbrella_forced_friday_meetings(real_data, 'Gosford') == 8

    def test_maitland_is_two(self, real_data):
        # Maitland Park PHL Friday umbrella count==2.
        assert club_umbrella_forced_friday_meetings(real_data, 'Maitland') == 2

    def test_central_club_is_zero(self, real_data):
        # Norths' home venue is the central NIHC venue → not a key in
        # home_field_map (club -> away venue) → no away-venue umbrella → 0.
        assert club_umbrella_forced_friday_meetings(real_data, 'Norths') == 0

    def test_unknown_or_empty_club_is_zero(self, real_data):
        assert club_umbrella_forced_friday_meetings(real_data, '') == 0
        assert club_umbrella_forced_friday_meetings(real_data, None) == 0
        assert club_umbrella_forced_friday_meetings(real_data, 'NoSuchClub') == 0


class TestSpec044RealConfigHasNoPairNamedFriday:
    """Documented invariant the umbrella oracles rely on."""

    def test_no_pair_named_phl_fridays(self, real_data):
        # The live 2026 config's PHL Friday forced entries are all venue-wide
        # umbrellas (no pair naming). So pair-named counts are 0 and the range
        # ceilings are NOT reduced by a pair term.
        assert phl_forced_friday_meetings(real_data, 'Gosford', 'Norths') == 0
        assert phl_forced_friday_meetings(real_data, 'Gosford', 'Souths') == 0
        assert phl_forced_friday_meetings(real_data, 'Maitland', 'Norths') == 0


class TestSpec044TeamPairSundayMeetingsRange:
    """`team_pair_sunday_meetings_range` umbrella-aware floor (spec-044)."""

    def test_phl_base_is_four(self, real_data):
        # PHL T=6, R=20 → base = 20 // 5 = 4. Sanity anchor for the oracles.
        assert _per_matchup_for_grade(real_data, 'PHL') == 4

    def test_gosford_souths_floor_clamped_to_zero(self, real_data):
        # Gosford umbrella=8, pair-named=0:
        #   tp_min = max(0, 4 - 0 - 8) = 0 ; tp_max = 4 + 1 - 0 = 5  → (0, 5).
        rng = team_pair_sunday_meetings_range(real_data, ('Gosford', 'Souths'), 'PHL')
        assert rng == (0, 5)

    def test_gosford_norths_real_config(self, real_data):
        # Spec worked example assumed a pair-named Gosford/Norths Friday (=> (0,4)),
        # but the LIVE config has no pair-named PHL Fridays, so pair term is 0:
        #   tp_min = max(0, 4 - 0 - 8) = 0 ; tp_max = 5 - 0 = 5  → (0, 5).
        rng = team_pair_sunday_meetings_range(real_data, ('Gosford', 'Norths'), 'PHL')
        assert rng == (0, 5)

    def test_maitland_norths_floor_two(self, real_data):
        # Maitland umbrella=2, pair-named=0:
        #   tp_min = max(0, 4 - 0 - 2) = 2 ; tp_max = 5  → (2, 5).
        rng = team_pair_sunday_meetings_range(real_data, ('Maitland', 'Norths'), 'PHL')
        assert rng == (2, 5)

    def test_central_pair_unchanged(self, real_data):
        # Both central (umbrella=0), pair-named=0 → range unchanged (4, 5).
        rng = team_pair_sunday_meetings_range(real_data, ('Norths', 'Souths'), 'PHL')
        assert rng == (4, 5)


class TestSpec044PairGradeSundayAlignedWeekendRange:
    """`pair_grade_sunday_aligned_weekend_range` applies the same lower-bound
    term (PHL is 1×1 → max_ab=1, so min_budget == tp_min, max_budget == tp_max)."""

    def test_gosford_souths_floor_clamped(self, real_data):
        rng = pair_grade_sunday_aligned_weekend_range(real_data, ('Gosford', 'Souths'), 'PHL')
        assert rng == (0, 5)

    def test_central_pair_unchanged(self, real_data):
        rng = pair_grade_sunday_aligned_weekend_range(real_data, ('Norths', 'Souths'), 'PHL')
        assert rng == (4, 5)


class TestSpec044NonPhlByteIdentical:
    """DoD-4 regression: non-PHL ranges are byte-identical pre/post fix.

    The umbrella helper short-circuits on grade != 'PHL', so every non-PHL
    pair's range equals the plain `(base, base+1)` formula — no umbrella term
    is ever applied.
    """

    def test_non_phl_pairs_equal_base_ceiling(self, real_data):
        # spec-044 review fix: the original loop paired Gosford WITH Maitland —
        # but Gosford fields 0 non-PHL teams, so every pair was silently skipped
        # and the umbrella clubs were never exercised in a non-PHL grade. Pair
        # the umbrella club Maitland (umb('Maitland')==2 in PHL) against central
        # clubs so the byte-identical assertion actually covers an umbrella club
        # in non-PHL grades, where the helper must short-circuit to 0.
        exercised = 0
        umbrella_club_exercised = 0
        for grade in ('2nd', '3rd', '4th', '5th', '6th'):
            base = _per_matchup_for_grade(real_data, grade)
            if base == 0:
                continue
            for club_pair in (
                ('Maitland', 'Norths'),
                ('Maitland', 'Souths'),
                ('Norths', 'Souths'),
            ):
                a, b = team_pair_counts(real_data, club_pair, grade)
                if a == 0 or b == 0:
                    continue  # pair not present in this grade
                tp = team_pair_sunday_meetings_range(real_data, club_pair, grade)
                wk = pair_grade_sunday_aligned_weekend_range(real_data, club_pair, grade)
                max_ab = max(a, b)
                assert tp == (base, base + 1), (
                    f'{club_pair} {grade}: tp {tp} != ({base},{base+1})'
                )
                assert wk == (max_ab * base, max_ab * (base + 1)), (
                    f'{club_pair} {grade}: wk {wk} != '
                    f'({max_ab*base},{max_ab*(base+1)})'
                )
                exercised += 1
                if 'Maitland' in club_pair:
                    umbrella_club_exercised += 1
        # Guard the coverage so the silent-skip gap (review finding) cannot recur.
        assert exercised > 0, 'no non-PHL pair was exercised — coverage gap'
        assert umbrella_club_exercised > 0, (
            'an umbrella club (Maitland) must be exercised in a non-PHL grade '
            'to prove the umbrella term is not applied off-PHL'
        )
