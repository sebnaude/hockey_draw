"""Tests for `constraints.atoms._phl_forced_friday_helper`.

After spec-037 this module exposes:
  - `phl_forced_friday_meetings(data, club_a, club_b)` (spec-005, unchanged)
  - `away_club_min_sundays_home(data, club)` (spec-037, NEW)
  - `away_club_max_sundays_home(data, club)` (spec-037, NEW)

The three legacy public functions (`phl_forced_friday_count`,
`away_club_required_sundays`, `away_club_total_weekends`) are GONE. Their
tests are deleted; the per-pair Friday meeting tests (still consumed by
`ClubVsClubStackedWeekends`) are retained verbatim.

All scenarios use REAL data shapes (no mocks). Hand-computed oracles in each
test docstring; the test asserts the helper output matches them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models import Club, Team
from constraints.atoms._phl_forced_friday_helper import (
    away_club_max_sundays_home,
    away_club_min_sundays_home,
    phl_forced_friday_meetings,
)


# ---------------------------------------------------------------------------
# away_club_min_sundays_home / away_club_max_sundays_home — spec-037 bounds
#
# Oracle table (hand-computed per spec-037 worked examples):
#
#   Maitland fields PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18
#     non-PHL home games: 9, 9, 8, 9 -> floor = max(9, 9, 8, 9) = 9
#     all-grade home games incl PHL: 10, 9, 9, 8, 9 -> ceiling = 10
#
#   Gosford fields only PHL=20
#     non-PHL home games: [] -> floor = 0
#     all-grade home games: 10 -> ceiling = 10
#
#   Hypothetical: club fields 3rd=22 (dominant), PHL=18
#     non-PHL home games: 11 -> floor = 11
#     all-grade home games: 11 (3rd), 9 (PHL) -> ceiling = max(11, 9) = 11
# ---------------------------------------------------------------------------


def _build_minimal_data(clubs_grades, num_rounds):
    """Build a minimal `data` dict with only what the bounds helpers read.

    `clubs_grades`: dict[club_name -> list[grade_name]] — which clubs field
                    which grades (one team per club/grade is enough for
                    `_grades_played_by_club`).
    `num_rounds`: dict[grade_name -> int] — required games per team in that
                  grade.
    """
    clubs = []
    teams = []
    for club_name, grades in clubs_grades.items():
        club = Club(name=club_name, home_field='dummy')
        clubs.append(club)
        for g in grades:
            teams.append(Team(name=f'{club_name} {g}', club=club, grade=g))
    return {
        'teams': teams,
        'clubs': clubs,
        'num_rounds': dict(num_rounds),
        'grades': [],
        'forced_games': [],
    }


class TestAwayClubMinSundaysHome:
    """Floor of the derived Sunday-home range."""

    def test_given_maitland_phl_and_4_other_grades_then_floor_is_nine(self):
        """GIVEN Maitland fields PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18,
        WHEN computing min_sundays_home,
        THEN floor = max(9, 9, 8, 9) = 9 (max across non-PHL).

        Hand-computed: 3rd 18//2=9, 4th 18//2=9, 5th 16//2=8, 6th 18//2=9.
        max(9, 9, 8, 9) = 9.
        """
        data = _build_minimal_data(
            {'Maitland': ['PHL', '3rd', '4th', '5th', '6th']},
            {'PHL': 20, '3rd': 18, '4th': 18, '5th': 16, '6th': 18},
        )
        assert away_club_min_sundays_home(data, 'Maitland') == 9

    def test_given_gosford_phl_only_then_floor_is_zero(self):
        """GIVEN Gosford fields only PHL=20,
        WHEN computing min_sundays_home,
        THEN floor = 0 (no non-PHL grades).

        Hand-computed: non-PHL grade list is empty -> 0.
        """
        data = _build_minimal_data(
            {'Gosford': ['PHL']},
            {'PHL': 20},
        )
        assert away_club_min_sundays_home(data, 'Gosford') == 0

    def test_given_club_with_third_grade_dominant_then_floor_eleven(self):
        """GIVEN a club fields PHL=18, 3rd=22,
        WHEN computing min_sundays_home,
        THEN floor = 11 (3rd grade drives, 22//2 = 11).
        """
        data = _build_minimal_data(
            {'TestClub': ['PHL', '3rd']},
            {'PHL': 18, '3rd': 22},
        )
        assert away_club_min_sundays_home(data, 'TestClub') == 11

    def test_given_unknown_club_then_floor_is_zero(self):
        """GIVEN no teams for the queried club,
        WHEN computing,
        THEN floor = 0 (grades list empty -> non-PHL list empty)."""
        data = _build_minimal_data({}, {})
        assert away_club_min_sundays_home(data, 'NoSuchClub') == 0


class TestAwayClubMaxSundaysHome:
    """Ceiling of the derived Sunday-home range."""

    def test_given_maitland_phl_and_4_other_grades_then_ceiling_is_ten(self):
        """GIVEN Maitland fields PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18,
        WHEN computing max_sundays_home,
        THEN ceiling = max(10, 9, 9, 8, 9) = 10 (PHL drives via 20//2=10).
        """
        data = _build_minimal_data(
            {'Maitland': ['PHL', '3rd', '4th', '5th', '6th']},
            {'PHL': 20, '3rd': 18, '4th': 18, '5th': 16, '6th': 18},
        )
        assert away_club_max_sundays_home(data, 'Maitland') == 10

    def test_given_gosford_phl_only_then_ceiling_is_ten(self):
        """GIVEN Gosford fields only PHL=20,
        WHEN computing max_sundays_home,
        THEN ceiling = 10 (PHL 20//2 = 10, ceil(20//2) = 10).
        """
        data = _build_minimal_data(
            {'Gosford': ['PHL']},
            {'PHL': 20},
        )
        assert away_club_max_sundays_home(data, 'Gosford') == 10

    def test_given_club_with_third_grade_dominant_then_ceiling_eleven(self):
        """GIVEN a club fields PHL=18, 3rd=22,
        WHEN computing max_sundays_home,
        THEN ceiling = max(11, 9) = 11 (3rd drives over PHL's 9 = 18//2).
        Per spec-037: PHL uses ceil, so (18+1)//2 = 9; 3rd uses floor 22//2 = 11.
        max(9, 11) = 11.
        """
        data = _build_minimal_data(
            {'TestClub': ['PHL', '3rd']},
            {'PHL': 18, '3rd': 22},
        )
        assert away_club_max_sundays_home(data, 'TestClub') == 11

    def test_given_unknown_club_then_ceiling_is_zero(self):
        """GIVEN no teams for the queried club, THEN ceiling = 0."""
        data = _build_minimal_data({}, {})
        assert away_club_max_sundays_home(data, 'NoSuchClub') == 0

    def test_given_odd_phl_num_rounds_then_ceiling_uses_ceil(self):
        """Defensive rounding: PHL=19 (odd) should give ceiling (19+1)//2 = 10.
        Floor for non-PHL stays at //2.

        GIVEN a club fields PHL=19 only,
        WHEN computing max_sundays_home,
        THEN ceiling = ceil(19/2) = 10.
        """
        data = _build_minimal_data(
            {'TestClub': ['PHL']},
            {'PHL': 19},
        )
        assert away_club_max_sundays_home(data, 'TestClub') == 10

    def test_given_odd_non_phl_num_rounds_then_floor_used(self):
        """Defensive rounding: non-PHL grade num_rounds=17 uses 17//2 = 8.
        (Production never sees this but the rule must be deterministic.)

        GIVEN a club fields PHL=20, 3rd=17,
        WHEN computing the two bounds,
        THEN floor = 17 // 2 = 8 (non-PHL uses floor),
             ceiling = max(20//2, 17//2) = max(10, 8) = 10 (PHL drives via 20).
        """
        data = _build_minimal_data(
            {'TestClub': ['PHL', '3rd']},
            {'PHL': 20, '3rd': 17},
        )
        assert away_club_min_sundays_home(data, 'TestClub') == 8
        assert away_club_max_sundays_home(data, 'TestClub') == 10


# ---------------------------------------------------------------------------
# phl_forced_friday_meetings — spec-005 per-pair Friday count (UNCHANGED)
# ---------------------------------------------------------------------------


class TestPhlForcedFridayMeetings:
    """Per-pair Friday count for `ClubVsClubStackedPHLSundayBudget` (spec-005).

    Same FORCED-aware greedy partition as the deleted `phl_forced_friday_count`,
    but narrowed to candidate vars between TWO specific clubs. Verifies:
      - Umbrella (count=2) for one club + per-pair (count=1) for that pair
        -> 1 (the per-pair entry pins one of the umbrella's two; total
        Maitland Fridays = 2 but only 1 is vs the specific opponent).
      - Per-pair-only entry -> its count.
      - Entry naming other clubs -> 0.
      - Self-pair (A == B) -> 0.
    """

    def test_given_no_forced_games_returns_zero(self, phl_data):
        """No FORCED -> 0."""
        phl_data['forced_games'] = []
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0

    def test_given_pair_explicit_when_counting_returns_entry_count(self, phl_data):
        """One per-pair Maitland-vs-Norths Friday entry, count=2 -> 2."""
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Maitland', 'Norths'],
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland-vs-Norths PHL Friday count = 2'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 2

    def test_given_umbrella_and_per_pair_returns_per_pair_count(self, phl_data):
        """Umbrella `count=2 Maitland Fridays` + per-pair `count=1 Maitland-vs-Norths`.

        Hand-computed: the umbrella `{club: Maitland, count: 2}` doesn't
        guarantee any Maitland-vs-Norths Friday (the 2 forced games could
        all be Maitland-vs-Tigers). Only the per-pair entry pins a
        Maitland-vs-Norths Friday. Result = 1 (the per-pair entry's count).

        This is the per-PAIR helper. See `_entry_targets_pair_phl_friday`
        docstring.
        """
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday', 'club': 'Maitland',
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland total Friday PHL count == 2'},
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Maitland', 'Norths'],
             'count': 1,
             'description': 'Maitland-vs-Norths Friday'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 1

    def test_given_other_pair_entry_returns_zero(self, phl_data):
        """A Norths-vs-Wests entry does NOT involve Maitland-vs-Norths.

        Hand-computed: candidate vars for (Maitland, Norths) don't intersect
        the (Norths, Wests) entry's matched set. Result = 0.
        """
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Norths', 'Wests'],
             'count': 1,
             'description': 'Norths-vs-Wests Friday — no Maitland'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0

    def test_given_self_pair_returns_zero(self, phl_data):
        """A==B is a degenerate input: a club doesn't play itself. Result = 0."""
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday', 'club': 'Maitland',
             'count': 2, 'constraint': 'equal',
             'description': 'Doesnt matter for self-pair'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Maitland') == 0

    def test_given_club_without_phl_teams_returns_zero(self, phl_data):
        """If either club has no PHL teams, no candidate vars exist -> 0."""
        phl_data['teams'] = [t for t in phl_data['teams'] if t.club.name != 'Norths']
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Maitland', 'Norths'],
             'count': 1,
             'description': 'Norths has no teams'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0

    def test_given_sunday_entry_returns_zero(self, phl_data):
        """Sunday FORCED for the pair -> 0 (Friday-only helper)."""
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Sunday',
             'teams': ['Maitland', 'Norths'],
             'count': 2,
             'description': 'Sunday — not Friday'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0
