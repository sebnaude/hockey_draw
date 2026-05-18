"""Tests for `constraints.atoms._phl_forced_friday_helper`.

All scenarios use REAL data shapes (no mocks). Hand-computed oracles in the
docstring of each test; the test asserts the helper output matches them.

Critical case (from spec-004 Clarification): one FORCED entry
`count==2 sum of Maitland Fridays` + one FORCED entry
`count==1 Maitland-vs-Tigers Friday` must yield
`phl_forced_friday_count('Maitland') == 2` — NOT 3.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms._phl_forced_friday_helper import (
    away_club_required_sundays,
    away_club_total_weekends,
    phl_forced_friday_count,
    phl_forced_friday_meetings,
)


# Reuse the small PHL fixture from conftest. It has Friday slots at BM/Maitland/
# Gosford for weeks 1-5 and 5 clubs (Tigers, Wests, Norths, Maitland, Gosford).


# ---------------------------------------------------------------------------
# phl_forced_friday_count
# ---------------------------------------------------------------------------


class TestPhlForcedFridayCount:
    """Counts PHL Friday games this club WILL play, FORCED-aware."""

    def test_given_no_forced_games_when_counting_then_returns_zero(self, phl_data):
        """Given no FORCED_GAMES entries, When counting, Then count == 0."""
        phl_data['forced_games'] = []
        assert phl_forced_friday_count(phl_data, 'Maitland') == 0
        assert phl_forced_friday_count(phl_data, 'Gosford') == 0

    def test_given_unrelated_forced_games_when_counting_then_returns_zero(
        self, phl_data
    ):
        """Given a Sunday FORCED entry, When counting Friday Maitland, Then 0."""
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Sunday', 'club': 'Maitland', 'count': 1,
             'description': 'Sunday Maitland — not Friday'},
        ]
        assert phl_forced_friday_count(phl_data, 'Maitland') == 0

    def test_given_single_umbrella_friday_count_two_when_counting_then_two(
        self, phl_data
    ):
        """Given `count=2 equal` Maitland Park Friday PHL umbrella,
        When counting Maitland, Then == 2.

        Oracle: the convenor pinned exactly 2 Maitland Park Fridays. No
        per-pair overlay → contribution = 2.
        """
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'field_location': 'Maitland Park',
             'count': 2, 'constraint': 'equal',
             'description': 'Exactly 2 PHL Fridays at Maitland Park'},
        ]
        assert phl_forced_friday_count(phl_data, 'Maitland') == 2

    def test_given_umbrella_count_two_plus_subset_per_pair_when_counting_then_two(
        self, phl_data
    ):
        """Critical case from spec-004 Clarification.

        Given a `count==2 equal` umbrella for Maitland Fridays AND a `count==1`
        per-pair Maitland-vs-Tigers Friday (which is a STRICT SUBSET of the
        umbrella), When counting Maitland Friday games, Then == 2 (not 3).

        The per-pair entry forces ONE specific variable to be played; that
        variable is one of the umbrella's two. Total Maitland Fridays = 2.
        """
        phl_data['forced_games'] = [
            # Umbrella: any Maitland Friday PHL game, exactly 2.
            {'grade': 'PHL', 'day': 'Friday', 'club': 'Maitland',
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland total Friday PHL count == 2'},
            # Per-pair: Maitland-vs-Tigers Friday PHL, exactly 1 (default).
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Maitland', 'Tigers'],
             'description': 'Maitland-vs-Tigers Friday — one of the two'},
        ]
        # Hand-computed: umbrella(2) covers per-pair(1) → 2.
        assert phl_forced_friday_count(phl_data, 'Maitland') == 2

    def test_given_two_disjoint_venue_scopes_when_counting_then_sum(self, phl_data):
        """Disjoint scopes (different venues) → sum of counts.

        Given Maitland: `Maitland Park count=2 equal` AND
        `NIHC count=1 (per-pair Maitland-vs-Souths)`, When counting,
        Then == 3 (2 at Maitland Park + 1 at NIHC — different venues, disjoint).
        """
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'field_location': 'Maitland Park',
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland Park PHL Fridays = 2'},
            {'grade': 'PHL', 'day': 'Friday',
             'field_location': 'Newcastle International Hockey Centre',
             'teams': ['Maitland', 'Norths'],
             'description': 'Maitland vs Norths NIHC Friday = 1 (default)'},
        ]
        # Hand-computed: candidate sets are disjoint by venue → 2 + 1 = 3.
        assert phl_forced_friday_count(phl_data, 'Maitland') == 3

    def test_given_entries_for_other_club_when_counting_then_excluded(
        self, phl_data
    ):
        """Per-pair entry naming OTHER clubs should not count toward this club.

        Given a `Norths-vs-Wests Friday` per-pair entry, When counting Maitland,
        Then == 0 (no Maitland involvement).
        """
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Norths', 'Wests'],
             'description': 'Norths-vs-Wests Friday — Maitland not involved'},
        ]
        assert phl_forced_friday_count(phl_data, 'Maitland') == 0
        # Same entry — Norths IS involved.
        assert phl_forced_friday_count(phl_data, 'Norths') == 1

    def test_given_no_phl_teams_when_counting_then_zero(self, phl_data):
        """Club not in PHL → count == 0 (no candidate vars exist)."""
        # Remove all Maitland teams to simulate "club doesn't field PHL".
        phl_data['teams'] = [t for t in phl_data['teams'] if t.club.name != 'Maitland']
        phl_data['games'] = [
            (t1, t2, g) for (t1, t2, g) in phl_data['games']
            if 'Maitland' not in t1 and 'Maitland' not in t2
        ]
        # An umbrella entry exists but no matching vars.
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday', 'club': 'Maitland', 'count': 2,
             'description': 'Maitland Fridays — but no Maitland team'},
        ]
        assert phl_forced_friday_count(phl_data, 'Maitland') == 0


# ---------------------------------------------------------------------------
# away_club_total_weekends
# ---------------------------------------------------------------------------


class TestAwayClubTotalWeekends:
    """Original max across grades, ignoring FORCED Fridays."""

    def test_given_phl_and_2nd_each_five_when_querying_maitland_then_five(
        self, phl_data
    ):
        """Fixture: num_rounds == {PHL: 5, 2nd: 5} → max = 5."""
        assert away_club_total_weekends(phl_data, 'Maitland') == 5

    def test_given_no_teams_when_querying_then_zero(self, phl_data):
        """Unknown club → 0."""
        assert away_club_total_weekends(phl_data, 'NoSuchClub') == 0


# ---------------------------------------------------------------------------
# away_club_required_sundays
# ---------------------------------------------------------------------------


class TestAwayClubRequiredSundays:
    """Sundays needed AFTER subtracting FORCED Fridays."""

    def test_given_no_forced_fridays_when_querying_then_total_weekends(
        self, phl_data
    ):
        """Given no FORCED Fridays, When querying Maitland (PHL=2nd=5),
        Then sundays == max(5-0, 5) == 5."""
        phl_data['forced_games'] = []
        assert away_club_required_sundays(phl_data, 'Maitland') == 5

    def test_given_two_forced_fridays_when_querying_then_reduced_by_two(
        self, phl_data
    ):
        """Given 2 FORCED Maitland Fridays, PHL=2nd=5, When querying Maitland,
        Then sundays == max(5-2, 5) == 5 (driven by 2nd grade — Friday
        absorbed into the same weekend count).

        This is the spec-004 edge case: when another grade requires >= PHL,
        FORCED Fridays don't reduce total Sundays (they're absorbed).
        """
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'field_location': 'Maitland Park',
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland Park Fridays = 2'},
        ]
        # Hand-computed: PHL=5, forced_fridays=2, max_other=5 → max(3, 5) = 5.
        assert away_club_required_sundays(phl_data, 'Maitland') == 5

    def test_given_phl_only_with_forced_fridays_when_querying_then_reduced(
        self, phl_data
    ):
        """When only PHL exists (no other grades), Sundays reduce by Friday count.

        Setup: drop all non-PHL teams so Maitland only fields PHL.
        With 2 FORCED Fridays, Sundays == max(5-2, 0) = 3.
        """
        # Strip all non-PHL Maitland teams.
        phl_data['teams'] = [
            t for t in phl_data['teams']
            if not (t.club.name == 'Maitland' and t.grade != 'PHL')
        ]
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'field_location': 'Maitland Park',
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland Park Fridays = 2'},
        ]
        # Hand-computed: grades = {'PHL'}; PHL=5; forced=2 → max(3, 0) = 3.
        assert away_club_required_sundays(phl_data, 'Maitland') == 3

    def test_given_phl18_other20_when_querying_then_twenty(self, phl_data):
        """Spec edge case verbatim: PHL=18, 3rd=20.

        Add a 3rd-grade team to Maitland, set num_rounds so PHL=18 / 3rd=20.
        Given 2 FORCED Fridays, When querying, Then sundays == 20 (driven by
        3rd grade — FORCED Fridays absorbed into the same 20 weekends).
        """
        from models import Team

        maitland_club = next(c for c in phl_data['clubs'] if c.name == 'Maitland')
        phl_data['teams'] = list(phl_data['teams']) + [
            Team(name='Maitland 3rd', club=maitland_club, grade='3rd'),
        ]
        phl_data['num_rounds'] = {'PHL': 18, '2nd': 18, '3rd': 20, 'max': 20}
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'field_location': 'Maitland Park',
             'count': 2, 'constraint': 'equal',
             'description': 'Maitland Park Fridays = 2'},
        ]
        # Hand-computed: PHL=18, fri=2, other_max=20 → max(18-2, 20) = 20.
        # And total_weekends = max(18, 20) = 20.
        assert away_club_required_sundays(phl_data, 'Maitland') == 20
        assert away_club_total_weekends(phl_data, 'Maitland') == 20

    def test_given_unknown_club_when_querying_then_zero(self, phl_data):
        assert away_club_required_sundays(phl_data, 'NoSuchClub') == 0


# ---------------------------------------------------------------------------
# phl_forced_friday_meetings — spec-005 per-pair Friday count
# ---------------------------------------------------------------------------


class TestPhlForcedFridayMeetings:
    """Per-pair Friday count for `ClubVsClubStackedPHLSundayBudget` (spec-005).

    Same FORCED-aware greedy partition as `phl_forced_friday_count`, but
    narrowed to candidate vars between TWO specific clubs. Verifies:
      - Umbrella (count=2) for one club + per-pair (count=1) for that pair
        → 1 (the per-pair entry pins one of the umbrella's two; total
        Maitland Fridays = 2 but only 1 is vs the specific opponent).
      - Per-pair-only entry → its count.
      - Entry naming other clubs → 0.
      - Self-pair (A == B) → 0.
    """

    def test_given_no_forced_games_returns_zero(self, phl_data):
        """No FORCED → 0."""
        phl_data['forced_games'] = []
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0

    def test_given_pair_explicit_when_counting_returns_entry_count(self, phl_data):
        """One per-pair Maitland-vs-Norths Friday entry, count=2 → 2."""
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

        This is the per-PAIR helper. The per-CLUB helper
        `phl_forced_friday_count(Maitland) == 2` — they answer different
        questions. See `_entry_targets_pair_phl_friday` docstring.
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
        """If either club has no PHL teams, no candidate vars exist → 0."""
        phl_data['teams'] = [t for t in phl_data['teams'] if t.club.name != 'Norths']
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Friday',
             'teams': ['Maitland', 'Norths'],
             'count': 1,
             'description': 'Norths has no teams'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0

    def test_given_sunday_entry_returns_zero(self, phl_data):
        """Sunday FORCED for the pair → 0 (Friday-only helper)."""
        phl_data['forced_games'] = [
            {'grade': 'PHL', 'day': 'Sunday',
             'teams': ['Maitland', 'Norths'],
             'count': 2,
             'description': 'Sunday — not Friday'},
        ]
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 0
