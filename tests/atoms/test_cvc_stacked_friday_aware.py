# spec-019: regression lock for ClubVsClubStackedWeekends per-pair Friday-awareness.
"""spec-019 — pin `ClubVsClubStackedWeekends`'s per-pair FORCED-Friday budget.

The convenor's question: if a club's PHL and 2nd both meet another club 3× each,
and ONE of those PHL meetings is FORCED onto a Friday, only 2 weekends should
require PHL/2nd alignment (the 3rd has only 2nd playing). This is already
implemented correctly and per-pair (`pair_grade_sunday_meetings` subtracts
`phl_forced_friday_meetings(a, b)` for PHL only). This file is a VERIFY-ONLY
regression guard — no production change — so a future refactor can't silently
revert to an aggregate Friday count.

The end-to-end SOLVER scenario already lives in
`tests/atoms/test_club_vs_club_stacked_alignment.py::TestScenarioTwoForcedFridayBudget`
(spec-005). This file adds (a) the convenor's exact budget numbers at the
helper level and (b) the A-SHARED cross-pair isolation gap that no existing
test covered: a FORCED Friday for pair (A, B) must NOT reduce pair (A, C)'s
budget when club A is shared.

Real data shapes, no mocks. Hand-computed oracles in each docstring.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms._club_vs_club_stacked_shared import (
    pair_grade_sunday_meetings,
    per_pair_grade_meeting_counts,
)
from constraints.atoms._phl_forced_friday_helper import phl_forced_friday_meetings
from constraints.atoms.base import MAITLAND

from tests.atoms.club_vs_club_stacked_fixture import build_stacked_fixture


# ---------------------------------------------------------------------------
# DoD 1 — the convenor's scenario, at the helper level (the 2 / 3 numbers)
# ---------------------------------------------------------------------------


class TestConvenorScenarioBudget:
    """PHL meets 3×, 2nd meets 3×, ONE PHL meeting FORCED to Friday →
    PHL Sunday budget 2, 2nd Sunday budget 3."""

    def test_one_forced_friday_reduces_only_phl_budget(self):
        """Given (Maitland, Norths) with PHL=3 and 2nd=3 meetings, and ONE
        pair-specific FORCED PHL Friday (`teams=['Maitland PHL','Norths PHL']`).
        Then phl_forced_friday_meetings == 1; PHL Sunday budget == 2 (3 − 1);
        2nd Sunday budget == 3 (unchanged — 2nd never plays Friday).

        Hand oracle: total PHL = 3 (T=2, R=3 → 3//1=3, 1 matchup). 1 forced
        Friday → 3 − 1 = 2. 2nd is not PHL so no subtraction → 3."""
        data = build_stacked_fixture({'PHL': 3, '2nd': 3})
        data['forced_games'] = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland PHL', 'Norths PHL'],
            'field_location': MAITLAND, 'count': 1, 'constraint': 'equal',
            'description': 'One Maitland-vs-Norths PHL Friday',
        }]
        pair = ('Maitland', 'Norths')

        # Sanity: the fixture really gives 3 PHL and 3 2nd meetings for the pair.
        counts = per_pair_grade_meeting_counts(data, pair)
        assert counts['PHL'] == 3
        assert counts['2nd'] == 3

        assert phl_forced_friday_meetings(data, *pair) == 1
        assert pair_grade_sunday_meetings(data, pair, 'PHL') == 2   # 3 − 1
        assert pair_grade_sunday_meetings(data, pair, '2nd') == 3   # unchanged

    def test_no_forced_friday_leaves_full_budget(self):
        """Control: with NO FORCED Friday, the PHL budget stays at the full 3."""
        data = build_stacked_fixture({'PHL': 3, '2nd': 3})
        data['forced_games'] = []
        pair = ('Maitland', 'Norths')
        assert phl_forced_friday_meetings(data, *pair) == 0
        assert pair_grade_sunday_meetings(data, pair, 'PHL') == 3
        assert pair_grade_sunday_meetings(data, pair, '2nd') == 3


# ---------------------------------------------------------------------------
# DoD 2 — A-SHARED cross-pair isolation (the regression gap)
# ---------------------------------------------------------------------------


class TestASharedPairIsolation:
    """A FORCED Friday for pair (A, B) must NOT reduce pair (A, C)'s budget,
    where club A is SHARED across the two pairs.

    The existing `test_phl_forced_friday_helper.py::...::test_given_other_pair_
    entry_returns_zero` only covers DISJOINT clubs (Norths-vs-Wests vs
    Maitland-vs-Norths). This is the A-shared variant.
    """

    def test_forced_friday_for_ab_does_not_touch_ac(self, phl_data):
        """Given 3 clubs (Maitland, Norths, Wests) all fielding PHL, and ONE
        FORCED Maitland-vs-Norths PHL Friday. Then:
          - phl_forced_friday_meetings(Maitland, Norths) == 1
          - phl_forced_friday_meetings(Maitland, Wests)  == 0   ← the gap
          - PHL Sunday budget (Maitland, Norths) == total − 1
          - PHL Sunday budget (Maitland, Wests)  == total       (unchanged)

        Hand oracle: phl_data PHL grade has 5 teams (T=5, odd). With
        num_rounds[PHL]=10 each pair has per_matchup = 10 // 5 = 2 meetings.
        The FORCED Maitland-vs-Norths Friday (count 1) reduces ONLY that pair:
        (Maitland,Norths) PHL budget = max(0, 2−1) = 1; (Maitland,Wests)
        budget = 2 (no FORCED Friday names that pair)."""
        # Override num_rounds so per-pair PHL meetings == 2 (clean reduction).
        phl_data['num_rounds'] = {'PHL': 10, '2nd': 10, 'max': 10}
        phl_data['forced_games'] = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland', 'Norths'],     # club names — pair-specific
            'count': 1, 'constraint': 'equal',
            'description': 'One Maitland-vs-Norths PHL Friday',
        }]

        # Both pairs start with 2 PHL meetings (hand oracle above).
        assert per_pair_grade_meeting_counts(phl_data, ('Maitland', 'Norths'))['PHL'] == 2
        assert per_pair_grade_meeting_counts(phl_data, ('Maitland', 'Wests'))['PHL'] == 2

        # The FORCED Friday touches ONLY the (Maitland, Norths) pair.
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Norths') == 1
        assert phl_forced_friday_meetings(phl_data, 'Maitland', 'Wests') == 0  # gap

        # Budget reduced for the named pair, untouched for the A-shared pair.
        assert pair_grade_sunday_meetings(phl_data, ('Maitland', 'Norths'), 'PHL') == 1
        assert pair_grade_sunday_meetings(phl_data, ('Maitland', 'Wests'), 'PHL') == 2
