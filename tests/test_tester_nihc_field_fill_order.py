"""Tests for the `_check_nihc_fill_wf_before_ef` and
`_check_nihc_fill_ef_before_sf` tester checks (spec-003).

Tester checks operate on a produced DrawStorage. No solver here -- the
checks are post-hoc statistical detectors.

Scenarios mirror the atom tests but drive the tester directly:

1. WF/EF violation in a single bucket.
2. EF/SF violation in a single bucket.
3. All fields properly filled -> no violations.
4. Field-fill order honoured across multiple slots in a day.
5. SF not a real option on the day (no SF games anywhere) -> no SF/EF
   violation flagged for an EF-only slot.
6. WF not a real option on the day -> no WF/EF violation flagged for an
   EF-only slot.
7. Non-NIHC games are not considered.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.tester import DrawTester
from analytics.storage import DrawStorage, StoredGame


BROADMEADOW = 'Newcastle International Hockey Centre'
MAITLAND = 'Maitland Park'


def _game(
    *,
    game_id: str,
    field_name: str,
    field_location: str = BROADMEADOW,
    date: str = '2026-03-22',
    day_slot: int = 1,
    week: int = 1,
    time: str = '11:30',
    t1: str = 'A',
    t2: str = 'B',
    grade: str = '3rd',
    day: str = 'Sunday',
    round_no: int = 1,
) -> StoredGame:
    return StoredGame(
        game_id=game_id,
        team1=t1,
        team2=t2,
        grade=grade,
        week=week,
        round_no=round_no,
        date=date,
        day=day,
        time=time,
        day_slot=day_slot,
        field_name=field_name,
        field_location=field_location,
    )


def _tester(*games: StoredGame) -> DrawTester:
    storage = DrawStorage(
        description='spec-003 tester scenario',
        num_weeks=1,
        num_games=len(games),
        games=list(games),
    )
    # Minimal data dict: the spec-003 checks don't touch teams / grades.
    data = {'clubs': [], 'teams': [], 'grades': []}
    return DrawTester(storage, data)


# ----------------------------------------------------------------------
# Scenario 1: WF/EF violation
# ----------------------------------------------------------------------


class TestWFBeforeEFCheck:
    def test_ef_with_no_wf_at_slot_flags_violation(self):
        """Given: WF has a game in slot 1 (proving WF is a real option
        for the date), and EF has a game in slot 2 with no WF in slot 2.
        Hand-computed expected: ONE violation flagged for slot 2 with
        constraint='NIHCFillWFBeforeEF'."""
        t = _tester(
            _game(game_id='G1', field_name='WF', day_slot=1, time='11:30'),
            _game(game_id='G2', field_name='EF', day_slot=2, time='13:00'),
        )
        violations = t._check_nihc_fill_wf_before_ef()
        assert len(violations) == 1
        v = violations[0]
        assert v.constraint == 'NIHCFillWFBeforeEF'
        assert 'G2' in v.affected_games

    def test_ef_with_wf_at_same_slot_no_violation(self):
        """Given: WF and EF both in slot 1.
        Hand-computed expected: zero violations."""
        t = _tester(
            _game(game_id='G1', field_name='WF', day_slot=1, time='11:30',
                  t1='A', t2='B'),
            _game(game_id='G2', field_name='EF', day_slot=1, time='11:30',
                  t1='C', t2='D'),
        )
        violations = t._check_nihc_fill_wf_before_ef()
        assert violations == []


# ----------------------------------------------------------------------
# Scenario 2: EF/SF violation
# ----------------------------------------------------------------------


class TestEFBeforeSFCheck:
    def test_sf_with_no_ef_at_slot_flags_violation(self):
        """Given: WF used somewhere (irrelevant), EF has a game in slot 1
        (so EF is a real option for the date), SF has a game in slot 2
        without EF in slot 2.
        Hand-computed expected: ONE violation under 'NIHCFillEFBeforeSF'."""
        t = _tester(
            _game(game_id='G1', field_name='EF', day_slot=1, time='11:30'),
            _game(game_id='G2', field_name='SF', day_slot=2, time='13:00'),
        )
        violations = t._check_nihc_fill_ef_before_sf()
        assert len(violations) == 1
        v = violations[0]
        assert v.constraint == 'NIHCFillEFBeforeSF'
        assert 'G2' in v.affected_games


# ----------------------------------------------------------------------
# Scenario 3: All three fields properly filled
# ----------------------------------------------------------------------


class TestAllThreeFilled:
    def test_perfect_order_no_violations(self):
        """Given: WF, EF, SF all carry a game at slot 1.
        Hand-computed: zero violations from either check."""
        t = _tester(
            _game(game_id='G1', field_name='WF', day_slot=1, t1='A', t2='B'),
            _game(game_id='G2', field_name='EF', day_slot=1, t1='C', t2='D'),
            _game(game_id='G3', field_name='SF', day_slot=1, t1='E', t2='F'),
        )
        assert t._check_nihc_fill_wf_before_ef() == []
        assert t._check_nihc_fill_ef_before_sf() == []


# ----------------------------------------------------------------------
# Scenario 4: Multiple slots in a day, all good
# ----------------------------------------------------------------------


class TestMultipleSlotsAllGood:
    def test_three_slots_all_filled_correctly(self):
        """Given: slot 1 has WF+EF, slot 2 has WF only, slot 3 has WF+EF+SF.
        Hand-computed: WF before EF holds for slot 1, vacuously for slot 2,
        holds for slot 3. EF before SF holds for slot 1 (no SF), vacuously
        for slot 2 (no SF), holds for slot 3. Zero violations."""
        t = _tester(
            _game(game_id='S1WF', field_name='WF', day_slot=1, t1='A', t2='B'),
            _game(game_id='S1EF', field_name='EF', day_slot=1, t1='C', t2='D'),
            _game(game_id='S2WF', field_name='WF', day_slot=2, t1='E', t2='F'),
            _game(game_id='S3WF', field_name='WF', day_slot=3, t1='G', t2='H'),
            _game(game_id='S3EF', field_name='EF', day_slot=3, t1='I', t2='J'),
            _game(game_id='S3SF', field_name='SF', day_slot=3, t1='K', t2='L'),
        )
        assert t._check_nihc_fill_wf_before_ef() == []
        assert t._check_nihc_fill_ef_before_sf() == []


# ----------------------------------------------------------------------
# Scenario 5: SF not a real option for the date (no SF games anywhere)
# ----------------------------------------------------------------------


class TestSFNotAValidOption:
    def test_no_sf_anywhere_means_no_sf_check_violations(self):
        """Given: a date with WF and EF games but NO SF games anywhere.
        That date never had SF as an option. An EF-only slot must not
        produce an `NIHCFillEFBeforeSF` violation (the check has nothing
        to assert)."""
        t = _tester(
            _game(game_id='G1', field_name='WF', day_slot=1, t1='A', t2='B'),
            _game(game_id='G2', field_name='EF', day_slot=1, t1='C', t2='D'),
            _game(game_id='G3', field_name='EF', day_slot=2, t1='E', t2='F'),
        )
        # WF/EF check: slot 1 fine; slot 2 has EF but no WF -- WF was an
        # option on the date (slot 1 had WF), so slot 2 IS a violation.
        # Hand-computed expected: 1 WF/EF violation, 0 EF/SF violations.
        wf_viols = t._check_nihc_fill_wf_before_ef()
        sf_viols = t._check_nihc_fill_ef_before_sf()
        assert len(wf_viols) == 1
        assert wf_viols[0].constraint == 'NIHCFillWFBeforeEF'
        assert sf_viols == []


# ----------------------------------------------------------------------
# Scenario 6: WF not a real option for the date
# ----------------------------------------------------------------------


class TestWFNotAValidOption:
    def test_no_wf_anywhere_means_no_wf_check_violations(self):
        """Given: a date with EF games only -- no WF in the entire date.
        Hand-computed: zero WF/EF violations (WF wasn't an option)."""
        t = _tester(
            _game(game_id='G1', field_name='EF', day_slot=1, t1='A', t2='B'),
            _game(game_id='G2', field_name='EF', day_slot=2, t1='C', t2='D'),
        )
        assert t._check_nihc_fill_wf_before_ef() == []
        assert t._check_nihc_fill_ef_before_sf() == []


# ----------------------------------------------------------------------
# Scenario 7: Non-NIHC games are ignored
# ----------------------------------------------------------------------


class TestNonNIHCIgnored:
    def test_maitland_park_games_dont_drive_check(self):
        """Given: a Maitland Park EF-named game (impossible in reality
        but useful to prove the location filter); no NIHC games.
        Hand-computed: zero violations because only NIHC games feed
        the bucket map."""
        t = _tester(
            _game(game_id='G1', field_name='Maitland Main Field',
                  field_location=MAITLAND, day_slot=1, t1='M', t2='N'),
        )
        assert t._check_nihc_fill_wf_before_ef() == []
        assert t._check_nihc_fill_ef_before_sf() == []


# ----------------------------------------------------------------------
# Scenario 8: Cache shared between WF/EF and EF/SF checks
# ----------------------------------------------------------------------


class TestCacheBetweenChecks:
    def test_running_both_checks_does_not_double_count(self):
        """Given: one slot triggers a WF/EF violation; one slot triggers
        an EF/SF violation. All other slots have WF present so they
        don't also trigger a WF/EF violation.
        Hand-computed: 1 + 1 = 2 violations total, each tagged distinctly.
        The internal `_nihc_field_usage` cache is shared between both
        checks; this test confirms no cross-pollination."""
        t = _tester(
            # WF at slot 1, EF at slot 2 -> WF/EF violation in slot 2.
            _game(game_id='G1', field_name='WF', day_slot=1, t1='A', t2='B'),
            _game(game_id='G2', field_name='EF', day_slot=2, t1='C', t2='D'),
            # Slot 3 has WF + EF (WF/EF clean), plus SF (EF/SF clean).
            _game(game_id='G3WF', field_name='WF', day_slot=3, t1='E', t2='F'),
            _game(game_id='G3EF', field_name='EF', day_slot=3, t1='G', t2='H'),
            # Slot 4: WF + SF without EF -> EF/SF violation; WF/EF clean
            # because EF isn't on the LHS of the WF/EF implication.
            _game(game_id='G4WF', field_name='WF', day_slot=4, t1='I', t2='J'),
            _game(game_id='G4SF', field_name='SF', day_slot=4, t1='K', t2='L'),
        )
        wf_viols = t._check_nihc_fill_wf_before_ef()
        sf_viols = t._check_nihc_fill_ef_before_sf()
        assert len(wf_viols) == 1
        assert wf_viols[0].constraint == 'NIHCFillWFBeforeEF'
        assert 'G2' in wf_viols[0].affected_games
        assert len(sf_viols) == 1
        assert sf_viols[0].constraint == 'NIHCFillEFBeforeSF'
        assert 'G4SF' in sf_viols[0].affected_games
