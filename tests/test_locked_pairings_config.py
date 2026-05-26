"""
spec-025 Unit A — LOCKED_PAIRINGS config, default, injection, validation.

Given/When/Then style, no mocks, hand-computed oracles. Reuses the real
model objects and the test_config_validation fixtures idiom.
"""

import pytest
from models import Team, Club, PlayingField, Timeslot
from utils import build_season_data, validate_game_config


# ============== Fixtures (mirroring test_config_validation.py) ==============

# All clubs home at NIHC so capacity/home-field phases stay clean — keeps the
# validation body free of unrelated FATALs so we can isolate LOCKED_PAIRINGS.
@pytest.fixture
def clubs():
    return {
        'Norths': Club(name='Norths', home_field='Newcastle International Hockey Centre'),
        'Souths': Club(name='Souths', home_field='Newcastle International Hockey Centre'),
        'Wests': Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        'Easts': Club(name='Easts', home_field='Newcastle International Hockey Centre'),
    }


@pytest.fixture
def teams(clubs):
    return [
        Team(name='Norths PHL', club=clubs['Norths'], grade='PHL'),
        Team(name='Norths 2nd', club=clubs['Norths'], grade='2nd'),
        Team(name='Souths PHL', club=clubs['Souths'], grade='PHL'),
        Team(name='Souths 2nd', club=clubs['Souths'], grade='2nd'),
        Team(name='Wests PHL', club=clubs['Wests'], grade='PHL'),
        Team(name='Easts PHL', club=clubs['Easts'], grade='PHL'),
    ]


@pytest.fixture
def fields():
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
    ]


@pytest.fixture
def timeslots(fields):
    ef, wf = fields[0], fields[1]
    slots = []
    dates = [
        ('2026-03-22', 1, 1),
        ('2026-03-29', 2, 2),
        ('2026-04-05', 3, 3),
        ('2026-04-12', 4, 4),
    ]
    for date, week, round_no in dates:
        for slot_idx, (time, field) in enumerate([
            ('10:00', ef), ('11:30', ef), ('13:00', wf), ('14:30', wf),
        ], start=1):
            slots.append(Timeslot(
                date=date, day='Sunday', time=time, week=week,
                day_slot=slot_idx, field=field, round_no=round_no,
            ))
    return slots


@pytest.fixture
def base_data(teams, timeslots, fields):
    """Minimal data dict — empty forced/blocked/preferred/club_days etc."""
    return {
        'teams': teams,
        'timeslots': timeslots,
        'fields': fields,
        'num_rounds': {'PHL': 3, '2nd': 3},
        'phl_game_times': {},
        'second_grade_times': {},
        'home_field_map': {},
        'forced_games': [],
        'blocked_games': [],
        'preferred_games': [],
        'club_days': {},
        'constraint_defaults': {},
        'grade_rounds_override': {},
        'locked_pairings': [],
    }


# ============== Round-trip (DoD 2) ==============

class TestRoundTrip:
    def test_locked_pairings_round_trips_through_build_season_data(self):
        """GIVEN a season config with a LOCKED_PAIRINGS entry,
        WHEN build_season_data runs (the load_season_data path),
        THEN data['locked_pairings'] contains that entry verbatim."""
        import config.season_2026 as s
        cfg = dict(s.SEASON_CONFIG)
        pin = {'teams': ['Norths', 'Souths'], 'grade': 'PHL',
               'date': '2026-03-22', 'description': 'pin'}
        cfg['locked_pairings'] = [pin]
        data = build_season_data(cfg)
        assert data['locked_pairings'] == [pin]

    def test_default_locked_pairings_unconfigured_is_empty(self):
        """An UNCONFIGURED season config exposes an empty locked_pairings list.

        (spec-025 Unit E populated season_2026's LOCKED_PAIRINGS with the 246
        migrated pins, so 2026 is no longer empty — assert the injection default
        on a config that supplies no pins instead.)"""
        import config.season_2026 as s
        cfg = dict(s.SEASON_CONFIG)
        cfg['locked_pairings'] = []
        data = build_season_data(cfg)
        assert data['locked_pairings'] == []

    def test_season_2026_locked_pairings_intentionally_empty(self):
        """final-form (commit 5761e88) intentionally emptied season_2026's
        LOCKED_PAIRINGS — the convenor edits that file directly for manual
        draw-generation testing. The live config therefore carries ZERO pins.
        The original 246-pin migration artefact is preserved (and its parity is
        still validated) in tests/fixtures/locked_pairings_premigration_2026.json
        — see tests/test_locked_pairings_migration_parity.py."""
        import json
        import os
        from config import load_season_data
        data = load_season_data(2026)
        # Live config: intentionally empty.
        assert data['locked_pairings'] == []
        # Frozen artefact: still the migrated 246 pins (parity coverage preserved).
        fixture = os.path.join(os.path.dirname(__file__), 'fixtures',
                               'locked_pairings_premigration_2026.json')
        with open(fixture) as f:
            assert len(json.load(f)['locked_pairings']) == 246


# ============== Forbidden-field FATAL (DoD 1, 6) ==============

class TestForbiddenFieldFatal:
    def test_pin_with_time_is_fatal_naming_field(self, base_data, capsys):
        """A pin carrying a forbidden 'time' key → FATAL naming the field."""
        base_data['locked_pairings'] = [
            {'teams': ['Norths', 'Souths'], 'grade': 'PHL',
             'date': '2026-03-22', 'time': '11:30', 'description': 'bad pin'},
        ]
        with pytest.raises(SystemExit):
            validate_game_config(base_data)
        out = capsys.readouterr().out
        assert 'LOCKED_PAIRINGS' in out
        assert 'time' in out
        assert 'forbidden' in out.lower()

    def test_pin_with_field_name_is_fatal(self, base_data, capsys):
        base_data['locked_pairings'] = [
            {'teams': ['Norths', 'Souths'], 'grade': 'PHL',
             'date': '2026-03-22', 'field_name': 'EF'},
        ]
        with pytest.raises(SystemExit):
            validate_game_config(base_data)
        out = capsys.readouterr().out
        assert 'field_name' in out


# ============== Unknown team FATAL (DoD 6) ==============

class TestUnknownTeamFatal:
    def test_pin_with_unknown_team_is_fatal(self, base_data, capsys):
        """A pin with an unresolvable team → FATAL (same as FORCED)."""
        base_data['locked_pairings'] = [
            {'teams': ['Nonexistent', 'Souths'], 'grade': 'PHL',
             'date': '2026-03-22'},
        ]
        with pytest.raises(SystemExit):
            validate_game_config(base_data)
        out = capsys.readouterr().out
        assert 'LOCKED_PAIRINGS' in out
        assert 'Nonexistent' in out


# ============== Locked-week filter (DoD 5) ==============

class TestLockedWeekFilter:
    def test_pin_in_locked_week_is_filtered_out(self, base_data):
        """GIVEN weeks 1-3 locked and a pin dated in week 2 (2026-03-29),
        WHEN validate_game_config runs,
        THEN no FATAL, and data['locked_pairings'] is emptied by the filter."""
        base_data['locked_weeks'] = {1, 2, 3}
        base_data['locked_pairings'] = [
            {'teams': ['Norths', 'Souths'], 'grade': 'PHL',
             'date': '2026-03-29', 'description': 'week-2 pin'},
        ]
        validate_game_config(base_data)  # must not raise
        assert base_data['locked_pairings'] == []


# ============== Guard update (DoD 6, M4) ==============

class TestGuardUpdate:
    def test_only_locked_pairings_still_enters_validation_body(self, base_data, capsys):
        """GIVEN every other config list empty but a non-empty locked_pairings,
        WHEN validate_game_config runs,
        THEN it enters the validation body (prints CONFIG VALIDATION header),
        proving the has_config_to_validate guard includes locked_pairings."""
        # Strip all other config so ONLY locked_pairings can trip the guard.
        base_data['home_field_map'] = {}
        base_data['phl_game_times'] = {}
        base_data['constraint_defaults'] = {}
        base_data['grade_rounds_override'] = {}
        base_data['locked_pairings'] = [
            {'teams': ['Norths', 'Souths'], 'grade': 'PHL',
             'date': '2026-03-22', 'description': 'sole config'},
        ]
        validate_game_config(base_data)  # valid pin → no raise
        out = capsys.readouterr().out
        assert 'CONFIG VALIDATION' in out
