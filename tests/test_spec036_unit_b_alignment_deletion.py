"""spec-036 Unit B — proof that deleting the legacy `_club_alignment_*` engine
path leaves club-vs-club alignment fully covered by the spec-005 stacked atoms.

No mocks, no patches: every assertion runs against the real registry, the real
solver-stage loader, a real CP-SAT build of the unified engine, and the real
DrawTester over a real fixture draw.

What this proves (DoD 11):
  (a) club-vs-club alignment is STILL enforced after deletion:
        - the production solver stages dispatch `ClubVsClubStackedWeekends` and
          `ClubVsClubStackedCoLocation` (and NOT the legacy `ClubVsClubAlignment`
          engine key), and
        - a deliberately mis-aligned draw is rejected by the tester's
          `ClubVsClubAlignment` violation check (the retained tester anchor).
  (b) the deleted engine methods no longer exist.
  (c) the engine still builds clean (no AttributeError from the removed
      groupings) through build_groupings + apply_all.
  (d) `len(CONSTRAINT_REGISTRY)` is unchanged and the `ClubVsClubAlignment`
      registry entry is still present (tester/name anchor).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.registry import CONSTRAINT_REGISTRY
from constraints.stages import (
    ENGINE_HARD_KEYS,
    ENGINE_SOFT_KEYS,
    load_solver_stages,
)
from constraints.unified import UnifiedConstraintEngine, BROADMEADOW, MAITLAND
from config import load_season_data
from models import PlayingField, Team, Club, Grade, Timeslot
from tests.conftest import create_model_and_vars


# spec-036 Unit B captured this count BEFORE the deletion (the `ClubVsClubAlignment`
# registry entry is retained, so it must NOT change).
REGISTRY_LEN_BEFORE = 49


def _mini_engine_data():
    """A small but realistic 4-club / 2-grade season that exercises the
    lower-grade (3rd/4th) path the deleted alignment methods used to walk."""
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    mf = PlayingField(location=MAITLAND, name='Maitland Main Field')
    clubs = [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
    ]
    teams = []
    for club in clubs:
        for grade in ['3rd', '4th']:
            teams.append(Team(
                name=f'{club.name} 3rd' if grade == '3rd' else f'{club.name} 4th',
                club=club, grade=grade,
            ))
    grades = [
        Grade(name='3rd', teams=[t.name for t in teams if t.grade == '3rd']),
        Grade(name='4th', teams=[t.name for t in teams if t.grade == '4th']),
    ]
    timeslots = []
    base_date = datetime(2025, 3, 23)
    for week in range(1, 5):
        date_str = (base_date + timedelta(weeks=week - 1)).strftime('%Y-%m-%d')
        for field in [ef, wf]:
            for slot, time in enumerate(['10:00', '11:30', '13:00'], 1):
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time=time, week=week,
                    day_slot=slot, field=field, round_no=week,
                ))
        for slot, time in enumerate(['10:00', '11:30'], 1):
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time=time, week=week,
                day_slot=slot, field=mf, round_no=week,
            ))
    games = []
    for grade_obj in grades:
        for t1, t2 in combinations(grade_obj.teams, 2):
            games.append((t1, t2, grade_obj.name))
    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': [ef, wf, mf],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'3rd': 4, '4th': 4, 'max': 4},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
    }


# ---------------------------------------------------------------------------
# (b) the deleted engine methods no longer exist
# ---------------------------------------------------------------------------

def test_legacy_alignment_methods_removed():
    assert not hasattr(UnifiedConstraintEngine, '_club_alignment_hard')
    assert not hasattr(UnifiedConstraintEngine, '_club_alignment_soft')


def test_legacy_alignment_groupings_removed_from_engine_key_sets():
    # The engine key was the only thing wiring the deleted methods into the
    # stage dispatcher. EqualMatchUpSpacing (the other key on the same line)
    # must stay.
    assert 'ClubVsClubAlignment' not in ENGINE_HARD_KEYS
    assert 'ClubVsClubAlignment' not in ENGINE_SOFT_KEYS
    assert 'EqualMatchUpSpacing' in ENGINE_HARD_KEYS
    assert 'EqualMatchUpSpacing' in ENGINE_SOFT_KEYS


# ---------------------------------------------------------------------------
# (c) the engine still builds clean (no AttributeError from removed groupings)
# ---------------------------------------------------------------------------

def test_engine_builds_clean_after_deletion():
    data = _mini_engine_data()
    model, X = create_model_and_vars(data['games'], data['timeslots'])
    data['penalties'] = {}
    engine = UnifiedConstraintEngine(model, X, data)
    # The removed groupings must NOT be attributes anymore...
    engine.build_groupings()
    assert not hasattr(engine, 'by_grade_clubpair_round')
    assert not hasattr(engine, 'by_sunday_clubpair_round_field')
    # ...and a full apply (hard + soft + objective groundwork) must not raise.
    engine.apply_all()
    # The legacy alignment penalty buckets are gone (no engine registers them).
    assert 'ClubVsClubAlignment' not in data['penalties']
    assert 'ClubVsClubAlignmentField' not in data['penalties']


def test_hard_stage_registers_no_penalty_buckets():
    data = _mini_engine_data()
    model, X = create_model_and_vars(data['games'], data['timeslots'])
    data['penalties'] = {}
    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()
    engine.apply_stage_1_hard()
    assert set(data['penalties']) == set()


# ---------------------------------------------------------------------------
# (d) registry count unchanged + ClubVsClubAlignment entry retained
# ---------------------------------------------------------------------------

def test_registry_count_unchanged_and_anchor_retained():
    assert len(CONSTRAINT_REGISTRY) == REGISTRY_LEN_BEFORE
    assert 'ClubVsClubAlignment' in CONSTRAINT_REGISTRY


# ---------------------------------------------------------------------------
# (a) alignment is STILL enforced — stacked atoms dispatched + tester rejects
# ---------------------------------------------------------------------------

def test_stacked_atoms_dispatched_and_legacy_key_not():
    """The production solver stages enforce alignment via the spec-005 stacked
    atoms; the legacy `ClubVsClubAlignment` engine key is dispatched nowhere."""
    data = load_season_data(2026)
    stages = load_solver_stages(data)
    applied = set()
    for st in stages:
        for atom in st.get('atoms', []):
            applied.add(atom if isinstance(atom, str)
                        else getattr(atom, '__name__', str(atom)))
    assert 'ClubVsClubStackedWeekends' in applied
    assert 'ClubVsClubStackedCoLocation' in applied
    # The deleted legacy engine path is not dispatched as an atom.
    assert 'ClubVsClubAlignment' not in applied


def test_misaligned_draw_rejected_by_tester_anchor():
    """A deliberately mis-aligned (non-coincident club-pair) draw is rejected
    by the retained `ClubVsClubAlignment` tester check — proving the anchor
    still flags the violation the deleted engine method used to push against."""
    # Reuse the real fixture + plumbing the violation-metadata suite uses.
    import json
    from analytics.storage import DrawStorage, StoredGame
    from analytics.tester import DrawTester
    from tests.test_violation_fixtures import _build_data_for_fixture

    fixtures_dir = Path(__file__).parent / 'fixtures' / 'violations'
    with open(fixtures_dir / 'club_vs_club_non_coincident.json') as f:
        raw = json.load(f)
    raw.pop('_violations', None)
    raw.pop('_description', None)
    overrides = raw.pop('_teams_override', None)
    games = [StoredGame(**g) for g in raw['games']]
    draw = DrawStorage(
        description=raw.get('description', 'spec036-unit-b-misaligned'),
        num_weeks=raw.get('num_weeks', 1),
        num_games=len(games),
        games=games,
    )
    data = _build_data_for_fixture(draw.games, teams_override=overrides)
    report = DrawTester(draw, data).run_violation_check()

    aligned_violations = [v for v in report.violations
                          if v.constraint == 'ClubVsClubAlignment']
    assert aligned_violations, \
        "tester anchor failed to flag the mis-aligned draw after engine deletion"
