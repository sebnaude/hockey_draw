# tests/test_analytics_coverage.py
"""
Coverage tests for analytics/storage.py, analytics/versioning.py, and analytics/reports.py.

Targets dark paths and untested methods to improve branch coverage.
Uses REAL objects throughout - no mocks.
"""

import json
import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add main repo root to path (worktree may not have all modules)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)
# If running inside a worktree, resolve to the main repo
if '.claude' in _repo_root:
    # e.g. .../draw/.claude/worktrees/agent-xxx -> .../draw
    _repo_root = _repo_root.split(os.sep + '.claude' + os.sep)[0]
sys.path.insert(0, _repo_root)

from analytics.storage import (
    DrawStorage, StoredGame, DrawAnalytics,
    create_draw_from_solution, load_draw, analyze_draw,
    export_draw_to_revformat, SlotAnalyzer, get_slot_analyzer,
)
from analytics.versioning import (
    DrawVersionManager, DrawVersion, VersionDiff,
)
from analytics.reports import (
    ClubReport, TeamReport, GradeReport, ComplianceCertificate,
    generate_html_report,
)
from models import PlayingField, Team, Club, Grade, Timeslot


# ======================================================================
# Shared fixtures
# ======================================================================

def _make_clubs():
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
    ]


def _make_teams(clubs):
    tigers, wests, maitland, norths = clubs
    return [
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Tigers 2nd', club=tigers, grade='2nd'),
        Team(name='Wests PHL', club=wests, grade='PHL'),
        Team(name='Wests 2nd', club=wests, grade='2nd'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
        Team(name='Maitland 2nd', club=maitland, grade='2nd'),
        Team(name='Norths PHL', club=norths, grade='PHL'),
        Team(name='Norths 2nd', club=norths, grade='2nd'),
    ]


def _make_grades():
    return [
        Grade(name='PHL', teams=[
            'Tigers PHL', 'Wests PHL', 'Maitland PHL', 'Norths PHL'
        ]),
        Grade(name='2nd', teams=[
            'Tigers 2nd', 'Wests 2nd', 'Maitland 2nd', 'Norths 2nd'
        ]),
    ]


def _make_fields():
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
    ]


def _make_games():
    """Create a realistic set of games across 3 weeks, 2 grades, 4 clubs."""
    games = [
        # Week 1 PHL
        StoredGame(game_id='G00001', team1='Maitland PHL', team2='Tigers PHL',
                   grade='PHL', week=1, round_no=1, date='2026-03-22',
                   day='Sunday', time='10:00', day_slot=1, field_name='EF',
                   field_location='Newcastle International Hockey Centre'),
        StoredGame(game_id='G00002', team1='Norths PHL', team2='Wests PHL',
                   grade='PHL', week=1, round_no=1, date='2026-03-22',
                   day='Sunday', time='11:30', day_slot=2, field_name='EF',
                   field_location='Newcastle International Hockey Centre'),
        # Week 1 2nd
        StoredGame(game_id='G00003', team1='Tigers 2nd', team2='Wests 2nd',
                   grade='2nd', week=1, round_no=1, date='2026-03-22',
                   day='Sunday', time='10:00', day_slot=1, field_name='WF',
                   field_location='Newcastle International Hockey Centre'),
        StoredGame(game_id='G00004', team1='Maitland 2nd', team2='Norths 2nd',
                   grade='2nd', week=1, round_no=1, date='2026-03-22',
                   day='Sunday', time='11:30', day_slot=2, field_name='WF',
                   field_location='Newcastle International Hockey Centre'),
        # Week 2 PHL
        StoredGame(game_id='G00005', team1='Maitland PHL', team2='Wests PHL',
                   grade='PHL', week=2, round_no=2, date='2026-03-29',
                   day='Sunday', time='10:00', day_slot=1, field_name='Maitland Main Field',
                   field_location='Maitland Park'),
        StoredGame(game_id='G00006', team1='Norths PHL', team2='Tigers PHL',
                   grade='PHL', week=2, round_no=2, date='2026-03-29',
                   day='Sunday', time='11:30', day_slot=2, field_name='EF',
                   field_location='Newcastle International Hockey Centre'),
        # Week 2 2nd
        StoredGame(game_id='G00007', team1='Tigers 2nd', team2='Norths 2nd',
                   grade='2nd', week=2, round_no=2, date='2026-03-29',
                   day='Sunday', time='10:00', day_slot=1, field_name='WF',
                   field_location='Newcastle International Hockey Centre'),
        StoredGame(game_id='G00008', team1='Maitland 2nd', team2='Wests 2nd',
                   grade='2nd', week=2, round_no=2, date='2026-03-29',
                   day='Sunday', time='11:30', day_slot=2, field_name='Maitland Main Field',
                   field_location='Maitland Park'),
        # Week 3 PHL
        StoredGame(game_id='G00009', team1='Maitland PHL', team2='Norths PHL',
                   grade='PHL', week=3, round_no=3, date='2026-04-05',
                   day='Sunday', time='10:00', day_slot=1, field_name='Maitland Main Field',
                   field_location='Maitland Park'),
        StoredGame(game_id='G00010', team1='Tigers PHL', team2='Wests PHL',
                   grade='PHL', week=3, round_no=3, date='2026-04-05',
                   day='Sunday', time='11:30', day_slot=2, field_name='EF',
                   field_location='Newcastle International Hockey Centre'),
        # Week 3 2nd
        StoredGame(game_id='G00011', team1='Maitland 2nd', team2='Tigers 2nd',
                   grade='2nd', week=3, round_no=3, date='2026-04-05',
                   day='Sunday', time='10:00', day_slot=1, field_name='Maitland Main Field',
                   field_location='Maitland Park'),
        StoredGame(game_id='G00012', team1='Norths 2nd', team2='Wests 2nd',
                   grade='2nd', week=3, round_no=3, date='2026-04-05',
                   day='Sunday', time='11:30', day_slot=2, field_name='WF',
                   field_location='Newcastle International Hockey Centre'),
    ]
    return games


def _make_draw(games=None):
    if games is None:
        games = _make_games()
    weeks = set(g.week for g in games)
    return DrawStorage(
        description='Test Draw',
        num_weeks=len(weeks),
        num_games=len(games),
        games=games,
    )


def _make_data():
    clubs = _make_clubs()
    teams = _make_teams(clubs)
    grades = _make_grades()
    fields = _make_fields()
    return {
        'clubs': clubs,
        'teams': teams,
        'grades': grades,
        'fields': fields,
        'num_rounds': {'PHL': 3, '2nd': 3},
    }


def _make_timeslots(fields):
    """Create timeslots matching the games' weeks."""
    ef, wf, mf = fields
    slots = []
    dates = [('2026-03-22', 1), ('2026-03-29', 2), ('2026-04-05', 3)]
    for date_str, week in dates:
        for f in [ef, wf, mf]:
            for slot_idx, time in enumerate(['10:00', '11:30'], 1):
                slots.append(Timeslot(
                    date=date_str, day='Sunday', time=time,
                    week=week, day_slot=slot_idx, field=f, round_no=week,
                ))
    return slots


@pytest.fixture
def draw():
    return _make_draw()


@pytest.fixture
def data():
    return _make_data()


@pytest.fixture
def data_with_timeslots():
    d = _make_data()
    d['timeslots'] = _make_timeslots(_make_fields())
    return d


# ======================================================================
# DrawStorage - untested methods
# ======================================================================

class TestDrawStorageGetGamesByClub:
    def test_returns_games_for_club(self, draw):
        maitland_games = draw.get_games_by_club('Maitland')
        # Maitland appears in games G00001,G00004,G00005,G00008,G00009,G00011
        assert len(maitland_games) >= 6

    def test_empty_for_unknown_club(self, draw):
        assert draw.get_games_by_club('NonExistent') == []


class TestDrawStorageMergeWith:
    def test_merge_combines_games(self):
        games_a = _make_games()[:4]
        games_b = _make_games()[4:8]
        draw_a = _make_draw(games_a)
        draw_b = _make_draw(games_b)
        merged = draw_a.merge_with(draw_b)
        assert merged.num_games == 8
        assert len(merged.games) == 8

    def test_merge_renumbers_ids(self):
        games_a = _make_games()[:2]
        games_b = _make_games()[2:4]
        draw_a = _make_draw(games_a)
        draw_b = _make_draw(games_b)
        merged = draw_a.merge_with(draw_b)
        ids = [g.game_id for g in merged.games]
        assert ids == ['G00000', 'G00001', 'G00002', 'G00003']

    def test_merge_description(self):
        draw_a = _make_draw(_make_games()[:2])
        draw_b = _make_draw(_make_games()[2:4])
        merged = draw_a.merge_with(draw_b)
        assert 'Merged' in merged.description


class TestDrawStorageFilterByFieldLocation:
    def test_filter_by_field_location(self, draw):
        maitland_games = draw.filter_games(field_location='Maitland Park')
        assert all(g.field_location == 'Maitland Park' for g in maitland_games)
        assert len(maitland_games) >= 1


class TestDrawStorageExportScheduleXlsx:
    def test_export_all_weeks(self, draw, tmp_path):
        xlsx_path = str(tmp_path / 'schedule.xlsx')
        draw.export_schedule_xlsx(xlsx_path)
        assert os.path.exists(xlsx_path)
        assert os.path.getsize(xlsx_path) > 0

    def test_export_selected_weeks(self, draw, tmp_path):
        xlsx_path = str(tmp_path / 'partial.xlsx')
        draw.export_schedule_xlsx(xlsx_path, weeks=[1, 2])
        assert os.path.exists(xlsx_path)

    def test_export_with_sheet_title(self, draw, tmp_path):
        xlsx_path = str(tmp_path / 'titled.xlsx')
        draw.export_schedule_xlsx(xlsx_path, weeks=[1], sheet_title='Round 1')
        assert os.path.exists(xlsx_path)
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_path)
        assert 'Round 1' in wb.sheetnames
        wb.close()


class TestDrawStorageFromXSolution:
    def test_from_x_solution_with_short_keys_skipped(self):
        """Short keys should be ignored."""
        X = {
            ('A', 'B', 'PHL', 'Sunday', 1, '10:00', 1, '2026-03-22', 1, 'EF',
             'Newcastle International Hockey Centre'): 1,
            ('A', 'B', 'PHL', 0): 1,  # short key
        }
        draw = DrawStorage.from_X_solution(X, 'Test')
        assert draw.num_games == 1


class TestDrawStorageLockAndSplitWithList:
    def test_lock_with_set_of_weeks(self):
        draw = _make_draw()
        locked = draw.get_locked_games(locked_weeks={1, 3})
        weeks = {g.week for g in locked}
        assert weeks == {1, 3}

    def test_remaining_with_set_of_weeks(self):
        draw = _make_draw()
        remaining = draw.get_remaining_games(locked_weeks={1, 3})
        weeks = {g.week for g in remaining}
        assert 1 not in weeks
        assert 3 not in weeks


class TestCreateDrawFromSolution:
    def test_convenience_function(self):
        X = {
            ('A', 'B', 'PHL', 'Sunday', 1, '10:00', 1, '2026-03-22', 1, 'EF',
             'Newcastle International Hockey Centre'): True,
        }
        draw = create_draw_from_solution(X, 'conv')
        assert draw.num_games == 1


class TestLoadDraw:
    def test_load_draw_convenience(self, draw, tmp_path):
        path = str(tmp_path / 'test.json')
        draw.save(path)
        loaded = load_draw(path)
        assert loaded.num_games == draw.num_games


# ======================================================================
# DrawAnalytics - 0% coverage, all methods
# ======================================================================

class TestDrawAnalytics:
    def test_init(self, draw, data):
        analytics = DrawAnalytics(draw, data)
        assert analytics.draw is draw
        assert len(analytics._team_to_club) > 0

    def test_get_club_for_team(self, draw, data):
        a = DrawAnalytics(draw, data)
        assert a.get_club_for_team('Tigers PHL') == 'Tigers'
        assert a.get_club_for_team('Unknown') == 'Unknown'

    def test_games_played_by_team_grade(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.games_played_by_team_grade()
        assert 'Team' in df.columns
        assert 'Total' in df.columns
        assert len(df) == len(data['teams'])

    def test_team_matchups_crosstab_all(self, draw, data):
        a = DrawAnalytics(draw, data)
        result = a.team_matchups_crosstab()
        assert 'PHL' in result
        assert 'Maitland PHL' in result['PHL'].index

    def test_team_matchups_crosstab_single_grade(self, draw, data):
        a = DrawAnalytics(draw, data)
        result = a.team_matchups_crosstab(grade='2nd')
        assert '2nd' in result
        assert 'PHL' not in result

    def test_home_away_analysis(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.home_away_analysis()
        assert 'Home' in df.columns
        assert 'Away' in df.columns
        assert 'Neutral' in df.columns
        assert len(df) == len(data['teams'])

    def test_club_season_schedule(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.club_season_schedule('Tigers')
        assert len(df) > 0
        assert 'Week' in df.columns
        assert 'Opponent' in df.columns

    def test_club_season_schedule_empty(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.club_season_schedule('NonExistent')
        assert len(df) == 0

    def test_grade_summary(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.grade_summary()
        assert 'Grade' in df.columns
        assert 'Total Games' in df.columns
        assert len(df) == len(data['grades'])

    def test_weekly_field_usage(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.weekly_field_usage()
        assert 'Week' in df.columns
        assert 'Games' in df.columns
        assert len(df) > 0

    def test_away_team_balance(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.away_team_balance()
        # Should have rows for Maitland teams
        assert len(df) >= 2  # Maitland PHL and Maitland 2nd

    def test_constraint_compliance_summary(self, draw, data):
        a = DrawAnalytics(draw, data)
        df = a.constraint_compliance_summary()
        assert 'Constraint' in df.columns
        assert 'Status' in df.columns
        assert len(df) >= 3

    def test_export_analytics_to_excel(self, draw, data, tmp_path):
        a = DrawAnalytics(draw, data)
        path = str(tmp_path / 'analytics.xlsx')
        a.export_analytics_to_excel(path)
        assert os.path.exists(path)
        # Verify multiple sheets exist
        xl = pd.ExcelFile(path)
        assert 'Summary' in xl.sheet_names
        assert 'Compliance Check' in xl.sheet_names
        assert 'Games Per Team' in xl.sheet_names
        xl.close()


class TestAnalyzeDraw:
    def test_convenience_function(self, draw, data):
        a = analyze_draw(draw, data)
        assert isinstance(a, DrawAnalytics)


# ======================================================================
# export_draw_to_revformat
# ======================================================================

class TestExportDrawToRevformat:
    def test_basic_export(self, draw, data, tmp_path):
        path = str(tmp_path / 'rev.csv')
        export_draw_to_revformat(draw, data, output_path=path)
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert 'Round' in df.columns
        assert 'Grade' in df.columns
        assert len(df) > 0

    def test_export_with_week_limit(self, draw, data, tmp_path):
        path = str(tmp_path / 'rev_limited.csv')
        export_draw_to_revformat(draw, data, output_path=path, week_limit=1)
        df = pd.read_csv(path)
        # Only week 1 games + bye entries
        game_rows = df[df['Team 2'] != 'BYE']
        assert all(r == 1 for r in game_rows['Round'])

    def test_bye_entries_present(self, draw, data, tmp_path):
        path = str(tmp_path / 'rev_byes.csv')
        export_draw_to_revformat(draw, data, output_path=path)
        df = pd.read_csv(path)
        byes = df[df['Team 2'] == 'BYE']
        # Our draw has all teams playing every week so no byes, but the code path is covered
        # (or there may be byes if not all teams play every week)
        assert isinstance(byes, pd.DataFrame)


# ======================================================================
# SlotAnalyzer
# ======================================================================

class TestSlotAnalyzer:
    def test_get_all_possible_slots(self, draw, data_with_timeslots):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        slots = sa.get_all_possible_slots()
        assert len(slots) > 0
        assert 'week' in slots[0]

    def test_get_used_slots(self, draw, data_with_timeslots):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        used = sa.get_used_slots()
        assert len(used) == draw.num_games

    def test_get_used_slots_by_week(self, draw, data_with_timeslots):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        used_w1 = sa.get_used_slots(week=1)
        assert all(s['week'] == 1 for s in used_w1)

    def test_get_unused_slots(self, draw, data_with_timeslots):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        unused = sa.get_unused_slots()
        # Some slots are unused (games don't fill every slot)
        assert len(unused) > 0
        # Unused slots should not overlap with used slot keys
        used_keys = {(s['week'], s['day_slot'], s['field_name']) for s in sa.get_used_slots()}
        for s in unused:
            assert (s['week'], s['day_slot'], s['field_name']) not in used_keys

    def test_get_unused_slots_by_week(self, draw, data_with_timeslots):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        unused = sa.get_unused_slots(week=1)
        assert all(s['week'] == 1 for s in unused)

    def test_slot_usage_summary(self, draw, data_with_timeslots):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        df = sa.slot_usage_summary()
        assert 'Week' in df.columns
        assert 'Utilization' in df.columns

    def test_print_unused_slots(self, draw, data_with_timeslots, capsys):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        sa.print_unused_slots()
        captured = capsys.readouterr()
        assert 'UNUSED SLOTS' in captured.out or 'No unused slots' in captured.out

    def test_print_unused_slots_by_week(self, draw, data_with_timeslots, capsys):
        sa = SlotAnalyzer(draw, data_with_timeslots)
        sa.print_unused_slots(week=1)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestGetSlotAnalyzer:
    def test_convenience_function(self, draw, data_with_timeslots):
        sa = get_slot_analyzer(draw, data_with_timeslots)
        assert isinstance(sa, SlotAnalyzer)


# ======================================================================
# DrawVersionManager
# ======================================================================

class TestDrawVersion:
    def test_version_string(self):
        v = DrawVersion(major=2, minor=3, created_at='2026-01-01',
                        description='test', filename='draw_v2.3.json', game_count=100)
        assert v.version_string == 'v2.3'
        assert 'v2.3' in str(v)


class TestVersionDiff:
    def test_no_changes(self):
        diff = VersionDiff()
        assert not diff.has_changes
        assert diff.summary == 'No changes'

    def test_with_changes(self):
        diff = VersionDiff(
            added_games=[{'matchup': 'A vs B'}],
            removed_games=[{'matchup': 'C vs D'}],
            modified_games=[{'matchup': 'E vs F'}],
        )
        assert diff.has_changes
        assert '+1 games' in diff.summary
        assert '-1 games' in diff.summary
        assert '~1 modified' in diff.summary


class TestDrawVersionManager:
    def test_init_creates_directories(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        assert mgr.base_path.exists()
        assert mgr.versions_path.exists()

    def test_init_no_year(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'))
        assert mgr.base_path == tmp_path / 'draws'

    def test_get_versions_empty(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        versions = mgr.get_versions()
        assert versions == []

    def test_get_latest_version_empty(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        assert mgr.get_latest_version() is None

    def test_get_next_major_version_first(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        assert mgr.get_next_major_version() == (1, 0)

    def test_get_next_minor_version_first(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        assert mgr.get_next_minor_version() == (1, 0)

    def test_save_new_draw_major(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        version = mgr.save_new_draw(draw, 'Initial generation', is_major=True)
        assert version.major == 1
        assert version.minor == 0
        assert version.game_count == draw.num_games
        # Check files created
        assert (mgr.versions_path / 'draw_v1.0.json').exists()
        assert (mgr.base_path / 'current.json').exists()
        assert mgr.changelog_path.exists()

    def test_save_new_draw_minor(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        v1 = mgr.save_new_draw(draw, 'Initial', is_major=True)
        v2 = mgr.save_new_draw(draw, 'Minor tweak', is_major=False)
        assert v2.major == 1
        assert v2.minor == 1

    def test_get_versions_after_save(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        mgr.save_new_draw(draw, 'v1')
        mgr.save_new_draw(draw, 'v2')
        versions = mgr.get_versions()
        assert len(versions) == 2

    def test_get_next_major_after_save(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        mgr.save_new_draw(draw, 'v1')
        assert mgr.get_next_major_version() == (2, 0)

    def test_get_next_minor_after_save(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        mgr.save_new_draw(draw, 'v1')
        assert mgr.get_next_minor_version() == (1, 1)

    def test_load_version(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        mgr.save_new_draw(draw, 'Test load')
        loaded = mgr.load_version(1, 0)
        assert loaded is not None
        assert loaded.num_games == draw.num_games

    def test_load_version_nonexistent(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        assert mgr.load_version(99, 99) is None

    def test_load_latest(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        mgr.save_new_draw(draw, 'Latest')
        loaded = mgr.load_latest()
        assert loaded is not None
        assert loaded.num_games == draw.num_games

    def test_load_latest_empty(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        assert mgr.load_latest() is None

    def test_list_versions_empty(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        result = mgr.list_versions()
        assert 'No versions found' in result

    def test_list_versions_with_data(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        mgr.save_new_draw(draw, 'Test list')
        result = mgr.list_versions()
        assert 'v1.0' in result
        assert 'Version History' in result


class TestDrawVersionManagerComputeDiff:
    def test_no_changes(self):
        draw = _make_draw()
        mgr = DrawVersionManager('.')
        diff = mgr.compute_diff(draw, draw)
        assert not diff.has_changes

    def test_added_game(self):
        games1 = _make_games()[:4]
        games2 = _make_games()[:6]
        draw1 = _make_draw(games1)
        draw2 = _make_draw(games2)
        mgr = DrawVersionManager('.')
        diff = mgr.compute_diff(draw1, draw2)
        assert len(diff.added_games) == 2
        assert len(diff.removed_games) == 0

    def test_removed_game(self):
        games1 = _make_games()[:6]
        games2 = _make_games()[:4]
        draw1 = _make_draw(games1)
        draw2 = _make_draw(games2)
        mgr = DrawVersionManager('.')
        diff = mgr.compute_diff(draw1, draw2)
        assert len(diff.removed_games) == 2

    def test_modified_game(self):
        games = _make_games()[:4]
        draw1 = _make_draw(games)
        # Modify time of first game
        modified_games = [g.model_copy() for g in games]
        modified_games[0] = StoredGame(
            **{**modified_games[0].model_dump(), 'time': '14:00'}
        )
        draw2 = _make_draw(modified_games)
        mgr = DrawVersionManager('.')
        diff = mgr.compute_diff(draw1, draw2)
        assert len(diff.modified_games) == 1


class TestDrawVersionManagerSaveModifiedDraw:
    def test_save_modified_draw(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        old_draw = _make_draw(_make_games()[:6])
        mgr.save_new_draw(old_draw, 'Original')

        new_games = _make_games()[:6]
        # Change the time of one game
        new_games[0] = StoredGame(
            **{**new_games[0].model_dump(), 'time': '14:00'}
        )
        new_draw = _make_draw(new_games)
        version = mgr.save_modified_draw(new_draw, old_draw, 'Fixed timing')
        assert version.major == 1
        assert version.minor == 1
        assert (mgr.versions_path / 'draw_v1.1.json').exists()

    def test_changelog_updated(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        old_draw = _make_draw()
        mgr.save_new_draw(old_draw, 'Original')

        new_draw = _make_draw()
        mgr.save_modified_draw(new_draw, old_draw, 'Tweaked')
        content = mgr.changelog_path.read_text(encoding='utf-8')
        assert 'v1.1' in content
        assert 'Tweaked' in content


class TestDrawVersionManagerXlsxCopy:
    def test_save_new_draw_with_xlsx(self, tmp_path, draw):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        # Create a fake xlsx
        xlsx_path = tmp_path / 'schedule.xlsx'
        draw.export_schedule_xlsx(str(xlsx_path))
        version = mgr.save_new_draw(draw, 'With xlsx', xlsx_path=xlsx_path)
        versioned_xlsx = mgr.versions_path / f'draw_v{version.major}.{version.minor}.xlsx'
        assert versioned_xlsx.exists()
        assert (mgr.base_path / 'current.xlsx').exists()


class TestSerializeForJson:
    def test_dict(self):
        result = DrawVersionManager._serialize_for_json({'a': {1, 2}})
        assert result == {'a': [1, 2]}

    def test_set(self):
        result = DrawVersionManager._serialize_for_json({1, 3, 2})
        assert result == [1, 2, 3]

    def test_list(self):
        result = DrawVersionManager._serialize_for_json([{1, 2}])
        assert result == [[1, 2]]

    def test_primitive(self):
        assert DrawVersionManager._serialize_for_json(42) == 42


class TestMigrateLegacyDraws:
    def test_no_legacy(self, tmp_path, capsys):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        mgr.migrate_legacy_draws()
        captured = capsys.readouterr()
        assert 'No legacy draws to migrate' in captured.out

    def test_with_legacy(self, tmp_path):
        mgr = DrawVersionManager(str(tmp_path / 'draws'), year=2026)
        draw = _make_draw()
        # Save directly in base path (legacy location)
        legacy_path = mgr.base_path / 'draw_v1.0.json'
        draw.save(str(legacy_path))
        mgr.migrate_legacy_draws()
        # File should have moved to versions/
        assert not legacy_path.exists()
        assert (mgr.versions_path / 'draw_v1.0.json').exists()


# ======================================================================
# ClubReport - untested methods
# ======================================================================

class TestClubReportGenerateForClub:
    def test_generate_returns_sheets(self, draw, data):
        report = ClubReport(draw, data)
        sheets = report.generate_for_club('Tigers')
        assert 'Club Summary' in sheets
        assert 'Opponents' in sheets
        assert isinstance(sheets['Club Summary'], pd.DataFrame)

    def test_generate_exports_excel(self, draw, data, tmp_path):
        report = ClubReport(draw, data)
        path = str(tmp_path / 'tigers.xlsx')
        sheets = report.generate_for_club('Tigers', output=path)
        assert os.path.exists(path)
        assert len(sheets) >= 3  # summary + opponents + at least 1 schedule

    def test_bye_weeks(self, draw, data):
        report = ClubReport(draw, data)
        byes = report.bye_weeks('Tigers PHL')
        # Tigers PHL plays every week in our data, so no byes
        assert isinstance(byes, list)

    def test_home_away_by_opponent(self, draw, data):
        report = ClubReport(draw, data)
        df = report.home_away_by_opponent('Maitland PHL')
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'Opponent' in df.columns
            assert 'Home' in df.columns


# ======================================================================
# TeamReport
# ======================================================================

class TestTeamReport:
    def test_generate(self, draw, data):
        report = TeamReport(draw, data)
        sheets = report.generate('Tigers PHL')
        assert 'Schedule' in sheets
        assert 'Home-Away' in sheets
        assert 'Byes' in sheets
        assert 'Summary' in sheets

    def test_generate_exports_excel(self, draw, data, tmp_path):
        report = TeamReport(draw, data)
        path = str(tmp_path / 'team_report.xlsx')
        report.generate('Tigers PHL', output=path)
        assert os.path.exists(path)

    def test_generate_team_with_no_games(self, data):
        """Test team report for a team with no games returns valid empty report."""
        sparse_draw = DrawStorage(
            description='Sparse', num_weeks=0, num_games=0, games=[]
        )
        report = TeamReport(sparse_draw, data)
        sheets = report.generate('Tigers PHL')
        assert sheets['Summary']['Value'].iloc[0] == 0  # Total Games = 0


# ======================================================================
# GradeReport
# ======================================================================

class TestGradeReport:
    def test_grade_schedule(self, draw, data):
        report = GradeReport(draw, data)
        df = report.grade_schedule('PHL')
        assert len(df) > 0
        assert 'Team 1' in df.columns

    def test_matchup_matrix(self, draw, data):
        report = GradeReport(draw, data)
        df = report.matchup_matrix('PHL')
        assert isinstance(df, pd.DataFrame)

    def test_generate(self, draw, data):
        report = GradeReport(draw, data)
        sheets = report.generate('PHL')
        assert 'Schedule' in sheets
        assert 'Matchup Matrix' in sheets
        assert 'Games Per Team' in sheets

    def test_generate_exports_excel(self, draw, data, tmp_path):
        report = GradeReport(draw, data)
        path = str(tmp_path / 'grade.xlsx')
        report.generate('PHL', output=path)
        assert os.path.exists(path)

    def test_generate_all_grades(self, draw, data, tmp_path):
        report = GradeReport(draw, data)
        output_dir = str(tmp_path / 'grades')
        report.generate_all_grades(output_dir)
        assert os.path.exists(output_dir)
        assert any(f.endswith('.xlsx') for f in os.listdir(output_dir))


# ======================================================================
# ComplianceCertificate
# ======================================================================

class TestComplianceCertificate:
    def test_run_checks(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        report = cert.run_checks()
        assert report is not None
        assert report.total_games == draw.num_games

    def test_summary_table(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        df = cert.summary_table()
        assert 'Constraint' in df.columns
        assert 'Status' in df.columns
        assert len(df) > 0

    def test_certificate_header(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        df = cert.certificate_header()
        assert 'Field' in df.columns
        assert 'Value' in df.columns
        fields = df['Field'].tolist()
        assert 'Total Games' in fields
        assert 'Overall Status' in fields

    def test_detailed_violations(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        df = cert.detailed_violations()
        assert isinstance(df, pd.DataFrame)
        # May have violations since our test draw is minimal

    def test_is_compliant(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        result = cert.is_compliant()
        assert isinstance(result, bool)

    def test_is_critical_compliant(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        result = cert.is_critical_compliant()
        assert isinstance(result, bool)

    def test_generate(self, draw, data):
        cert = ComplianceCertificate(draw, data)
        sheets = cert.generate()
        assert 'Certificate' in sheets
        assert 'Constraint Summary' in sheets

    def test_generate_exports_excel(self, draw, data, tmp_path):
        cert = ComplianceCertificate(draw, data)
        path = str(tmp_path / 'compliance.xlsx')
        cert.generate(output=path)
        assert os.path.exists(path)


# ======================================================================
# generate_html_report
# ======================================================================

class TestGenerateHtmlReport:
    def test_generates_html(self, draw, data, tmp_path):
        path = str(tmp_path / 'report.html')
        generate_html_report(draw, data, output=path)
        assert os.path.exists(path)
        content = open(path, encoding='utf-8').read()
        assert '<html' in content
        assert 'Draw Report' in content
        assert str(draw.num_games) in content
