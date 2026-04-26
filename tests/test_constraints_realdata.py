# tests/test_constraints_realdata.py
"""
Tests for DrawTester constraint checks using real 2026 fixture data (first 6 weeks).

Each test class covers one constraint check method, running it against real schedule
data to verify it catches known violations. This avoids the fragility of hand-crafted
synthetic data that may not trigger real constraint paths.

Fixture: tests/fixtures/draw_2026_first6weeks.json (108 games, weeks 1,2,4,5,6)

Known violation counts in the fixture (determined empirically 2026-04-07):
  - NoDoubleBookingTeams: 0 (clean)
  - NoDoubleBookingFields: 0 (clean)
  - ClubGradeAdjacency: 9 (7 adjacent-grade + 2 duplicate-team)
  - PHLSecondGradeAdjacency: 1
  - MaitlandBackToBack: 0 (clean)
  - MaitlandAwayClubsLimit: 1
  - TeamConflict: 0 (clean)
  - EqualMatchUpSpacing: 1
  - FiftyFiftyHomeAway: 1
  - ClubGameSpread: 23
  - MinClubsOnFieldBroadmeadow: 15
  - MaxClubsPerTimeslotBroadmeadow: 0 (clean)
  - EnsureBestTimeslotChoices: 0 (clean)
  - PreferredTimes: 0 (clean)

Constraints that trivially fail with partial data (6 of 22 weeks) — tested separately:
  - EqualGames: 48 (teams can't reach full game count)
  - BalancedMatchups: 58 (not enough meetings)
  - ClubDay: 12 (club days fall outside the 6-week window)
  - ClubVsClubAlignment: 28 (insufficient rounds for coincidence)
  - ClubFieldConcentration: 37 (diagnostic, not a solver constraint)
  - PHLSecondGradeTimes: 1 (Gosford Friday count can't match with partial data)
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from config import load_season_data


# ============== Fixtures ==============

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'draw_2026_first6weeks.json')


@pytest.fixture(scope='module')
def season_data():
    """Load full 2026 season config (teams, clubs, grades, etc.)."""
    return load_season_data(2026)


@pytest.fixture(scope='module')
def fixture_draw():
    """Load the first-6-weeks fixture as a DrawStorage."""
    with open(FIXTURE_PATH) as f:
        raw = json.load(f)
    games = [StoredGame(**g) for g in raw['games']]
    return DrawStorage(
        description=raw['description'],
        num_weeks=raw['num_weeks'],
        num_games=raw['num_games'],
        games=games,
    )


@pytest.fixture(scope='module')
def tester(fixture_draw, season_data):
    """Create a DrawTester from fixture data."""
    return DrawTester(fixture_draw, season_data)


def _make_modified_draw(fixture_draw, filter_fn):
    """Return a new DrawStorage excluding games where filter_fn(game) is True."""
    kept = [g for g in fixture_draw.games if not filter_fn(g)]
    return DrawStorage(
        description='Modified fixture',
        num_weeks=fixture_draw.num_weeks,
        num_games=len(kept),
        games=kept,
    )


# ============== Level 1: CRITICAL ==============


class TestNoDoubleBookingTeams:
    """A team should never play more than one game in the same week."""

    def test_clean_pass(self, tester):
        violations = tester._check_no_double_booking_teams()
        assert len(violations) == 0

    def test_injected_double_booking(self, fixture_draw, season_data):
        """Duplicate a game for a team in the same week -> must be caught."""
        original_game = fixture_draw.games[0]
        dupe = StoredGame(
            game_id='G_DUPE',
            team1=original_game.team1,
            team2='Norths 4th',  # different opponent
            grade=original_game.grade,
            week=original_game.week,
            round_no=original_game.round_no,
            date=original_game.date,
            day=original_game.day,
            time='15:00',
            day_slot=5,
            field_name='SF',
            field_location=original_game.field_location,
        )
        draw = DrawStorage(
            description='Double-booked test',
            num_weeks=fixture_draw.num_weeks,
            num_games=fixture_draw.num_games + 1,
            games=list(fixture_draw.games) + [dupe],
        )
        t = DrawTester(draw, season_data)
        violations = t._check_no_double_booking_teams()
        team_msgs = [v for v in violations if original_game.team1 in v.message]
        assert len(team_msgs) >= 1, f"Expected double-booking violation for {original_game.team1}"


class TestNoDoubleBookingFields:
    """A field should never have two games at the same date+slot."""

    def test_clean_pass(self, tester):
        violations = tester._check_no_double_booking_fields()
        assert len(violations) == 0

    def test_injected_field_conflict(self, fixture_draw, season_data):
        """Two games on same field, same date, same slot -> must be caught."""
        ref = fixture_draw.games[0]
        dupe = StoredGame(
            game_id='G_FIELD_DUPE',
            team1='Norths 3rd', team2='Souths 3rd', grade='3rd',
            week=ref.week, round_no=ref.round_no, date=ref.date,
            day=ref.day, time=ref.time, day_slot=ref.day_slot,
            field_name=ref.field_name, field_location=ref.field_location,
        )
        draw = DrawStorage(
            description='Field conflict test',
            num_weeks=fixture_draw.num_weeks,
            num_games=fixture_draw.num_games + 1,
            games=list(fixture_draw.games) + [dupe],
        )
        t = DrawTester(draw, season_data)
        violations = t._check_no_double_booking_fields()
        assert len(violations) >= 1


class TestEqualGames:
    """With only 6 weeks of data, teams will be short of their full game count."""

    def test_partial_data_expected_violations(self, tester):
        violations = tester._check_equal_games()
        # 48 teams with fewer games than expected
        assert len(violations) >= 40, f"Expected many EqualGames violations with partial data, got {len(violations)}"
        # Every violation message should mention expected vs actual
        for v in violations:
            assert 'expected' in v.message.lower() or 'games' in v.message.lower()


class TestBalancedMatchups:
    """With partial data, many pairs won't have met enough times."""

    def test_partial_data_expected_violations(self, tester):
        violations = tester._check_balanced_matchups()
        assert len(violations) >= 40, f"Expected many BalancedMatchups violations with partial data, got {len(violations)}"


class TestFiftyFiftyHomeAway:
    """Per-pair home/away balance for Maitland/Gosford teams."""

    def test_known_violation(self, tester):
        violations = tester._check_fifty_fifty_home_away()
        assert len(violations) >= 1
        # Known: Maitland 3rd vs Wests 3rd is 0H/2A
        mait_wests = [v for v in violations if 'Maitland 3rd' in v.message and 'Wests 3rd' in v.message]
        assert len(mait_wests) == 1, f"Expected Maitland 3rd vs Wests 3rd violation, found: {[v.message for v in violations]}"


class TestMaitlandBackToBack:
    """No consecutive Maitland home weekends (sliding window)."""

    def test_clean_pass(self, tester):
        violations = tester._check_maitland_back_to_back()
        assert len(violations) == 0


class TestEqualMatchUpSpacing:
    """Matchups should be evenly spaced across rounds."""

    def test_known_violation(self, tester):
        violations = tester._check_equal_matchup_spacing()
        assert len(violations) >= 1
        # Known: Maitland 3rd vs Wests 3rd gap of 3 (min 4)
        mait = [v for v in violations if 'Maitland 3rd' in v.message and 'Wests 3rd' in v.message]
        assert len(mait) >= 1, f"Expected Maitland 3rd vs Wests 3rd spacing violation"


# ============== Level 2: HIGH ==============


class TestMaitlandAwayClubsLimit:
    """Max away clubs at Maitland per week."""

    def test_known_violation(self, tester):
        violations = tester._check_maitland_away_clubs_limit()
        assert len(violations) >= 1
        # Known: Week 4 has 3 away clubs (Wests, Crusaders, Souths)
        week4 = [v for v in violations if 'Week 4' in v.message]
        assert len(week4) == 1
        assert 'Wests' in week4[0].message or 'Crusaders' in week4[0].message or 'Souths' in week4[0].message

    def test_violation_mentions_club_names(self, tester):
        violations = tester._check_maitland_away_clubs_limit()
        for v in violations:
            assert 'away clubs' in v.message.lower() or 'max' in v.message.lower()


class TestTeamConflict:
    """Teams with shared players can't play at same time."""

    def test_clean_pass(self, tester):
        violations = tester._check_team_conflict()
        assert len(violations) == 0


class TestClubDay:
    """Club days fall outside the 6-week window, so violations are expected."""

    def test_partial_data_expected_violations(self, tester):
        violations = tester._check_club_day()
        # Club days (Crusaders 2026-06-14, University 2026-07-26) are outside weeks 1-6
        # so missing-team and no-games violations are expected
        assert len(violations) >= 1


# ============== Level 3: MEDIUM ==============


class TestClubGradeAdjacency:
    """Adjacent grades from same club should not play at the same timeslot.

    This is the constraint that prompted this entire test overhaul.
    The fixture has 9 known violations:
      - 7 adjacent-grade violations (mostly Wests, one Tigers)
      - 2 duplicate-team violations (University 6th teams)
    """

    def test_total_violation_count(self, tester):
        violations = tester._check_club_grade_adjacency()
        assert len(violations) == 9, (
            f"Expected 9 ClubGradeAdjacency violations, got {len(violations)}: "
            + "; ".join(v.message for v in violations)
        )

    def test_wests_adjacent_grade_violations(self, tester):
        """Wests has the most violations — 7 adjacent-grade entries."""
        violations = tester._check_club_grade_adjacency()
        wests_adj = [v for v in violations if 'Wests' in v.message and 'adjacent' in v.message]
        assert len(wests_adj) == 7, (
            f"Expected 7 Wests adjacent-grade violations, got {len(wests_adj)}: "
            + "; ".join(v.message for v in wests_adj)
        )

    def test_tigers_adjacent_grade_violation(self, tester):
        """Tigers has one 5th/6th violation on 2026-04-12."""
        violations = tester._check_club_grade_adjacency()
        tigers = [v for v in violations if 'Tigers' in v.message and 'adjacent' in v.message]
        assert len(tigers) == 1
        assert '5th/6th' in tigers[0].message
        assert '2026-04-12' in tigers[0].message

    def test_university_duplicate_team_violation(self, tester):
        """University has duplicate 6th teams at same slot (not playing each other)."""
        violations = tester._check_club_grade_adjacency()
        uni = [v for v in violations if 'University' in v.message and 'duplicate' in v.message]
        assert len(uni) == 1
        assert 'University Gentlemen 6th' in uni[0].message
        assert 'University Seapigs 6th' in uni[0].message

    def test_specific_grade_pairs_found(self, tester):
        """Verify the specific grade pairs that appear in violations."""
        violations = tester._check_club_grade_adjacency()
        adj_messages = [v.message for v in violations if 'adjacent' in v.message]
        pairs_found = set()
        for msg in adj_messages:
            for pair in ['PHL/2nd', '2nd/3rd', '3rd/4th', '4th/5th', '5th/6th']:
                if pair in msg:
                    pairs_found.add(pair)
        # Known pairs in the fixture
        assert '4th/5th' in pairs_found
        assert '5th/6th' in pairs_found
        assert '2nd/3rd' in pairs_found
        assert '3rd/4th' in pairs_found

    def test_removing_offending_game_reduces_violations(self, fixture_draw, season_data):
        """Remove a Wests 4th game from slot 6 on 2026-04-26 -> one fewer violation."""
        # The violation: Wests 4th/5th at 2026-04-26, slot 6
        modified_games = []
        removed = False
        for g in fixture_draw.games:
            if (g.date == '2026-04-26' and g.day_slot == 6 and g.grade == '4th'
                    and ('Wests' in g.team1 or 'Wests' in g.team2) and not removed):
                removed = True
                continue  # skip this game
            modified_games.append(g)

        assert removed, "Should have found and removed the offending Wests 4th game"
        draw = DrawStorage(
            description='Modified', num_weeks=fixture_draw.num_weeks,
            num_games=len(modified_games), games=modified_games,
        )
        t = DrawTester(draw, season_data)
        violations = t._check_club_grade_adjacency()
        assert len(violations) == 8, f"Removing one game should reduce violations from 9 to 8, got {len(violations)}"


class TestClubVsClubAlignment:
    """Club matchup rounds should coincide across grades."""

    def test_partial_data_violations(self, tester):
        violations = tester._check_club_vs_club_alignment()
        # With only 6 weeks, many club pairs can't achieve alignment
        assert len(violations) >= 20


class TestClubGameSpread:
    """Games should be spread across the day (not all bunched or gapped)."""

    def test_known_violations(self, tester):
        violations = tester._check_club_game_spread()
        assert len(violations) >= 15, f"Expected many ClubGameSpread violations, got {len(violations)}"

    def test_violations_reference_clubs(self, tester):
        """Every violation should mention a club name."""
        violations = tester._check_club_game_spread()
        known_clubs = {'Colts', 'Wests', 'Norths', 'Tigers', 'Maitland', 'Gosford',
                       'Souths', 'Crusaders', 'University', 'Port Stephens'}
        for v in violations:
            assert any(club in v.message for club in known_clubs), f"No club name in: {v.message}"


class TestClubFieldConcentration:
    """Diagnostic check — clubs shouldn't be too spread across fields."""

    def test_known_violations(self, tester):
        violations = tester._check_club_field_concentration()
        assert len(violations) >= 20


# ============== Level 4: LOW ==============


class TestMaximiseClubsPerTimeslotBroadmeadow:
    """Prefer maximising club diversity in each Broadmeadow timeslot."""

    def test_clean_pass(self, tester):
        violations = tester._check_maximise_clubs_per_timeslot_broadmeadow()
        assert len(violations) == 0


class TestMinimiseClubsOnAFieldBroadmeadow:
    """Prefer minimising the number of clubs assigned to a single field."""

    def test_known_violations(self, tester):
        violations = tester._check_minimise_clubs_on_a_field_broadmeadow()
        assert len(violations) >= 10

    def test_specific_violation_content(self, tester):
        """Violations should mention field, week, and club count."""
        violations = tester._check_minimise_clubs_on_a_field_broadmeadow()
        for v in violations:
            assert 'clubs' in v.message.lower()
            assert 'max' in v.message.lower() or 'NIHC' in v.message


# ============== Level 5: VERY LOW ==============


class TestEnsureBestTimeslotChoices:
    """Prefer best timeslots for each grade."""

    def test_clean_pass(self, tester):
        violations = tester._check_ensure_best_timeslot_choices()
        assert len(violations) == 0


class TestPreferredTimes:
    """Prefer team-specific preferred times."""

    def test_clean_pass(self, tester):
        violations = tester._check_preferred_times()
        assert len(violations) == 0


# ============== Cross-cutting: PHLSecondGrade ==============


class TestPHLSecondGradeAdjacency:
    """PHL and 2nd grade from same club must satisfy 180-min + same-location rule."""

    def test_known_violation(self, tester):
        violations = tester._check_phl_second_grade_adjacency()
        assert len(violations) >= 1
        # Known: Wests PHL (13:00) and 2nd (17:30) at same location but 270min apart
        wests = [v for v in violations if 'Wests' in v.message]
        assert len(wests) >= 1
        assert '270min' in wests[0].message or '270' in wests[0].message

    def test_violation_severity(self, tester):
        violations = tester._check_phl_second_grade_adjacency()
        for v in violations:
            assert v.severity_level == 1  # CRITICAL


class TestPHLSecondGradeTimes:
    """PHL and 2nd grade timing constraints (Gosford Friday games, same-slot, etc.)."""

    def test_partial_data_violation(self, tester):
        violations = tester._check_phl_second_grade_times()
        # With partial data, Gosford Friday count won't match
        assert len(violations) >= 1


# ============== Integration: run_violation_check ==============


class TestRunViolationCheckIntegration:
    """Test the full orchestrated run_violation_check() against real data."""

    def test_returns_violation_report(self, tester):
        report = tester.run_violation_check()
        assert report is not None
        assert report.has_violations

    def test_constraint_results_populated(self, tester):
        report = tester.run_violation_check()
        # Should have results for each checked constraint
        assert len(report.constraint_results) >= 10

    def test_club_grade_adjacency_in_results(self, tester):
        """The constraint that prompted this overhaul should appear in results."""
        report = tester.run_violation_check()
        cga = [cr for cr in report.constraint_results if cr.constraint == 'ClubGradeAdjacency']
        assert len(cga) == 1
        assert cga[0].status == 'VIOLATED'
        assert len(cga[0].violations) >= 1

    def test_no_double_booking_passes(self, tester):
        report = tester.run_violation_check()
        ndb_teams = [cr for cr in report.constraint_results if cr.constraint == 'NoDoubleBookingTeams']
        ndb_fields = [cr for cr in report.constraint_results if cr.constraint == 'NoDoubleBookingFields']
        if ndb_teams:
            assert ndb_teams[0].status == 'PASSED'
        if ndb_fields:
            assert ndb_fields[0].status == 'PASSED'
