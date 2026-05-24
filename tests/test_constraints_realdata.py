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
  - PHLSecondGradeTimes: 1 (Gosford Friday count can't match with partial data)
  - (spec-031: ClubFieldConcentration removed — diagnostic deleted)
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


# spec-018: TestMaitlandBackToBack removed — the `_check_maitland_back_to_back`
# tester method (and the venue back-to-back rule) were deleted.


class TestEqualMatchUpSpacing:
    """Matchups should be evenly spaced across rounds.

    spec-008 (Part A) aligned the tester's threshold with the solver's
    threshold via `constraints.atoms._spacing.effective_spacing`. The
    convenor-facing number S = "free rounds between meetings" and the
    hard rule forbids `gap = r2 - r1 <= S`.

    For 3rd grade (T=6 in the 2026 fixture):
      ideal_gap(6) = legacy_min_gap(6) - 1 = 3 - 1 = 2
    so a gap of 3 is allowed (3 > 2). The previous tester used `ideal=T-2`
    with `floor=min(T//2, T-2)` which over-reported a gap of 3 as a
    violation (its min was 4) even though the solver itself accepted gap=3
    — the tester was strictly tighter than the solver. The spec-008 fix
    removes that false positive, so the previous "Maitland 3rd vs Wests
    3rd gap of 3" entry is no longer flagged.
    """

    def test_no_false_positive_at_legacy_min_gap(self, tester):
        """Gaps >= the new solver threshold must NOT be flagged.

        Hand calc for the fixture: 3rd has T=6 teams; ideal_gap(6) = 2;
        hard rule forbids gap <= 2. The fixture's tightest 3rd-grade
        repeat (gap=3) sits above the threshold, so the tester emits
        zero violations under the spec-008-aligned math.
        """
        violations = tester._check_equal_matchup_spacing()
        # Any remaining violations must be real (gap <= 2 = ideal_gap(6))
        # for a T=6 grade. Filter the messages to confirm none are at
        # gap=3 (the legacy false-positive band).
        for v in violations:
            # Message: "...gap of N rounds...". Pull the integer that
            # follows "gap of" — guard against trailing punctuation.
            import re
            m = re.search(r'gap of (\d+)', v.message)
            assert m, f"Unexpected message format: {v.message}"
            gap = int(m.group(1))
            assert gap <= 2, (
                f"Tester reported gap={gap} as a violation but the spec-008 "
                f"solver threshold for T=6 is gap<=2; got: {v.message}"
            )


# ============== Level 2: HIGH ==============


# spec-018: TestMaitlandAwayClubsLimit removed — the
# `_check_maitland_away_clubs_limit` tester method (and the away-clubs-per-week
# cap rule) were deleted.


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
    """Same-grade-same-club teams should not play at the same timeslot.

    spec-007 changed the rule: adjacent-grade concurrency is NO LONGER a
    violation (the convenor reported it was over-restrictive). Only the
    same-grade-same-club case survives. The fixture's known 9-violation
    count has shrunk to 2 -- the two University 6th duplicate-team
    appearances. The 7 previous adjacent-grade entries (Wests x6, Tigers
    x1) are gone by design.
    """

    def test_total_violation_count(self, tester):
        """Expected: 1 same-grade-same-club violation (down from 9 pre-spec-007).

        Empirically the fixture contains exactly one University 6th
        duplicate-team conflict (the original headline number of 2 turned
        out to be a single distinct slot reported via the tester's
        per-club-grade-slot iteration; same-grade-same-club only fires
        once per (date, slot, club, grade) bucket).
        """
        violations = tester._check_club_grade_adjacency()
        assert len(violations) == 1, (
            f"Expected 1 ClubGradeAdjacency violation after spec-007 "
            f"(adjacent-grade removed), got {len(violations)}: "
            + "; ".join(v.message for v in violations)
        )

    def test_no_adjacent_grade_violations(self, tester):
        """spec-007: adjacent-grade is not a violation any more."""
        violations = tester._check_club_grade_adjacency()
        adj = [v for v in violations if 'adjacent' in v.message]
        assert adj == [], (
            f"Expected zero adjacent-grade violations after spec-007, got "
            + "; ".join(v.message for v in adj)
        )

    def test_university_duplicate_team_violations(self, tester):
        """University has duplicate 6th teams at same slot (not playing each other).

        This is the surviving same-grade-same-club case. The fixture has two
        such events: the Gentlemen + Seapigs pair (and another offset).
        """
        violations = tester._check_club_grade_adjacency()
        uni = [v for v in violations if 'University' in v.message and 'duplicate' in v.message]
        assert len(uni) >= 1, (
            f"Expected at least one University duplicate-team violation, got: "
            + "; ".join(v.message for v in violations)
        )
        # The Gentlemen + Seapigs pair must appear at least once.
        msgs = "; ".join(v.message for v in uni)
        assert 'University Gentlemen 6th' in msgs
        assert 'University Seapigs 6th' in msgs

    def test_removing_university_duplicate_reduces_violations(self, fixture_draw, season_data):
        """Removing one University 6th game from the conflict slot drops one violation."""
        # Find the slot with two University 6th teams (not playing each other).
        target_slot = None
        for game in fixture_draw.games:
            if game.grade != '6th':
                continue
            if not any('University' in t for t in (game.team1, game.team2)):
                continue
            # Check sibling games in same date+slot.
            siblings = [
                g for g in fixture_draw.games
                if g.grade == '6th' and g.date == game.date
                and g.day_slot == game.day_slot and g.game_id != game.game_id
                and any('University' in t for t in (g.team1, g.team2))
            ]
            if siblings:
                # Make sure the two University teams aren't playing each other.
                uni_teams_in_slot = {
                    t for g2 in [game, *siblings]
                    for t in (g2.team1, g2.team2)
                    if 'University' in t
                }
                if len(uni_teams_in_slot) >= 2:
                    target_slot = (game, game.date, game.day_slot)
                    break

        assert target_slot is not None, (
            "Fixture should contain at least one University duplicate-6th "
            "conflict for spec-007 to flag"
        )
        bad_game = target_slot[0]
        modified_games = [g for g in fixture_draw.games if g.game_id != bad_game.game_id]

        draw = DrawStorage(
            description='Modified', num_weeks=fixture_draw.num_weeks,
            num_games=len(modified_games), games=modified_games,
        )
        t = DrawTester(draw, season_data)
        before = tester_baseline_count()  # type: ignore[name-defined]  -- declared below in module-level helper

        violations_after = t._check_club_grade_adjacency()
        assert len(violations_after) < before, (
            f"Removing one University 6th game should reduce ClubGradeAdjacency "
            f"violations below {before}; got {len(violations_after)}"
        )


def tester_baseline_count() -> int:
    """Baseline (pre-removal) ClubGradeAdjacency violations: hard-coded to 1.

    Recompute when the fixture is regenerated. Keeping this as a module-level
    helper rather than a fixture so it's plain to find when the count drifts.
    """
    return 1


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


# spec-031: TestClubFieldConcentration removed (_check_club_field_concentration deleted).

# ============== Level 4: LOW ==============


# spec-024: TestMaximiseClubsPerTimeslotBroadmeadow and
# TestMinimiseClubsOnAFieldBroadmeadow removed (constraints deleted).


class TestVenueEarliestSlotFill:
    """spec-021: venues fill earliest slots — no gaps, anchored to earliest."""

    def test_clean_pass(self, tester):
        violations = tester._check_venue_earliest_slot_fill()
        assert len(violations) == 0


class TestPreferredTimes:
    """Prefer team-specific preferred times."""

    def test_clean_pass(self, tester):
        violations = tester._check_preferred_times()
        assert len(violations) == 0


# ============== Cross-cutting: PHLSecondGrade ==============


class TestPHLSecondGradeAdjacency:
    """spec-014: same-club PHL/2nd must be back-to-back on one field at a
    single venue, OR (across venues) start >= 150-min apart (spec-030: 180->150)."""

    @staticmethod
    def _expected_violation_count(tester):
        """Independent re-derivation of the rule over the fixture games — the
        oracle the tester method is checked against (no shared code path)."""
        from collections import defaultdict
        cross_min = tester.data.get('constraint_defaults', {}).get(
            'phl_2nd_cross_venue_min_minutes', 150)

        def mins(t):
            h, m = t.split(':')
            return int(h) * 60 + int(m)

        club_games = defaultdict(list)
        for g in tester.draw.games:
            for team in (g.team1, g.team2):
                c = tester._team_to_club.get(team)
                if c:
                    club_games[(g.week, g.day, c, g.grade)].append(g)
        per_club = defaultdict(dict)
        for (week, day, club, grade), gs in club_games.items():
            if grade in ('PHL', '2nd'):
                per_club[(week, day, club)][grade] = gs

        count = 0
        for _key, gg in per_club.items():
            if 'PHL' not in gg or '2nd' not in gg:
                continue
            for p in gg['PHL']:
                for q in gg['2nd']:
                    if p.field_location == q.field_location:
                        ok = (p.field_name == q.field_name
                              and abs(p.day_slot - q.day_slot) == 1)
                    else:
                        ok = abs(mins(p.time) - mins(q.time)) >= cross_min
                    if not ok:
                        count += 1
        return count

    def test_matches_independent_recompute(self, tester):
        violations = tester._check_phl_2nd_adjacency()
        assert len(violations) == self._expected_violation_count(tester)

    def test_violation_severity(self, tester):
        violations = tester._check_phl_2nd_adjacency()
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
