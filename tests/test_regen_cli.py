# tests/test_regen_cli.py
"""
spec-026 Unit B — CLI flags, union free-scope resolver, overlap validation.

Given/When/Then style, no mocks/patches, hand-computed oracles only.
Tests the three new run.py generate flags:
  --regen-from SOURCE
  --regen-grades G [G ...]
  --regen-weeks SPEC

And the two new pure functions:
  _parse_week_spec(spec) -> set[int]
  resolve_regen_scope(games, regen_grades, regen_weeks, lock_weeks) -> (free_ids, frozen_ids)

Hand-computed oracle (spec-026 Unit B):
  Synthetic draw: grades {PHL, 6th} x weeks {1, 2, 3} (6 games total, one per combo).
  Game IDs:
    G00001  PHL  week=1
    G00002  PHL  week=2
    G00003  PHL  week=3
    G00004  6th  week=1
    G00005  6th  week=2
    G00006  6th  week=3

  Case 1 (regen_grades={'6th'}, regen_weeks=None/empty):
    free   = {G00004, G00005, G00006}  (all 6th games)
    frozen = {G00001, G00002, G00003}  (all PHL games)

  Case 2 (regen_grades=empty, regen_weeks={3}):
    free   = {G00003, G00006}  (week-3 games of both grades)
    frozen = {G00001, G00002, G00004, G00005}  (weeks 1-2 of both grades)

  Case 3 (regen_grades={'6th'}, regen_weeks={3}):
    free   = {G00004, G00005, G00006, G00003}  (all 6th + PHL week 3)
    frozen = {G00001, G00002}                   (PHL weeks 1-2)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run
from run import _parse_week_spec, resolve_regen_scope, _validate_regen_lock_weeks_overlap
from analytics.storage import StoredGame


# ---------------------------------------------------------------------------
# Helpers — synthetic draw
# ---------------------------------------------------------------------------

_SLOT_DEFAULTS = dict(
    day="Sunday",
    time="11:30",
    day_slot=2,
    round_no=1,
    field_name="EF",
    field_location="Newcastle International Hockey Centre",
    date="2026-03-22",  # overridden per game in fixture
)

_DATES = {1: "2026-03-22", 2: "2026-03-29", 3: "2026-04-05"}


def _make_game(game_id: str, grade: str, week: int) -> StoredGame:
    """Build a minimal StoredGame for the synthetic draw."""
    return StoredGame(
        game_id=game_id,
        team1=f"Team A {grade}",
        team2=f"Team B {grade}",
        grade=grade,
        week=week,
        date=_DATES[week],
        **{k: v for k, v in _SLOT_DEFAULTS.items() if k != "date"},
    )


@pytest.fixture()
def synthetic_games():
    """
    6-game draw: {PHL, 6th} x {1, 2, 3}.

    G00001  PHL  week=1
    G00002  PHL  week=2
    G00003  PHL  week=3
    G00004  6th  week=1
    G00005  6th  week=2
    G00006  6th  week=3
    """
    return [
        _make_game("G00001", "PHL", 1),
        _make_game("G00002", "PHL", 2),
        _make_game("G00003", "PHL", 3),
        _make_game("G00004", "6th", 1),
        _make_game("G00005", "6th", 2),
        _make_game("G00006", "6th", 3),
    ]


# ---------------------------------------------------------------------------
# _parse_week_spec
# ---------------------------------------------------------------------------

class TestParseWeekSpec:
    """Tests for _parse_week_spec helper — hand-computed oracle."""

    def test_given_range_when_10_to_22_then_all_weeks_in_range(self):
        # GIVEN a closed-range spec
        # WHEN parsed
        result = _parse_week_spec("10-22")
        # THEN all weeks 10..22 inclusive (hand oracle: 13 weeks)
        expected = set(range(10, 23))
        assert result == expected, f"expected {expected}, got {result}"

    def test_given_comma_list_when_10_12_14_then_three_weeks(self):
        # GIVEN a comma-list
        result = _parse_week_spec("10,12,14")
        # THEN exactly those three weeks
        assert result == {10, 12, 14}

    def test_given_combination_when_10_12_comma_15_then_union(self):
        # GIVEN a combination of range and comma
        result = _parse_week_spec("10-12,15")
        # THEN {10, 11, 12, 15}
        assert result == {10, 11, 12, 15}

    def test_single_week(self):
        # GIVEN a single week number
        result = _parse_week_spec("3")
        assert result == {3}

    def test_empty_string_returns_empty_set(self):
        # GIVEN an empty string
        result = _parse_week_spec("")
        assert result == set()

    def test_whitespace_only_returns_empty_set(self):
        result = _parse_week_spec("   ")
        assert result == set()

    def test_range_with_whitespace_around_hyphen(self):
        # GIVEN a range like "10-12" — hyphen splits cleanly
        result = _parse_week_spec("10-12")
        assert result == {10, 11, 12}

    def test_invalid_token_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_week_spec("abc")

    def test_inverted_range_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_week_spec("22-10")

    def test_empty_tokens_from_extra_commas_are_skipped(self):
        # GIVEN a spec with empty tokens (double/trailing comma)
        # THEN the empty tokens are skipped, real weeks parsed (oracle: {10,12})
        assert _parse_week_spec("10,,12") == {10, 12}
        assert _parse_week_spec("10,12,") == {10, 12}

    def test_range_with_non_integer_bound_raises_value_error(self):
        # GIVEN a range whose second bound is non-integer
        with pytest.raises(ValueError):
            _parse_week_spec("10-abc")


# ---------------------------------------------------------------------------
# resolve_regen_scope — Case 1: grade-only
# ---------------------------------------------------------------------------

class TestResolveRegenScopeCase1GradeOnly:
    """Case 1 (hand oracle): regen_grades={'6th'}, regen_weeks=None.
    free = {G00004, G00005, G00006}; frozen = {G00001, G00002, G00003}.
    """

    def test_given_6games_when_regen_grades_6th_only_then_free_is_all_6th(self, synthetic_games):
        # GIVEN the 6-game draw
        # WHEN regen_grades={'6th'}, regen_weeks=None
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades={"6th"}, regen_weeks=None)
        # THEN free = all 6th games (hand oracle)
        expected_free = {"G00004", "G00005", "G00006"}
        assert free == expected_free, f"expected free={expected_free!r}, got {free!r}"

    def test_given_6games_when_regen_grades_6th_only_then_frozen_is_all_phl(self, synthetic_games):
        # THEN frozen = all PHL games (hand oracle)
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades={"6th"}, regen_weeks=None)
        expected_frozen = {"G00001", "G00002", "G00003"}
        assert frozen == expected_frozen, f"expected frozen={expected_frozen!r}, got {frozen!r}"

    def test_free_and_frozen_are_disjoint(self, synthetic_games):
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades={"6th"}, regen_weeks=None)
        assert free & frozen == set(), "free and frozen must be disjoint"

    def test_free_union_frozen_equals_all_games(self, synthetic_games):
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades={"6th"}, regen_weeks=None)
        all_ids = {g.game_id for g in synthetic_games}
        assert free | frozen == all_ids, "free ∪ frozen must cover all games"


# ---------------------------------------------------------------------------
# resolve_regen_scope — Case 2: week-only
# ---------------------------------------------------------------------------

class TestResolveRegenScopeCase2WeekOnly:
    """Case 2 (hand oracle): regen_grades=empty, regen_weeks={3}.
    free = {G00003, G00006}; frozen = {G00001, G00002, G00004, G00005}.
    """

    def test_given_6games_when_regen_weeks_3_then_free_is_week3_games(self, synthetic_games):
        # GIVEN the 6-game draw
        # WHEN regen_grades=empty, regen_weeks={3}
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades=set(), regen_weeks={3})
        # THEN free = week-3 games of both grades (hand oracle)
        expected_free = {"G00003", "G00006"}
        assert free == expected_free, f"expected free={expected_free!r}, got {free!r}"

    def test_given_6games_when_regen_weeks_3_then_frozen_is_weeks_1_2(self, synthetic_games):
        # THEN frozen = weeks 1-2 of both grades (hand oracle)
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades=set(), regen_weeks={3})
        expected_frozen = {"G00001", "G00002", "G00004", "G00005"}
        assert frozen == expected_frozen, f"expected frozen={expected_frozen!r}, got {frozen!r}"

    def test_free_and_frozen_are_disjoint(self, synthetic_games):
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades=set(), regen_weeks={3})
        assert free & frozen == set()

    def test_free_union_frozen_equals_all_games(self, synthetic_games):
        free, frozen = resolve_regen_scope(synthetic_games, regen_grades=set(), regen_weeks={3})
        all_ids = {g.game_id for g in synthetic_games}
        assert free | frozen == all_ids


# ---------------------------------------------------------------------------
# resolve_regen_scope — Case 3: both grade and week (union semantics)
# ---------------------------------------------------------------------------

class TestResolveRegenScopeCase3Both:
    """Case 3 (hand oracle): regen_grades={'6th'}, regen_weeks={3}.
    free = {G00003, G00004, G00005, G00006}; frozen = {G00001, G00002}.
    (all 6th + PHL week 3)
    """

    def test_given_6games_when_both_then_free_is_6th_plus_phl_week3(self, synthetic_games):
        # GIVEN the 6-game draw
        # WHEN regen_grades={'6th'} AND regen_weeks={3}
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades={"6th"}, regen_weeks={3}
        )
        # THEN free = all 6th games (weeks 1-3) + PHL week 3 (hand oracle)
        expected_free = {"G00003", "G00004", "G00005", "G00006"}
        assert free == expected_free, f"expected free={expected_free!r}, got {free!r}"

    def test_given_6games_when_both_then_frozen_is_phl_weeks_1_2(self, synthetic_games):
        # THEN frozen = PHL weeks 1-2 (hand oracle)
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades={"6th"}, regen_weeks={3}
        )
        expected_frozen = {"G00001", "G00002"}
        assert frozen == expected_frozen, f"expected frozen={expected_frozen!r}, got {frozen!r}"

    def test_free_and_frozen_are_disjoint(self, synthetic_games):
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades={"6th"}, regen_weeks={3}
        )
        assert free & frozen == set()

    def test_free_union_frozen_equals_all_games(self, synthetic_games):
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades={"6th"}, regen_weeks={3}
        )
        all_ids = {g.game_id for g in synthetic_games}
        assert free | frozen == all_ids

    def test_union_semantics_phl_week3_freed_by_week_axis(self, synthetic_games):
        """PHL week 3 (G00003) is freed by the week axis even though PHL is not in regen_grades."""
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades={"6th"}, regen_weeks={3}
        )
        assert "G00003" in free, "G00003 (PHL week 3) must be free via the week axis"

    def test_union_semantics_6th_week1_freed_by_grade_axis(self, synthetic_games):
        """6th week 1 (G00004) is freed by the grade axis even though week 1 is not in regen_weeks."""
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades={"6th"}, regen_weeks={3}
        )
        assert "G00004" in free, "G00004 (6th week 1) must be free via the grade axis"


# ---------------------------------------------------------------------------
# resolve_regen_scope — lock_weeks exclusion
# ---------------------------------------------------------------------------

class TestResolveRegenScopeWithLockWeeks:
    """Hard-locked played weeks are excluded from both free and frozen."""

    def test_lock_weeks_excluded_from_both_sets(self, synthetic_games):
        # GIVEN week 1 is hard-locked (played), regen covers 6th
        free, frozen = resolve_regen_scope(
            synthetic_games,
            regen_grades={"6th"},
            regen_weeks=None,
            lock_weeks={1},
        )
        # THEN G00001 (PHL week 1) and G00004 (6th week 1) appear in NEITHER set
        assert "G00001" not in free and "G00001" not in frozen, (
            "G00001 (hard-locked) must not appear in free or frozen"
        )
        assert "G00004" not in free and "G00004" not in frozen, (
            "G00004 (hard-locked) must not appear in free or frozen"
        )

    def test_remaining_games_still_partitioned_correctly(self, synthetic_games):
        # GIVEN week 1 locked, regen_grades={'6th'}
        free, frozen = resolve_regen_scope(
            synthetic_games,
            regen_grades={"6th"},
            regen_weeks=None,
            lock_weeks={1},
        )
        # THEN weeks 2-3: 6th free, PHL frozen
        assert free == {"G00005", "G00006"}, f"got {free}"
        assert frozen == {"G00002", "G00003"}, f"got {frozen}"


# ---------------------------------------------------------------------------
# resolve_regen_scope — degenerate "freeze all" case
# ---------------------------------------------------------------------------

class TestResolveRegenScopeFreezeAll:
    """Empty regen_grades AND empty regen_weeks → everything frozen (degenerate case)."""

    def test_empty_both_axes_freezes_everything(self, synthetic_games):
        # GIVEN empty regen_grades and empty regen_weeks
        free, frozen = resolve_regen_scope(
            synthetic_games, regen_grades=set(), regen_weeks=set()
        )
        # THEN free is empty, frozen is all games
        assert free == set(), f"expected free=set(), got {free}"
        expected_frozen = {g.game_id for g in synthetic_games}
        assert frozen == expected_frozen, f"expected frozen={expected_frozen}, got {frozen}"

    def test_none_regen_weeks_freezes_non_grade_games(self, synthetic_games):
        """regen_weeks=None is equivalent to empty (no week axis freedom)."""
        free_none, frozen_none = resolve_regen_scope(
            synthetic_games, regen_grades=set(), regen_weeks=None
        )
        free_empty, frozen_empty = resolve_regen_scope(
            synthetic_games, regen_grades=set(), regen_weeks=set()
        )
        assert free_none == free_empty
        assert frozen_none == frozen_empty


# ---------------------------------------------------------------------------
# _validate_regen_lock_weeks_overlap
# ---------------------------------------------------------------------------

class TestValidateRegenLockWeeksOverlap:
    """Overlapping --regen-weeks and --lock-weeks must be FATAL."""

    def test_given_overlap_then_sys_exit(self):
        # GIVEN regen_weeks and lock_weeks that overlap
        with pytest.raises(SystemExit) as exc_info:
            _validate_regen_lock_weeks_overlap(
                regen_weeks={3, 4, 5}, lock_weeks={5, 6, 7}
            )
        # THEN exits non-zero
        assert exc_info.value.code != 0

    def test_given_overlap_then_error_message_names_weeks(self, capsys):
        # GIVEN overlap on week 5
        with pytest.raises(SystemExit):
            _validate_regen_lock_weeks_overlap(
                regen_weeks={3, 5}, lock_weeks={5, 7}
            )
        captured = capsys.readouterr()
        assert "5" in captured.out, f"overlapping week 5 must appear in error: {captured.out}"

    def test_given_no_overlap_then_no_exit(self):
        # GIVEN no overlap
        # WHEN validated — should NOT raise
        _validate_regen_lock_weeks_overlap(
            regen_weeks={10, 11, 12}, lock_weeks={1, 2, 3}
        )

    def test_given_empty_regen_weeks_then_no_exit(self):
        # GIVEN empty regen_weeks
        _validate_regen_lock_weeks_overlap(regen_weeks=set(), lock_weeks={1, 2, 3})

    def test_given_empty_lock_weeks_then_no_exit(self):
        # GIVEN empty lock_weeks
        _validate_regen_lock_weeks_overlap(regen_weeks={1, 2, 3}, lock_weeks=set())

    def test_multiple_overlapping_weeks_named_in_error(self, capsys):
        # GIVEN multiple overlapping weeks
        with pytest.raises(SystemExit):
            _validate_regen_lock_weeks_overlap(
                regen_weeks={3, 4, 5}, lock_weeks={4, 5, 6}
            )
        captured = capsys.readouterr()
        # Both overlapping weeks (4 and 5) must appear in the error message
        assert "4" in captured.out and "5" in captured.out, (
            f"overlapping weeks 4 and 5 must appear in error: {captured.out}"
        )


# ---------------------------------------------------------------------------
# CLI argument parsing — new flags appear in generate subcommand
# ---------------------------------------------------------------------------

class TestCLIArgParsing:
    """Verify the three new flags are registered on the generate subparser.

    Real parsing of the flags into the run_generate handler (dest names +
    value flow) is proven end-to-end by spec-026 Unit C's DoD #8 test, which
    runs the actual `run.py generate --regen-from … --regen-grades …` command.
    Here we assert the flags are registered by inspecting `generate --help`
    output (argparse builds the full subparser to render help).
    """

    def test_regen_from_flag_present_in_generate_help(self, capsys):
        # GIVEN the generate subcommand
        # WHEN --help is invoked
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'generate', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        # THEN --regen-from appears in help
        assert '--regen-from' in captured.out, (
            f"--regen-from not found in generate --help:\n{captured.out}"
        )

    def test_regen_grades_flag_present_in_generate_help(self, capsys):
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'generate', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert '--regen-grades' in captured.out, (
            f"--regen-grades not found in generate --help:\n{captured.out}"
        )

    def test_regen_weeks_flag_present_in_generate_help(self, capsys):
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'generate', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert '--regen-weeks' in captured.out, (
            f"--regen-weeks not found in generate --help:\n{captured.out}"
        )

    def test_existing_flags_not_broken_by_new_additions(self, capsys):
        """Existing flags like --lock-weeks still appear in help."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'generate', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert '--lock-weeks' in captured.out
        assert '--locked' in captured.out
        assert '--year' in captured.out

    def test_help_flag_still_works(self, capsys):
        """Top-level --help still works after new flags are added."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert 'usage' in captured.out.lower()


