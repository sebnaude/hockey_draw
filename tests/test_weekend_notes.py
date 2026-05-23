# tests/test_weekend_notes.py
"""
No-mock tests for analytics/notes.py :: build_weekend_notes().

All six scenarios from the spec-028 Unit A test outline, with hand-computed
oracles. DrawStorage instances are built synthetically via StoredGame — no
draw file is loaded from disk.

Given/When/Then naming convention used throughout.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.notes import build_weekend_notes
from analytics.storage import DrawStorage, StoredGame


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_game(date: str, week: int, game_id: str = "G00001") -> StoredGame:
    """Return a minimal StoredGame with the given date and week."""
    return StoredGame(
        game_id=game_id,
        team1="Norths PHL",
        team2="Wests PHL",
        grade="PHL",
        week=week,
        round_no=week,
        date=date,
        day="Sunday",
        time="11:30",
        day_slot=2,
        field_name="EF",
        field_location="Newcastle International Hockey Centre",
    )


def _make_draw(*games: StoredGame) -> DrawStorage:
    """Return a DrawStorage containing the supplied games."""
    return DrawStorage(
        description="synthetic test draw",
        num_weeks=len({g.week for g in games}),
        num_games=len(games),
        games=list(games),
    )


def _write_notes_json(tmp_dir: str, content: dict) -> str:
    """Write *content* as JSON to a temp file and return its path."""
    path = os.path.join(tmp_dir, "notes.json")
    Path(path).write_text(json.dumps(content), encoding="utf-8")
    return path


def _minimal_data(
    blocked_games=None,
    forced_games=None,
    preferred_weekends=None,
) -> dict:
    """Minimal data dict sufficient for build_weekend_notes()."""
    return {
        "year": 2026,
        "blocked_games": blocked_games or [],
        "forced_games": forced_games or [],
        "preferred_weekends": preferred_weekends or [],
    }


# ---------------------------------------------------------------------------
# Scenario 1 — hand-authored notes.json maps date to week via draw map
# ---------------------------------------------------------------------------

class TestScenario1_HandAuthoredNotesMappedViaDrawMap:
    """
    Given a synthetic DrawStorage with one StoredGame on date '2026-05-17' at
    week=9, and a notes.json containing {"2026-05-17": [{"category": "Field",
    "text": "Test note"}]}, when build_weekend_notes() is called, then
    result[9] == ["Field: Test note"].

    Oracle: the synthetic game explicitly maps 2026-05-17 → week 9 in the draw
    map, so no bucketing is needed.
    """

    def test_given_draw_game_on_date_when_notes_json_has_same_date_then_week_is_from_draw_map(self, tmp_path):
        # Given
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()
        notes_content = {
            "2026-05-17": [{"category": "Field", "text": "Test note"}]
        }
        notes_path = _write_notes_json(str(tmp_path), notes_content)

        # When
        result = build_weekend_notes(draw, data, notes_path=notes_path)

        # Then
        assert 9 in result, f"Expected week 9 in result, got keys: {list(result.keys())}"
        assert result[9] == ["Field: Test note"]


# ---------------------------------------------------------------------------
# Scenario 2 — dedup + opt-in: only the one 'note'-flagged entry surfaces,
#              and only once despite 10 sibling entries on the same date
# ---------------------------------------------------------------------------

class TestScenario2_DedupAndOptIn:
    """
    Given a data dict whose blocked_games contain:
      - ONE entry {'date': '2026-05-17', 'description': 'Masters SC weekend',
                   'note': True}
      - NINE sibling entries on the same date with the same description but NO
        'note' key
    and a synthetic DrawStorage with a game on '2026-05-17' at week=9,
    when build_weekend_notes() is called with no notes.json,
    then result[9] has exactly one "Blocked: Masters SC weekend" line.

    Oracle:
      - 2026-05-17 is 56 days after season start 2026-03-22; bucket = 56//7+1 = 9.
      - Only the entry with 'note': True is surfaced.
      - Dedup reduces identical formatted strings to one.
      - len(result[9]) == 1.
    """

    def test_given_one_noted_and_nine_unnoted_entries_when_built_then_exactly_one_blocked_line(self, tmp_path):
        # Given: one opted-in blocked entry + 9 siblings without 'note'
        description = "Masters SC weekend"
        opted_in_entry = {
            "date": "2026-05-17",
            "description": description,
            "note": True,
        }
        silent_siblings = [
            {"date": "2026-05-17", "description": description}
            for _ in range(9)
        ]
        blocked_games = [opted_in_entry] + silent_siblings

        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data(blocked_games=blocked_games)
        # No notes.json
        notes_path = _write_notes_json(str(tmp_path), {})

        # When
        result = build_weekend_notes(draw, data, notes_path=notes_path)

        # Then
        assert 9 in result, f"Expected week 9 in result, got keys: {list(result.keys())}"
        assert result[9] == ["Blocked: Masters SC weekend"], (
            f"Expected exactly one line, got: {result[9]}"
        )
        assert len(result[9]) == 1

    def test_entry_without_note_key_never_surfaces(self, tmp_path):
        """An entry with no 'note' key at all should not appear in any week."""
        blocked_games = [
            {"date": "2026-05-17", "description": "Should be invisible"},
        ]
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data(blocked_games=blocked_games)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        for week, lines in result.items():
            for line in lines:
                assert "Should be invisible" not in line

    def test_note_true_falls_back_to_reason_when_no_description(self, tmp_path):
        """'note': True with no description but a reason uses reason as text."""
        blocked_games = [
            {
                "date": "2026-05-17",
                "reason": "Fallback reason text",
                "note": True,
            }
        ]
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data(blocked_games=blocked_games)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 9 in result
        assert "Blocked: Fallback reason text" in result[9]


# ---------------------------------------------------------------------------
# Scenario 3 — un-dateable entry (only 'day', no 'date') never surfaces
# ---------------------------------------------------------------------------

class TestScenario3_UndateableEntryAbsent:
    """
    Given a FORCED_GAMES-style entry {'day': 'Friday', 'note': True,
    'description': 'solver-picks-date'} (no 'date' key),
    when build_weekend_notes() is called,
    then the formatted string "Forced: solver-picks-date" is absent from all
    weeks in the result.

    Oracle: entries with only 'day' and no 'date' are skipped per spec — the
    solver picks the date, so it cannot be placed on a specific weekend.
    """

    def test_given_forced_entry_with_day_only_when_built_then_absent_from_all_weeks(self, tmp_path):
        # Given
        forced_games = [
            {
                "day": "Friday",
                "grade": "PHL",
                "description": "solver-picks-date",
                "note": True,
            }
        ]
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data(forced_games=forced_games)
        notes_path = _write_notes_json(str(tmp_path), {})

        # When
        result = build_weekend_notes(draw, data, notes_path=notes_path)

        # Then
        all_lines = [line for lines in result.values() for line in lines]
        assert "Forced: solver-picks-date" not in all_lines, (
            f"Un-dateable entry should not appear in any week. Got: {all_lines}"
        )

    def test_entry_with_both_date_and_day_is_resolvable(self, tmp_path):
        """Entry with BOTH 'date' and 'day' is resolved via 'date' (not skipped)."""
        forced_games = [
            {
                "date": "2026-05-17",
                "day": "Sunday",
                "grade": "PHL",
                "description": "Has both date and day",
                "note": True,
            }
        ]
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data(forced_games=forced_games)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 9 in result
        assert "Forced: Has both date and day" in result[9]


# ---------------------------------------------------------------------------
# Scenario 4 — preferred_weekends entry with 'note': True and a game in draw
# ---------------------------------------------------------------------------

class TestScenario4_PreferredWeekendNoteSurfaces:
    """
    Given a preferred_weekends entry {'date': '2026-06-12', 'note': True,
    'description': 'Norths home — 80th Anniversary', 'mode': 'avoid'} and a
    synthetic DrawStorage with a game on '2026-06-12' at week=12,
    when build_weekend_notes() is called,
    then result[12] contains "Preferred: Norths home — 80th Anniversary".

    Oracle:
      - 2026-06-12 is 82 days after 2026-03-22; bucket = 82//7+1 = 12.
      - Game-based resolution also yields week 12 (identical result).
    """

    def test_given_preferred_entry_with_note_and_game_on_same_date_then_week12_has_preferred_line(self, tmp_path):
        # Given
        preferred_weekends = [
            {
                "date": "2026-06-12",
                "mode": "avoid",
                "description": "Norths home — 80th Anniversary",
                "note": True,
            }
        ]
        draw = _make_draw(_make_game("2026-06-12", week=12))
        data = _minimal_data(preferred_weekends=preferred_weekends)
        notes_path = _write_notes_json(str(tmp_path), {})

        # When
        result = build_weekend_notes(draw, data, notes_path=notes_path)

        # Then
        assert 12 in result, f"Expected week 12 in result, got keys: {list(result.keys())}"
        assert "Preferred: Norths home — 80th Anniversary" in result[12]

    def test_preferred_entry_note_string_uses_verbatim_text(self, tmp_path):
        """'note': '<string>' uses that string verbatim, not description."""
        preferred_weekends = [
            {
                "date": "2026-06-12",
                "mode": "avoid",
                "description": "Should not appear",
                "note": "Custom verbatim note text",
            }
        ]
        draw = _make_draw(_make_game("2026-06-12", week=12))
        data = _minimal_data(preferred_weekends=preferred_weekends)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 12 in result
        assert "Preferred: Custom verbatim note text" in result[12]
        all_lines = [line for lines in result.values() for line in lines]
        assert "Preferred: Should not appear" not in all_lines

    def test_preferred_entry_dates_plural_list_supported(self, tmp_path):
        """'dates': [...] plural list also produces notes per date."""
        preferred_weekends = [
            {
                "dates": ["2026-05-17", "2026-06-12"],
                "mode": "avoid",
                "description": "Multi-date preferred note",
                "note": True,
            }
        ]
        draw = _make_draw(
            _make_game("2026-05-17", week=9, game_id="G00001"),
            _make_game("2026-06-12", week=12, game_id="G00002"),
        )
        data = _minimal_data(preferred_weekends=preferred_weekends)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 9 in result
        assert "Preferred: Multi-date preferred note" in result[9]
        assert 12 in result
        assert "Preferred: Multi-date preferred note" in result[12]


# ---------------------------------------------------------------------------
# Scenario 5 — malformed JSON raises ValueError naming the file
# ---------------------------------------------------------------------------

class TestScenario5_MalformedJsonRaisesValueError:
    """
    Given a notes_path pointing at a file containing invalid JSON,
    when build_weekend_notes() is called,
    then a ValueError is raised whose message names the file path.
    """

    def test_given_malformed_json_when_built_then_value_error_with_filename(self, tmp_path):
        # Given
        bad_json_path = os.path.join(str(tmp_path), "notes.json")
        Path(bad_json_path).write_text("{this is not valid json", encoding="utf-8")

        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        # When / Then
        with pytest.raises(ValueError) as exc_info:
            build_weekend_notes(draw, data, notes_path=bad_json_path)

        error_message = str(exc_info.value)
        assert "notes.json" in error_message or bad_json_path in error_message, (
            f"ValueError should mention the file path. Got: {error_message!r}"
        )

    def test_given_json_array_at_root_when_built_then_value_error(self, tmp_path):
        """notes.json with a JSON array (not object) at root raises ValueError."""
        array_json_path = os.path.join(str(tmp_path), "notes.json")
        Path(array_json_path).write_text("[]", encoding="utf-8")

        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        with pytest.raises(ValueError):
            build_weekend_notes(draw, data, notes_path=array_json_path)


# ---------------------------------------------------------------------------
# Scenario 6 — note dated outside the season window is dropped, no crash
# ---------------------------------------------------------------------------

class TestScenario6_OffWindowNotesDropped:
    """
    Given a notes.json with a date far before the season start ('2026-01-01'),
    when build_weekend_notes() is called,
    then the note is absent from all weeks in the result and no exception is
    raised.

    Oracle:
      - 2026-01-01 is 80 days before 2026-03-22; bucket = (-80)//7+1 = -11.
      - Week -11 is below the season floor (week 1) → dropped.
    """

    def test_given_pre_season_date_when_built_then_note_dropped_silently(self, tmp_path):
        # Given
        notes_content = {
            "2026-01-01": [{"category": "Field", "text": "Pre-season note to drop"}]
        }
        notes_path = _write_notes_json(str(tmp_path), notes_content)

        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        # When — should not raise
        result = build_weekend_notes(draw, data, notes_path=notes_path)

        # Then — the off-window note is absent from all weeks
        all_lines = [line for lines in result.values() for line in lines]
        assert "Field: Pre-season note to drop" not in all_lines, (
            f"Off-window note should be dropped. Got all_lines: {all_lines}"
        )

    def test_given_post_season_date_when_built_then_note_dropped(self, tmp_path):
        """A date after end_date is also dropped."""
        notes_content = {
            "2027-01-01": [{"category": "Field", "text": "Post-season note"}]
        }
        notes_path = _write_notes_json(str(tmp_path), notes_content)

        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        all_lines = [line for lines in result.values() for line in lines]
        assert "Field: Post-season note" not in all_lines


# ---------------------------------------------------------------------------
# Additional behavioural tests covering spec requirements
# ---------------------------------------------------------------------------

class TestCategoryOrdering:
    """Notes within a week are ordered Field < Request < Preferred < Blocked < Forced."""

    def test_category_order_is_stable(self, tmp_path):
        # Given: one note per category, all on the same date/week
        notes_content = {
            "2026-05-17": [
                {"category": "Request", "text": "R note"},
                {"category": "Field", "text": "F note"},
            ]
        }
        notes_path = _write_notes_json(str(tmp_path), notes_content)

        blocked_games = [{"date": "2026-05-17", "description": "B note", "note": True}]
        forced_games = [{"date": "2026-05-17", "description": "Fo note", "note": True}]
        preferred_weekends = [
            {"date": "2026-05-17", "description": "P note", "note": True}
        ]

        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data(
            blocked_games=blocked_games,
            forced_games=forced_games,
            preferred_weekends=preferred_weekends,
        )

        # When
        result = build_weekend_notes(draw, data, notes_path=notes_path)

        # Then: order must be Field, Request, Preferred, Blocked, Forced
        assert 9 in result
        lines = result[9]
        categories = [line.split(":")[0] for line in lines]
        expected_order = ["Field", "Request", "Preferred", "Blocked", "Forced"]
        assert categories == expected_order, (
            f"Expected category order {expected_order}, got {categories}"
        )


class TestMissingNotesJson:
    """A missing notes.json file is treated as empty — no error."""

    def test_given_nonexistent_notes_path_then_no_error(self, tmp_path):
        nonexistent = os.path.join(str(tmp_path), "does_not_exist.json")
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        # Should not raise
        result = build_weekend_notes(draw, data, notes_path=nonexistent)
        # Result may be empty (no entries) — that's fine
        assert isinstance(result, dict)


class TestCommentKeySkipped:
    """A leading _comment key in notes.json is silently skipped."""

    def test_comment_key_is_skipped_gracefully(self, tmp_path):
        notes_content = {
            "_comment": "Schema description — ignore this key",
            "2026-05-17": [{"category": "Field", "text": "Real note"}],
        }
        notes_path = _write_notes_json(str(tmp_path), notes_content)
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 9 in result
        assert "Field: Real note" in result[9]


class TestNoteStringShorthand:
    """Bare strings in notes.json entries get category 'Note'."""

    def test_bare_string_gets_note_category(self, tmp_path):
        notes_content = {
            "2026-05-17": ["plain string note"],
        }
        notes_path = _write_notes_json(str(tmp_path), notes_content)
        draw = _make_draw(_make_game("2026-05-17", week=9))
        data = _minimal_data()

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 9 in result
        assert "Note: plain string note" in result[9]


class TestOffDrawDateBucketingOracle:
    """Off-draw dates (no game in draw) resolve via 7-day bucket from season start."""

    def test_blocked_date_not_in_draw_resolved_by_bucketing(self, tmp_path):
        """
        Date 2026-05-17 has no game in draw. Bucket: (56 days) // 7 + 1 = 9.
        The entry should appear in week 9.
        """
        # Draw has a game on a DIFFERENT date so 2026-05-17 is off-draw
        draw = _make_draw(_make_game("2026-04-05", week=3))
        blocked_games = [
            {"date": "2026-05-17", "description": "Off-draw blocked", "note": True}
        ]
        data = _minimal_data(blocked_games=blocked_games)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 9 in result, f"Expected week 9 from bucketing 2026-05-17, got keys: {list(result.keys())}"
        assert "Blocked: Off-draw blocked" in result[9]

    def test_preferred_date_not_in_draw_resolved_by_bucketing(self, tmp_path):
        """
        Date 2026-06-12 has no game in draw. Bucket: (82 days) // 7 + 1 = 12.
        """
        draw = _make_draw(_make_game("2026-04-05", week=3))
        preferred_weekends = [
            {"date": "2026-06-12", "description": "Norths 80th Anniversary", "note": True}
        ]
        data = _minimal_data(preferred_weekends=preferred_weekends)
        notes_path = _write_notes_json(str(tmp_path), {})

        result = build_weekend_notes(draw, data, notes_path=notes_path)

        assert 12 in result, f"Expected week 12 from bucketing 2026-06-12, got keys: {list(result.keys())}"
        assert "Preferred: Norths 80th Anniversary" in result[12]
