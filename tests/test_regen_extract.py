# tests/test_regen_extract.py
"""
spec-026 Unit A — extract_locked_pairings.

Given/When/Then style, no mocks/patches, hand-computed oracles only.
DrawStorage/StoredGame instances are built synthetically — no draw file is
loaded from disk.

Hand oracle for the primary scenario:
    3-game draw:
        G00001  4th  week=2  date='2026-04-05'  team1='Norths 4th'  team2='Souths 4th'
        G00002  4th  week=2  date='2026-04-05'  team1='Easts 4th'   team2='Wests 4th'
        G00003  6th  week=2  date='2026-04-05'  team1='Norths 6th'  team2='Souths 6th'

    extract_locked_pairings(draw, freeze_grades={'4th'}, freeze_weeks={2},
                             exclude_weeks=frozenset())
    → 2 pins, both grade='4th', both date='2026-04-05'
    → keys of each pin == {'teams', 'grade', 'date', 'description'}
    → no 'time', 'day_slot', 'field_name', 'field_location' keys

    With exclude_weeks={2}: 0 pins.
    With freeze_grades=frozenset(): 0 pins.
    With freeze_grades={'4th'}, freeze_weeks={3}: 0 pins (week 3 not in draw).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.storage import DrawStorage, StoredGame, extract_locked_pairings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLOT_DEFAULTS = dict(
    day="Sunday",
    time="11:30",
    day_slot=2,
    field_name="EF",
    field_location="Newcastle International Hockey Centre",
)

_EXPECTED_KEYS = {"teams", "grade", "date", "description"}
_FORBIDDEN_KEYS = {"time", "day_slot", "field_name", "field_location"}


def _make_game(
    game_id: str,
    team1: str,
    team2: str,
    grade: str,
    week: int,
    date: str,
    round_no: int = 1,
    **overrides,
) -> StoredGame:
    """Build a minimal StoredGame with slot/field defaults."""
    kwargs = {**_SLOT_DEFAULTS, **overrides}
    return StoredGame(
        game_id=game_id,
        team1=team1,
        team2=team2,
        grade=grade,
        week=week,
        round_no=round_no,
        date=date,
        **kwargs,
    )


def _make_draw(*games: StoredGame) -> DrawStorage:
    """Wrap games in a DrawStorage."""
    return DrawStorage(
        description="synthetic regen-extract test draw",
        num_weeks=len({g.week for g in games}),
        num_games=len(games),
        games=list(games),
    )


# ---------------------------------------------------------------------------
# Fixtures — canonical 3-game draw (hand oracle)
# ---------------------------------------------------------------------------

@pytest.fixture()
def draw_3games() -> DrawStorage:
    """
    3-game draw:
      G00001  4th  week=2  2026-04-05  Norths 4th  vs Souths 4th
      G00002  4th  week=2  2026-04-05  Easts 4th   vs Wests 4th
      G00003  6th  week=2  2026-04-05  Norths 6th  vs Souths 6th
    """
    return _make_draw(
        _make_game("G00001", "Norths 4th", "Souths 4th", "4th", week=2, date="2026-04-05"),
        _make_game("G00002", "Easts 4th", "Wests 4th", "4th", week=2, date="2026-04-05"),
        _make_game("G00003", "Norths 6th", "Souths 6th", "6th", week=2, date="2026-04-05"),
    )


# ---------------------------------------------------------------------------
# Scenario 1 — primary oracle: 2 pins for grade='4th', week=2
# ---------------------------------------------------------------------------

class TestPrimaryOracle:
    def test_given_3game_draw_when_freeze_4th_week2_then_exactly_2_pins(self, draw_3games):
        # GIVEN the 3-game draw
        # WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN exactly 2 pins
        assert len(pins) == 2, f"expected 2 pins, got {len(pins)}: {pins}"

    def test_each_pin_has_exactly_the_required_keys(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN each pin has exactly {teams, grade, date, description}
        for pin in pins:
            assert set(pin) == _EXPECTED_KEYS, (
                f"pin has wrong keys: {set(pin)!r} (expected {_EXPECTED_KEYS!r})"
            )

    def test_no_slot_or_field_keys_present(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN none of the forbidden keys appear
        for pin in pins:
            for key in _FORBIDDEN_KEYS:
                assert key not in pin, f"forbidden key '{key}' found in pin: {pin}"

    def test_all_pins_have_grade_4th_and_correct_date(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN grade and date match the hand oracle
        for pin in pins:
            assert pin["grade"] == "4th", f"expected grade '4th', got {pin['grade']!r}"
            assert pin["date"] == "2026-04-05", f"expected date '2026-04-05', got {pin['date']!r}"

    def test_teams_field_is_a_list_of_two_strings(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN 'teams' is a list with exactly 2 non-empty strings
        for pin in pins:
            assert isinstance(pin["teams"], list), "teams must be a list"
            assert len(pin["teams"]) == 2, "teams must have exactly 2 elements"
            assert all(isinstance(t, str) and t for t in pin["teams"]), (
                "each team must be a non-empty string"
            )

    def test_teams_preserve_storedgame_alphabetical_order(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN team ordering is preserved from StoredGame (not re-sorted)
        # G00001: Norths 4th / Souths 4th — alphabetical by construction
        # G00002: Easts 4th / Wests 4th — alphabetical by construction
        team_pairs = {tuple(pin["teams"]) for pin in pins}
        assert ("Norths 4th", "Souths 4th") in team_pairs
        assert ("Easts 4th", "Wests 4th") in team_pairs

    def test_6th_grade_game_is_not_included(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN no pin for 6th grade
        for pin in pins:
            assert pin["grade"] != "6th", f"6th grade pin should not appear: {pin}"


# ---------------------------------------------------------------------------
# Scenario 2 — exclude_weeks removes the entire week → 0 pins
# ---------------------------------------------------------------------------

class TestExcludeWeeks:
    def test_given_3game_draw_when_exclude_weeks_includes_week2_then_0_pins(self, draw_3games):
        # GIVEN the 3-game draw
        # WHEN week=2 is excluded (it's a hard-locked played week)
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks={2},
        )
        # THEN 0 pins (double-pinning prevented)
        assert pins == [], f"expected [], got {pins}"

    def test_exclude_weeks_not_in_freeze_has_no_effect(self, draw_3games):
        # GIVEN / WHEN excluding week=1 (not in draw) doesn't remove the week-2 pins
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks={1},
        )
        assert len(pins) == 2, f"expected 2 pins, got {len(pins)}"


# ---------------------------------------------------------------------------
# Scenario 3 — bye/placeholder guard: team2='' produces no pin
# ---------------------------------------------------------------------------

class TestByeGuard:
    def test_bye_game_produces_no_pin(self):
        # GIVEN a draw with one bye (team2='') and one normal game, both matching grade/week
        bye_game = _make_game("G00001", "Norths 4th", "", "4th", week=2, date="2026-04-05")
        normal_game = _make_game("G00002", "Easts 4th", "Wests 4th", "4th", week=2, date="2026-04-05")
        draw = _make_draw(bye_game, normal_game)
        # WHEN
        pins = extract_locked_pairings(
            draw,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN only the normal game is pinned (bye game produces no pin)
        assert len(pins) == 1, f"expected 1 pin (bye skipped), got {len(pins)}: {pins}"
        assert pins[0]["teams"] == ["Easts 4th", "Wests 4th"]

    def test_draw_with_only_bye_produces_empty_list(self):
        # GIVEN a draw with only a bye game
        bye_game = _make_game("G00001", "Norths 4th", "", "4th", week=2, date="2026-04-05")
        draw = _make_draw(bye_game)
        # WHEN
        pins = extract_locked_pairings(
            draw,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN 0 pins
        assert pins == []

    def test_team1_empty_also_skipped(self):
        # GIVEN a game where team1 is empty (defensive guard)
        weird_game = _make_game("G00001", "", "Souths 4th", "4th", week=2, date="2026-04-05")
        draw = _make_draw(weird_game)
        # WHEN
        pins = extract_locked_pairings(
            draw,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN 0 pins
        assert pins == []


# ---------------------------------------------------------------------------
# Scenario 4 — empty freeze_grades → 0 pins (empty-set semantics)
# ---------------------------------------------------------------------------

class TestEmptyFreezeGrades:
    def test_empty_freeze_grades_returns_empty_list(self, draw_3games):
        # GIVEN the 3-game draw
        # WHEN freeze_grades is empty
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades=frozenset(),
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN 0 pins — nothing matches the grade axis
        assert pins == [], f"expected [], got {pins}"


# ---------------------------------------------------------------------------
# Scenario 5 — empty freeze_weeks → 0 pins
# ---------------------------------------------------------------------------

class TestEmptyFreezeWeeks:
    def test_empty_freeze_weeks_returns_empty_list(self, draw_3games):
        # GIVEN the 3-game draw
        # WHEN freeze_weeks is empty
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks=frozenset(),
            exclude_weeks=frozenset(),
        )
        # THEN 0 pins — nothing matches the week axis
        assert pins == [], f"expected [], got {pins}"


# ---------------------------------------------------------------------------
# Scenario 6 — grade frozen but different week → 0 pins
# ---------------------------------------------------------------------------

class TestGradeFrozenDifferentWeek:
    def test_freeze_week3_on_draw_with_only_week2_games_returns_empty(self, draw_3games):
        # GIVEN the 3-game draw (all week=2)
        # WHEN freeze_weeks={3} (no week-3 games exist in draw)
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={3},
            exclude_weeks=frozenset(),
        )
        # THEN 0 pins — grade matches but week doesn't
        assert pins == [], f"expected [], got {pins}"


# ---------------------------------------------------------------------------
# Scenario 7 — freeze both grades → 3 pins
# ---------------------------------------------------------------------------

class TestFreezeBothGrades:
    def test_freeze_4th_and_6th_returns_all_3_games(self, draw_3games):
        # GIVEN the 3-game draw
        # WHEN both grades frozen
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th", "6th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN 3 pins (all 3 games match)
        assert len(pins) == 3, f"expected 3 pins, got {len(pins)}: {pins}"
        for pin in pins:
            assert set(pin) == _EXPECTED_KEYS

    def test_freeze_both_grades_correct_grade_distribution(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th", "6th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        grades = [pin["grade"] for pin in pins]
        assert grades.count("4th") == 2
        assert grades.count("6th") == 1


# ---------------------------------------------------------------------------
# Scenario 8 — description field is a non-empty string
# ---------------------------------------------------------------------------

class TestDescriptionField:
    def test_description_is_non_empty_string(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN each pin has a non-empty description string
        for pin in pins:
            assert isinstance(pin["description"], str)
            assert len(pin["description"]) > 0, "description must be non-empty"

    def test_description_contains_grade_and_date(self, draw_3games):
        # GIVEN / WHEN
        pins = extract_locked_pairings(
            draw_3games,
            freeze_grades={"4th"},
            freeze_weeks={2},
            exclude_weeks=frozenset(),
        )
        # THEN description references the grade and date (convenience check)
        for pin in pins:
            assert "4th" in pin["description"], f"grade missing from description: {pin['description']}"
            assert "2026-04-05" in pin["description"], f"date missing from description: {pin['description']}"
