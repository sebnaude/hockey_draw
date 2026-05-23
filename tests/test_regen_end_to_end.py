# tests/test_regen_end_to_end.py
"""
spec-026 Unit C — orchestration wiring, dedup, metadata, versioning.

Given/When/Then style, NO mocks/patches, hand-computed oracles only.

Tested at the right seams with REAL objects:
  - Frozen-set computation + pin extraction (run.py functions) up to the
    solver call (DoD #4/#7 setup).
  - Concatenation + dedup merge helper (main_staged._merge_regen_pins) (DoD #3).
  - Metadata `regen` block via the real _build_draw_metadata (DoD #6),
    including the games_changed diff helper.
  - Versioning MAJOR bump via real save_solver_output against a tmp draws dir
    (DoD #5).
  - Overlap + missing-week validation (FATAL) (Unit B review L3).

NOT covered here (delegated to post-merge DoD #8 verification):
  - "solver runs to feasible + frozen pairings preserved in OUTPUT" — a full
    real CP-SAT solve takes hours and cannot run in pytest. The orchestrator
    runs `run.py generate --regen-from ... --regen-grades 6th --year 2026
    --simple` in the background per the spec. Everything UP TO the solve is
    exercised here with real objects (no mocks).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.storage import (
    DrawStorage,
    StoredGame,
    count_games_changed,
)
from analytics.versioning import DrawVersionManager
import run as run_mod
from run import (
    resolve_regen_scope,
    _build_regen_pins,
    _validate_regen_lock_weeks_overlap,
    _validate_regen_weeks_in_source,
    _compute_regen_state,
    _select_regen_group,
)


class _Args:
    """Minimal argparse-Namespace stand-in for _compute_regen_state."""
    def __init__(self, **kw):
        self.regen_from = kw.get("regen_from")
        self.regen_grades = kw.get("regen_grades")
        self.regen_weeks = kw.get("regen_weeks")
from main_staged import _merge_regen_pins


# ---------------------------------------------------------------------------
# Helpers — synthetic but structurally valid DrawStorage
# ---------------------------------------------------------------------------

_SLOT_DEFAULTS = dict(
    day="Sunday",
    time="11:30",
    day_slot=2,
    field_name="EF",
    field_location="Newcastle International Hockey Centre",
)


def _g(game_id, t1, t2, grade, week, date, **overrides):
    kwargs = {**_SLOT_DEFAULTS, **overrides}
    return StoredGame(
        game_id=game_id, team1=t1, team2=t2, grade=grade,
        week=week, round_no=week, date=date, **kwargs,
    )


def _draw(*games):
    return DrawStorage(
        description="synthetic regen E2E test draw",
        num_weeks=len({g.week for g in games}),
        num_games=len(games),
        games=list(games),
    )


# ---------------------------------------------------------------------------
# Source draw fixture — multiple grades incl. 6th, multiple weeks.
#
# Grades {PHL, 4th, 6th} x weeks {1, 2}, dates by week.
#   week 1 -> '2026-03-22'   week 2 -> '2026-03-29'
#
#   G01  PHL  w1  Maitland PHL / Norths PHL
#   G02  PHL  w2  Maitland PHL / Norths PHL
#   G03  4th  w1  Easts 4th    / Wests 4th
#   G04  4th  w2  Easts 4th    / Wests 4th
#   G05  6th  w1  Souths 6th   / Tigers 6th
#   G06  6th  w2  Souths 6th   / Tigers 6th
# ---------------------------------------------------------------------------

W1 = "2026-03-22"
W2 = "2026-03-29"


@pytest.fixture()
def source_draw():
    return _draw(
        _g("G01", "Maitland PHL", "Norths PHL", "PHL", 1, W1),
        _g("G02", "Maitland PHL", "Norths PHL", "PHL", 2, W2),
        _g("G03", "Easts 4th", "Wests 4th", "4th", 1, W1),
        _g("G04", "Easts 4th", "Wests 4th", "4th", 2, W2),
        _g("G05", "Souths 6th", "Tigers 6th", "6th", 1, W1),
        _g("G06", "Souths 6th", "Tigers 6th", "6th", 2, W2),
    )


# ===========================================================================
# 1. Frozen-set / pin correctness (DoD #4/#7 setup)
# ===========================================================================

class TestFrozenSetAndPins:
    def test_regen_grades_6th_freezes_all_non_6th_games(self, source_draw):
        # GIVEN --regen-grades 6th, no regen-weeks, no lock-weeks.
        # Oracle: free = G05,G06 (6th); frozen = G01,G02,G03,G04 (PHL+4th).
        free_ids, frozen_ids = resolve_regen_scope(
            source_draw.games, regen_grades={"6th"}, regen_weeks=set(), lock_weeks=set()
        )
        assert free_ids == {"G05", "G06"}
        assert frozen_ids == {"G01", "G02", "G03", "G04"}

    def test_pins_are_exactly_one_per_frozen_non_6th_game(self, source_draw):
        # GIVEN the frozen set from --regen-grades 6th
        _, frozen_ids = resolve_regen_scope(
            source_draw.games, regen_grades={"6th"}, regen_weeks=set(), lock_weeks=set()
        )
        # WHEN building pins
        pins = _build_regen_pins(source_draw, frozen_ids)
        # THEN exactly 4 pins (one per frozen game), zero for 6th.
        assert len(pins) == 4
        # Hand oracle: exact (teams, grade, date) set.
        got = {(tuple(p["teams"]), p["grade"], p["date"]) for p in pins}
        oracle = {
            (("Maitland PHL", "Norths PHL"), "PHL", W1),
            (("Maitland PHL", "Norths PHL"), "PHL", W2),
            (("Easts 4th", "Wests 4th"), "4th", W1),
            (("Easts 4th", "Wests 4th"), "4th", W2),
        }
        assert got == oracle
        # And NO 6th-grade pin.
        assert all(p["grade"] != "6th" for p in pins)

    def test_pins_only_carry_teams_grade_date_description(self, source_draw):
        _, frozen_ids = resolve_regen_scope(
            source_draw.games, regen_grades={"6th"}, regen_weeks=set(), lock_weeks=set()
        )
        pins = _build_regen_pins(source_draw, frozen_ids)
        for p in pins:
            assert set(p) == {"teams", "grade", "date", "description"}

    def test_union_rule_does_not_overpin_freed_grade_in_frozen_week(self, source_draw):
        # GIVEN --regen-grades 6th AND --regen-weeks 1 (union frees 6th + all w1).
        # Oracle:
        #   free  = 6th all weeks (G05,G06) UNION week-1 games (G01,G03,G05)
        #         = {G01, G03, G05, G06}
        #   frozen = {G02 (PHL w2), G04 (4th w2)}
        free_ids, frozen_ids = resolve_regen_scope(
            source_draw.games, regen_grades={"6th"}, regen_weeks={1}, lock_weeks=set()
        )
        assert free_ids == {"G01", "G03", "G05", "G06"}
        assert frozen_ids == {"G02", "G04"}
        # Pins must be EXACTLY the frozen games — NOT the freed 6th-w1 / w2 games.
        pins = _build_regen_pins(source_draw, frozen_ids)
        got = {(tuple(p["teams"]), p["grade"], p["date"]) for p in pins}
        assert got == {
            (("Maitland PHL", "Norths PHL"), "PHL", W2),
            (("Easts 4th", "Wests 4th"), "4th", W2),
        }
        # Critically: no 6th pin (6th is freed) and no week-1 pin (week 1 freed).
        assert all(p["grade"] != "6th" for p in pins)
        assert all(p["date"] != W1 for p in pins)

    def test_bye_in_frozen_set_produces_no_pin(self):
        # GIVEN a draw with a bye (team2='') and a normal game, both in the
        # frozen set (4th grade, frozen because we regen 6th only).
        draw = _draw(
            _g("B1", "Norths 4th", "", "4th", 1, W1),          # bye
            _g("B2", "Easts 4th", "Wests 4th", "4th", 1, W1),  # normal
            _g("B3", "Souths 6th", "Tigers 6th", "6th", 1, W1),  # free (6th)
        )
        _, frozen_ids = resolve_regen_scope(
            draw.games, regen_grades={"6th"}, regen_weeks=set(), lock_weeks=set()
        )
        assert frozen_ids == {"B1", "B2"}
        pins = _build_regen_pins(draw, frozen_ids)
        # THEN only the normal game is pinned (bye skipped by the guard).
        assert len(pins) == 1
        assert pins[0]["teams"] == ["Easts 4th", "Wests 4th"]

    def test_hard_locked_week_yields_no_pin(self, source_draw):
        # GIVEN week 1 is hard-locked (played). It is in neither free nor frozen.
        free_ids, frozen_ids = resolve_regen_scope(
            source_draw.games, regen_grades={"6th"}, regen_weeks=set(), lock_weeks={1}
        )
        # Week-1 games (G01,G03,G05) excluded from both sets.
        assert free_ids == {"G06"}            # 6th w2 only (6th w1 is hard-locked)
        assert frozen_ids == {"G02", "G04"}   # PHL w2 + 4th w2
        pins = _build_regen_pins(source_draw, frozen_ids)
        # No pin should reference a hard-locked (week-1 / W1) game.
        assert all(p["date"] != W1 for p in pins)
        assert len(pins) == 2


# ===========================================================================
# 2. Concatenation + dedup (DoD #3 injection) — _merge_regen_pins on real dicts
# ===========================================================================

class TestMergeAndDedup:
    def test_config_pin_duplicating_extracted_pin_is_collapsed(self):
        # GIVEN a data dict with a hand-authored config pin that duplicates one
        # extracted regen pin (same teams, grade, date).
        config_pin = {
            "teams": ["Maitland PHL", "Norths PHL"], "grade": "PHL", "date": W1,
            "description": "hand-authored",
        }
        data = {"locked_pairings": [config_pin]}
        extracted = [
            {"teams": ["Maitland PHL", "Norths PHL"], "grade": "PHL", "date": W1,
             "description": "Regen pin dup"},
            {"teams": ["Easts 4th", "Wests 4th"], "grade": "4th", "date": W2,
             "description": "Regen pin unique"},
        ]
        # WHEN merged
        _merge_regen_pins(data, extracted)
        merged = data["locked_pairings"]
        # THEN the duplicate is collapsed: 1 config + 1 unique extracted = 2.
        assert len(merged) == 2
        keys = {(tuple(p["teams"]), p["grade"], p["date"]) for p in merged}
        assert keys == {
            (("Maitland PHL", "Norths PHL"), "PHL", W1),
            (("Easts 4th", "Wests 4th"), "4th", W2),
        }
        # The CONFIG pin wins (kept its description), not the regen dup.
        phl_pins = [p for p in merged
                    if tuple(p["teams"]) == ("Maitland PHL", "Norths PHL")]
        assert len(phl_pins) == 1
        assert phl_pins[0]["description"] == "hand-authored"

    def test_no_duplicates_keeps_all_pins(self):
        data = {"locked_pairings": [
            {"teams": ["A", "B"], "grade": "PHL", "date": W1, "description": "c"},
        ]}
        extracted = [
            {"teams": ["C", "D"], "grade": "4th", "date": W2, "description": "r"},
        ]
        _merge_regen_pins(data, extracted)
        assert len(data["locked_pairings"]) == 2

    def test_empty_extracted_leaves_config_untouched(self):
        config = [{"teams": ["A", "B"], "grade": "PHL", "date": W1, "description": "c"}]
        data = {"locked_pairings": list(config)}
        _merge_regen_pins(data, None)
        assert data["locked_pairings"] == config
        _merge_regen_pins(data, [])
        assert data["locked_pairings"] == config

    def test_merge_when_no_config_pins_present(self):
        # GIVEN data with no locked_pairings key at all.
        data = {}
        extracted = [
            {"teams": ["A", "B"], "grade": "PHL", "date": W1, "description": "r"},
        ]
        _merge_regen_pins(data, extracted)
        assert data["locked_pairings"] == extracted

    def test_scope_only_config_pin_never_deduped(self):
        # A hand-authored scope-only pin (no 'teams') must pass through and not
        # collide with team-shaped extracted pins.
        scope_pin = {"grade": "PHL", "date": W1, "description": "scope-only"}
        data = {"locked_pairings": [scope_pin]}
        extracted = [
            {"teams": ["A", "B"], "grade": "PHL", "date": W1, "description": "r"},
        ]
        _merge_regen_pins(data, extracted)
        assert len(data["locked_pairings"]) == 2


# ===========================================================================
# 3. games_changed diff helper (DoD #6)
# ===========================================================================

class TestGamesChangedDiff:
    def test_retimed_games_counted_pairing_preserved(self, source_draw):
        # GIVEN a result identical to source EXCEPT two games are re-timed
        # (G05 6th time changed; G06 6th field changed). Pairings preserved.
        result = _draw(
            _g("R01", "Maitland PHL", "Norths PHL", "PHL", 1, W1),
            _g("R02", "Maitland PHL", "Norths PHL", "PHL", 2, W2),
            _g("R03", "Easts 4th", "Wests 4th", "4th", 1, W1),
            _g("R04", "Easts 4th", "Wests 4th", "4th", 2, W2),
            _g("R05", "Souths 6th", "Tigers 6th", "6th", 1, W1, time="14:00"),
            _g("R06", "Souths 6th", "Tigers 6th", "6th", 2, W2, field_name="WF"),
        )
        # Oracle: exactly 2 games changed (G05 time, G06 field).
        assert count_games_changed(source_draw, result) == 2

    def test_identical_draw_zero_changes(self, source_draw):
        # Same pairings/slots -> 0.
        assert count_games_changed(source_draw, source_draw) == 0

    def test_changed_pairing_not_counted_only_common_pairings(self, source_draw):
        # GIVEN a result where a 6th pairing is REPLACED (new opponent) — that
        # pairing identity differs so it is not a "common pairing"; not counted.
        result = _draw(
            _g("R01", "Maitland PHL", "Norths PHL", "PHL", 1, W1),
            _g("R02", "Maitland PHL", "Norths PHL", "PHL", 2, W2),
            _g("R03", "Easts 4th", "Wests 4th", "4th", 1, W1),
            _g("R04", "Easts 4th", "Wests 4th", "4th", 2, W2),
            # 6th week 1 now a different matchup (new identity)
            _g("R05", "Crusaders 6th", "Tigers 6th", "6th", 1, W1, time="14:00"),
            _g("R06", "Souths 6th", "Tigers 6th", "6th", 2, W2),  # unchanged
        )
        # Only common pairings (G01-G04, G06) are compared; all unchanged -> 0.
        assert count_games_changed(source_draw, result) == 0


# ===========================================================================
# 4. Metadata `regen` block (DoD #6) — real _build_draw_metadata
# ===========================================================================

def _solution_from_draw(draw):
    """Build an X-solution dict (key->1) from a DrawStorage for metadata tests."""
    return {g.to_key(): 1 for g in draw.games}


class TestMetadataRegenBlock:
    def test_regen_block_has_all_six_keys_with_correct_values(self, source_draw, tmp_path):
        # GIVEN a source draw saved to disk + a result draw (2 6th games retimed).
        source_path = tmp_path / "source.json"
        source_draw.save(str(source_path))
        result = _draw(
            _g("R01", "Maitland PHL", "Norths PHL", "PHL", 1, W1),
            _g("R02", "Maitland PHL", "Norths PHL", "PHL", 2, W2),
            _g("R03", "Easts 4th", "Wests 4th", "4th", 1, W1),
            _g("R04", "Easts 4th", "Wests 4th", "4th", 2, W2),
            _g("R05", "Souths 6th", "Tigers 6th", "6th", 1, W1, time="14:00"),
            _g("R06", "Souths 6th", "Tigers 6th", "6th", 2, W2, field_name="WF"),
        )
        # WHEN building metadata with a regen_info block, as run_generate would set.
        data = {
            "year": 2026,
            "_regen_info": {
                "source_draw": str(source_path),
                "regen_grades": ["6th"],
                "regen_weeks": [],
                "frozen_pin_count": 4,
                "hard_locked_weeks": [],
            },
        }
        vm = DrawVersionManager(str(tmp_path / "draws"), year=2026)
        solution = _solution_from_draw(result)
        meta = vm._build_draw_metadata(result, solution, data, "unified", "ts")
        # THEN the regen block exists with exactly the 6 required keys + values.
        assert "regen" in meta
        regen = meta["regen"]
        assert set(regen) == {
            "source_draw", "regen_grades", "regen_weeks",
            "frozen_pin_count", "hard_locked_weeks", "games_changed",
        }
        assert regen["source_draw"] == str(source_path)
        assert regen["regen_grades"] == ["6th"]
        assert regen["regen_weeks"] == []
        assert regen["frozen_pin_count"] == 4
        assert regen["hard_locked_weeks"] == []
        # games_changed oracle: G05 time + G06 field = 2.
        assert regen["games_changed"] == 2

    def test_missing_source_path_yields_zero_games_changed(self, source_draw, tmp_path, capsys):
        # GIVEN a regen_info pointing at a non-existent source draw file.
        data = {
            "year": 2026,
            "_regen_info": {
                "source_draw": str(tmp_path / "does_not_exist.json"),
                "regen_grades": ["6th"],
                "regen_weeks": [],
                "frozen_pin_count": 4,
                "hard_locked_weeks": [],
            },
        }
        vm = DrawVersionManager(str(tmp_path / "draws"), year=2026)
        meta = vm._build_draw_metadata(
            source_draw, _solution_from_draw(source_draw), data, "unified", "ts")
        # THEN the block still exists; games_changed defaults to 0 with a warning.
        assert meta["regen"]["games_changed"] == 0
        assert "could not compute regen games_changed" in capsys.readouterr().out

    def test_non_regen_run_has_no_regen_block(self, source_draw, tmp_path):
        # GIVEN data with NO _regen_info.
        data = {"year": 2026}
        vm = DrawVersionManager(str(tmp_path / "draws"), year=2026)
        solution = _solution_from_draw(source_draw)
        meta = vm._build_draw_metadata(source_draw, solution, data, "unified", "ts")
        # THEN no regen block (additive / opt-in).
        assert "regen" not in meta


# ===========================================================================
# 5. Versioning MAJOR bump (DoD #5) — real save_solver_output, tmp draws dir
# ===========================================================================

class TestVersioningMajorBump:
    """DoD #5: a regen solve bumps MAJOR (vN.M -> v{N+1}.0).

    `save_solver_output(is_major=True)` delegates the version computation to
    `save_new_draw(is_major=True)` (versioning.py:608-613). A full
    save_solver_output requires a complete season `data` (roster export);
    rather than mock that, we drive the REAL version-bumping path it calls —
    `save_new_draw(is_major=True)` — against a tmp draws dir. This is the exact
    code that computes the MAJOR bump for a regen save (no mocks).
    """

    def test_regen_save_bumps_major(self, source_draw, tmp_path):
        # GIVEN an existing v3.2 draw in a tmp draws dir.
        draws_dir = tmp_path / "draws"
        vm = DrawVersionManager(str(draws_dir), year=2026)
        seed = _draw(*source_draw.games)
        seed.metadata["version"] = "v3.2"
        vm.versions_path.mkdir(parents=True, exist_ok=True)
        seed.save(str(vm.versions_path / "draw_v3.2.json"))
        # The next MAJOR must be v4.0 (what is_major=True selects).
        assert vm.get_next_major_version() == (4, 0)

        # WHEN saving the regen output via the real is_major=True path.
        result = _draw(*source_draw.games)
        version = vm.save_new_draw(result, "regen test", is_major=True)
        # THEN MAJOR bumped v3.2 -> v4.0 (NOT v3.3 — that would be a MINOR/hand-edit).
        assert version.version_string == "v4.0"

    def test_minor_path_would_have_been_v3_3(self, source_draw, tmp_path):
        # Sanity oracle: confirm the bump is genuinely MAJOR, not MINOR — a
        # MINOR save of the same seed would have produced v3.3.
        vm = DrawVersionManager(str(tmp_path / "draws"), year=2026)
        seed = _draw(*source_draw.games)
        vm.versions_path.mkdir(parents=True, exist_ok=True)
        seed.save(str(vm.versions_path / "draw_v3.2.json"))
        assert vm.get_next_minor_version() == (3, 3)
        assert vm.get_next_major_version() == (4, 0)


# ===========================================================================
# 6. Overlap + missing-week validation (FATAL)
# ===========================================================================

class TestValidation:
    def test_regen_weeks_overlapping_lock_weeks_is_fatal(self):
        with pytest.raises(SystemExit) as exc:
            _validate_regen_lock_weeks_overlap({3, 4}, {4, 5})
        assert exc.value.code == 1

    def test_regen_weeks_no_overlap_passes(self):
        # No exception.
        _validate_regen_lock_weeks_overlap({3, 4}, {5, 6})

    def test_regen_week_absent_from_source_is_fatal(self):
        with pytest.raises(SystemExit) as exc:
            _validate_regen_weeks_in_source({3, 99}, {1, 2, 3})
        assert exc.value.code == 1

    def test_regen_week_present_in_source_passes(self):
        _validate_regen_weeks_in_source({2, 3}, {1, 2, 3})

    def test_empty_regen_weeks_always_passes(self):
        _validate_regen_weeks_in_source(set(), {1, 2, 3})


# ===========================================================================
# 7. Full regen-state orchestration (DoD #3) — _compute_regen_state +
#    group-selection guard. Real source draw on disk, NO solver, NO mocks.
# ===========================================================================

class TestComputeRegenState:
    def test_non_regen_run_returns_all_none(self):
        # GIVEN no --regen-from -> non-regen run is completely unaffected.
        pins, info, groups = _compute_regen_state(_Args(regen_from=None), set())
        assert pins is None and info is None and groups is None

    def test_regen_scope_flags_without_regen_from_warn_and_no_op(self, capsys):
        # GIVEN --regen-grades but NO --regen-from (the enabler is missing).
        args = _Args(regen_from=None, regen_grades=["6th"], regen_weeks=None)
        # WHEN computing regen state.
        pins, info, groups = _compute_regen_state(args, set())
        # THEN it is a no-op AND warns the user their scope flag was ignored.
        assert pins is None and info is None and groups is None
        assert "--regen-grades/--regen-weeks ignored" in capsys.readouterr().out

    def test_regen_grades_6th_produces_expected_state(self, source_draw, tmp_path):
        # GIVEN a source draw on disk + --regen-grades 6th.
        src = tmp_path / "src.json"
        source_draw.save(str(src))
        args = _Args(regen_from=str(src), regen_grades=["6th"], regen_weeks=None)
        # WHEN computing regen state (no hard-locked weeks).
        pins, info, groups = _compute_regen_state(args, set())
        # THEN 4 pins (PHL+4th, both weeks), regen_info populated.
        assert len(pins) == 4
        assert all(p["grade"] != "6th" for p in pins)
        assert info == {
            "source_draw": str(src),
            "regen_grades": ["6th"],
            "regen_weeks": [],
            "frozen_pin_count": 4,
            "hard_locked_weeks": [],
        }
        # spec-023 absent -> group guard returns None.
        assert groups is None

    def test_group_guard_fires_warning_when_spec023_absent(self, source_draw, tmp_path, capsys):
        # GIVEN spec-023 (constraints.groups.resolve_groups) is NOT importable.
        src = tmp_path / "src.json"
        source_draw.save(str(src))
        args = _Args(regen_from=str(src), regen_grades=["6th"], regen_weeks=None)
        # WHEN computing regen state.
        _compute_regen_state(args, set())
        # THEN the documented WARNING is printed.
        out = capsys.readouterr().out
        assert "WARNING: spec-027 regen group not available" in out

    def test_select_regen_group_returns_none_and_warns(self, capsys):
        # Direct unit on the guard (spec-023 absent in this worktree).
        assert _select_regen_group() is None
        assert "WARNING: spec-027 regen group not available" in capsys.readouterr().out

    def test_lock_weeks_excluded_from_pins(self, source_draw, tmp_path):
        # GIVEN week 1 hard-locked + --regen-grades 6th.
        src = tmp_path / "src.json"
        source_draw.save(str(src))
        args = _Args(regen_from=str(src), regen_grades=["6th"], regen_weeks=None)
        pins, info, _ = _compute_regen_state(args, {1})
        # THEN no pin references week-1 date (hard-locked, not pinned).
        assert all(p["date"] != W1 for p in pins)
        assert info["hard_locked_weeks"] == [1]
        # frozen = PHL w2 + 4th w2 = 2 pins.
        assert len(pins) == 2
        assert info["frozen_pin_count"] == 2

    def test_regen_weeks_overlap_lock_weeks_fatal_via_compute(self, source_draw, tmp_path):
        src = tmp_path / "src.json"
        source_draw.save(str(src))
        args = _Args(regen_from=str(src), regen_grades=None, regen_weeks="1")
        with pytest.raises(SystemExit) as exc:
            _compute_regen_state(args, {1})
        assert exc.value.code == 1

    def test_regen_week_absent_from_source_fatal_via_compute(self, source_draw, tmp_path):
        src = tmp_path / "src.json"
        source_draw.save(str(src))
        # Source has weeks {1,2}; request week 9.
        args = _Args(regen_from=str(src), regen_grades=None, regen_weeks="9")
        with pytest.raises(SystemExit) as exc:
            _compute_regen_state(args, set())
        assert exc.value.code == 1
