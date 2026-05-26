"""spec-035 Unit A — raw-core e2e launcher + forced-free test-config checks.

Given/When/Then with hand-computed oracles. No mocks/patches: the real
`config.season_test.get_season_data()` builds the real data dict, the real
`run._resolve_group_selection` resolves the real registry, and the engine-skip
oracle uses the real `constraints.stages` helpers. No solve is run (build-only).

Covers the Unit A "No-mock test outline":
  1. Data via get_season_data() is forced-free AND its team set == the 2026 base
     team set (oracle: the 48 names hand-listed below from season_2026's CSVs).
  2. The launcher's resolved profile (no exclude) has groups==['core'],
     workers==10, fix_round_1 falsy, no locked weeks, exclude==[].
  3. With exclude=['ClubGameSpread'] the profile is identical EXCEPT exclude,
     AND the engine's skip_constraints contains 'ClubGameSpread' while
     'ClubNoConcurrentSlot' stays applied (in the resolved constraint set and
     never skipped — it is a non-engine atom).
"""
from scripts.run_core_e2e import build_run_config
from config.season_test import get_season_data
from config import load_season_data
from run import _resolve_group_selection
from constraints.stages import collect_engine_keys, ALL_ENGINE_KEYS


# ---------------------------------------------------------------------------
# Oracle: the 48 base teams of the 2026 season, hand-listed from the 2026 team
# CSVs (data/2026/teams). season_test deep-copies season_2026's config, so its
# team set must equal this exactly. If a team is added/removed in 2026, this
# list (and the count) must be updated deliberately — that is the point of a
# hand-listed oracle.
# ---------------------------------------------------------------------------
EXPECTED_2026_TEAMS = {
    'Colts 4th', 'Colts 6th', 'Colts Gold 5th', 'Colts Green 5th',
    'Crusaders 3rd', 'Crusaders 4th', 'Crusaders 5th', 'Crusaders 6th',
    'Gosford PHL',
    'Maitland 3rd', 'Maitland 4th', 'Maitland 5th', 'Maitland 6th', 'Maitland PHL',
    'Norths 2nd', 'Norths 3rd', 'Norths 4th', 'Norths 5th', 'Norths PHL',
    'Port Stephens 3rd', 'Port Stephens 4th', 'Port Stephens 6th',
    'Souths 2nd', 'Souths 3rd', 'Souths 4th', 'Souths 6th', 'Souths PHL',
    'Tigers 2nd', 'Tigers 3rd', 'Tigers 4th', 'Tigers 5th',
    'Tigers Black 6th', 'Tigers PHL', 'Tigers Yellow 6th',
    'University 3rd', 'University 5th', 'University Gentlemen 6th',
    'University Redhogs 4th', 'University Seapigs 4th', 'University Seapigs 6th',
    'Wests 2nd', 'Wests 3rd', 'Wests 6th',
    'Wests Green 4th', 'Wests Green 5th', 'Wests PHL',
    'Wests Red 4th', 'Wests Red 5th',
}


def _team_names(data):
    return {getattr(t, 'name', str(t)) for t in data['teams']}


# ---------------------------------------------------------------------------
# Test 1 — DoD-1: forced-free + team set == 2026 base set.
# ---------------------------------------------------------------------------
def test_season_test_is_forced_free_and_matches_2026_teams():
    # Given: data built via the documented Option-(b) path.
    data = get_season_data()

    # Then: it is forced-free (the whole point of season_test).
    assert data.get('forced_games') == [], (
        f"season_test must be forced-free; got {data.get('forced_games')!r}")

    # And: its team set equals the hand-listed 2026 base team set (oracle).
    names = _team_names(data)
    assert len(names) == 48, f"expected 48 base teams, got {len(names)}"
    assert names == EXPECTED_2026_TEAMS, (
        f"season_test team set drifted from 2026 base set; "
        f"only in test: {sorted(names - EXPECTED_2026_TEAMS)}; "
        f"only in 2026-oracle: {sorted(EXPECTED_2026_TEAMS - names)}")

    # And: it equals the live season_2026 build too (guards the oracle itself
    # against silent CSV drift in either direction).
    assert names == _team_names(load_season_data(2026))


# ---------------------------------------------------------------------------
# Test 2 — DoD-2: the raw-core profile with no exclude, field-by-field.
# ---------------------------------------------------------------------------
def test_build_run_config_no_exclude_is_raw_core_profile():
    # Given/When: resolve the profile with no exclude.
    cfg = build_run_config()

    # Then: each fixed flag matches the raw-core profile exactly.
    assert cfg['groups'] == ['core']
    assert cfg['workers'] == 10
    assert not cfg['fix_round_1']            # falsy — week 1 NOT fixed
    assert cfg['fix_round_1'] is False
    assert cfg['locked_weeks'] == []         # raw — no locks
    assert cfg['forced_games'] == []         # inherent to season_test
    assert cfg['exclude'] == []
    assert cfg['no_symmetry_breakers'] is False   # symmetry_breakers stay ON

    # And: the resolved constraint set keeps the always-on symmetry breakers
    # (spec-032) — core itself does not contain them, so this proves the union.
    _, constraint_names = _resolve_group_selection(
        cfg['groups'], cfg['exclude'], cfg['no_symmetry_breakers'])
    for sb in ('NIHCFillWFBeforeEF', 'NIHCFillEFBeforeSF', 'SoftLexMatchupOrdering'):
        assert sb in constraint_names, f"symmetry breaker {sb} must stay applied"


# ---------------------------------------------------------------------------
# Test 3 — DoD-2b: exclude=['ClubGameSpread'] is the ONLY delta, and the engine
# skip / applied sets behave correctly.
# ---------------------------------------------------------------------------
def test_build_run_config_exclude_is_only_delta_and_skips_clubgamespread():
    cfg0 = build_run_config()
    cfg1 = build_run_config(exclude=['ClubGameSpread'])

    # Then: the two profiles are identical except for `exclude`.
    delta = [k for k in cfg0 if cfg0[k] != cfg1[k]]
    assert delta == ['exclude'], f"expected only 'exclude' to differ, got {delta}"
    assert cfg1['exclude'] == ['ClubGameSpread']

    # Resolve both selections through the real CLI helper + registry.
    _, cn0 = _resolve_group_selection(cfg0['groups'], cfg0['exclude'], False)
    _, cn1 = _resolve_group_selection(cfg1['groups'], cfg1['exclude'], False)

    # ClubGameSpread is the only constraint removed from the resolved set.
    assert 'ClubGameSpread' in cn0
    assert 'ClubGameSpread' not in cn1
    assert (set(cn0) - set(cn1)) == {'ClubGameSpread'}, (
        f"exclude should drop exactly ClubGameSpread; dropped {set(cn0) - set(cn1)}")

    # ClubNoConcurrentSlot stays applied in BOTH (it is a separate core_hard
    # atom, NOT part of the ClubGameSpread method — must not be dropped).
    assert 'ClubNoConcurrentSlot' in cn0
    assert 'ClubNoConcurrentSlot' in cn1

    # Engine skip oracle (mirrors apply_constraint_set:
    #   engine.skip_constraints = ALL_ENGINE_KEYS - selected_engine_keys).
    skip1 = ALL_ENGINE_KEYS - collect_engine_keys(cn1)[0]
    # ClubGameSpread IS an engine key — excluded => it lands in skip_constraints.
    assert 'ClubGameSpread' in skip1
    # ClubNoConcurrentSlot is a NON-engine atom — it is never an engine skip key
    # (so "not in skip_constraints" holds), and it stays in the applied set
    # (asserted above) i.e. it remains applied, not relaxed.
    assert 'ClubNoConcurrentSlot' not in skip1
