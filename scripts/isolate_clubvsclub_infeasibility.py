#!/usr/bin/env python
"""Isolate the EXACT cause of ClubVsClubStackedWeekends ("club alignment")
infeasibility on the real 2026 config — open Step-1 from
docs/todo/spec-035-realconfig-stage5-handoff.md.

Established already (analytical pass, mode=analytical):
  - The atom is NOT self-infeasible: Layer-6 nested chain has a feasible
    prefix-intersection for every (pair, grade). So the blocker is an
    INTERACTION with the always-on fundamentals (EqualGames / NoDoubleBooking)
    + the forced-game sums baked into generate_X.

This script assembles the MINIMAL reproduction and bisects it:
  fundamentals (EqualGamesAndBalanceMatchUps + NoDoubleBookingTeams/Fields)
  + the alignment atom, controllable by `max_layer` and by `pairs` subset.

Modes:
  analytical   — instant Layer-6 prefix-intersection check (no solver).
  reproduce    — fundamentals + full atom; confirm INFEASIBLE; layer-bisect (2/4/5/6).
  pairs        — at the offending max_layer, drop one club_pair at a time to find
                 the pair(s) whose removal restores feasibility.

Run (default = all modes):
  .venv\\Scripts\\python.exe scripts\\isolate_clubvsclub_infeasibility.py [mode] [--layer N] [--time S]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ortools.sat.python import cp_model  # noqa: E402

from config import load_season_data  # noqa: E402
from utils import generate_X, get_nearest_week_by_date  # noqa: E402
from constraints.helper_vars import HelperVarRegistry  # noqa: E402
from constraints.atoms._club_day_shared import parse_club_day_entries  # noqa: E402
from constraints.archived.original import (  # noqa: E402
    EnsureEqualGamesAndBalanceMatchUps,
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
)
from constraints.atoms._club_vs_club_stacked_shared import (  # noqa: E402
    STACK_PLAY_PREFIX,
    STACK_TEAM_PAIR_PLAY_PREFIX,
    enumerate_club_pairs,
    enumerate_team_pairs_in_pair_grade,
    pair_grade_sunday_aligned_weekend_range,
    per_pair_grade_meeting_counts,
    team_pair_counts,
    team_pair_sunday_meetings_range,
)

YEAR = 2026


def build_data_and_X(quiet=True):
    if quiet:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = load_season_data(YEAR)
            model = cp_model.CpModel()
            X, conflicts = generate_X(model, data)
    else:
        data = load_season_data(YEAR)
        model = cp_model.CpModel()
        X, conflicts = generate_X(model, data)
    data['team_conflicts'] = conflicts
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    return data, model, X


def _sunday_index(X, data):
    locked_weeks = set(data.get('locked_weeks', set()) or set())
    sunday_weeks = sorted({
        key[6] for key in X
        if len(key) >= 11 and key[3] == 'Sunday' and key[6] not in locked_weeks
    })
    tp_week_vars = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11 or not key[3] or key[3] != 'Sunday' or key[6] in locked_weeks:
            continue
        tp_week_vars[(key[0], key[1], key[6])].append(var)
    return sunday_weeks, tp_week_vars


def _cd_hosts(data):
    out = defaultdict(set)
    for club_name, date_str, _opp in parse_club_day_entries(data):
        out[get_nearest_week_by_date(date_str, data.get('timeslots', []))].add(club_name)
    return out


def _pair_specs(data, pair, sunday_weeks, tp_week_vars):
    specs = []
    for grade in per_pair_grade_meeting_counts(data, pair).keys():
        min_b, max_b = pair_grade_sunday_aligned_weekend_range(data, pair, grade)
        if max_b == 0:
            continue
        tp_min, tp_max = team_pair_sunday_meetings_range(data, pair, grade)
        if tp_max == 0:
            continue
        a, b = team_pair_counts(data, pair, grade)
        tps = enumerate_team_pairs_in_pair_grade(data, pair, grade)
        if not tps:
            continue
        avail = {w for w in sunday_weeks
                 if any(tp_week_vars.get((tp[0], tp[1], w)) for tp in tps)}
        specs.append((grade, min_b, max_b, tp_min, tp_max, a, b, tps, avail))
    return specs


def apply_stacked(model, X, data, registry, pairs, max_layer=6):
    """Faithful replica of ClubVsClubStackedWeekends.apply, restricted to `pairs`
    and capped at `max_layer`. Returns constraint count added."""
    sunday_weeks, tp_week_vars = _sunday_index(X, data)
    if not sunday_weeks:
        return 0
    cd_hosts = _cd_hosts(data)
    n = 0
    for pair in pairs:
        specs = _pair_specs(data, pair, sunday_weeks, tp_week_vars)
        if not specs:
            continue
        team_pair_play = {}
        for grade, *_rest, tps, _avail in specs:
            for tp in tps:
                for w in sunday_weeks:
                    if (tp, w) in team_pair_play:
                        continue
                    team_pair_play[(tp, w)] = registry.get_or_create_bool(
                        (STACK_TEAM_PAIR_PLAY_PREFIX, tp, w),
                        tp_week_vars.get((tp[0], tp[1], w), []),
                        f'tpp_{tp[0]}_{tp[1]}_w{w}')
        if max_layer >= 2:
            for grade, _mb, _xb, tp_min, tp_max, _a, _b, tps, _av in specs:
                for tp in tps:
                    v = [team_pair_play[(tp, w)] for w in sunday_weeks]
                    model.Add(sum(v) >= tp_min)
                    model.Add(sum(v) <= tp_max)
                    n += 2
        play_pg = {}
        for grade, _mb, _xb, _tpm, _tpx, a, b, tps, _av in specs:
            min_ab = min(a, b)
            for w in sunday_weeks:
                tpv = [team_pair_play[(tp, w)] for tp in tps]
                ppg = registry.get_or_create_bool(
                    (STACK_PLAY_PREFIX, pair, grade, w), tpv,
                    f'ppg_{pair[0]}_{pair[1]}_{grade}_w{w}')
                play_pg[(grade, w)] = ppg
                if max_layer >= 4:
                    model.Add(sum(tpv) == min_ab).OnlyEnforceIf(ppg)
                    model.Add(sum(tpv) == 0).OnlyEnforceIf(ppg.Not())
                    n += 2
        if max_layer >= 5:
            for grade, min_b, max_b, _tpm, _tpx, _a, _b, _tps, _av in specs:
                pv = [play_pg[(grade, w)] for w in sunday_weeks]
                model.Add(sum(pv) >= min_b)
                model.Add(sum(pv) <= max_b)
                n += 2
        if max_layer >= 6:
            chain = sorted(specs, key=lambda s: (-s[2], s[0]))
            for i in range(len(chain) - 1):
                hi, lo = chain[i][0], chain[i + 1][0]
                for w in sunday_weeks:
                    hosts = cd_hosts.get(w)
                    if hosts and (pair[0] in hosts or pair[1] in hosts):
                        continue
                    model.Add(play_pg[(lo, w)] <= play_pg[(hi, w)])
                    n += 1
    return n


def apply_fundamentals(model, X, data):
    EnsureEqualGamesAndBalanceMatchUps().apply(model, X, data)
    NoDoubleBookingTeamsConstraint().apply(model, X, data)
    NoDoubleBookingFieldsConstraint().apply(model, X, data)


def solve(model, t=30, workers=4):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = t
    s.parameters.num_search_workers = workers
    st = s.Solve(model)
    return s.status_name(st)


# ---------------------------------------------------------------- analytical
def analytical_check(data, X):
    sunday_weeks, tp_week_vars = _sunday_index(X, data)
    cd_hosts = _cd_hosts(data)
    flagged = []
    for pair in enumerate_club_pairs(data):
        specs = _pair_specs(data, pair, sunday_weeks, tp_week_vars)
        if len(specs) < 2:
            continue
        chain = sorted(specs, key=lambda s: (-s[2], s[0]))
        exempt = {w for w in sunday_weeks
                  if cd_hosts.get(w) and (pair[0] in cd_hosts[w] or pair[1] in cd_hosts[w])}
        higher = None
        for spec in chain:
            grade, min_b, max_b = spec[0], spec[1], spec[2]
            avail = spec[8]
            feasible = set(avail) if higher is None else (avail & higher) | (avail & exempt)
            if len(feasible) < min_b:
                flagged.append((pair, grade, min_b, sorted(feasible)))
            higher = set(avail) if higher is None else (higher & avail) | exempt
    return flagged


# ----------------------------------------------------------------- runners
def run_reproduce(layer_list, t):
    print("\n=== REPRODUCE: fundamentals + alignment atom ===")
    # sanity: fundamentals only
    data, model, X = build_data_and_X()
    apply_fundamentals(model, X, data)
    print(f"  [sanity] fundamentals only            -> {solve(model, t)}")
    for ml in layer_list:
        data, model, X = build_data_and_X()
        apply_fundamentals(model, X, data)
        reg = HelperVarRegistry(model)
        pairs = enumerate_club_pairs(data)
        n = apply_stacked(model, X, data, reg, pairs, max_layer=ml)
        print(f"  fundamentals + alignment(max_layer={ml}, {n} con) -> {solve(model, t)}")


def run_pairs(layer, t):
    print(f"\n=== PAIR BISECT at max_layer={layer}: drop one club_pair at a time ===")
    data0, _m, X0 = build_data_and_X()
    all_pairs = enumerate_club_pairs(data0)
    print(f"  {len(all_pairs)} club pairs. Baseline (all pairs) expected INFEASIBLE.")
    restored = []
    for drop in all_pairs:
        data, model, X = build_data_and_X()
        apply_fundamentals(model, X, data)
        reg = HelperVarRegistry(model)
        kept = [p for p in all_pairs if p != drop]
        apply_stacked(model, X, data, reg, kept, max_layer=layer)
        st = solve(model, t)
        flag = '  <== removing this pair RELIEVES infeasibility' if st not in ('INFEASIBLE',) else ''
        if flag:
            restored.append((drop, st))
        print(f"  drop {str(drop):<34} -> {st}{flag}")
    print(f"\n  Pairs whose removal restores feasibility: {restored or 'NONE (conflict is global / multi-pair)'}")
    return restored


def run_single_pairs(layer, t):
    print(f"\n=== SINGLE-PAIR at max_layer={layer}: fundamentals + alignment for ONE pair ===")
    data0, _m, X0 = build_data_and_X()
    all_pairs = enumerate_club_pairs(data0)
    bad = []
    for only in all_pairs:
        data, model, X = build_data_and_X()
        apply_fundamentals(model, X, data)
        reg = HelperVarRegistry(model)
        apply_stacked(model, X, data, reg, [only], max_layer=layer)
        st = solve(model, t)
        if st == 'INFEASIBLE':
            bad.append(only)
            print(f"  only {str(only):<34} -> {st}  <== self-conflicting pair")
    if not bad:
        print("  No single pair + fundamentals is infeasible alone (conflict needs >=2 pairs).")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["all", "analytical", "reproduce", "pairs", "single"])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--time", type=int, default=30)
    args = ap.parse_args()

    print(f"Loading real {YEAR} config + generating X ...")
    data, _m, X = build_data_and_X(quiet=False)
    print(f"\n  {len(X)} vars; {len(data['teams'])} teams; games={len(data['games'])}")

    if args.mode in ("all", "analytical"):
        flagged = analytical_check(data, X)
        print(f"\n=== ANALYTICAL Layer-6 contradictions: {len(flagged)} ===")
        for pair, grade, mb, feas in flagged:
            print(f"  {pair} {grade}: need {mb}, feasible weeks {feas}")

    if args.mode in ("all", "reproduce"):
        run_reproduce([2, 4, 5, 6], args.time)

    if args.mode in ("all", "single"):
        run_single_pairs(args.layer, args.time)

    if args.mode in ("all", "pairs"):
        run_pairs(args.layer, args.time)


if __name__ == "__main__":
    main()
