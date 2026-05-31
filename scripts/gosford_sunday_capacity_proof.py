#!/usr/bin/env python
"""Counting proof of the ClubVsClubStackedWeekends (Layer-2) infeasibility on the
real 2026 config.

Mechanism: the atom's per-team-pair Sunday floor `tp_min` only subtracts FORCED
Friday meetings that NAME BOTH clubs (`phl_forced_friday_meetings` under-counts
umbrella scopes by design). The 8 Gosford Friday games are an UMBRELLA scope
(`{grade:PHL, field_location:CCHP, day:Friday}`, sum==8, no teams), so they are
subtracted from NO pair. EqualGames still forces those 8 onto Fridays, leaving
only R-8 Sundays for the club — but the atom demands Σ_opp tp_min Sunday meetings.

For each PHL club, compare:
  atom Sunday FLOOR   = Σ_opponents tp_min(club, opp)
  Sunday CAPACITY     = R - (forced Friday games the club MUST play)
If floor > capacity, the atom is infeasible against EqualGames — proven by
counting, no solver needed.
"""
from __future__ import annotations

import io
import contextlib
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ortools.sat.python import cp_model  # noqa: E402
from config import load_season_data  # noqa: E402
from utils import generate_X  # noqa: E402
from constraints.atoms._club_vs_club_stacked_shared import (  # noqa: E402
    enumerate_club_pairs, team_pair_sunday_meetings_range, team_pair_counts,
)
from constraints.atoms._phl_forced_friday_helper import phl_forced_friday_meetings  # noqa: E402

YEAR = 2026


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = load_season_data(YEAR)
        model = cp_model.CpModel()
        X, _c = generate_X(model, data)
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']

    num_rounds = data.get('num_rounds', {})
    teams = data['teams']
    phl_clubs = sorted({t.club.name for t in teams if t.grade == 'PHL'})
    R = num_rounds.get('PHL')
    T = sum(1 for t in teams if t.grade == 'PHL')
    base = R // (T - 1) if T % 2 == 0 else R // T
    print(f"PHL: T={T} teams, R={R} games/team, base meetings/pair={base} "
          f"(EqualGames forces each pair to meet in [{base},{base+1}])")
    print(f"PHL clubs: {phl_clubs}\n")

    # forced PHL-Friday lower bound PER CLUB, ROBUST: per (club, venue) take the
    # LARGEST single forced-Friday scope count — specific-date / pair scopes are
    # SUBSETS of the venue umbrella (e.g. the 7 specific CCHP dates are within the
    # `sum==8` CCHP umbrella), so they must not be summed on top of it. Away-venue
    # umbrellas (CCHP/Maitland Park) guarantee the home club is a participant of
    # every game in the scope (home-venue filter); NIHC umbrellas name no single
    # club so are skipped (conservative — under-counts, keeping this a LOWER bound).
    home_club = {'Central Coast Hockey Park': 'Gosford', 'Maitland Park': 'Maitland'}
    venue_scope_max = defaultdict(int)   # (club, venue) -> max forced count at that venue
    for e in (data.get('forced_games') or []):
        if e.get('day') != 'Friday':
            continue
        if e.get('grade') != 'PHL' and 'PHL' not in (e.get('grades') or []):
            continue
        if e.get('constraint', 'equal') != 'equal':
            continue  # lesse/greatere don't pin a hard minimum here
        cnt = int(e.get('count', 1))
        loc = e.get('field_location')
        hc = home_club.get(loc)
        if hc:
            venue_scope_max[(hc, loc)] = max(venue_scope_max[(hc, loc)], cnt)
    forced_fri_min = defaultdict(int)
    for (club, _loc), cnt in venue_scope_max.items():
        forced_fri_min[club] += cnt

    print(f"{'club':<14}{'Σ tp_min (atom floor)':<24}{'forced Fri (min)':<18}"
          f"{'Sunday cap = R-Fri':<20}{'verdict'}")
    print("-" * 88)
    for club in phl_clubs:
        opps = set()
        floor = 0
        detail = []
        for pair in enumerate_club_pairs(data):
            if club not in pair:
                continue
            a, b = team_pair_counts(data, pair, 'PHL')
            if a == 0 or b == 0:
                continue
            tp_min, tp_max = team_pair_sunday_meetings_range(data, pair, 'PHL')
            ff = phl_forced_friday_meetings(data, pair[0], pair[1])
            opp = pair[1] if pair[0] == club else pair[0]
            opps.add(opp)
            floor += tp_min
            detail.append((opp, tp_min, tp_max, ff))
        ff_min = forced_fri_min.get(club, 0)
        cap = R - ff_min
        verdict = 'INFEASIBLE  (floor > capacity)' if floor > cap else 'ok'
        print(f"{club:<14}{floor:<24}{ff_min:<18}{cap:<20}{verdict}")
        if floor > cap:
            print(f"    -> {club} must play {floor} Sunday meetings (atom floor) but only "
                  f"{cap} Sundays exist after {ff_min} forced Fridays.")
            for opp, mn, mx, ff in sorted(detail):
                print(f"       vs {opp:<12} tp_min={mn} tp_max={mx}  "
                      f"(pair-named forced Fri subtracted={ff})")


if __name__ == "__main__":
    main()
