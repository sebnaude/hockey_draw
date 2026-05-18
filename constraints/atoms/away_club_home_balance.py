"""AwayClubPerOpponentAndAggregateHomeBalance (spec-004).

For each team in each away-based club, enforce TWO home-balance constraints:

  1. Per-opponent: for each opponent in this team's grade, home games against
     that opponent ∈ [floor(total/2), ceil(total/2)] — applied as two
     `model.Add` inequalities so the IntVar lands on the correct integer for
     odd/even totals automatically.

  2. Aggregate: across all of this team's games, home games ∈
     [floor(total_games/2), ceil(total_games/2)] — same construction.

The intersection of the two gives the best outcome: pairs are individually
balanced AND the aggregate is balanced, automatically handling the case where
a team plays an opponent 3 times (per-pair lands 1H/2A or 2H/1A; aggregate
balance ensures no team is stuck at 3H/0A across opponents).

## Why this replaces `FiftyFiftyHomeandAway`

The legacy class (in `constraints/archived/original.py`) did:
  * Per-pair home/away balance: `home_games * 2 in [total - 1, total + 1]`.
  * Aggregate home/away balance: `agg_home * 2 in [agg_total - 1, agg_total + 1]`.

The per-pair block is preserved verbatim by this atom's "per-opponent" rule
(the `* 2 in [t-1, t+1]` form is mathematically identical to
`in [floor(t/2), ceil(t/2)]`). The aggregate block is preserved by the
"aggregate" rule. spec-004 also adds the home-WEEKEND-count via the sister
`AwayClubHomeWeekendsCount` atom — together they give the convenor exactly
the per-pair, per-team-aggregate, AND per-club-weekend balance.

## What's NOT here

Per-club home-weekend totals live in `AwayClubHomeWeekendsCount`. Consecutive
home weekends spacing lives in `NonDefaultHomeGrouping`. Locked weeks are
handled the same way as the legacy class — the sums see the locked vars as
constants (the solver will already have fixed them via `model.Add(X[k]==1)`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from constraints.atoms.base import Atom


class AwayClubPerOpponentAndAggregateHomeBalance(Atom):
    """Per-opponent + aggregate home/away balance for every away-club team."""

    canonical_name = 'AwayClubPerOpponentAndAggregateHomeBalance'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        home_field_map: Dict[str, str] = data.get('home_field_map', {}) or {}
        if not home_field_map:
            return 0

        team_to_club = {team.name: team.club.name for team in data.get('teams', [])}
        constraints_added = 0

        # Group home and away vars per (away_team, opponent_team).
        # home: game scheduled at the away_team's home venue.
        # away: game scheduled elsewhere.
        # away_team is always a member of one of the home_field_map clubs.
        per_pair_home: Dict[tuple, List] = defaultdict(list)
        per_pair_away: Dict[tuple, List] = defaultdict(list)

        away_venues = set(home_field_map.values())
        venue_owner = {venue: club for club, venue in home_field_map.items()}

        for key, var in X.items():
            if len(key) < 11:
                continue
            if not key[3]:
                continue
            t1, t2 = key[0], key[1]
            venue = key[10]
            club1 = team_to_club.get(t1)
            club2 = team_to_club.get(t2)

            # Determine if either team belongs to an away-based club.
            t1_away_club = club1 if club1 in home_field_map else None
            t2_away_club = club2 if club2 in home_field_map else None

            if not t1_away_club and not t2_away_club:
                continue  # neither team is away-based

            # Skip games where both teams are from the SAME away club
            # (intra-club derbies — there's no meaningful home/away balance
            # since the venue is mutual).
            if t1_away_club and t2_away_club and t1_away_club == t2_away_club:
                continue

            for away_team, opponent, away_club in (
                (t1, t2, t1_away_club),
                (t2, t1, t2_away_club),
            ):
                if not away_club:
                    continue
                # Skip when the opponent's team is from the same club (handled above).
                if team_to_club.get(opponent) == away_club:
                    continue
                pair_key = (away_team, opponent)
                home_venue = home_field_map[away_club]
                if venue == home_venue:
                    per_pair_home[pair_key].append(var)
                elif venue in away_venues and venue_owner.get(venue) == team_to_club.get(opponent):
                    # The opponent is also from an away-based club; this game
                    # is at the opponent's home venue — counts as AWAY for our
                    # away_team. (e.g. Maitland vs Gosford at Gosford.)
                    per_pair_away[pair_key].append(var)
                else:
                    # Neutral venue (Broadmeadow / other) — AWAY for our away_team.
                    per_pair_away[pair_key].append(var)

        # Apply per-pair home/away balance.
        # For each (away_team, opponent), home_games in [floor(t/2), ceil(t/2)].
        # Track per-team aggregates for the second constraint.
        per_team_home: Dict[str, List] = defaultdict(list)
        per_team_all: Dict[str, List] = defaultdict(list)

        for pair_key, home_vars in per_pair_home.items():
            away_vars = per_pair_away.get(pair_key, [])
            if not home_vars or not away_vars:
                # No degree of freedom — skip silently to match legacy behavior.
                # (Legacy printed a warning; the warning is noisy in tests, drop.)
                per_team_home[pair_key[0]].extend(home_vars)
                per_team_all[pair_key[0]].extend(home_vars)
                per_team_all[pair_key[0]].extend(away_vars)
                continue

            home_var = model.NewIntVar(
                0, len(home_vars), f'pair_home_{pair_key[0]}_vs_{pair_key[1]}',
            )
            model.Add(home_var == sum(home_vars))
            total_var = model.NewIntVar(
                0, len(home_vars) + len(away_vars),
                f'pair_total_{pair_key[0]}_vs_{pair_key[1]}',
            )
            model.Add(total_var == home_var + sum(away_vars))

            # home_var ∈ [floor(total/2), ceil(total/2)] — implemented as the
            # equivalent `home_var * 2 in [total - 1, total + 1]` form so the
            # constraint stays linear without needing CP-SAT's division.
            model.Add(home_var * 2 >= total_var - 1)
            model.Add(home_var * 2 <= total_var + 1)
            constraints_added += 2

            per_team_home[pair_key[0]].extend(home_vars)
            per_team_all[pair_key[0]].extend(home_vars)
            per_team_all[pair_key[0]].extend(away_vars)

        # Apply aggregate per-team home/away balance.
        for team, h_vars in per_team_home.items():
            all_vars = per_team_all.get(team, [])
            if not h_vars or not all_vars:
                continue
            agg_home = model.NewIntVar(0, len(h_vars), f'agg_home_{team}')
            agg_total = model.NewIntVar(0, len(all_vars), f'agg_total_{team}')
            model.Add(agg_home == sum(h_vars))
            model.Add(agg_total == sum(all_vars))
            model.Add(agg_home * 2 >= agg_total - 1)
            model.Add(agg_home * 2 <= agg_total + 1)
            constraints_added += 2

        return constraints_added
