"""SOFT analogue of ClubDayParticipation (spec-027 regen-soft).

The hard atom ``ClubDayParticipation`` forbids any team in a club from having
ZERO games on its designated club-day date (it asserts ``sum(team_vars) >= 1``
for every such team). That is a hard clause: with FORCED placements or other
constraints it can make the model infeasible.

This SOFT analogue keeps the model feasible for ANY assignment of X. Instead of
forbidding non-participation, it emits one penalty BoolVar per club-day team
that is 1 exactly when that team plays NO game on its club-day date (the
violation amount), weighted by ``penalty_weights['regen_club_day_participation']``.

Penalty unit: 1 unit per club-day team that fails to play on its club-day.

The linearization pins the penalty var ``v`` exactly for 0/1 inputs:
  v >= 1 - sum(team_vars)        # if the team has no game that day, v must be 1
  v <= 1 - team_var  (each var)  # if the team plays any game that day, v is 0

All skip / no-op conditions and the club-day lookup are inherited from the
shared helpers, mirroring the hard atom.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries,
)

# Default penalty weight when `penalty_weights['regen_club_day_participation']`
# is unset. Large — a regen-soft analogue of a CRITICAL/HIGH hard rule, so the
# solver only ever leaves a club-day team idle when there is no other choice.
REGEN_CLUB_DAY_PARTICIPATION_DEFAULT_WEIGHT = 40000


class ClubDayParticipationRegenSoft(Atom):
    """SOFT: penalise each club-day team with no game on its club-day date."""

    canonical_name = 'ClubDayParticipationRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_club_day_participation',
            REGEN_CLUB_DAY_PARTICIPATION_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_club_day_participation',
            {'weight': weight, 'penalties': []},
        )

        club_team_lookup = {}
        for team in data['teams']:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        n = 0
        for club_name, date_str, _opponent in parse_club_day_entries(data):
            club_team_names = club_team_lookup.get(club_name, [])
            game_keys = club_day_game_keys(X, club_team_names, date_str)
            if not game_keys:
                raise ValueError(
                    f'No games found for club {club_name} on {date_str}'
                )
            for team in club_team_names:
                team_vars = [
                    X[key] for key in game_keys
                    if team in (key[0], key[1])
                ]
                if not team_vars:
                    continue
                # v == 1 iff the team plays NO game on its club-day date.
                v = model.NewBoolVar(
                    f'regen_club_day_absent_{team}_{date_str}'
                )
                model.Add(v >= 1 - sum(team_vars))
                for tv in team_vars:
                    model.Add(v <= 1 - tv)
                bucket['penalties'].append(v)
                n += 1
        return n
