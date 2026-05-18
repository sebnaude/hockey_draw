"""MaitlandAlternateHomeAway — soft penalty for non-alternating Maitland weekends.

## Why this atom exists (spec-012 Part B audit + decision)

### Audit

The existing combo of constraints around Maitland home/away weekends:

1. **`NonDefaultHomeGrouping`** (canonical, aka `MaitlandHomeGrouping`).
   Sliding window of size `max_consecutive_home + 1`. With the default
   `CONSTRAINT_DEFAULTS['maitland_max_consecutive_home'] = 1` the window size
   is 2 and the constraint reads `sum(home_indicators[w, w+1]) <= 1` for every
   consecutive playable-week pair. That **hard-forbids** two consecutive home
   weekends ("HH").

2. **`AwayAtNonDefaultGrouping`** (canonical, aka `AwayAtMaitlandGrouping`).
   Caps the number of *distinct away clubs* visiting the Maitland venue per
   weekend. Affects density-per-weekend, **not** the alternation pattern.

3. **`AwayClubHomeWeekendsCount`** (spec-004).
   Pins the exact **count** of home weekends but does not dictate their
   distribution among the playable weeks.

### What the existing combo does NOT enforce

Maitland plays roughly 10 home + 12 away weekends across 22 playable weeks.
With HH hard-forbidden, the home weekends are forced spaced (gap ≥ 1 between
them). But two consecutive *away* weekends ("AA") are completely
unconstrained. Patterns like `H A H A H A H A A A A A H A A A A A H A A H`
are legal — the home weekends spread out at the start and clump-away at the
end. The convenor's intent (pairs of consecutive H/A weekends rather than
long runs) is not actively enforced — only one direction (HH) is hard-blocked.

### Decision: SHIP the soft atom

This atom adds a soft penalty per consecutive playable-week pair `(w, w+1)`
where both weekends are home OR both are away. Since HH is already
hard-forbidden, the HH branch always contributes 0 in production — coding it
symmetrically keeps the atom robust to future slack relaxations of the
hard rule. The AA branch is what does the work, discouraging long away
runs and pushing the solver toward an interleaved H A H A pattern.

Severity 4 (LOW), weight from `PENALTY_WEIGHTS['maitland_alternate_home_away']`
(default 50_000). Never blocks feasibility.

## Implementation notes

- **Scope.** Maitland only (per spec-012 "out of scope: Generalising H/A
  alternation to all away-based clubs"). Gosford has its own H/A semantics
  (`gosford_friday_games` budget + tiny Sunday-slot count) — a separate plan.
- **Indicators.** `home_ind[w] = OR(X-vars where field_location = Maitland
  Park AND Maitland team involved AND week = w)`. `any_ind[w] = OR(X-vars
  where Maitland team involved AND week = w)`. The "away" weekend indicator
  is `any_ind - home_ind` (effectively `away_ind = 1 iff any_ind = 1 AND
  home_ind = 0`).
- **Penalty per consecutive pair.** For each ordered pair `(w, w_next)` of
  playable weeks, define `both_home = AND(home[w], home[w_next])` and
  `both_away = AND(away[w], away[w_next])`. The total penalty is the sum of
  these BoolVars across all consecutive pairs.
- **Locked weeks** are skipped — the constraint only applies to pairs where
  BOTH weeks are unlocked. (If either week is locked, its indicators are
  fixed by the prior solution and the alternation isn't a free choice.)
- **Bye weeks.** When Maitland has no game in a given week (a bye), neither
  `home_ind` nor `away_ind` is 1 → neither `both_home` nor `both_away` fires
  → no penalty contribution. The pair is effectively skipped.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from constraints.atoms.base import Atom


# The single club this atom targets (per spec-012 scope).
TARGET_CLUB = 'Maitland'


class MaitlandAlternateHomeAway(Atom):
    """Soft penalty for consecutive Maitland weekends of the same type (HH or AA)."""

    canonical_name = 'MaitlandAlternateHomeAway'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        weight = (
            data.get('penalty_weights', {})
            .get('maitland_alternate_home_away', 50_000)
        )
        if weight == 0:
            return 0

        home_field_map: Dict[str, str] = data.get('home_field_map', {}) or {}
        home_venue = home_field_map.get(TARGET_CLUB)
        if not home_venue:
            # Maitland isn't a non-default-home club in this season config —
            # nothing to do.
            return 0

        locked_weeks: Set[int] = set(data.get('locked_weeks', set()) or set())
        team_to_club = {team.name: team.club.name for team in data.get('teams', [])}

        # Collect X-vars per week: any-week (Maitland involved) and home-week
        # (Maitland involved AND at Maitland Park).
        any_vars_by_week: Dict[int, List] = defaultdict(list)
        home_vars_by_week: Dict[int, List] = defaultdict(list)

        for key, var in X.items():
            if len(key) < 11:
                continue  # dummy
            if not key[3]:
                continue  # no day (defensive)
            t1, t2 = key[0], key[1]
            club1 = team_to_club.get(t1)
            club2 = team_to_club.get(t2)
            if TARGET_CLUB != club1 and TARGET_CLUB != club2:
                continue
            week = key[6]
            if week in locked_weeks:
                continue
            any_vars_by_week[week].append(var)
            if key[10] == home_venue:
                home_vars_by_week[week].append(var)

        # Build indicator BoolVars per week.
        weeks = sorted(any_vars_by_week.keys())
        if len(weeks) < 2:
            return 0

        home_ind: Dict[int, object] = {}
        away_ind: Dict[int, object] = {}
        for week in weeks:
            any_vars = any_vars_by_week[week]
            home_vars = home_vars_by_week.get(week, [])

            # any_ind = OR(any_vars)
            any_i = model.NewBoolVar(f'maitland_any_w{week}')
            if len(any_vars) == 1:
                model.Add(any_i == any_vars[0])
            else:
                model.AddMaxEquality(any_i, any_vars)

            # home_ind = OR(home_vars) — 0 when no home var that week.
            if home_vars:
                home_i = model.NewBoolVar(f'maitland_home_w{week}')
                if len(home_vars) == 1:
                    model.Add(home_i == home_vars[0])
                else:
                    model.AddMaxEquality(home_i, home_vars)
            else:
                # No home option this week → home_ind hard-fixed to 0.
                home_i = model.NewConstant(0)

            # away_ind = any_ind AND NOT home_ind. CP-SAT formulation:
            #   away_ind <= any_ind
            #   away_ind <= 1 - home_ind
            #   away_ind >= any_ind - home_ind
            away_i = model.NewBoolVar(f'maitland_away_w{week}')
            model.Add(away_i <= any_i)
            # away_i <= 1 - home_i  (only safe when home_i is a BoolVar/IntVar)
            model.Add(away_i + home_i <= 1)
            model.Add(away_i >= any_i - home_i)

            home_ind[week] = home_i
            away_ind[week] = away_i

        # Build the soft-penalty bucket.
        data.setdefault('penalties', {})
        bucket = data['penalties'].setdefault(
            'maitland_alternate_home_away',
            {'weight': weight, 'penalties': []},
        )

        # Penalty per consecutive playable-week pair.
        # We iterate pairs of *adjacent indices* in the sorted week list, which
        # respects no-play weeks (those have no X-vars and are absent from
        # `weeks`). Two playing weeks that are calendar-adjacent OR separated
        # only by no-play weeks are treated as consecutive — the convenor
        # evaluates alternation in terms of playable rounds, not raw calendar
        # weeks (see CLAUDE.md §"Weeks vs Rounds").
        #
        # Each pair contributes two BoolVars:
        #   both_home <= home[w0]
        #   both_home <= home[w1]
        #   both_home >= home[w0] + home[w1] - 1     (channels AND via arithmetic)
        # and symmetric for both_away. The objective sums them; each `1` adds
        # `weight` to the cost.
        n = 0
        for i in range(len(weeks) - 1):
            w0, w1 = weeks[i], weeks[i + 1]

            both_home = model.NewBoolVar(f'maitland_both_home_w{w0}_{w1}')
            model.Add(both_home <= home_ind[w0])
            model.Add(both_home <= home_ind[w1])
            model.Add(both_home >= home_ind[w0] + home_ind[w1] - 1)

            both_away = model.NewBoolVar(f'maitland_both_away_w{w0}_{w1}')
            model.Add(both_away <= away_ind[w0])
            model.Add(both_away <= away_ind[w1])
            model.Add(both_away >= away_ind[w0] + away_ind[w1] - 1)

            bucket['penalties'].append(both_home)
            bucket['penalties'].append(both_away)
            n += 2

        return n
