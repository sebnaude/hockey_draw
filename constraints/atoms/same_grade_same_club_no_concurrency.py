"""Same-grade, same-club teams must not play simultaneously (HARD).

Extracted from the legacy `ClubGradeAdjacencyConstraint`. The adjacent-grade
soft portion was removed entirely under spec-007; only the genuinely-
fundamental same-grade-same-club no-concurrency rule survives.

A club that fields multiple teams in the same grade (e.g. Tigers 3rd-A and
Tigers 3rd-B) must not have those two teams play different opponents at the
same (week, day_slot). The rule is enforced per (club, grade, week, day_slot)
bucket: the sum of cross-club games involving any duplicate-set team from
that club must be <= 1.

Intra-club derby games (Tigers 3rd-A vs Tigers 3rd-B) involve both duplicate
teams in one variable — there is nothing concurrent about them. NoDoubleBookingTeams
already prevents that single variable conflicting with other team appearances.

Severity 1 (CRITICAL): one parent literally cannot be in two simultaneous
games of their own kid's grade.
"""
from collections import defaultdict

from constraints.atoms.base import Atom


class SameGradeSameClubNoConcurrency(Atom):
    canonical_name = 'SameGradeSameClubNoConcurrency'
    atom_group = 'ClubGradeAdjacency'  # legacy lineage; the cluster is obsolete

    def apply(self, model, X, data, registry) -> int:
        teams = data['teams']
        locked_weeks = set(data.get('locked_weeks', set()))

        # team -> club name lookup
        team_club = {t.name: t.club.name for t in teams}

        # (club, grade) -> set of duplicate team names (clubs with >=2 teams in
        # the same grade). Only these participate in the constraint.
        per_club_grade = defaultdict(list)
        for t in teams:
            per_club_grade[(t.club.name, t.grade)].append(t.name)
        dup_team_names = {
            key: set(names) for key, names in per_club_grade.items()
            if len(names) > 1
        }
        if not dup_team_names:
            return 0

        # (club, grade, week, day_slot) -> list of decision variables for
        # cross-club games whose own-club participant is a duplicate-set team.
        buckets = defaultdict(list)

        for key, var in X.items():
            if len(key) < 11:
                continue
            if not key[3]:  # dummy / no-day
                continue
            t1, t2, grade, _day, day_slot, _time, week, _date, _round_no, _fname, _floc = key
            if locked_weeks and week in locked_weeks:
                continue
            c1 = team_club.get(t1)
            c2 = team_club.get(t2)
            if c1 is None or c2 is None:
                continue
            if c1 == c2:
                # Intra-club derby uses one variable for both teams -- nothing
                # to constrain; the rule fires only when two different
                # variables put two duplicate-set teammates in the same slot.
                continue
            dup_set1 = dup_team_names.get((c1, grade))
            if dup_set1 and t1 in dup_set1:
                buckets[(c1, grade, week, day_slot)].append(var)
            dup_set2 = dup_team_names.get((c2, grade))
            if dup_set2 and t2 in dup_set2:
                buckets[(c2, grade, week, day_slot)].append(var)

        n = 0
        for vars_list in buckets.values():
            if len(vars_list) > 1:
                model.Add(sum(vars_list) <= 1)
                n += 1
        return n
