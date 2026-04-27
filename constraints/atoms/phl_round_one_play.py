"""Every PHL team plays at least one game in round 1."""
from collections import defaultdict

from constraints.atoms.base import Atom, iter_phl_keys


class PHLRoundOnePlay(Atom):
    canonical_name = 'PHLRoundOnePlay'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        per_team = defaultdict(list)
        for key, var in iter_phl_keys(X, data):
            if key[8] != 1:
                continue
            per_team[key[0]].append(var)
            per_team[key[1]].append(var)

        if not per_team:
            return 0

        n = 0
        phl_teams = [t.name for t in data['teams'] if t.grade == 'PHL']
        for team in phl_teams:
            vars_list = per_team.get(team)
            if vars_list:
                model.Add(sum(vars_list) >= 1)
                n += 1
        return n
