"""Force exactly one PHL Friday Gosford game in each configured round."""
from collections import defaultdict

from constraints.atoms.base import Atom, GOSFORD, iter_phl_keys


class GosfordFridayRoundsForced(Atom):
    canonical_name = 'GosfordFridayRoundsForced'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        defaults = data.get('constraint_defaults', {})
        rounds = set(defaults.get('gosford_friday_rounds', [2, 4, 5, 9, 10]))
        if not rounds:
            return 0

        per_round = defaultdict(list)
        for key, var in iter_phl_keys(X, data):
            if key[3] == 'Friday' and key[10] == GOSFORD:
                per_round[key[8]].append(var)

        n = 0
        for round_no, vars_list in per_round.items():
            if round_no in rounds:
                model.Add(sum(vars_list) == 1)
                n += 1
        return n
