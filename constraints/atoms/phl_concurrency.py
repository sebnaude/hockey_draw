"""PHL games at Broadmeadow cannot share a (week, day, day_slot)."""
from collections import defaultdict

from constraints.atoms.base import Atom, BROADMEADOW, iter_phl_keys


class PHLConcurrencyAtBroadmeadow(Atom):
    canonical_name = 'PHLConcurrencyAtBroadmeadow'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        groups = defaultdict(list)
        for key, var in iter_phl_keys(X, data):
            location = key[10]
            if location != BROADMEADOW:
                continue
            week, day, day_slot = key[6], key[3], key[4]
            groups[(week, day, day_slot, location)].append(var)

        n = 0
        for vars_list in groups.values():
            if len(vars_list) > 1:
                model.Add(sum(vars_list) <= 1)
                n += 1
        return n
