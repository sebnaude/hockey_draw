"""Cap PHL Friday-night games at Broadmeadow (NIHC)."""
from constraints.atoms.base import Atom, BROADMEADOW, iter_phl_keys


class BroadmeadowFridayCount(Atom):
    canonical_name = 'BroadmeadowFridayCount'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        defaults = data.get('constraint_defaults', {})
        max_friday = defaults.get('max_friday_broadmeadow', 3)

        friday_vars = [
            var
            for key, var in iter_phl_keys(X, data)
            if key[3] == 'Friday' and key[10] == BROADMEADOW
        ]
        if not friday_vars:
            return 0
        model.Add(sum(friday_vars) <= max_friday)
        return 1
