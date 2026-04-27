"""Total PHL Friday-night games at Gosford equals season target."""
from constraints.atoms.base import Atom, GOSFORD, iter_phl_keys


class GosfordFridayCount(Atom):
    canonical_name = 'GosfordFridayCount'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        defaults = data.get('constraint_defaults', {})
        target = defaults.get('gosford_friday_games', 8)

        gosford_vars = [
            var
            for key, var in iter_phl_keys(X, data)
            if key[3] == 'Friday' and key[10] == GOSFORD
        ]
        if not gosford_vars:
            return 0
        model.Add(sum(gosford_vars) == target)
        return 1
