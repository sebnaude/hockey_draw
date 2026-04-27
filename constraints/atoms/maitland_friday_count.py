"""Total PHL Friday games at Maitland Park equals season target.

Only Gosford-vs-Maitland games can land on Friday at Maitland Park (enforced by
`home_field_map` + season BLOCKED_GAMES); this atom counts the resulting vars.
"""
from constraints.atoms.base import Atom, MAITLAND, iter_phl_keys


class MaitlandFridayCount(Atom):
    canonical_name = 'MaitlandFridayCount'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        defaults = data.get('constraint_defaults', {})
        target = defaults.get('maitland_friday_games', 2)

        maitland_vars = [
            var
            for key, var in iter_phl_keys(X, data)
            if key[3] == 'Friday' and key[10] == MAITLAND
        ]
        if not maitland_vars:
            return 0
        model.Add(sum(maitland_vars) == target)
        return 1
