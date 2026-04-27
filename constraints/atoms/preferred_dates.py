"""Soft penalty for missing the PHL preferred-date target of one game per date."""
from collections import defaultdict

from constraints.atoms.base import Atom, iter_phl_keys
from utils import get_nearest_week_by_date


class PreferredDates(Atom):
    canonical_name = 'PreferredDates'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        phl_prefs = data.get('phl_preferences', {})
        allowed_keys = {'preferred_dates'}
        invalid = set(phl_prefs.keys()) - allowed_keys
        if invalid:
            raise ValueError(
                f"Invalid keys found: {invalid}, currently do not support any keys "
                f"other than {allowed_keys}"
            )

        pref_dates = {
            d.date().strftime('%Y-%m-%d') for d in phl_prefs.get('preferred_dates', [])
        }
        if not pref_dates:
            return 0

        weight = data.get('penalty_weights', {}).get('phl_preferences', 10000)
        data.setdefault('penalties', {})
        data['penalties']['phl_preferences'] = {'weight': weight, 'penalties': []}

        per_date = defaultdict(list)
        for key, var in iter_phl_keys(X, data):
            date_str = key[7]
            if date_str in pref_dates:
                per_date[date_str].append(var)

        locked_weeks = set(data.get('locked_weeks', set()))
        timeslots = data['timeslots']
        n = 0
        for date_str, vars_list in per_date.items():
            if not vars_list:
                continue
            week_no = get_nearest_week_by_date(date_str, timeslots)
            if week_no in locked_weeks:
                continue
            pen = model.NewIntVar(0, len(vars_list), f'u_pref_date_{date_str}')
            model.AddAbsEquality(pen, sum(vars_list) - 1)
            data['penalties']['phl_preferences']['penalties'].append(pen)
            n += 1
        return n
