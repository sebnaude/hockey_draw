# config/season_test.py
"""
"test" season configuration.

A clean, constraint-free twin of the 2026 season. It reuses EVERYTHING from
config/season_2026.py — same teams (data/2026/teams), same fields, timeslots,
field availabilities, club days, soft preferences, penalty weights and grade
round settings — but with NO hard game pins or eliminations:

    - forced_games   = []   (no FORCED_GAMES)
    - blocked_games  = []   (no BLOCKED_GAMES, including no perennial blocks)
    - locked_pairings = []  (no LOCKED_PAIRINGS pins)

Use this for baseline / unconstrained full-season solves to see what the model
produces before real-world club requests and venue rules are layered back in.

Because it derives from season_2026's SEASON_CONFIG, any later change to the
2026 fields/timeslots/teams flows through automatically.

Loading:
    from config import load_season_config, load_season_data
    cfg  = load_season_config('test')   # SEASON_CONFIG dict
    data = load_season_data('test')      # full solver data dict

Note: run.py's --year flag is typed int, so the standard CLI cannot select
this config as `--year test`; load it programmatically as shown above.
"""

import copy

from config.season_2026 import SEASON_CONFIG as _BASE_2026_CONFIG

# Deep-copy so mutating the constraint lists here never touches the shared
# 2026 config object (the modules are imported into the same process).
SEASON_CONFIG = copy.deepcopy(_BASE_2026_CONFIG)

# Strip every hard game pin / elimination — the whole point of this config.
SEASON_CONFIG['forced_games'] = []
SEASON_CONFIG['blocked_games'] = []
SEASON_CONFIG['locked_pairings'] = []


def get_season_data() -> dict:
    """Build the complete solver data dict for the constraint-free test season."""
    from utils import build_season_data
    return build_season_data(SEASON_CONFIG)
