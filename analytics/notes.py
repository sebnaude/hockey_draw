# analytics/notes.py
"""
Weekend notes builder for the published season draw export.

Implements ``build_weekend_notes()`` which merges hand-authored ``notes.json``
entries with opt-in ``'note'`` fields from BLOCKED_GAMES, FORCED_GAMES, and
PREFERRED_WEEKENDS into a ``{week_number: [formatted note lines]}`` dict
suitable for passing to ``DrawStorage.export_schedule_xlsx(weekend_notes=...)``.

See spec-028 for full design rationale and category/opt-in rules.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date as _date
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from analytics.storage import DrawStorage

logger = logging.getLogger(__name__)

# Stable category ordering for output lines within a week.
# "Club Day" (spec-029) slots between Request and Preferred; the existing
# categories keep their relative order — only their absolute ints shift down.
_CATEGORY_ORDER: Dict[str, int] = {
    "Field":     0,
    "Request":   1,
    "Club Day":  2,
    "Preferred": 3,
    "Blocked":   4,
    "Forced":    5,
}
_DEFAULT_CATEGORY_ORDER = 6  # unknown categories go last


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_weekend_notes(
    draw: "DrawStorage",
    data: dict,
    notes_path: Optional[str] = None,
) -> Dict[int, List[str]]:
    """Return {week_number: [formatted note lines]} for the export Notes column.

    Sources, merged then deduped per week (see spec-028, spec-029):
      - hand-authored notes_path JSON (default ``data/{year}/notes.json``)
      - opt-in ``'note'`` entries in ``data['blocked_games']``,
        ``['forced_games']``, and ``['preferred_weekends']``
      - opt-in ``'note'`` entries in ``data['club_days']`` (spec-029): a club
        day whose value is a dict carrying a truthy ``'note'`` → ``Club Day:``.

    Each output line is ``"Category: text"``.  Order within a week is stable:
    Field, Request, Club Day, Preferred, Blocked, Forced; ties broken by
    insertion order.  Duplicates (exact formatted string) are removed.

    Args:
        draw:        A ``DrawStorage`` instance — used to build the date→week map.
        data:        The season data dict returned by ``load_season_data(year)``.
                     Must contain ``'year'``, ``'blocked_games'``,
                     ``'forced_games'``, ``'preferred_weekends'``, and
                     ``'club_days'`` keys.
        notes_path:  Path to the hand-authored JSON file.  Defaults to
                     ``data/{year}/notes.json``.  A missing file is treated as
                     ``{}``; malformed JSON raises a clear ``ValueError``.

    Returns:
        Dict mapping week number to a deduplicated, stable-ordered list of
        ``"Category: text"`` strings.  Weeks with no notes are absent from the
        dict (not present with an empty list).
    """
    year = data["year"]

    # --- resolve notes_path default ---
    if notes_path is None:
        notes_path = str(Path("data") / str(year) / "notes.json")

    # --- build date → week map from the draw ---
    date_to_week = _date_to_week_map(draw)

    # --- get season start for off-draw resolution ---
    start_date = _get_season_start(year)
    end_date = _get_season_end(year)

    # raw list of (category_sort_key, category, text, week) tuples
    raw: List[tuple] = []

    # 1. Hand-authored notes.json
    hand_notes = _load_notes_json(notes_path)
    for iso_date, entries in hand_notes.items():
        week = _resolve_week(iso_date, date_to_week, start_date, end_date, source=f"notes.json:{iso_date}")
        if week is None:
            continue
        for entry in entries:
            category = entry.get("category", "Note")
            text = entry["text"]
            raw.append((_category_sort(category), category, text, week))

    # 2. Auto-derived from blocked_games, forced_games, preferred_weekends
    raw.extend(_derive_from_config(
        data.get("blocked_games", []),
        "Blocked",
        date_to_week,
        start_date,
        end_date,
    ))
    raw.extend(_derive_from_config(
        data.get("forced_games", []),
        "Forced",
        date_to_week,
        start_date,
        end_date,
    ))
    raw.extend(_derive_from_preferred(
        data.get("preferred_weekends", []),
        date_to_week,
        start_date,
        end_date,
    ))
    # 3. Auto-derived from club_days (spec-029). NOTE: a dict, not a list.
    raw.extend(_derive_from_club_days(
        data.get("club_days", {}),
        date_to_week,
        start_date,
        end_date,
    ))

    # --- group, dedup, sort ---
    # accumulate per-week; preserve insertion order within each sort bucket
    week_lines: Dict[int, List[tuple]] = defaultdict(list)
    for sort_key, category, text, week in raw:
        week_lines[week].append((sort_key, category, text))

    result: Dict[int, List[str]] = {}
    for week, tuples in week_lines.items():
        seen: set = set()
        ordered: List[str] = []
        # stable sort by category sort key (insertion order preserved for ties
        # because Python's sort is stable and we appended in insertion order)
        for sort_key, category, text in sorted(tuples, key=lambda t: t[0]):
            line = f"{category}: {text}"
            if line not in seen:
                seen.add(line)
                ordered.append(line)
        if ordered:
            result[week] = ordered

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_notes_json(notes_path: str) -> Dict[str, List[Dict]]:
    """Load and normalise the hand-authored notes.json file.

    Returns a dict keyed by ISO date string → list of ``{"category", "text"}``
    dicts.  Missing file → ``{}``.  Malformed JSON → ``ValueError`` naming the
    file.  Non-date keys (e.g. ``_comment``) are silently skipped.
    """
    path = Path(notes_path)
    if not path.exists():
        logger.info("notes.json not found at %s — treating as empty", notes_path)
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"notes.json at '{notes_path}' contains malformed JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"notes.json at '{notes_path}' must be a JSON object (dict), "
            f"got {type(raw).__name__}"
        )

    result: Dict[str, List[Dict]] = {}
    for key, value in raw.items():
        # Skip non-date keys such as _comment
        if not _is_iso_date(key):
            logger.info("notes.json: skipping non-date key '%s'", key)
            continue

        entries: List[Dict] = []
        if not isinstance(value, list):
            logger.info("notes.json: date '%s' value is not a list — skipping", key)
            continue

        for item in value:
            if isinstance(item, str):
                # shorthand bare string → category "Note"
                entries.append({"category": "Note", "text": item})
            elif isinstance(item, dict):
                if "text" not in item:
                    logger.info(
                        "notes.json: entry under '%s' missing 'text' key — skipping: %s",
                        key, item,
                    )
                    continue
                entries.append({
                    "category": item.get("category", "Note"),
                    "text": item["text"],
                })
            else:
                logger.info(
                    "notes.json: unexpected item type %s under '%s' — skipping",
                    type(item).__name__, key,
                )

        if entries:
            result[key] = entries

    return result


def _date_to_week_map(draw: "DrawStorage") -> Dict[str, int]:
    """Build a ``{ISO-date-str: week}`` map from a ``DrawStorage`` instance."""
    mapping: Dict[str, int] = {}
    for game in draw.games:
        # game.date is a string attribute per StoredGame definition
        d = game.date
        w = game.week
        if d not in mapping:
            mapping[d] = w
        # If multiple games on the same date (should all have same week), keep first.
    return mapping


def _resolve_week(
    iso_date: str,
    date_to_week: Dict[str, int],
    start_date: _date,
    end_date: _date,
    source: str = "",
) -> Optional[int]:
    """Resolve an ISO date string to a week number.

    Uses the draw's date→week map first.  For dates not in the draw (e.g. a
    blocked date where zero games were scheduled), falls back to 7-day bucketing
    from the season start: ``week = (date - start_date).days // 7 + 1``.

    Returns ``None`` (and logs INFO) if the resolved week is outside the season
    window or if the date cannot be parsed.
    """
    if iso_date in date_to_week:
        return date_to_week[iso_date]

    # Fallback: 7-day bucket
    try:
        d = _date.fromisoformat(iso_date)
    except ValueError:
        logger.info("Cannot parse date '%s' from %s — dropping note", iso_date, source)
        return None

    delta = (d - start_date).days
    week = delta // 7 + 1

    # Check bounds — season runs from week 1 to the end_date's bucket
    end_delta = (end_date - start_date).days
    max_week = end_delta // 7 + 1

    if week < 1 or week > max_week:
        logger.info(
            "Note date '%s' (%s) resolves to week %d which is outside the "
            "season window [1, %d] — dropping",
            iso_date, source, week, max_week,
        )
        return None

    return week


def _derive_from_config(
    entries: List[dict],
    category: str,
    date_to_week: Dict[str, int],
    start_date: _date,
    end_date: _date,
) -> List[tuple]:
    """Yield (sort_key, category, text, week) tuples from BLOCKED/FORCED entries.

    Opt-in rule: entry must have a truthy ``'note'`` key.
    Date requirement: entry must have a ``'date'`` key (ISO string).
    Entries without ``'date'`` but with ``'day'`` only are skipped (un-dateable).
    ``'note': True`` → fall back to ``description`` then ``reason``.
    """
    raw: List[tuple] = []
    sort_key = _category_sort(category)

    for entry in entries:
        note_val = entry.get("note")
        if not note_val:
            continue  # not opted in

        iso_date = entry.get("date")
        if iso_date is None:
            # Un-dateable — has 'day' but no 'date'
            logger.info(
                "%s entry has 'note' but no 'date' (un-dateable) — skipping: %s",
                category, entry.get("description", entry),
            )
            continue

        week = _resolve_week(
            str(iso_date),
            date_to_week,
            start_date,
            end_date,
            source=f"{category}:{entry.get('description', iso_date)}",
        )
        if week is None:
            continue

        if isinstance(note_val, str):
            text = note_val
        else:
            # note_val is True → fall back to description or reason
            text = entry.get("description") or entry.get("reason") or ""
            if not text:
                logger.info(
                    "%s entry has 'note': True but no description/reason — skipping: %s",
                    category, entry,
                )
                continue

        raw.append((sort_key, category, text, week))

    return raw


def _derive_from_preferred(
    entries: List[dict],
    date_to_week: Dict[str, int],
    start_date: _date,
    end_date: _date,
) -> List[tuple]:
    """Yield (sort_key, category, text, week) tuples from PREFERRED_WEEKENDS.

    Supports ``'date'`` (singular) and ``'dates'`` (plural list).
    Opt-in rule: entry must have a truthy ``'note'`` key.
    """
    category = "Preferred"
    sort_key = _category_sort(category)
    raw: List[tuple] = []

    for entry in entries:
        note_val = entry.get("note")
        if not note_val:
            continue

        if isinstance(note_val, str):
            text = note_val
        else:
            text = entry.get("description") or entry.get("reason") or ""
            if not text:
                logger.info(
                    "Preferred entry has 'note': True but no description/reason — skipping: %s",
                    entry,
                )
                continue

        # Collect date(s) from 'date' or 'dates'
        iso_dates: List[str] = []
        if "date" in entry:
            iso_dates.append(str(entry["date"]))
        if "dates" in entry and isinstance(entry["dates"], list):
            iso_dates.extend(str(d) for d in entry["dates"])

        if not iso_dates:
            # Opted-in but un-dateable (no 'date'/'dates') — mirror the
            # blocked/forced un-dateable log so the path is never silent.
            logger.info(
                "Preferred entry has 'note' but no 'date'/'dates' (un-dateable) "
                "— skipping: %s",
                entry.get("description", entry),
            )
            continue

        for iso_date in iso_dates:
            week = _resolve_week(
                iso_date,
                date_to_week,
                start_date,
                end_date,
                source=f"Preferred:{entry.get('description', iso_date)}",
            )
            if week is None:
                continue
            raw.append((sort_key, category, text, week))

    return raw


def _derive_from_club_days(
    club_days: dict,
    date_to_week: Dict[str, int],
    start_date: _date,
    end_date: _date,
) -> List[tuple]:
    """Yield (sort_key, category, text, week) tuples from CLUB_DAYS entries.

    ``club_days`` is a ``{club_name: value}`` **dict** (unlike the
    blocked/forced/preferred sources, which are lists).

    Opt-in rule (mirrors the other auto-derived sources): a club day is
    surfaced **iff** its value is a dict carrying a truthy ``'note'`` key.  A
    bare ``datetime``/``str`` value, or a dict without ``'note'``, is never
    surfaced — so seasons whose ``CLUB_DAYS`` are still bare datetimes stay
    silent.
      - ``'note': <str>`` → that string is the note text.
      - ``'note': True``  → falls back to ``"<ClubName> Club Day"``.
    """
    from utils import normalize_club_day

    category = "Club Day"
    sort_key = _category_sort(category)
    raw: List[tuple] = []

    for club_name, value in club_days.items():
        # Opt-in: only dict values carrying a truthy 'note' surface.
        if not isinstance(value, dict):
            continue
        note_val = value.get("note")
        if not note_val:
            continue

        text = note_val if isinstance(note_val, str) else f"{club_name} Club Day"

        # Resolve the club-day date via the canonical parser, then to ISO.
        date_val = normalize_club_day(value)[0]
        iso = _club_day_iso(date_val)
        if iso is None:
            logger.info(
                "Club Day for %s has an unparseable date (%r) — skipping note",
                club_name, date_val,
            )
            continue

        week = _resolve_week(
            iso, date_to_week, start_date, end_date,
            source=f"Club Day:{club_name}",
        )
        if week is None:
            continue

        raw.append((sort_key, category, text, week))

    return raw


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def _club_day_iso(date_val) -> Optional[str]:
    """Normalise a CLUB_DAYS date value to an ISO ``'YYYY-MM-DD'`` string.

    Handles ``datetime``/``date`` (via ``strftime``) and an already-ISO ``str``.
    Returns ``None`` for anything unparseable (caller logs + skips).
    """
    if isinstance(date_val, str):
        return date_val if _is_iso_date(date_val) else None
    if hasattr(date_val, "strftime"):  # datetime or date
        return date_val.strftime("%Y-%m-%d")
    return None


def _category_sort(category: str) -> int:
    return _CATEGORY_ORDER.get(category, _DEFAULT_CATEGORY_ORDER)


def _is_iso_date(s: str) -> bool:
    """Return True if ``s`` looks like an ISO date (YYYY-MM-DD)."""
    if len(s) != 10:
        return False
    try:
        _date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _get_season_start(year: int) -> _date:
    """Retrieve the season start date from the season config."""
    from config import load_season_config
    cfg = load_season_config(year)
    start = cfg["start_date"]
    if hasattr(start, "date"):
        return start.date()
    return start  # already a date object


def _get_season_end(year: int) -> _date:
    """Retrieve the season end date from the season config."""
    from config import load_season_config
    cfg = load_season_config(year)
    end = cfg["end_date"]
    if hasattr(end, "date"):
        return end.date()
    return end
