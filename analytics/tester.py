# analytics/tester.py
"""
Draw Testing and Modification Tool.

This module provides utilities for:
1. Loading and modifying draws
2. Running constraint violation tests
3. Generating violation reports
4. What-if analysis for game moves

Usage:
    # Load a draw
    tester = DrawTester.from_file("draw.json", data)
    
    # Move a game and test
    tester.move_game("G00123", new_week=5, new_day_slot=2, new_time="14:00")
    report = tester.run_violation_check()
    print(report)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.storage import DrawStorage, StoredGame
from models import Team, Club, Grade


def _registry_normalize(name):
    """Lazy-import wrapper for normalize_constraint_name."""
    from constraints.registry import normalize_constraint_name
    return normalize_constraint_name(name)

def _registry_canonical_for_solver(name):
    """Lazy-import wrapper for get_canonical_for_solver_name."""
    from constraints.registry import get_canonical_for_solver_name
    return get_canonical_for_solver_name(name)

def _registry_get_slack_key(name):
    """Lazy-import wrapper for get_slack_key."""
    from constraints.registry import get_slack_key
    return get_slack_key(name)

def _registry_get_info(canonical_name):
    """Lazy-import wrapper to get ConstraintInfo from registry."""
    from constraints.registry import CONSTRAINT_REGISTRY
    return CONSTRAINT_REGISTRY.get(canonical_name)


# Severity levels mapping - lower number = more severe
# Level 1: Core constraints that must never be broken
# Level 2: Important structural constraints (club days, team conflicts)
# Level 3: Team preferences and spacing constraints
# Level 4: Soft optimization constraints
CONSTRAINT_SEVERITY_LEVELS = {
    # Level 1 - CRITICAL (must never break)
    'NoDoubleBookingTeams': 1,
    'NoDoubleBookingFields': 1,
    'EqualGames': 1,
    'BalancedMatchups': 1,
    'PHLAnd2ndAdjacency': 1,  # spec-014 (was PHLAndSecondGradeAdjacency)
    'PHLAndSecondGradeTimes': 1,
    'FiftyFiftyHomeAway': 1,
    # spec-018: `MaxMaitlandHomeWeekends` / `MaitlandHomeGrouping` removed —
    # venue back-to-back home-weekend rule deleted.
    # spec-016: NIHC fill order is now a SOFT symmetry-breaker (severity 5,
    # was 1), reported as soft pressure (see the metric_value on its
    # violations). Listed at level 5 below.

    'EqualMatchUpSpacing': 1,

    # Level 2 - HIGH (structural, club-specific)
    'ClubDayConstraint': 2,
    # spec-018: `AwayAtMaitlandGrouping` removed — away-clubs-per-week cap deleted.
    'TeamConflict': 2,

    # Level 3 - MEDIUM (spacing, alignment, game spread)
    'ClubGradeAdjacency': 3,
    'ClubVsClubAlignment': 3,
    'ClubGameSpread': 3,

    # spec-024: Level 4 MaximiseClubsPerTimeslotBroadmeadow /
    # MinimiseClubsOnAFieldBroadmeadow entries removed (constraints deleted).
    
    # spec-021: HARD anchored earliest-slot fill (was soft EnsureBestTimeslotChoices).
    'VenueEarliestSlotFill': 2,
    # spec-021: HARD cross-grade club no-concurrency (extracted from ClubGameSpread).
    'ClubNoConcurrentSlot': 2,
    # Level 5 - VERY LOW (timeslot preferences)
    'PreferredTimesConstraint': 5,
    # spec-016: NIHC field-fill order — soft symmetry-breaker.
    'NIHCFillWFBeforeEF': 5,
    'NIHCFillEFBeforeSF': 5,
    # spec-020: PreferredGames — soft, weighted FORCED analogue. Deviations are
    # soft pressure (metric_value = deviation penalty), never hard violations.
    'PreferredGames': 5,

    # Config-driven checks
    'ForcedGames': 1,   # CRITICAL - forced games must happen
    'BlockedGames': 1,  # CRITICAL - blocked games must not happen
    'LockedPairings': 1,  # CRITICAL - spec-025 pins must keep their date
}

# Mapping from severity level to label
SEVERITY_LEVEL_LABELS = {
    1: 'CRITICAL',
    2: 'HIGH',
    3: 'MEDIUM',
    4: 'LOW',
    5: 'VERY LOW',
}


def _severity_level_for(constraint_name: str) -> int:
    """spec-023 (Unit D): resolve a constraint/check name to its severity level,
    reading FROM THE REGISTRY (``ConstraintInfo.severity_level``) as the single
    source of truth — the same field ``resolve_group('severity_N')`` resolves
    over. Falls back to the local ``CONSTRAINT_SEVERITY_LEVELS`` map only for
    tester-only check labels that have no canonical registry entry
    (``EqualGames``, ``BalancedMatchups``, ``FiftyFiftyHomeAway``). Behaviour is
    identical to the old map for every registry-backed name (verified: zero
    deltas across all tester labels)."""
    from constraints.registry import CONSTRAINT_REGISTRY, normalize_constraint_name
    canonical = normalize_constraint_name(constraint_name)
    if canonical and canonical in CONSTRAINT_REGISTRY:
        return CONSTRAINT_REGISTRY[canonical].severity_level
    return CONSTRAINT_SEVERITY_LEVELS.get(constraint_name, 5)


@dataclass
class Violation:
    """Represents a single constraint violation."""
    constraint: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'VERY LOW'
    message: str
    affected_games: List[str] = field(default_factory=list)
    week: Optional[int] = None
    severity_level: int = 5  # 1-5, lower = worse
    # Phase 7a: structured aggregation hooks. Atoms populate these so
    # ViolationReport.breakdown can roll up by club / metric without
    # re-deriving from `message`.
    affected_clubs: List[str] = field(default_factory=list)
    metric_value: Optional[float] = None

    def __str__(self) -> str:
        games_str = f" [{', '.join(self.affected_games)}]" if self.affected_games else ""
        return f"[L{self.severity_level}-{self.severity}] {self.constraint}: {self.message}{games_str}"

    @classmethod
    def create(cls, constraint: str, message: str,
               affected_games: List[str] = None, week: Optional[int] = None,
               affected_clubs: List[str] = None,
               metric_value: Optional[float] = None) -> 'Violation':
        """Factory method that auto-determines severity from constraint name."""
        level = _severity_level_for(constraint)
        severity = SEVERITY_LEVEL_LABELS.get(level, 'VERY LOW')
        return cls(
            constraint=constraint,
            severity=severity,
            message=message,
            affected_games=affected_games or [],
            week=week,
            severity_level=level,
            affected_clubs=affected_clubs or [],
            metric_value=metric_value,
        )


@dataclass
class ViolationBreakdown:
    """Phase 7a: structured aggregation of violations.

    `by_club`: club_name -> violations involving that club.
    `by_type`: canonical constraint name -> violations of that type.
    `by_severity`: severity label ('CRITICAL', 'HIGH', ...) -> violations.
    `soft_pressure`: canonical_name -> rollup of how close clubs are to limit.
        keys per entry: at_limit (int), over_limit (int), total_penalty (number),
        worst_club (str|None), worst_value (number|None).
    """
    by_club: Dict[str, List['Violation']] = field(default_factory=dict)
    by_type: Dict[str, List['Violation']] = field(default_factory=dict)
    by_severity: Dict[str, List['Violation']] = field(default_factory=dict)
    soft_pressure: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_violations(cls, violations: List['Violation']) -> 'ViolationBreakdown':
        by_club: Dict[str, List['Violation']] = defaultdict(list)
        by_type: Dict[str, List['Violation']] = defaultdict(list)
        by_severity: Dict[str, List['Violation']] = defaultdict(list)
        soft: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'at_limit': 0, 'over_limit': 0, 'total_penalty': 0,
                     'worst_club': None, 'worst_value': None}
        )

        for v in violations:
            by_type[v.constraint].append(v)
            by_severity[v.severity].append(v)
            for c in v.affected_clubs:
                by_club[c].append(v)

            if v.metric_value is not None:
                bucket = soft[v.constraint]
                bucket['over_limit'] += 1
                bucket['total_penalty'] += float(v.metric_value)
                if (bucket['worst_value'] is None
                        or v.metric_value > bucket['worst_value']):
                    bucket['worst_value'] = float(v.metric_value)
                    bucket['worst_club'] = (
                        v.affected_clubs[0] if v.affected_clubs else None
                    )

        return cls(
            by_club=dict(by_club), by_type=dict(by_type),
            by_severity=dict(by_severity), soft_pressure=dict(soft),
        )


@dataclass
class ConstraintResult:
    """Result of checking a single constraint."""
    constraint: str
    status: str  # 'PASSED', 'VIOLATED', 'SKIPPED'
    skip_reason: str = ''
    violations: List[Violation] = field(default_factory=list)
    slack_value: int = 0


@dataclass
class ViolationReport:
    """Complete violation report for a draw."""
    draw_description: str
    total_games: int
    violations: List[Violation] = field(default_factory=list)
    constraint_results: List[ConstraintResult] = field(default_factory=list)
    metadata_source: str = ''  # 'draw_json', 'checkpoint', 'manual', 'none'

    @property
    def breakdown(self) -> 'ViolationBreakdown':
        """Phase 7a: structured aggregation by club / type / severity / soft-pressure."""
        return ViolationBreakdown.from_violations(self.violations)
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def highest_severity_level(self) -> int:
        """Return the highest (worst) severity level. Lower number = worse."""
        if not self.violations:
            return 0  # No violations
        return min(v.severity_level for v in self.violations)
    
    @property
    def highest_severity_label(self) -> str:
        """Return the label for the highest severity level."""
        level = self.highest_severity_level
        if level == 0:
            return "NONE"
        return SEVERITY_LEVEL_LABELS.get(level, "UNKNOWN")
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'CRITICAL')
    
    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'HIGH')
    
    @property
    def medium_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'MEDIUM')
    
    @property
    def low_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'LOW')
    
    def count_by_level(self, level: int) -> int:
        """Count violations at a specific severity level."""
        return sum(1 for v in self.violations if v.severity_level == level)
    
    def violations_by_level(self, level: int) -> List[Violation]:
        """Get all violations at a specific severity level."""
        return [v for v in self.violations if v.severity_level == level]
    
    def summary(self) -> str:
        """Return a concise summary."""
        if not self.has_violations:
            return f"[PASS] No violations found in {self.total_games} games."
        
        return (
            f"[FAIL] {len(self.violations)} violations found\n"
            f"   Highest Severity: Level {self.highest_severity_level} ({self.highest_severity_label})\n"
            f"   Level 1 (CRITICAL): {self.count_by_level(1)}\n"
            f"   Level 2 (HIGH): {self.count_by_level(2)}\n"
            f"   Level 3 (MEDIUM): {self.count_by_level(3)}\n"
            f"   Level 4 (LOW): {self.count_by_level(4)}\n"
            f"   Level 5 (VERY LOW): {self.count_by_level(5)}"
        )
    
    def full_report(self) -> str:
        """Return a detailed report."""
        lines = [
            "=" * 60,
            "CONSTRAINT VIOLATION REPORT",
            "=" * 60,
            f"Draw: {self.draw_description}",
            f"Total Games: {self.total_games}",
            f"Total Violations: {len(self.violations)}",
            f"Highest Severity: Level {self.highest_severity_level} ({self.highest_severity_label})",
            "-" * 60
        ]
        
        if not self.violations:
            lines.append("[PASS] ALL CONSTRAINTS SATISFIED")
        else:
            # Group by severity level (1-5)
            for level in [1, 2, 3, 4, 5]:
                level_violations = self.violations_by_level(level)
                if level_violations:
                    label = SEVERITY_LEVEL_LABELS.get(level, 'UNKNOWN')
                    lines.append(f"\nLevel {level} - {label} ({len(level_violations)}):")
                    lines.append("-" * 40)
                    for v in level_violations:
                        lines.append(f"  • {v.constraint}")
                        lines.append(f"    {v.message}")
                        if v.affected_games:
                            lines.append(f"    Games: {', '.join(v.affected_games[:5])}")
                            if len(v.affected_games) > 5:
                                lines.append(f"    ... and {len(v.affected_games) - 5} more")

        # Constraint results summary (if available)
        if self.constraint_results:
            lines.append("")
            lines.append("-" * 60)
            lines.append(self.constraints_summary())
            if self.metadata_source:
                lines.append(f"Metadata source: {self.metadata_source}")
            skipped = [r for r in self.constraint_results if r.status == 'SKIPPED']
            if skipped:
                lines.append(f"Skipped: {', '.join(r.constraint for r in skipped)}")

        lines.append("=" * 60)
        return "\n".join(lines)
    
    def by_constraint(self) -> Dict[str, List[Violation]]:
        """Group violations by constraint name."""
        result = defaultdict(list)
        for v in self.violations:
            result[v.constraint].append(v)
        return dict(result)
    
    def compare_to(self, other: 'ViolationReport') -> Tuple[int, int, str]:
        """
        Compare this report to another. Used for finding best slot.
        
        Returns:
            Tuple of (severity_comparison, count_comparison, explanation)
            - severity_comparison: -1 if this is better, 0 if equal, 1 if worse
            - count_comparison: difference in total violations (negative = better)
            - explanation: Human readable comparison
        """
        # First compare by highest severity level (lower = worse, so we want higher)
        self_level = self.highest_severity_level if self.has_violations else 5
        other_level = other.highest_severity_level if other.has_violations else 5
        
        if self_level > other_level:  # Higher number = less severe = better
            return -1, 0, f"Better severity (L{self_level} vs L{other_level})"
        elif self_level < other_level:
            return 1, 0, f"Worse severity (L{self_level} vs L{other_level})"
        
        # Same severity level - compare by count at that level
        self_count_at_level = self.count_by_level(self_level) if self_level <= 4 else 0
        other_count_at_level = other.count_by_level(other_level) if other_level <= 4 else 0
        
        count_diff = self_count_at_level - other_count_at_level
        if count_diff < 0:
            return -1, count_diff, f"Fewer L{self_level} violations ({self_count_at_level} vs {other_count_at_level})"
        elif count_diff > 0:
            return 1, count_diff, f"More L{self_level} violations ({self_count_at_level} vs {other_count_at_level})"
        
        # Same count at highest level - compare total violations
        total_diff = len(self.violations) - len(other.violations)
        if total_diff < 0:
            return -1, total_diff, f"Fewer total violations ({len(self.violations)} vs {len(other.violations)})"
        elif total_diff > 0:
            return 1, total_diff, f"More total violations ({len(self.violations)} vs {len(other.violations)})"
        
        return 0, 0, "Equal"

    def constraints_summary(self) -> str:
        """Summary of passed/violated/skipped constraints."""
        passed = sum(1 for r in self.constraint_results if r.status == 'PASSED')
        violated = sum(1 for r in self.constraint_results if r.status == 'VIOLATED')
        skipped = sum(1 for r in self.constraint_results if r.status == 'SKIPPED')
        return f"Constraints: {passed} passed, {violated} violated, {skipped} skipped"


class DrawTester:
    """Main class for testing and modifying draws."""
    
    GRADE_ORDER = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
    
    def __init__(self, draw: DrawStorage, data: Dict,
                 constraints_applied=None, excluded_constraints=None):
        """
        Initialize tester with a draw and data.

        Args:
            draw: DrawStorage object (will be copied for modification)
            data: Data dict containing teams, grades, clubs, etc.
            constraints_applied: Optional list of solver constraint names that were applied.
                If provided, only matching tester checks will run. None = run all (legacy).
            excluded_constraints: Optional list of constraint names to skip.
        """
        # Deep copy to avoid modifying original
        self.original_draw = draw
        self.draw = DrawStorage(**draw.model_dump())
        self.data = data
        self.teams: List[Team] = data.get('teams', [])
        self.grades: List[Grade] = data.get('grades', [])
        self.clubs: List[Club] = data.get('clubs', [])

        # Build lookups
        self._team_to_club = {t.name: t.club.name for t in self.teams}
        self._team_to_grade = {t.name: t.grade for t in self.teams}

        # Constraint slack - loaded from data dict (set by solver via --slack flag)
        self.constraint_slack = dict(data.get('constraint_slack', {}))

        # Merge metadata slack from draw (caller overrides metadata)
        if hasattr(draw, 'metadata') and draw.metadata:
            meta_slack = draw.metadata.get('solver_config', {}).get('constraint_slack', {})
            if meta_slack:
                merged = dict(meta_slack)
                merged.update(self.constraint_slack)  # caller overrides metadata
                self.constraint_slack = merged

        # Constraint filtering: which constraints to run/skip
        self._constraints_applied = None  # None = run all (legacy mode)
        self._excluded_constraints = set()

        if constraints_applied is not None:
            self._constraints_applied = set()
            for name in constraints_applied:
                canonical = _registry_canonical_for_solver(name) or _registry_normalize(name)
                if canonical:
                    self._constraints_applied.add(canonical)

        if excluded_constraints:
            for name in excluded_constraints:
                canonical = _registry_canonical_for_solver(name) or _registry_normalize(name)
                if canonical:
                    self._excluded_constraints.add(canonical)

        # Build playable-week sets from timeslots (accounts for no-play weekends)
        # sunday_weeks: weeks with Sunday games (all grades play)
        # all_play_weeks: weeks with any games (includes Friday-only PHL weeks)
        self._sunday_weeks = set()
        self._all_play_weeks = set()
        for t in data.get('timeslots', []):
            if t.day:
                self._all_play_weeks.add(t.week)
                if t.day == 'Sunday':
                    self._sunday_weeks.add(t.week)

        # Modification log
        self.modifications: List[str] = []
    
    def _should_check(self, canonical_name: str) -> tuple:
        """Return (should_run: bool, skip_reason: str)."""
        if canonical_name in self._excluded_constraints:
            return False, 'excluded'
        if self._constraints_applied is not None:
            # Tester-only diagnostics always run (they have no solver equivalent)
            info = _registry_get_info(canonical_name)
            if info and info.tester_only:
                return True, ''
            if canonical_name not in self._constraints_applied:
                return False, 'not in constraints_applied'
        return True, ''

    @classmethod
    def _extract_metadata(cls, draw):
        """Extract tester-relevant config from draw metadata."""
        meta = getattr(draw, 'metadata', None) or {}
        solver_config = meta.get('solver_config', {})
        applied = meta.get('constraints_applied', [])
        applied_names = [c['name'] if isinstance(c, dict) else c for c in applied]
        return {
            'constraints_applied': applied_names or None,
            'excluded_constraints': solver_config.get('excluded_constraints', []),
            'constraint_slack': solver_config.get('constraint_slack', {}),
            'mode': meta.get('mode', ''),
        }

    @classmethod
    def from_file(cls, path: str, data: Dict, use_metadata: bool = False) -> "DrawTester":
        """Create tester from a saved draw file.

        Args:
            path: Path to draw JSON file.
            data: Season data dict.
            use_metadata: If True, extract constraints_applied/excluded/slack from
                draw metadata and pass them to the tester.
        """
        draw = DrawStorage.load(path)
        if use_metadata:
            meta = cls._extract_metadata(draw)
            return cls(draw, data,
                       constraints_applied=meta['constraints_applied'],
                       excluded_constraints=meta['excluded_constraints'])
        return cls(draw, data)
    
    @classmethod
    def from_X_solution(cls, X_solution: Dict, data: Dict, description: str = "") -> "DrawTester":
        """Create tester from X solution dict."""
        draw = DrawStorage.from_X_solution(X_solution, description)
        return cls(draw, data)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, data, description='Checkpoint'):
        """Load solution + metadata from checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint directory containing solution.pkl
                and optionally metadata.json.
            data: Season data dict.
            description: Description for the tester.
        """
        import pickle
        import json
        from pathlib import Path as _Path

        cp = _Path(checkpoint_dir)

        pkl_path = cp / 'solution.pkl'
        if not pkl_path.exists():
            raise FileNotFoundError(f"No solution.pkl in {checkpoint_dir}")
        with open(pkl_path, 'rb') as f:
            solution = pickle.load(f)

        tester = cls.from_X_solution(solution, data, description=description)

        # Load metadata and apply constraint info
        meta_path = cp / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

            applied = meta.get('constraints_applied', [])
            applied_names = [c['name'] if isinstance(c, dict) else c for c in applied]
            if applied_names:
                tester._constraints_applied = set()
                for name in applied_names:
                    canonical = _registry_canonical_for_solver(name) or _registry_normalize(name)
                    if canonical:
                        tester._constraints_applied.add(canonical)

            excluded = meta.get('excluded_constraints', [])
            for name in excluded:
                canonical = _registry_canonical_for_solver(name) or _registry_normalize(name)
                if canonical:
                    tester._excluded_constraints.add(canonical)

            # Merge slack
            meta_slack = meta.get('constraint_slack', {})
            if meta_slack:
                merged = dict(meta_slack)
                merged.update(tester.constraint_slack)
                tester.constraint_slack = merged

        return tester
    
    def reset(self) -> None:
        """Reset to original draw, discarding modifications."""
        self.draw = DrawStorage(**self.original_draw.model_dump())
        self.modifications = []
    
    # ============== Game Modification Methods ==============
    
    def move_game(
        self,
        game_id: str,
        new_week: Optional[int] = None,
        new_day: Optional[str] = None,
        new_day_slot: Optional[int] = None,
        new_time: Optional[str] = None,
        new_date: Optional[str] = None,
        new_field_name: Optional[str] = None,
        new_field_location: Optional[str] = None
    ) -> bool:
        """
        Move a game to a new timeslot.
        
        Args:
            game_id: The game ID to move
            new_week: Optional new week number
            new_day: Optional new day name
            new_day_slot: Optional new day slot
            new_time: Optional new time
            new_date: Optional new date
            new_field_name: Optional new field name
            new_field_location: Optional new field location
        
        Returns:
            True if game was found and modified, False otherwise
        """
        for i, game in enumerate(self.draw.games):
            if game.game_id == game_id:
                # Create modified game
                game_dict = game.model_dump()
                
                if new_week is not None:
                    game_dict['week'] = new_week
                if new_day is not None:
                    game_dict['day'] = new_day
                if new_day_slot is not None:
                    game_dict['day_slot'] = new_day_slot
                if new_time is not None:
                    game_dict['time'] = new_time
                if new_date is not None:
                    game_dict['date'] = new_date
                if new_field_name is not None:
                    game_dict['field_name'] = new_field_name
                if new_field_location is not None:
                    game_dict['field_location'] = new_field_location
                
                self.draw.games[i] = StoredGame(**game_dict)
                
                # Log modification
                mod_desc = f"Moved game {game_id}"
                changes = []
                if new_week is not None:
                    changes.append(f"week={new_week}")
                if new_day is not None:
                    changes.append(f"day={new_day}")
                if new_day_slot is not None:
                    changes.append(f"slot={new_day_slot}")
                if new_time is not None:
                    changes.append(f"time={new_time}")
                if new_field_name is not None:
                    changes.append(f"field={new_field_name}")
                
                self.modifications.append(f"{mod_desc}: {', '.join(changes)}")
                return True
        
        return False
    
    def swap_games(self, game_id_1: str, game_id_2: str) -> bool:
        """
        Swap the timeslots of two games.
        
        Returns:
            True if both games were found and swapped, False otherwise
        """
        game1 = None
        game2 = None
        idx1 = idx2 = -1
        
        for i, game in enumerate(self.draw.games):
            if game.game_id == game_id_1:
                game1 = game
                idx1 = i
            elif game.game_id == game_id_2:
                game2 = game
                idx2 = i
        
        if game1 is None or game2 is None:
            return False
        
        # Swap timeslot attributes
        swap_attrs = ['week', 'round_no', 'date', 'day', 'time', 'day_slot', 'field_name', 'field_location']
        
        g1_dict = game1.model_dump()
        g2_dict = game2.model_dump()
        
        for attr in swap_attrs:
            g1_dict[attr], g2_dict[attr] = g2_dict[attr], g1_dict[attr]
        
        self.draw.games[idx1] = StoredGame(**g1_dict)
        self.draw.games[idx2] = StoredGame(**g2_dict)
        
        self.modifications.append(f"Swapped games {game_id_1} and {game_id_2}")
        return True
    
    def find_game(
        self,
        team: Optional[str] = None,
        opponent: Optional[str] = None,
        week: Optional[int] = None,
        grade: Optional[str] = None
    ) -> List[StoredGame]:
        """Find games matching criteria."""
        results = []
        for game in self.draw.games:
            if team and team not in (game.team1, game.team2):
                continue
            if opponent and opponent not in (game.team1, game.team2):
                continue
            if team and opponent and not (
                (game.team1 == team and game.team2 == opponent) or
                (game.team2 == team and game.team1 == opponent)
            ):
                continue
            if week is not None and game.week != week:
                continue
            if grade and game.grade != grade:
                continue
            results.append(game)
        return results
    
    def find_available_slots(self, week: int, field_location: Optional[str] = None) -> List[Dict]:
        """
        Find available timeslots in a given week.
        
        Args:
            week: Week number to check
            field_location: Optional filter by location
            
        Returns:
            List of dicts with available slot information from timeslots config.
        """
        # Get all possible slots from data if available
        timeslots = self.data.get('timeslots', [])
        
        # Get slots used in this week
        used_slot_keys = set()
        for game in self.draw.games:
            if game.week == week:
                key = (game.day_slot, game.field_name, game.field_location)
                used_slot_keys.add(key)
        
        available = []
        for t in timeslots:
            if t.week != week:
                continue
            if field_location and t.field.location != field_location:
                continue
            
            key = (t.day_slot, t.field.name, t.field.location)
            if key not in used_slot_keys:
                available.append({
                    'week': t.week,
                    'day': t.day,
                    'day_slot': t.day_slot,
                    'time': t.time,
                    'date': t.date,
                    'round_no': t.round_no,
                    'field_name': t.field.name,
                    'field_location': t.field.location
                })
        
        return available
    
    def move_game_to_available_slot(
        self,
        game_id: str,
        target_week: int,
        target_day_slot: Optional[int] = None,
        target_field_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Move a game to an available slot.
        
        If day_slot or field_name not specified, picks first available.
        
        Args:
            game_id: ID of game to move
            target_week: Week to move the game to
            target_day_slot: Optional specific slot (auto-selects if None)
            target_field_name: Optional specific field (auto-selects if None)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Find the game
        game = None
        for g in self.draw.games:
            if g.game_id == game_id:
                game = g
                break
        
        if not game:
            return False, f"Game {game_id} not found"
        
        # Get available slots in target week
        available = self.find_available_slots(target_week)
        
        if not available:
            return False, f"No available slots in week {target_week}"
        
        # Filter by preferences if specified
        if target_day_slot is not None:
            available = [s for s in available if s['day_slot'] == target_day_slot]
        if target_field_name is not None:
            available = [s for s in available if s['field_name'] == target_field_name]
        
        if not available:
            return False, f"No matching available slots in week {target_week}"
        
        # Pick first available slot
        slot = available[0]
        
        # Move the game
        success = self.move_game(
            game_id,
            new_week=slot['week'],
            new_day=slot['day'],
            new_day_slot=slot['day_slot'],
            new_time=slot['time'],
            new_date=slot['date'],
            new_field_name=slot['field_name'],
            new_field_location=slot['field_location']
        )
        
        if success:
            msg = f"Moved {game_id} to week {target_week}, slot {slot['day_slot']}, {slot['field_name']}"
            return True, msg
        else:
            return False, "Failed to move game"
    
    def list_unused_slots(self, week: Optional[int] = None) -> List[Dict]:
        """
        List all unused slots, optionally filtered by week.
        
        Args:
            week: Optional week filter
            
        Returns:
            List of unused slot dicts
        """
        if week is not None:
            return self.find_available_slots(week)
        
        # Get all weeks
        weeks = sorted(set(g.week for g in self.draw.games))
        all_unused = []
        for w in weeks:
            all_unused.extend(self.find_available_slots(w))
        return all_unused
    
    def print_unused_slots(self, week: Optional[int] = None) -> None:
        """Print unused slots to console."""
        from collections import defaultdict
        
        if week is not None:
            unused = self.find_available_slots(week)
        else:
            weeks = sorted(set(t.week for t in self.data.get('timeslots', [])))
            unused = []
            for w in weeks:
                unused.extend(self.find_available_slots(w))
        
        if not unused:
            print("No unused slots found" + (f" in week {week}" if week else ""))
            return
        
        print(f"\n{'='*60}")
        print(f"UNUSED SLOTS" + (f" - Week {week}" if week else " - All Weeks"))
        print(f"{'='*60}")
        
        by_week = defaultdict(list)
        for slot in unused:
            by_week[slot['week']].append(slot)
        
        for wk in sorted(by_week.keys()):
            print(f"\nWeek {wk}:")
            for slot in sorted(by_week[wk], key=lambda x: (x['day_slot'], x['field_name'])):
                print(f"  Slot {slot['day_slot']}: {slot['field_name']} @ {slot['field_location']} "
                      f"({slot['day']} {slot['time']})")
        
        print(f"\nTotal unused: {len(unused)} slots")
    
    # ============== Find Best Slot ==============
    
    def find_best_slot_for_game(
        self,
        game_id: str,
        weeks: Optional[List[int]] = None,
        field_locations: Optional[List[str]] = None,
        max_results: int = 10,
        include_swaps: bool = True,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Find the best available slots for a game, ranked by severity level and violation count.
        
        This method tests each potential slot (both empty slots and swaps) and ranks them
        by:
        1. Lowest severity level of violations (higher number = better, no violations = best)
        2. Fewest violations at that severity level
        3. Fewest total violations
        
        Args:
            game_id: ID of the game to find a new slot for
            weeks: Optional list of weeks to consider (default: all weeks)
            field_locations: Optional list of field locations to consider
            max_results: Maximum number of results to return (default: 10)
            include_swaps: Whether to include potential game swaps (default: True)
            verbose: Print progress updates (default: True)
            
        Returns:
            List of dicts with slot info and violation report, sorted best to worst:
            [
                {
                    'slot': {...slot info...},
                    'type': 'empty' | 'swap',
                    'swap_with': game_id (if swap),
                    'report': ViolationReport,
                    'severity_level': int (0 = no violations, 1-5 = worst to least severe),
                    'violation_count': int,
                    'rank_explanation': str
                },
                ...
            ]
        """
        # Find the game to move
        game_to_move = None
        for game in self.draw.games:
            if game.game_id == game_id:
                game_to_move = game
                break
        
        if not game_to_move:
            raise ValueError(f"Game {game_id} not found")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"FINDING BEST SLOT FOR GAME: {game_id}")
            print(f"  {game_to_move.team1} vs {game_to_move.team2} ({game_to_move.grade})")
            print(f"  Currently: Week {game_to_move.week}, Slot {game_to_move.day_slot}, {game_to_move.field_name}")
            print(f"{'='*60}")
        
        # Get current violation report for comparison
        current_report = self.run_violation_check()
        if verbose:
            print(f"\nCurrent violations: {len(current_report.violations)}")
            print(f"  Highest severity: Level {current_report.highest_severity_level} ({current_report.highest_severity_label})")
        
        # Determine weeks to search
        if weeks is None:
            weeks = sorted(set(t.week for t in self.data.get('timeslots', [])))
        
        candidates = []
        tested_count = 0
        
        # Test empty slots
        if verbose:
            print(f"\nSearching empty slots in weeks: {weeks[:5]}..." if len(weeks) > 5 else f"\nSearching empty slots in weeks: {weeks}")
        
        for week in weeks:
            available_slots = self.find_available_slots(week, field_locations[0] if field_locations and len(field_locations) == 1 else None)
            
            # Filter by field location if specified
            if field_locations:
                available_slots = [s for s in available_slots if s['field_location'] in field_locations]
            
            for slot in available_slots:
                tested_count += 1
                
                # Create a fresh copy and test the move
                test_tester = DrawTester(self.draw, self.data)
                test_tester.move_game(
                    game_id,
                    new_week=slot['week'],
                    new_day=slot['day'],
                    new_day_slot=slot['day_slot'],
                    new_time=slot['time'],
                    new_date=slot['date'],
                    new_field_name=slot['field_name'],
                    new_field_location=slot['field_location']
                )
                
                report = test_tester.run_violation_check()
                
                candidates.append({
                    'slot': slot,
                    'type': 'empty',
                    'swap_with': None,
                    'report': report,
                    'severity_level': report.highest_severity_level if report.has_violations else 0,
                    'violation_count': len(report.violations),
                    'violations_by_level': {
                        1: report.count_by_level(1),
                        2: report.count_by_level(2),
                        3: report.count_by_level(3),
                        4: report.count_by_level(4),
                    }
                })
        
        if verbose:
            print(f"  Tested {tested_count} empty slots")
        
        # Test swaps with other games
        if include_swaps:
            swap_count = 0
            if verbose:
                print(f"\nSearching potential swaps...")
            
            for other_game in self.draw.games:
                if other_game.game_id == game_id:
                    continue
                
                # Skip if week not in search range
                if other_game.week not in weeks:
                    continue
                
                # Skip if field location not in filter
                if field_locations and other_game.field_location not in field_locations:
                    continue
                
                swap_count += 1
                
                # Create a fresh copy and test the swap
                test_tester = DrawTester(self.draw, self.data)
                test_tester.swap_games(game_id, other_game.game_id)
                
                report = test_tester.run_violation_check()
                
                slot_info = {
                    'week': other_game.week,
                    'day': other_game.day,
                    'day_slot': other_game.day_slot,
                    'time': other_game.time,
                    'date': other_game.date,
                    'round_no': other_game.round_no,
                    'field_name': other_game.field_name,
                    'field_location': other_game.field_location,
                }
                
                candidates.append({
                    'slot': slot_info,
                    'type': 'swap',
                    'swap_with': other_game.game_id,
                    'swap_game_info': f"{other_game.team1} vs {other_game.team2}",
                    'report': report,
                    'severity_level': report.highest_severity_level if report.has_violations else 0,
                    'violation_count': len(report.violations),
                    'violations_by_level': {
                        1: report.count_by_level(1),
                        2: report.count_by_level(2),
                        3: report.count_by_level(3),
                        4: report.count_by_level(4),
                    }
                })
            
            if verbose:
                print(f"  Tested {swap_count} potential swaps")
        
        # Sort candidates: 
        # 1. By severity_level ascending (0 = no violations, best; then 4, 3, 2, 1)
        # 2. By violation count at that level
        # 3. By total violation count
        def sort_key(c):
            level = c['severity_level']
            # Convert 0 to -1 so it sorts first (no violations is best)
            sort_level = -1 if level == 0 else level
            count_at_level = c['violations_by_level'].get(level, 0) if level > 0 else 0
            return (sort_level, count_at_level, c['violation_count'])
        
        candidates.sort(key=sort_key, reverse=True)  # Reverse because higher is better for our sort_level
        
        # Actually we want:
        # - severity_level 0 first (no violations)
        # - then severity_level 4, 3, 2, 1 (higher = less severe = better)
        # So we need to sort descending by a "goodness" score
        def goodness_key(c):
            level = c['severity_level']
            # Goodness: 5 for no violations, then 4,3,2,1 based on level
            goodness = 5 if level == 0 else (5 - level)
            count_at_level = c['violations_by_level'].get(level, 0) if level > 0 else 0
            # Return tuple for sorting (higher goodness = better, lower counts = better)
            return (goodness, -count_at_level, -c['violation_count'])
        
        candidates.sort(key=goodness_key, reverse=True)
        
        # Add rank explanation
        for i, c in enumerate(candidates):
            level = c['severity_level']
            if level == 0:
                c['rank_explanation'] = "No violations"
            else:
                label = SEVERITY_LEVEL_LABELS.get(level, 'UNKNOWN')
                count_at_level = c['violations_by_level'].get(level, 0)
                c['rank_explanation'] = f"Level {level} ({label}): {count_at_level} violations, {c['violation_count']} total"
        
        # Return top results
        results = candidates[:max_results]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"TOP {len(results)} RESULTS (of {len(candidates)} tested)")
            print(f"{'='*60}")
            
            for i, r in enumerate(results, 1):
                slot = r['slot']
                print(f"\n{i}. {r['rank_explanation']}")
                if r['type'] == 'empty':
                    print(f"   Type: Empty slot")
                else:
                    print(f"   Type: Swap with {r['swap_with']} ({r.get('swap_game_info', '')})")
                print(f"   Week {slot['week']}, Slot {slot['day_slot']}, {slot['field_name']} @ {slot['field_location']}")
                print(f"   {slot['day']} {slot['time']}")
                
                if r['violation_count'] > 0:
                    print(f"   Violations: L1={r['violations_by_level'][1]}, L2={r['violations_by_level'][2]}, "
                          f"L3={r['violations_by_level'][3]}, L4={r['violations_by_level'][4]}")
        
        return results
    
    def print_best_slot_report(self, game_id: str, **kwargs) -> None:
        """
        Find and print a detailed report of best slots for a game.
        
        This is a convenience wrapper around find_best_slot_for_game with verbose output.
        """
        results = self.find_best_slot_for_game(game_id, verbose=True, **kwargs)
        
        if not results:
            print("\n[X] No suitable slots found!")
            return
        
        best = results[0]
        print(f"\n{'='*60}")
        print("RECOMMENDATION")
        print(f"{'='*60}")
        
        if best['severity_level'] == 0:
            print("[OK] BEST OPTION: No constraint violations!")
        else:
            print(f"[WARNING] BEST OPTION: {best['rank_explanation']}")
        
        slot = best['slot']
        if best['type'] == 'empty':
            print(f"\nMove game to empty slot:")
        else:
            print(f"\nSwap with game {best['swap_with']} ({best.get('swap_game_info', '')}):")
        
        print(f"  Week {slot['week']}, Slot {slot['day_slot']}")
        print(f"  {slot['field_name']} @ {slot['field_location']}")
        print(f"  {slot['day']} {slot['time']}")
        
        if best['violation_count'] > 0:
            print(f"\nViolations that would result:")
            for v in best['report'].violations[:5]:
                print(f"  • [{v.severity_level}] {v.constraint}: {v.message}")
            if len(best['report'].violations) > 5:
                print(f"  ... and {len(best['report'].violations) - 5} more")
    
    # ============== Constraint Checking ==============
    
    def run_violation_check(self) -> ViolationReport:
        """Run constraint checks, respecting metadata if available.

        When constraints_applied is set (from draw metadata or explicit param),
        only those constraints will be checked. Excluded constraints are skipped.
        Tester-only diagnostics (e.g., ClubFieldConcentration) always run unless
        explicitly excluded.
        """
        violations = []
        constraint_results = []

        # Define the check mapping: (canonical_name, check_method)
        # Order matches the original flat list for consistent output
        checks = [
            # Level 1 - CRITICAL
            ('NoDoubleBookingTeams', self._check_no_double_booking_teams),
            ('NoDoubleBookingFields', self._check_no_double_booking_fields),
            # spec-003: NIHC field-fill ordering (WF -> EF -> SF). One method
            # covers both implications and returns per (date, day_slot)
            # violations tagged with the firing implication. Each canonical
            # name routes through `_check_nihc_field_fill_order` filtered to
            # its own implication so violations aren't double-counted.
            ('NIHCFillWFBeforeEF', self._check_nihc_fill_wf_before_ef),
            ('NIHCFillEFBeforeSF', self._check_nihc_fill_ef_before_sf),
            ('EqualGamesAndBalanceMatchUps', self._check_equal_games),
            ('EqualGamesAndBalanceMatchUps', self._check_balanced_matchups),
            ('FiftyFiftyHomeandAway', self._check_fifty_fifty_home_away),
            # spec-018: `_check_maitland_back_to_back` removed — venue
            # back-to-back home-weekend rule deleted.
            ('PHLAnd2ndAdjacency', self._check_phl_2nd_adjacency),
            ('PHLAndSecondGradeTimes', self._check_phl_second_grade_times),
            ('EqualMatchUpSpacing', self._check_equal_matchup_spacing),
            # spec-008 Part B: byes-as-first-class spacing.
            ('BalancedByeSpacing', self._check_balanced_bye_spacing),
            # Level 2 - HIGH
            # spec-018: `_check_maitland_away_clubs_limit` removed —
            # away-clubs-per-week cap deleted.
            ('ClubDay', self._check_club_day),
            ('TeamConflict', self._check_team_conflict),
            # Level 3 - MEDIUM
            ('ClubGradeAdjacency', self._check_club_grade_adjacency),
            ('TeamPairNoConcurrency', self._check_team_pair_no_concurrency),
            ('ClubVsClubAlignment', self._check_club_vs_club_alignment),
            ('ClubGameSpread', self._check_club_game_spread),
            # spec-021: cross-grade club no-concurrency (capacity-aware).
            ('ClubNoConcurrentSlot', self._check_club_no_concurrent_slot),
            ('ClubFieldConcentration', self._check_club_field_concentration),
            # spec-024: MaximiseClubsPerTimeslotBroadmeadow /
            # MinimiseClubsOnAFieldBroadmeadow checks removed (constraints deleted).
            # Level 5 - VERY LOW
            ('VenueEarliestSlotFill', self._check_venue_earliest_slot_fill),
            ('PreferredTimes', self._check_preferred_times),
            # Config-driven checks (forced/blocked games)
            ('ForcedGames', self._check_forced_games),
            ('BlockedGames', self._check_blocked_games),
            # spec-025: locked-pairing date pins must keep their date
            ('LockedPairings', self._check_locked_pairings),
            # spec-020: soft preferred-games deviation (reported as soft pressure)
            ('PreferredGames', self._check_preferred_games),
        ]

        checked_canonicals = set()
        for canonical_name, check_method in checks:
            should_run, skip_reason = self._should_check(canonical_name)

            if not should_run:
                if canonical_name not in checked_canonicals:
                    constraint_results.append(ConstraintResult(
                        constraint=canonical_name, status='SKIPPED',
                        skip_reason=skip_reason
                    ))
                    checked_canonicals.add(canonical_name)
                continue

            check_violations = check_method()
            violations.extend(check_violations)

            if canonical_name not in checked_canonicals:
                slack_val = self.constraint_slack.get(
                    _registry_get_slack_key(canonical_name) or canonical_name, 0)
                constraint_results.append(ConstraintResult(
                    constraint=canonical_name,
                    status='VIOLATED' if check_violations else 'PASSED',
                    violations=check_violations,
                    slack_value=slack_val
                ))
                checked_canonicals.add(canonical_name)
            elif check_violations:
                # Update existing result (e.g., BalancedMatchups after EqualGames)
                for r in constraint_results:
                    if r.constraint == canonical_name:
                        r.violations.extend(check_violations)
                        r.status = 'VIOLATED'

        metadata_source = 'none'
        if self._constraints_applied is not None:
            metadata_source = 'draw_json'

        return ViolationReport(
            draw_description=self.draw.description or "Modified Draw",
            total_games=len(self.draw.games),
            violations=violations,
            constraint_results=constraint_results,
            metadata_source=metadata_source,
        )
    
    def _check_no_double_booking_teams(self) -> List[Violation]:
        """Check that no team plays more than once per week."""
        violations = []
        team_games_per_week = defaultdict(list)
        
        for game in self.draw.games:
            team_games_per_week[(game.week, game.team1)].append(game.game_id)
            team_games_per_week[(game.week, game.team2)].append(game.game_id)
        
        for (week, team), games in team_games_per_week.items():
            if len(games) > 1:
                violations.append(Violation.create(
                    constraint="NoDoubleBookingTeams",
                    message=f"Team '{team}' plays {len(games)} games in week {week}",
                    affected_games=games,
                    week=week
                ))
        
        return violations
    
    def _check_no_double_booking_fields(self) -> List[Violation]:
        """Check that no field hosts more than one game per slot."""
        violations = []
        field_games = defaultdict(list)

        for game in self.draw.games:
            key = (game.date, game.day_slot, game.field_name)
            field_games[key].append(game.game_id)

        for (week, slot, field), games in field_games.items():
            if len(games) > 1:
                violations.append(Violation.create(
                    constraint="NoDoubleBookingFields",
                    message=f"Field '{field}' has {len(games)} games in week {week}, slot {slot}",
                    affected_games=games,
                    week=week
                ))

        return violations

    # ------------------------------------------------------------------
    # spec-003: NIHC field-fill order (WF -> EF -> SF).
    # ------------------------------------------------------------------

    def _nihc_field_usage(
        self,
    ) -> 'Dict[Tuple[str, int], Dict[str, List[str]]]':
        """Build {(date, day_slot): {field_name: [game_ids]}} for NIHC games.

        Cached on the instance so the WF/EF and EF/SF checks share work.
        The set of fields *appearing* in this map is the only source of
        truth for "is this field a real slot for this (date, day_slot)?"
        -- a field whose slot doesn't exist for the day simply has no
        scheduled games and therefore no entry. This mirrors the atom's
        skip rule.
        """
        cache = getattr(self, '_nihc_usage_cache', None)
        if cache is not None:
            return cache
        from constraints.atoms.base import BROADMEADOW
        usage: Dict[Tuple[str, int], Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for game in self.draw.games:
            if game.field_location != BROADMEADOW:
                continue
            usage[(game.date, game.day_slot)][game.field_name].append(
                game.game_id
            )
        cache = {k: dict(v) for k, v in usage.items()}
        self._nihc_usage_cache = cache
        return cache

    def _check_nihc_fill_wf_before_ef(self) -> List[Violation]:
        """Per (date, day_slot) at NIHC: violation if EF has a game and WF
        does not.

        Only applies where BOTH WF and EF would normally appear in the
        day's schedule. If WF is not a real slot for the day (i.e. no
        game at WF appears across the entire date in any draw, which we
        cannot reliably infer from a single (date, day_slot)), we still
        flag because at NIHC every slot has WF available. A real
        unavailability (e.g. field unavailability) is handled by the
        atom skipping the bucket -- since this is a tester-side check
        against a produced draw, the absence of any WF game across all
        slots of a date is the heuristic for "WF wasn't an option."
        """
        violations: List[Violation] = []
        usage = self._nihc_field_usage()
        # Build set of dates that had ANY WF game at all (used as a proxy
        # for "WF was a real option this day"). If WF was completely
        # unavailable across a date, we don't flag the bucket.
        wf_dates = {date for (date, _), fields in usage.items() if 'WF' in fields}

        for (date, day_slot), fields_at_slot in sorted(usage.items()):
            if 'EF' not in fields_at_slot:
                continue  # nothing on the LHS
            if 'WF' in fields_at_slot:
                continue  # WF used at this slot -> implication holds
            if date not in wf_dates:
                continue  # WF wasn't a real option this date at all
            ef_games = fields_at_slot['EF']
            violations.append(Violation.create(
                constraint='NIHCFillWFBeforeEF',
                message=(
                    f"NIHC field-fill order: date {date} slot {day_slot} "
                    f"uses EF without WF (prefer WF before EF — soft)"
                ),
                affected_games=ef_games,
                # spec-016: soft symmetry-breaker. metric_value rolls this into
                # the soft_pressure breakdown rather than a hard-failure count.
                metric_value=1.0,
            ))
        return violations

    def _check_nihc_fill_ef_before_sf(self) -> List[Violation]:
        """Per (date, day_slot) at NIHC: violation if SF has a game and EF
        does not.

        Same heuristic as the WF/EF check: a date with no EF game at all
        is treated as "EF wasn't an option," so the bucket is skipped.
        """
        violations: List[Violation] = []
        usage = self._nihc_field_usage()
        ef_dates = {date for (date, _), fields in usage.items() if 'EF' in fields}

        for (date, day_slot), fields_at_slot in sorted(usage.items()):
            if 'SF' not in fields_at_slot:
                continue
            if 'EF' in fields_at_slot:
                continue
            if date not in ef_dates:
                continue
            sf_games = fields_at_slot['SF']
            violations.append(Violation.create(
                constraint='NIHCFillEFBeforeSF',
                message=(
                    f"NIHC field-fill order: date {date} slot {day_slot} "
                    f"uses SF without EF (prefer EF before SF — soft)"
                ),
                affected_games=sf_games,
                # spec-016: soft symmetry-breaker — rolls into soft_pressure.
                metric_value=1.0,
            ))
        return violations

    def _check_equal_games(self) -> List[Violation]:
        """Check each team plays the expected number of games."""
        violations = []
        team_counts = defaultdict(int)
        
        for game in self.draw.games:
            team_counts[game.team1] += 1
            team_counts[game.team2] += 1
        
        num_rounds = self.data.get('num_rounds', {})
        
        for team in self.teams:
            expected = num_rounds.get(team.grade, 0)
            actual = team_counts.get(team.name, 0)
            
            if expected > 0 and actual != expected:
                violations.append(Violation.create(
                    constraint="EqualGames",
                    message=f"Team '{team.name}' has {actual} games (expected {expected})"
                ))
        
        return violations
    
    def _check_balanced_matchups(self) -> List[Violation]:
        """Check pair matchups are balanced (base or base+1)."""
        violations = []
        pair_counts = defaultdict(int)
        
        for game in self.draw.games:
            pair = tuple(sorted([game.team1, game.team2]))
            pair_counts[(game.grade, pair)] += 1
        
        num_rounds = self.data.get('num_rounds', {})
        
        for grade in self.grades:
            T = grade.num_teams
            R = num_rounds.get(grade.name, 0)
            
            if T < 2 or R == 0:
                continue
            
            base = R // (T - 1) if T % 2 == 0 else R // T
            
            for (g, pair), count in pair_counts.items():
                if g != grade.name:
                    continue
                
                if count < base:
                    violations.append(Violation.create(
                        constraint="BalancedMatchups",
                        message=f"Pair {pair} in {g} meets {count} times (min {base})"
                    ))
                elif count > base + 1:
                    violations.append(Violation.create(
                        constraint="BalancedMatchups",
                        message=f"Pair {pair} in {g} meets {count} times (max {base+1})"
                    ))
        
        return violations
    
    def _check_fifty_fifty_home_away(self) -> List[Violation]:
        """Check Maitland/Gosford per-pair home/away balance.

        The solver enforces balance PER PAIR (each team vs each opponent),
        not aggregate across all opponents. This check matches that logic:
        home_count * 2 >= total - 1 AND home_count * 2 <= total + 1.
        """
        violations = []

        for club_prefix, home_field in [('Maitland', 'Maitland Park'), ('Gosford', 'Central Coast Hockey Park')]:
            pair_balance = defaultdict(lambda: {'home': 0, 'away': 0})

            for game in self.draw.games:
                for team in [game.team1, game.team2]:
                    if club_prefix in team:
                        other = game.team2 if team == game.team1 else game.team1
                        if club_prefix in other:
                            continue
                        if game.field_location == home_field:
                            pair_balance[(team, other)]['home'] += 1
                        else:
                            pair_balance[(team, other)]['away'] += 1

            for (team, opponent), counts in pair_balance.items():
                home = counts['home']
                away = counts['away']
                total = home + away
                if total > 0 and not (home * 2 >= total - 1 and home * 2 <= total + 1):
                    team_club = self._team_to_club.get(team) or club_prefix
                    violations.append(Violation.create(
                        constraint="FiftyFiftyHomeAway",
                        message=f"'{team}' vs '{opponent}': {home}H/{away}A out of {total} (not balanced)",
                        affected_clubs=[team_club],
                        metric_value=abs(home - away),
                    ))

        return violations
    
    # spec-018: `_check_maitland_back_to_back` (consecutive home-weekend
    # window check) and `_check_maitland_away_clubs_limit` (away-clubs-per-week
    # cap check) deleted — the venue-sequencing rules they verified were
    # removed. Per-club home-weekend counts are verified by the spec-004
    # `AwayClubHomeWeekendsCount` atom's own unit test; per-pair/aggregate
    # home/away balance remains in `_check_fifty_fifty_home_away`.
    
    def _check_club_grade_adjacency(self) -> List[Violation]:
        """Check that same-grade, same-club teams never play simultaneously.

        spec-007: the adjacent-grade check (e.g. PHL + 2nd same slot) was
        REMOVED ENTIRELY. The new prod rule is `SameGradeSameClubNoConcurrency`,
        which only forbids same-grade-same-club concurrency. Violations are
        still emitted under the `ClubGradeAdjacency` constraint name so
        existing report templates and the registry's tester mapping keep
        working without renaming downstream consumers.
        """
        violations = []

        games_per_slot = defaultdict(list)
        for game in self.draw.games:
            games_per_slot[(game.date, game.day_slot)].append(game)

        # Build lookup of clubs with multiple teams in the same grade.
        club_grade_teams = defaultdict(list)
        for team in self.teams:
            club_grade_teams[(team.club.name, team.grade)].append(team.name)
        multi_team_clubs = {k: v for k, v in club_grade_teams.items() if len(v) > 1}

        for (date, slot), games in games_per_slot.items():
            club_grade_team_games = defaultdict(lambda: defaultdict(set))
            for game in games:
                for team in [game.team1, game.team2]:
                    club = self._team_to_club.get(team)
                    if club:
                        club_grade_team_games[club][game.grade].add(team)

            for club, grade_team_map in club_grade_team_games.items():
                for grade, teams_in_slot in grade_team_map.items():
                    if (club, grade) not in multi_team_clubs:
                        continue
                    if len(teams_in_slot) >= 2:
                        # If the two teams are playing each other, that's a
                        # single shared game — not concurrency.
                        playing_each_other = any(
                            g.team1 in teams_in_slot and g.team2 in teams_in_slot
                            for g in games if g.grade == grade
                        )
                        if not playing_each_other:
                            violations.append(Violation.create(
                                constraint="ClubGradeAdjacency",
                                message=(
                                    f"Club '{club}' has duplicate {grade} teams "
                                    f"{sorted(teams_in_slot)} at {date}, slot {slot} "
                                    f"(not playing each other)"
                                ),
                                affected_clubs=[club],
                            ))

        return violations


    def _check_team_pair_no_concurrency(self) -> List[Violation]:
        """Flag every (week, day_slot) where a TEAM_PAIR_NO_CONCURRENCY pair co-occurs.

        Soft constraint (spec-007). The atom imposes a penalty per
        co-occurrence; the tester surfaces the same events as informational
        violations so reports and severity rollups can see them.
        """
        violations: List[Violation] = []
        pairs_raw = self.data.get('constraint_defaults', {}).get(
            'TEAM_PAIR_NO_CONCURRENCY'
        )
        if pairs_raw is None:
            pairs_raw = self.data.get('TEAM_PAIR_NO_CONCURRENCY', [])
        if not pairs_raw:
            return violations

        # Normalise to (team_a, team_b) tuples; ignore weights for tester purposes.
        pairs: List[tuple] = []
        for entry in pairs_raw:
            if len(entry) >= 2:
                pairs.append(tuple(sorted((entry[0], entry[1]))))

        # Map (team, week, day_slot) -> set of game_ids the team plays in.
        team_slot_games = defaultdict(set)
        for game in self.draw.games:
            slot = (game.week, game.day_slot)
            team_slot_games[(game.team1, *slot)].add(game.game_id)
            team_slot_games[(game.team2, *slot)].add(game.game_id)

        # For each declared pair, look for slots where both teams play.
        all_slots = {(w, s) for (_t, w, s) in team_slot_games}
        for team_a, team_b in pairs:
            for w, s in all_slots:
                if team_slot_games.get((team_a, w, s)) and team_slot_games.get((team_b, w, s)):
                    violations.append(Violation.create(
                        constraint='TeamPairNoConcurrency',
                        message=(
                            f"'{team_a}' and '{team_b}' both play in week {w}, "
                            f"day_slot {s} -- soft TEAM_PAIR_NO_CONCURRENCY conflict"
                        ),
                    ))
        return violations

    def _check_phl_2nd_adjacency(self) -> List[Violation]:
        """Check same-club PHL/2nd adjacency (spec-014).

        Per (week, day, club) where the club fields BOTH a PHL and a 2nd-grade
        game, the new rule (matching `PHLAnd2ndAdjacency`) requires:
        - **Same venue**: same FIELD and ADJACENT day_slots (back-to-back).
        - **Different venue**: start times >= `phl_2nd_cross_venue_min_minutes`
          (default 180) apart, measured in real minutes-since-midnight.
        Any pair breaking its applicable rule is a violation.
        """
        violations = []
        cross_venue_min = self.data.get('constraint_defaults', {}).get(
            'phl_2nd_cross_venue_min_minutes', 180
        )

        def _minutes(t):
            hh, mm = t.split(':')
            return int(hh) * 60 + int(mm)

        # Group games by (week, day, club, grade) - must be same day to compare
        club_games = defaultdict(list)
        for game in self.draw.games:
            for team in [game.team1, game.team2]:
                club = self._team_to_club.get(team)
                if club:
                    club_games[(game.week, game.day, club, game.grade)].append(game)

        # Find week+day combos where club has both PHL and 2nd
        weeks_clubs = defaultdict(dict)
        for (week, day, club, grade), games in club_games.items():
            if grade in ['PHL', '2nd']:
                weeks_clubs[(week, day, club)][grade] = games

        for (week, day, club), grade_games in weeks_clubs.items():
            if 'PHL' not in grade_games or '2nd' not in grade_games:
                continue

            for phl_game in grade_games['PHL']:
                phl_min = _minutes(phl_game.time)
                phl_loc = phl_game.field_location

                for second_game in grade_games['2nd']:
                    second_min = _minutes(second_game.time)
                    second_loc = second_game.field_location

                    if phl_loc == second_loc:
                        same_field = phl_game.field_name == second_game.field_name
                        adjacent = abs(phl_game.day_slot - second_game.day_slot) == 1
                        if not (same_field and adjacent):
                            violations.append(Violation.create(
                                constraint="PHLAnd2ndAdjacency",
                                message=(f"Club '{club}' week {week} ({day}): PHL "
                                         f"({phl_game.time} {phl_game.field_name}) and 2nd "
                                         f"({second_game.time} {second_game.field_name}) at same "
                                         f"venue {phl_loc} but not back-to-back on the same field"),
                                week=week
                            ))
                    else:
                        gap = abs(phl_min - second_min)
                        if gap < cross_venue_min:
                            violations.append(Violation.create(
                                constraint="PHLAnd2ndAdjacency",
                                message=(f"Club '{club}' week {week} ({day}): PHL "
                                         f"({phl_game.time} @ {phl_loc}) and 2nd "
                                         f"({second_game.time} @ {second_loc}) at different venues "
                                         f"but only {gap}min apart (< {cross_venue_min}min)"),
                                week=week
                            ))

        return violations

    def _check_phl_second_grade_times(self) -> List[Violation]:
        """Check PHLAndSecondGradeTimes constraints:

        1. No concurrent PHL games at Broadmeadow (same day_slot)
        2. PHL and 2nd grade from same club not at same day_slot at Broadmeadow
        3. Max 3 Friday night PHL games at NIHC
        4. Gosford Friday night count = 8
        5. Every PHL team plays in round 1
        """
        violations = []
        NIHC = 'Newcastle International Hockey Centre'
        CCHP = 'Central Coast Hockey Park'

        # 1. No concurrent PHL games at Broadmeadow (same week, day, day_slot)
        phl_broadmeadow_slots = defaultdict(list)
        for game in self.draw.games:
            if game.grade == 'PHL' and game.field_location == NIHC:
                phl_broadmeadow_slots[(game.week, game.day, game.day_slot)].append(game.game_id)

        for (week, day, slot), game_ids in phl_broadmeadow_slots.items():
            if len(game_ids) > 1:
                violations.append(Violation.create(
                    constraint="PHLAndSecondGradeTimes",
                    message=f"Week {week} ({day}), slot {slot}: {len(game_ids)} concurrent PHL games at Broadmeadow",
                    affected_games=game_ids,
                    week=week
                ))

        # 2. PHL and 2nd grade from same club not at same day_slot at Broadmeadow
        club_grade_slots = defaultdict(list)
        for game in self.draw.games:
            if game.grade in ['PHL', '2nd'] and game.field_location == NIHC:
                for team in [game.team1, game.team2]:
                    club = self._team_to_club.get(team)
                    if club:
                        club_grade_slots[(club, game.week, game.day, game.day_slot, game.grade)].append(game.game_id)

        # Group by (club, week, day, day_slot) and check for both PHL and 2nd
        club_slot_grades = defaultdict(lambda: defaultdict(list))
        for (club, week, day, slot, grade), gids in club_grade_slots.items():
            club_slot_grades[(club, week, day, slot)][grade].extend(gids)

        for (club, week, day, slot), grade_map in club_slot_grades.items():
            if 'PHL' in grade_map and '2nd' in grade_map:
                all_ids = grade_map['PHL'] + grade_map['2nd']
                violations.append(Violation.create(
                    constraint="PHLAndSecondGradeTimes",
                    message=f"Club '{club}' week {week} ({day}), slot {slot}: PHL and 2nd grade at same time at Broadmeadow",
                    affected_games=all_ids,
                    week=week
                ))

        # 3. Max 3 Friday night PHL games at NIHC
        friday_nihc_phl = [g.game_id for g in self.draw.games
                           if g.grade == 'PHL' and g.day == 'Friday' and g.field_location == NIHC]
        if len(friday_nihc_phl) > 3:
            violations.append(Violation.create(
                constraint="PHLAndSecondGradeTimes",
                message=f"Friday night PHL at NIHC: {len(friday_nihc_phl)} games (max 3)",
                affected_games=friday_nihc_phl
            ))

        # 4. Gosford Friday night count should be exactly 8
        friday_gosford_phl = [g.game_id for g in self.draw.games
                              if g.grade == 'PHL' and g.day == 'Friday' and g.field_location == CCHP]
        if friday_gosford_phl and len(friday_gosford_phl) != 8:
            violations.append(Violation.create(
                constraint="PHLAndSecondGradeTimes",
                message=f"Friday night PHL at Gosford: {len(friday_gosford_phl)} games (expected 8)",
                affected_games=friday_gosford_phl
            ))

        # 5. Every PHL team plays in round 1
        phl_teams = {t.name for t in self.teams if t.grade == 'PHL'}
        round1_teams = set()
        for game in self.draw.games:
            if game.grade == 'PHL' and game.round_no == 1:
                round1_teams.add(game.team1)
                round1_teams.add(game.team2)

        missing_round1 = phl_teams - round1_teams
        for team in missing_round1:
            violations.append(Violation.create(
                constraint="PHLAndSecondGradeTimes",
                message=f"PHL team '{team}' does not play in round 1"
            ))

        return violations

    def _check_equal_matchup_spacing(self) -> List[Violation]:
        """Check matchup spacing: consecutive meetings of the same pair should be spread out.

        spec-008 Part A: uses `constraints.atoms._spacing.effective_spacing`
        so the tester and the solver agree on the threshold. The hard rule
        forbids `gap = r2 - r1 <= S`, where S is the convenor-facing
        "rounds between meetings" number. With default slack, S equals
        `ideal_gap(T)` which is `legacy_min_gap - 1` — the off-by-one
        fix preserves the physical schedule a healthy solver produces.
        """
        from constraints.atoms._spacing import effective_spacing

        violations = []
        defaults = self.data.get('constraint_defaults', {})
        base_slack = int(defaults.get('spacing_base_slack', 0) or 0)
        config_slack = int(
            self.constraint_slack.get('EqualMatchUpSpacingConstraint', 0) or 0
        )

        # Build per-grade team counts
        grade_team_counts = {}
        for grade in self.grades:
            grade_team_counts[grade.name] = grade.num_teams

        # Gather meetings: (pair, grade) -> sorted list of round_nos
        pair_rounds = defaultdict(list)
        for game in self.draw.games:
            pair = tuple(sorted([game.team1, game.team2]))
            pair_rounds[(pair, game.grade)].append(game.round_no)

        for (pair, grade), rounds in pair_rounds.items():
            T = grade_team_counts.get(grade, 0)
            if T < 3:
                continue
            S = effective_spacing(
                T, base_slack=base_slack, config_slack=config_slack
            )
            if S <= 0:
                continue

            sorted_rounds = sorted(rounds)
            for i in range(len(sorted_rounds) - 1):
                gap = sorted_rounds[i + 1] - sorted_rounds[i]
                if gap <= S:
                    violations.append(Violation.create(
                        constraint="EqualMatchUpSpacing",
                        message=(
                            f"{pair[0]} vs {pair[1]} ({grade}): gap of {gap} rounds "
                            f"between meetings at rounds {sorted_rounds[i]} and "
                            f"{sorted_rounds[i+1]} (need gap > {S} — at least "
                            f"{S} round(s) between meetings)"
                        )
                    ))

        return violations

    def _check_balanced_bye_spacing(self) -> List[Violation]:
        """spec-008 Part B: each team's bye rounds should be spread evenly.

        Mirrors the solver atom `constraints.atoms.balanced_bye_spacing.
        BalancedByeSpacing`. For each team in each grade:
          R           = number of playable rounds (data['num_rounds']['max'])
          games_t     = games per team for the team's grade
          byes_t      = R - games_t
          S           = max(0, ideal_bye_gap(R, byes_t) - base_slack - config_slack)
        A team's bye-rounds are the rounds in which it does not appear in
        the published draw. The check flags every pair of consecutive byes
        whose gap is `<= S`.
        """
        from constraints.atoms._spacing import ideal_bye_gap

        violations = []
        defaults = self.data.get('constraint_defaults', {}) or {}
        base_slack = int(defaults.get('bye_spacing_base_slack', 0) or 0)
        config_slack = int(
            self.constraint_slack.get('BalancedByeSpacing', 0) or 0
        )

        grade_team_counts: Dict[str, int] = {}
        per_grade_team_names: Dict[str, List[str]] = defaultdict(list)
        for grade in self.grades:
            grade_team_counts[grade.name] = grade.num_teams
            per_grade_team_names[grade.name] = list(grade.teams)

        # Per-team rounds in which they appeared.
        team_rounds: Dict[Tuple[str, str], set] = defaultdict(set)
        for game in self.draw.games:
            team_rounds[(game.team1, game.grade)].add(game.round_no)
            if game.team2 != game.team1:
                team_rounds[(game.team2, game.grade)].add(game.round_no)

        num_rounds = self.data.get('num_rounds') or {}
        override = self.data.get('GRADE_ROUNDS_OVERRIDE') or {}
        max_r = int(num_rounds.get('max', 0) or 0)
        if max_r <= 0:
            return violations
        all_rounds = list(range(1, max_r + 1))

        for grade_name, team_names in per_grade_team_names.items():
            if grade_name in override:
                games_t = int(override[grade_name])
            else:
                games_t = int(num_rounds.get(grade_name, max_r))
            byes_t = max_r - games_t
            if byes_t < 2:
                continue
            S = max(0, ideal_bye_gap(max_r, byes_t) - base_slack - config_slack)
            if S <= 0:
                continue
            for team in team_names:
                played = team_rounds.get((team, grade_name), set())
                byes = sorted(r for r in all_rounds if r not in played)
                for i in range(len(byes) - 1):
                    gap = byes[i + 1] - byes[i]
                    if gap <= S:
                        violations.append(Violation.create(
                            constraint='BalancedByeSpacing',
                            message=(
                                f"{team} ({grade_name}): bye gap of {gap} rounds "
                                f"between bye-rounds {byes[i]} and {byes[i+1]} "
                                f"(need gap > {S} — at least {S} round(s) between "
                                f"byes; {byes_t} byes across {max_r} rounds)"
                            )
                        ))
        return violations

    def _check_team_conflict(self) -> List[Violation]:
        """Check that conflicting teams do not play at the same timeslot (week + day_slot)."""
        violations = []
        conflicts = self.data.get('team_conflicts', [])
        if not conflicts:
            return violations

        # Build lookup: team -> list of (week, day_slot) where they play
        team_slots = defaultdict(list)
        for game in self.draw.games:
            for team in [game.team1, game.team2]:
                team_slots[team].append((game.week, game.day_slot, game.game_id))

        for team1, team2 in conflicts:
            slots1 = {(w, s): gid for w, s, gid in team_slots.get(team1, [])}
            for w, s, gid2 in team_slots.get(team2, []):
                if (w, s) in slots1:
                    violations.append(Violation.create(
                        constraint="TeamConflict",
                        message=f"Conflicting teams '{team1}' and '{team2}' both play in week {w}, slot {s}",
                        affected_games=[slots1[(w, s)], gid2],
                        week=w
                    ))

        return violations

    def _check_club_day(self) -> List[Violation]:
        """Check club day constraints: on a club's designated day, all club teams must play,
        intra-club matchups should occur where possible, all on the same field, contiguous slots.
        """
        violations = []
        club_days = self.data.get('club_days', {})
        if not club_days:
            return violations

        from utils import get_teams_from_club

        for club_name, desired_date in club_days.items():
            date_str = desired_date.strftime('%Y-%m-%d') if hasattr(desired_date, 'strftime') else str(desired_date)

            # Get all club teams
            club_teams = set()
            for t in self.teams:
                if t.club.name.lower() == club_name.lower():
                    club_teams.add(t.name)

            if not club_teams:
                continue

            # Find games on this date involving the club
            club_games_on_date = []
            for game in self.draw.games:
                if game.date == date_str:
                    if game.team1 in club_teams or game.team2 in club_teams:
                        club_games_on_date.append(game)

            # Check every team plays
            playing_teams = set()
            for game in club_games_on_date:
                if game.team1 in club_teams:
                    playing_teams.add(game.team1)
                if game.team2 in club_teams:
                    playing_teams.add(game.team2)

            missing_teams = club_teams - playing_teams
            for team in missing_teams:
                violations.append(Violation.create(
                    constraint="ClubDayConstraint",
                    message=f"Club day '{club_name}' ({date_str}): team '{team}' does not play",
                    affected_games=[g.game_id for g in club_games_on_date]
                ))

            if not club_games_on_date:
                violations.append(Violation.create(
                    constraint="ClubDayConstraint",
                    message=f"Club day '{club_name}' ({date_str}): no games found for the club"
                ))
                continue

            # Check all games on same field
            fields_used = set(g.field_name for g in club_games_on_date)
            if len(fields_used) > 1:
                violations.append(Violation.create(
                    constraint="ClubDayConstraint",
                    message=f"Club day '{club_name}' ({date_str}): games on {len(fields_used)} fields ({fields_used}), should be 1",
                    affected_games=[g.game_id for g in club_games_on_date]
                ))

            # Check contiguity of slots
            slots_used = sorted(set(g.day_slot for g in club_games_on_date))
            if len(slots_used) >= 2:
                for i in range(len(slots_used) - 1):
                    if slots_used[i + 1] - slots_used[i] > 1:
                        violations.append(Violation.create(
                            constraint="ClubDayConstraint",
                            message=f"Club day '{club_name}' ({date_str}): non-contiguous slots {slots_used}",
                            affected_games=[g.game_id for g in club_games_on_date]
                        ))
                        break

        return violations

    def _check_club_vs_club_alignment(self) -> List[Violation]:
        """Check ClubVsClubAlignment: when two clubs meet across grades on a Sunday,
        their games should coincide in the same round and ideally same field.

        Reports hard violations (coincidences below minimum) and soft warnings
        (non-coincident rounds, multi-field usage).
        Skips PHL and 2nd grade (solver only applies to 3rd-6th).
        """
        violations = []
        config_slack = self.constraint_slack.get('ClubVsClubAlignment', 0)

        # Build games by (club_pair, grade, round_no)
        games_by_pair_grade_round = defaultdict(list)
        for game in self.draw.games:
            if game.grade in ['PHL', '2nd']:
                continue
            if game.day != 'Sunday':
                continue
            club1 = self._team_to_club.get(game.team1)
            club2 = self._team_to_club.get(game.team2)
            if club1 and club2 and club1 != club2:
                pair = tuple(sorted([club1, club2]))
                games_by_pair_grade_round[(pair, game.grade, game.round_no)].append(game)

        # Group by club pair: which rounds does each grade play?
        pair_grade_rounds = defaultdict(lambda: defaultdict(set))
        for (pair, grade, round_no), glist in games_by_pair_grade_round.items():
            pair_grade_rounds[pair][grade].add(round_no)

        # Per-team games calc for min coincidences
        num_rounds = self.data.get('num_rounds', {})
        per_team_games = {}
        for grade in self.grades:
            if grade.name in ['PHL', '2nd']:
                continue
            T = grade.num_teams
            if T < 2:
                continue
            R = num_rounds.get(grade.name, 0)
            per_team_games[grade.name] = R // (T - 1) if T % 2 == 0 else R // T

        # Check coincidences between grade pairs
        ordered_grades = sorted(per_team_games.items(), key=lambda x: x[1])
        used_grades = []
        ini_num = 0

        for grade, num_games in ordered_grades:
            used_grades.append(grade)
            if num_games <= ini_num:
                continue
            ini_num = num_games

            for grade2, _ in ordered_grades:
                if grade2 in used_grades:
                    continue

                for pair in pair_grade_rounds:
                    rounds_g1 = pair_grade_rounds[pair].get(grade, set())
                    rounds_g2 = pair_grade_rounds[pair].get(grade2, set())

                    if not rounds_g1 or not rounds_g2:
                        continue

                    coincident = rounds_g1 & rounds_g2
                    min_required = max(0, num_games - config_slack)

                    if len(coincident) < min_required:
                        violations.append(Violation.create(
                            constraint="ClubVsClubAlignment",
                            message=f"Clubs {pair[0]} vs {pair[1]}, grades {grade}/{grade2}: "
                                    f"{len(coincident)} coincident rounds (min {min_required}, "
                                    f"target {num_games})",
                            affected_clubs=[pair[0], pair[1]],
                            metric_value=min_required - len(coincident),
                        ))

        # Soft: check field alignment on coincident rounds
        for (pair, grade, round_no), glist in games_by_pair_grade_round.items():
            # Find other grades same pair/round
            for grade2 in [g.name for g in self.grades if g.name not in ['PHL', '2nd'] and g.name != grade]:
                other_games = games_by_pair_grade_round.get((pair, grade2, round_no), [])
                if not other_games:
                    continue
                fields1 = set(g.field_name for g in glist)
                fields2 = set(g.field_name for g in other_games)
                all_fields = fields1 | fields2
                if len(all_fields) > 2:
                    violations.append(Violation.create(
                        constraint="ClubVsClubAlignment",
                        message=f"Clubs {pair[0]} vs {pair[1]} round {round_no}, "
                                f"grades {grade}/{grade2}: using {len(all_fields)} fields ({all_fields}), max 2",
                        affected_clubs=[pair[0], pair[1]],
                        metric_value=len(all_fields),
                    ))

        return violations

    def _check_club_game_spread(self) -> List[Violation]:
        """Check ClubGameSpread (spec-024): per-FIELD near-contiguity + off-primary.

        Per (club, week, day, field):
          holes = (max_used_slot - min_used_slot + 1) - num_distinct_used_slots
          gap_cap = max(0, min(1, n_field - 3)) + slack
            (<=3 games on a field -> 0 holes; >=4 -> 1 hole allowed.)
          Hard: holes > gap_cap is a violation. Soft: any hole within cap warns.
        Plus a soft off-primary-field observation per (club, week, day) that spans
        more than one field (games not on the club's most-used field that day).
        Applies at ALL venues. Double-ups are NOT checked here — see
        `_check_club_no_concurrent_slot`.
        """
        violations = []
        config_slack = self.constraint_slack.get('ClubGameSpread', 0)

        # (club, week, day, field) -> {day_slot: set of game_ids}
        cwdf_slots = defaultdict(lambda: defaultdict(set))
        # (club, week, day) -> {field: set of game_ids}
        cwd_fields = defaultdict(lambda: defaultdict(set))
        for game in self.draw.games:
            for team in [game.team1, game.team2]:
                club = self._team_to_club.get(team)
                if not club:
                    continue
                cwdf_slots[(club, game.week, game.day, game.field_name)][game.day_slot].add(game.game_id)
                cwd_fields[(club, game.week, game.day)][game.field_name].add(game.game_id)

        # (1) per-field contiguity
        for (club, week, day, field), slot_games in cwdf_slots.items():
            all_game_ids = {gid for gids in slot_games.values() for gid in gids}
            n_field = len(all_game_ids)
            if n_field < 2:
                continue
            slots = sorted(slot_games.keys())
            holes = (slots[-1] - slots[0] + 1) - len(slots)
            gap_cap = max(0, min(1, n_field - 3)) + config_slack

            if holes > gap_cap:
                violations.append(Violation.create(
                    constraint="ClubGameSpread",
                    message=f"Club '{club}' week {week} ({day}) field {field}: {holes} hole(s) "
                            f"in slots {slots} exceeds cap {gap_cap} ({n_field} games)",
                    affected_games=list(all_game_ids)[:5],
                    week=week,
                    affected_clubs=[club],
                    metric_value=holes - gap_cap,
                ))
            elif holes > 0:
                violations.append(Violation.create(
                    constraint="ClubGameSpread",
                    message=f"[soft] Club '{club}' week {week} ({day}) field {field}: {holes} hole(s) "
                            f"in slots {slots} ({n_field} games, within cap {gap_cap})",
                    affected_games=list(all_game_ids)[:5],
                    week=week
                ))

        # (2) off-primary-field soft observation per (club, week, day)
        for (club, week, day), fields in cwd_fields.items():
            if len(fields) < 2:
                continue
            counts = {f: len(g) for f, g in fields.items()}
            off_primary = sum(counts.values()) - max(counts.values())
            if off_primary > 0:
                all_game_ids = {gid for g in fields.values() for gid in g}
                violations.append(Violation.create(
                    constraint="ClubGameSpread",
                    message=f"[soft] Club '{club}' week {week} ({day}): {off_primary} game(s) off the "
                            f"primary field (split across {len(fields)} fields: {counts})",
                    affected_games=list(all_game_ids)[:5],
                    week=week,
                    affected_clubs=[club],
                ))

        return violations

    def _check_club_no_concurrent_slot(self) -> List[Violation]:
        """Check ClubNoConcurrentSlot (spec-021): a club's games per (timeslot,
        venue) stay within a capacity-aware cap.

        cap = max(1, ceil(club_team_count / no_field_slots[location])).
        Allows forced double-ups when a club has more games at a venue than the
        venue has timeslots; flags any slot exceeding the cap.
        """
        violations = []
        no_field_slots = self.data.get('no_field_slots', {})
        club_team_count = defaultdict(set)
        for t in self.data.get('teams', []):
            club_team_count[t.club.name].add(t.name)

        # (club, week, day_slot, location) -> set of game_ids.
        groups = defaultdict(set)
        for game in self.draw.games:
            clubs = set()
            for team in [game.team1, game.team2]:
                club = self._team_to_club.get(team)
                if club:
                    clubs.add(club)
            for club in clubs:
                groups[(club, game.week, game.day_slot, game.field_location)].add(game.game_id)

        for (club, week, slot, location), gids in groups.items():
            count = len(gids)
            slots = no_field_slots.get(location, 0)
            n_teams = len(club_team_count.get(club, ()))
            cap = max(1, -(-n_teams // slots)) if slots > 0 else 1  # ceil(n/slots)
            if count > cap:
                violations.append(Violation.create(
                    constraint="ClubNoConcurrentSlot",
                    message=f"Club '{club}' week {week} slot {slot} at {location}: "
                            f"{count} concurrent games exceeds cap {cap}",
                    affected_games=list(gids)[:5],
                    week=week,
                    affected_clubs=[club],
                    metric_value=count - cap,
                ))
        return violations

    def _check_club_field_concentration(self) -> List[Violation]:
        """Check ClubFieldConcentration: clubs should play on the same field per day.

        field_spread = num_games - max_games_on_any_single_field
        Hard: field_spread <= max(0, num_games // 2 - 1) + slack
        Soft: reports any field_spread > 0 as warnings.
        """
        violations = []
        config_slack = self.constraint_slack.get('ClubGameSpread', 0)

        # Group games by (club, week, day) -> {field_name: [game_ids]}
        # Only Broadmeadow — away venues have 1 field so field concentration is irrelevant
        club_day_fields = defaultdict(lambda: defaultdict(list))
        for game in self.draw.games:
            if game.field_location != 'Newcastle International Hockey Centre':
                continue
            for team in [game.team1, game.team2]:
                club = self._team_to_club.get(team)
                if club:
                    club_day_fields[(club, game.week, game.day)][game.field_name].append(game.game_id)

        for (club, week, day), field_games in club_day_fields.items():
            num_games = sum(len(gids) for gids in field_games.values())
            if num_games < 2:
                continue

            max_on_field = max(len(gids) for gids in field_games.values())
            field_spread = num_games - max_on_field
            hard_cap = max(0, num_games // 2 - 1) + config_slack

            all_game_ids = [gid for gids in field_games.values() for gid in gids]

            if field_spread > hard_cap:
                violations.append(Violation.create(
                    constraint="ClubFieldConcentration",
                    message=f"Club '{club}' week {week} ({day}): field_spread={field_spread} exceeds "
                            f"hard cap {hard_cap} ({num_games} games across {len(field_games)} fields)",
                    affected_games=list(all_game_ids)[:5],
                    week=week
                ))
            elif field_spread > 0:
                field_summary = ', '.join(f'{f}:{len(g)}' for f, g in sorted(field_games.items()))
                violations.append(Violation.create(
                    constraint="ClubFieldConcentration",
                    message=f"[soft] Club '{club}' week {week} ({day}): {field_spread} game(s) off main field "
                            f"({field_summary})",
                    affected_games=list(all_game_ids)[:5],
                    week=week
                ))

        return violations

    # spec-024: _check_maximise_clubs_per_timeslot_broadmeadow and
    # _check_minimise_clubs_on_a_field_broadmeadow removed (constraints deleted).

    def _check_venue_earliest_slot_fill(self) -> List[Violation]:
        """Check VenueEarliestSlotFill (spec-021): games at a venue on a day pack
        into the EARLIEST timeslots — no interior gap AND anchored to the
        earliest slot the venue offers.

        Post-hoc, the earliest slot a (location, day) offers is taken as the
        minimum day_slot observed at that (location, day) across the whole draw
        (the slot set is the same every playing week). Two checks per
        (week, day, location):
          - earliest-start: if the venue has games but the earliest offered slot
            is unused, the block didn't anchor to slot 1 → violation.
          - no-gap: an unused slot between two used ones → violation.

        HARD rule — violations are real, not warnings.
        """
        violations = []

        # Group games by (week, day, field_location) -> set of day_slots used.
        location_slots = defaultdict(set)
        location_games = defaultdict(list)
        # Earliest day_slot the venue offers (min observed across all weeks).
        earliest_offered = {}
        for game in self.draw.games:
            key = (game.week, game.day, game.field_location)
            location_slots[key].add(game.day_slot)
            location_games[key].append(game.game_id)
            loc_day = (game.day, game.field_location)
            prev = earliest_offered.get(loc_day)
            if prev is None or game.day_slot < prev:
                earliest_offered[loc_day] = game.day_slot

        for (week, day, location), slots in location_slots.items():
            sorted_slots = sorted(slots)

            # Earliest-start: anchored to the earliest slot the venue offers.
            first_offered = earliest_offered.get((day, location))
            if first_offered is not None and sorted_slots[0] > first_offered:
                violations.append(Violation.create(
                    constraint="VenueEarliestSlotFill",
                    message=f"Week {week} ({day}) at {location}: earliest used slot "
                            f"{sorted_slots[0]} is later than offered slot {first_offered} "
                            f"— not anchored to earliest",
                    affected_games=location_games[(week, day, location)][:5],
                    week=week
                ))

            # No-gap: unused slot between two used ones.
            for i in range(1, len(sorted_slots)):
                prev_slot = sorted_slots[i - 1]
                curr_slot = sorted_slots[i]
                if curr_slot - prev_slot > 1:
                    violations.append(Violation.create(
                        constraint="VenueEarliestSlotFill",
                        message=f"Week {week} ({day}) at {location}: gap between "
                                f"slots {prev_slot} and {curr_slot}",
                        affected_games=location_games[(week, day, location)][:5],
                        week=week
                    ))

        return violations

    def _check_preferred_times(self) -> List[Violation]:
        """Check PreferredTimesConstraint: teams should not play on their no-play preference dates.

        This is a soft constraint - reports violations as warnings with counts.
        """
        violations = []
        noplay = self.data.get('preference_no_play', {})
        if not noplay:
            return violations

        for key, value in noplay.items():
            if isinstance(value, dict) and 'club' in value:
                club_name = value['club']
                dates = value.get('dates', [])
                if 'date' in value:
                    dates = [value['date']]

                # Get affected teams
                club_teams = set()
                grade_filter = None
                grades_filter = None
                if 'grade' in value:
                    grade_filter = value['grade']
                if 'grades' in value:
                    grades_filter = [g.lower() for g in value['grades']]

                for t in self.teams:
                    if t.club.name.lower() == club_name.lower():
                        if grade_filter and grade_filter.lower() not in t.name.lower():
                            continue
                        if grades_filter and not any(g in t.name.lower() for g in grades_filter):
                            continue
                        club_teams.add(t.name)

                for date in dates:
                    if hasattr(date, 'strftime'):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)

                    for game in self.draw.games:
                        if game.date == date_str:
                            if game.team1 in club_teams or game.team2 in club_teams:
                                violations.append(Violation.create(
                                    constraint="PreferredTimesConstraint",
                                    message=f"[soft] '{key}': game on no-play date {date_str} "
                                            f"({game.team1} vs {game.team2}, {game.grade})",
                                    affected_games=[game.game_id]
                                ))

            elif isinstance(value, list):
                club_name = key
                club_teams = {t.name for t in self.teams if t.club.name.lower() == club_name.lower()}

                for restriction in value:
                    date_str = restriction.get('date', '')
                    if not date_str:
                        continue
                    for game in self.draw.games:
                        if game.date == date_str:
                            if game.team1 in club_teams or game.team2 in club_teams:
                                # Check additional restriction fields
                                match = True
                                if 'field_location' in restriction and game.field_location != restriction['field_location']:
                                    match = False
                                if match:
                                    violations.append(Violation.create(
                                        constraint="PreferredTimesConstraint",
                                        message=f"[soft] '{club_name}': game on no-play date {date_str} "
                                                f"({game.team1} vs {game.team2}, {game.grade})",
                                        affected_games=[game.game_id]
                                    ))

        return violations

    # ============== Forced / Blocked Game Checks ==============

    @staticmethod
    def _resolve_team_for_check(name: str, grade, teams: list) -> list:
        """Resolve a club name to full team name(s) for post-hoc checking.

        Works like utils._resolve_team_name but doesn't need pre-built lookups.
        """
        team_names = {t.name for t in teams}
        if name in team_names:
            return [name]
        if grade and not isinstance(grade, (list, tuple)):
            full = f"{name} {grade}"
            if full in team_names:
                return [full]
            matches = [t.name for t in teams if t.club.name == name and t.grade == grade]
            if matches:
                return matches
        if grade and isinstance(grade, (list, tuple)):
            results = []
            for g in grade:
                results.extend(DrawTester._resolve_team_for_check(name, g, teams))
            return results
        results = [t.name for t in teams if t.club.name == name]
        return results if results else [name]

    @staticmethod
    def _game_matches_scope(game: 'StoredGame', entry: dict) -> bool:
        """Check if a StoredGame matches the scope fields of a forced/blocked entry."""
        field_map = {
            'grade': lambda g: g.grade,
            'day': lambda g: g.day,
            'day_slot': lambda g: g.day_slot,
            'time': lambda g: g.time,
            'week': lambda g: g.week,
            'date': lambda g: g.date,
            'round_no': lambda g: g.round_no,
            'field_name': lambda g: g.field_name,
            'field_location': lambda g: g.field_location,
        }
        for field_name, accessor in field_map.items():
            if field_name in entry:
                val = entry[field_name]
                game_val = accessor(game)
                if isinstance(val, list):
                    if game_val not in val:
                        return False
                else:
                    if game_val != val:
                        return False
        # Handle 'grades' (plural) when 'grade' not present
        if 'grades' in entry and 'grade' not in entry:
            if game.grade not in entry['grades']:
                return False
        return True

    @staticmethod
    def _game_matches_teams(game: 'StoredGame', entry: dict, teams: list) -> bool:
        """Check if a StoredGame matches the team specification of an entry."""
        raw_teams = entry.get('teams', [])
        club = entry.get('club')

        if not raw_teams and not club:
            # No team filter — matches any game in scope
            return True

        grade = entry.get('grade')
        grades = entry.get('grades', [])
        effective_grade = grades if grades else grade
        game_pair = tuple(sorted([game.team1, game.team2]))

        if raw_teams:
            if len(raw_teams) == 2:
                resolved_t1 = DrawTester._resolve_team_for_check(raw_teams[0], effective_grade, teams)
                resolved_t2 = DrawTester._resolve_team_for_check(raw_teams[1], effective_grade, teams)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        if game_pair == pair:
                            return True
            elif len(raw_teams) == 1:
                resolved = DrawTester._resolve_team_for_check(raw_teams[0], effective_grade, teams)
                for rt in resolved:
                    if rt in (game.team1, game.team2):
                        return True
        elif club:
            # 'club' key: match any team from that club
            club_teams = DrawTester._resolve_team_for_check(club, effective_grade, teams)
            for ct in club_teams:
                if ct in (game.team1, game.team2):
                    return True

        return False

    def _check_forced_games(self) -> List[Violation]:
        """Check that FORCED_GAMES constraints are satisfied in the draw.

        For each forced game entry, counts matching games and verifies the
        constraint type is respected (default: exactly 1).
        """
        violations = []
        forced_games = self.data.get('forced_games', [])
        if not forced_games:
            return violations

        for entry in forced_games:
            desc = entry.get('description', str({k: v for k, v in entry.items()
                                                  if k not in ('description', 'reason')}))
            ctype = entry.get('constraint', 'equal')

            matching_games = []
            for game in self.draw.games:
                if self._game_matches_scope(game, entry) and \
                   self._game_matches_teams(game, entry, self.teams):
                    matching_games.append(game.game_id)

            count = len(matching_games)
            threshold = entry.get('count', 1)
            violated = False
            if ctype == 'equal' and count != threshold:
                violated = True
                msg = f"Expected exactly {threshold} game(s), found {count}"
            elif ctype == 'lesse' and count > threshold:
                violated = True
                msg = f"Expected at most {threshold} game(s), found {count}"
            elif ctype == 'greatere' and count < threshold:
                violated = True
                msg = f"Expected at least {threshold} game(s), found {count}"
            elif ctype == 'greater' and count <= threshold:
                violated = True
                msg = f"Expected more than {threshold} game(s), found {count}"
            elif ctype == 'less' and count >= threshold:
                violated = True
                msg = f"Expected fewer than {threshold} game(s), found {count}"

            if violated:
                violations.append(Violation.create(
                    constraint="ForcedGames",
                    message=f"Forced game '{desc}': {msg}",
                    affected_games=matching_games,
                ))

        return violations

    def _check_blocked_games(self) -> List[Violation]:
        """Check that BLOCKED_GAMES are respected in the draw.

        For each blocked game entry, verifies no matching games exist.
        """
        violations = []
        blocked_games = self.data.get('blocked_games', [])
        if not blocked_games:
            return violations

        for entry in blocked_games:
            desc = entry.get('description', str({k: v for k, v in entry.items()
                                                  if k not in ('description', 'reason')}))

            for game in self.draw.games:
                if self._game_matches_scope(game, entry) and \
                   self._game_matches_teams(game, entry, self.teams):
                    violations.append(Violation.create(
                        constraint="BlockedGames",
                        message=f"Blocked game '{desc}': "
                                f"{game.team1} vs {game.team2} ({game.grade}) "
                                f"on {game.date} at {game.field_location}",
                        affected_games=[game.game_id],
                    ))

        return violations

    def _check_locked_pairings(self) -> List[Violation]:
        """Check that LOCKED_PAIRINGS date-pins are honoured in the draw (spec-025).

        Each pin is a mechanical date-pin: a pairing (+ grade) that must appear on
        its `date` in the finished draw, with time/slot/field left free. The
        generate_X pass enforces this as a hard `sum == 1` over candidate vars on
        the pin's date, so a violation here is structural (CRITICAL). Mirrors
        `_check_forced_games`' scope/team matching, but the threshold is fixed:
        exactly one matching game must exist on the pin's date.
        """
        violations = []
        locked_pairings = self.data.get('locked_pairings', [])
        if not locked_pairings:
            return violations

        for entry in locked_pairings:
            desc = entry.get('description', str({k: v for k, v in entry.items()
                                                  if k not in ('description', 'reason')}))

            matching_games = []
            for game in self.draw.games:
                if self._game_matches_scope(game, entry) and \
                   self._game_matches_teams(game, entry, self.teams):
                    matching_games.append(game.game_id)

            count = len(matching_games)
            if count != 1:
                violations.append(Violation.create(
                    constraint="LockedPairings",
                    message=f"Locked pairing '{desc}': expected exactly 1 game "
                            f"on {entry.get('date')}, found {count}",
                    affected_games=matching_games,
                ))

        return violations

    def _check_preferred_games(self) -> List[Violation]:
        """Report PREFERRED_GAMES deviations as SOFT pressure (spec-020).

        Mirrors `_check_forced_games`' scope/team matching but, because these
        are soft preferences, a deviation is NOT a hard violation. Each entry
        whose matched-game count deviates from its target emits a severity-5
        Violation carrying `metric_value` = the deviation penalty, so the
        breakdown rolls it up under `soft_pressure['PreferredGames']` rather
        than as a structural failure.
        """
        violations = []
        preferred_games = self.data.get('preferred_games', [])
        if not preferred_games:
            return violations

        for entry in preferred_games:
            desc = entry.get('description', str({k: v for k, v in entry.items()
                                                  if k not in ('description', 'reason')}))
            ctype = entry.get('constraint', 'equal')
            N = entry.get('count', 1)

            matching_games = []
            for game in self.draw.games:
                if self._game_matches_scope(game, entry) and \
                   self._game_matches_teams(game, entry, self.teams):
                    matching_games.append(game.game_id)

            count = len(matching_games)
            if ctype == 'equal':
                penalty = abs(count - N)
            elif ctype == 'lesse':
                penalty = max(0, count - N)
            elif ctype == 'less':
                penalty = max(0, count - (N - 1))
            elif ctype == 'greatere':
                penalty = max(0, N - count)
            elif ctype == 'greater':
                penalty = max(0, (N + 1) - count)
            else:
                penalty = abs(count - N)

            if penalty > 0:
                violations.append(Violation.create(
                    constraint="PreferredGames",
                    message=(f"Preferred game '{desc}' [{ctype} {N}]: "
                             f"found {count} game(s), deviation penalty {penalty}"),
                    affected_games=matching_games,
                    metric_value=float(penalty),
                ))

        return violations

    # ============== Reporting ==============

    def print_modifications(self) -> None:
        """Print all modifications made to the draw."""
        if not self.modifications:
            print("No modifications made.")
            return
        
        print(f"\n{'='*40}")
        print(f"MODIFICATIONS ({len(self.modifications)})")
        print('='*40)
        for i, mod in enumerate(self.modifications, 1):
            print(f"  {i}. {mod}")
    
    def save_modified_draw(self, path: str) -> None:
        """Save the modified draw to a file."""
        self.draw.description = f"{self.draw.description} (Modified: {len(self.modifications)} changes)"
        self.draw.save(path)


# ============== Convenience Functions ==============

def test_draw(draw_path: str, data: Dict) -> ViolationReport:
    """Load a draw and run violation checks."""
    tester = DrawTester.from_file(draw_path, data)
    return tester.run_violation_check()


def test_solution(X_solution: Dict, data: Dict) -> ViolationReport:
    """Test an X solution dict for violations."""
    tester = DrawTester.from_X_solution(X_solution, data)
    return tester.run_violation_check()


def what_if_move_game(
    draw_path: str,
    data: Dict,
    game_id: str,
    **new_slot_kwargs
) -> Tuple[ViolationReport, List[str]]:
    """
    Test what happens if a game is moved.
    
    Args:
        draw_path: Path to draw JSON file
        data: Data dictionary
        game_id: ID of game to move
        **new_slot_kwargs: New slot parameters (new_week, new_day_slot, etc.)
    
    Returns:
        Tuple of (ViolationReport, list of constraint names broken)
        The report shows violations caused by the move, with severity levels.
    """
    tester = DrawTester.from_file(draw_path, data)
    
    success = tester.move_game(game_id, **new_slot_kwargs)
    if not success:
        raise ValueError(f"Game {game_id} not found in draw")
    
    report = tester.run_violation_check()
    constraints_broken = list(set(v.constraint for v in report.violations))
    
    return report, constraints_broken


def what_if_swap_games(
    draw_path: str,
    data: Dict,
    game_id_1: str,
    game_id_2: str
) -> Tuple[ViolationReport, List[str]]:
    """
    Test what happens if two games are swapped.
    
    Args:
        draw_path: Path to draw JSON file
        data: Data dictionary
        game_id_1: ID of first game
        game_id_2: ID of second game
    
    Returns:
        Tuple of (ViolationReport, list of constraint names broken)
        The report shows violations caused by the swap, with severity levels.
    """
    tester = DrawTester.from_file(draw_path, data)
    
    success = tester.swap_games(game_id_1, game_id_2)
    if not success:
        raise ValueError(f"One or both games not found: {game_id_1}, {game_id_2}")
    
    report = tester.run_violation_check()
    constraints_broken = list(set(v.constraint for v in report.violations))
    
    return report, constraints_broken


def find_best_slot(
    draw_path: str,
    data: Dict,
    game_id: str,
    weeks: Optional[List[int]] = None,
    field_locations: Optional[List[str]] = None,
    max_results: int = 10,
    include_swaps: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """
    Find the best available slots for a game.
    
    This is a convenience function that loads a draw and calls find_best_slot_for_game.
    Results are ranked by:
    1. Lowest severity level (0 = no violations, then 4, 3, 2, 1)
    2. Fewest violations at that severity level
    3. Fewest total violations
    
    Args:
        draw_path: Path to draw JSON file
        data: Data dictionary
        game_id: ID of game to find new slot for
        weeks: Optional list of weeks to consider
        field_locations: Optional list of field locations to consider
        max_results: Maximum results to return (default: 10)
        include_swaps: Whether to include potential swaps (default: True)
        verbose: Print progress and results (default: True)
    
    Returns:
        List of slot options with violation reports, sorted best to worst
    """
    tester = DrawTester.from_file(draw_path, data)
    return tester.find_best_slot_for_game(
        game_id,
        weeks=weeks,
        field_locations=field_locations,
        max_results=max_results,
        include_swaps=include_swaps,
        verbose=verbose
    )


def compare_draws(
    draw1_path: str,
    draw2_path: str,
    data: Dict
) -> Tuple[ViolationReport, ViolationReport]:
    """Compare violation reports for two draws."""
    report1 = test_draw(draw1_path, data)
    report2 = test_draw(draw2_path, data)
    return report1, report2


def get_severity_level(constraint_name: str) -> int:
    """
    Get the severity level for a constraint name.
    
    Args:
        constraint_name: Name of the constraint
        
    Returns:
        Severity level (1-5, lower = more severe)
    """
    return _severity_level_for(constraint_name)


def get_severity_label(level: int) -> str:
    """
    Get the label for a severity level.
    
    Args:
        level: Severity level (1-4)
        
    Returns:
        Label string ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    """
    return SEVERITY_LEVEL_LABELS.get(level, 'UNKNOWN')
