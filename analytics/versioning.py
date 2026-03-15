# analytics/versioning.py
"""
Draw versioning system with automatic CHANGELOG generation.

This module provides:
1. Semantic versioning for draws (major.minor)
2. Automatic CHANGELOG generation with diffs
3. Version history tracking
4. Safe storage with version collision prevention
5. Unified output saving for all solver modes

Version Scheme:
- Major (X.0): Complete regeneration or structural changes
- Minor (X.Y): Incremental updates (game modifications, swaps, fixes)

Directory Structure:
    draws/{year}/
    ├── current.json              # Always the latest draw (auto-updated)
    ├── current.xlsx              # Always the latest schedule Excel
    ├── current_analytics.xlsx    # Always the latest analytics
    ├── CHANGELOG.md              # Auto-generated version history
    └── versions/                 # All versioned draws
        ├── draw_v1.0.json
        ├── draw_v1.0.xlsx
        ├── draw_v2.0.json
        └── draw_v2.0.xlsx

Usage:
    from analytics.versioning import DrawVersionManager
    
    manager = DrawVersionManager("draws", year=2026)
    
    # Save a new major version (fresh generation)
    version = manager.save_new_draw(draw, "Initial generation")
    
    # Save a minor update (modification to existing)
    version = manager.save_modified_draw(draw, old_draw, "Fixed Maitland clash")
    
    # Save full solver output (draw + Excel + analytics)
    version = manager.save_solver_output(solution, data, "Staged solve")
"""

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .storage import DrawStorage, StoredGame


@dataclass
class DrawVersion:
    """Represents a specific version of a draw."""
    major: int
    minor: int
    created_at: str
    description: str
    filename: str
    game_count: int
    
    @property
    def version_string(self) -> str:
        return f"v{self.major}.{self.minor}"
    
    def __str__(self) -> str:
        return f"{self.version_string} ({self.filename})"


@dataclass
class VersionDiff:
    """Tracks differences between two draw versions."""
    added_games: List[Dict[str, Any]] = field(default_factory=list)
    removed_games: List[Dict[str, Any]] = field(default_factory=list)
    modified_games: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def has_changes(self) -> bool:
        return bool(self.added_games or self.removed_games or self.modified_games)
    
    @property
    def summary(self) -> str:
        parts = []
        if self.added_games:
            parts.append(f"+{len(self.added_games)} games")
        if self.removed_games:
            parts.append(f"-{len(self.removed_games)} games")
        if self.modified_games:
            parts.append(f"~{len(self.modified_games)} modified")
        return ", ".join(parts) if parts else "No changes"


class DrawVersionManager:
    """
    Manages versioned draw storage with automatic CHANGELOG.
    
    Directory structure:
        draws/{year}/
        ├── current.json              # Always the latest draw
        ├── current.xlsx              # Always the latest schedule Excel
        ├── current_analytics.xlsx    # Always the latest analytics
        ├── CHANGELOG.md              # Auto-generated version history
        └── versions/                 # All versioned draws
            ├── draw_v1.0.json
            ├── draw_v1.0.xlsx
            ├── draw_v2.0.json
            └── draw_v2.0.xlsx
    """
    
    VERSION_PATTERN = re.compile(r"draw_v(\d+)\.(\d+)\.json")
    
    def __init__(self, base_dir: str, year: Optional[int] = None):
        """
        Initialize version manager.
        
        Args:
            base_dir: Base directory for draws (e.g., "draws")
            year: Optional year (creates draws/{year}/ structure)
        """
        if year:
            self.base_path = Path(base_dir) / str(year)
        else:
            self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Versions subfolder for all versioned draws
        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)
        
        self.changelog_path = self.base_path / "CHANGELOG.md"
    
    def get_versions(self) -> List[DrawVersion]:
        """Get all existing versions, sorted by version number.
        
        Looks in both versions/ subfolder and base path for backward compatibility.
        """
        versions = []
        seen = set()
        
        # Check versions/ subfolder first (new location)
        for search_path in [self.versions_path, self.base_path]:
            for path in search_path.glob("draw_v*.json"):
                match = self.VERSION_PATTERN.match(path.name)
                if match and path.name not in seen:
                    seen.add(path.name)
                    major, minor = int(match.group(1)), int(match.group(2))
                    try:
                        draw = DrawStorage.load(str(path))
                        versions.append(DrawVersion(
                            major=major,
                            minor=minor,
                            created_at=draw.created_at,
                            description=draw.description,
                            filename=path.name,
                            game_count=draw.num_games
                        ))
                    except Exception as e:
                        print(f"Warning: Could not load {path}: {e}")
        
        return sorted(versions, key=lambda v: (v.major, v.minor))
    
    def get_latest_version(self) -> Optional[DrawVersion]:
        """Get the most recent version."""
        versions = self.get_versions()
        return versions[-1] if versions else None
    
    def get_next_major_version(self) -> Tuple[int, int]:
        """Get the next major version number."""
        latest = self.get_latest_version()
        if latest:
            return (latest.major + 1, 0)
        return (1, 0)
    
    def get_next_minor_version(self) -> Tuple[int, int]:
        """Get the next minor version number."""
        latest = self.get_latest_version()
        if latest:
            return (latest.major, latest.minor + 1)
        return (1, 0)  # First version is always 1.0
    
    def _version_filename(self, major: int, minor: int) -> str:
        """Generate filename for a version."""
        return f"draw_v{major}.{minor}.json"
    
    def _game_key(self, game: StoredGame) -> str:
        """Generate a unique key for a game (for diffing)."""
        return f"{game.team1}|{game.team2}|{game.grade}|{game.week}|{game.round_no}"
    
    def _game_full_key(self, game: StoredGame) -> str:
        """Generate a full key including schedule details."""
        return (f"{game.team1}|{game.team2}|{game.grade}|{game.week}|"
                f"{game.date}|{game.time}|{game.field_location}")
    
    def compute_diff(
        self,
        old_draw: DrawStorage,
        new_draw: DrawStorage
    ) -> VersionDiff:
        """
        Compute differences between two draws.
        
        Returns:
            VersionDiff with added, removed, and modified games
        """
        diff = VersionDiff()
        
        # Create lookup dicts
        old_games = {self._game_key(g): g for g in old_draw.games}
        new_games = {self._game_key(g): g for g in new_draw.games}
        
        old_full = {self._game_key(g): self._game_full_key(g) for g in old_draw.games}
        new_full = {self._game_key(g): self._game_full_key(g) for g in new_draw.games}
        
        # Find removed games
        for key in old_games:
            if key not in new_games:
                g = old_games[key]
                diff.removed_games.append({
                    "matchup": f"{g.team1} vs {g.team2}",
                    "grade": g.grade,
                    "week": g.week,
                    "date": g.date,
                    "time": g.time,
                    "venue": g.field_location
                })
        
        # Find added games
        for key in new_games:
            if key not in old_games:
                g = new_games[key]
                diff.added_games.append({
                    "matchup": f"{g.team1} vs {g.team2}",
                    "grade": g.grade,
                    "week": g.week,
                    "date": g.date,
                    "time": g.time,
                    "venue": g.field_location
                })
        
        # Find modified games (same matchup but different schedule)
        for key in new_games:
            if key in old_games and old_full[key] != new_full[key]:
                old_g = old_games[key]
                new_g = new_games[key]
                diff.modified_games.append({
                    "matchup": f"{new_g.team1} vs {new_g.team2}",
                    "grade": new_g.grade,
                    "week": new_g.week,
                    "old": f"{old_g.date} {old_g.time} @ {old_g.field_location}",
                    "new": f"{new_g.date} {new_g.time} @ {new_g.field_location}"
                })
        
        return diff
    
    def _update_changelog(
        self,
        version: DrawVersion,
        description: str,
        diff: Optional[VersionDiff] = None,
        is_major: bool = True
    ):
        """Update the CHANGELOG.md file with a new entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Build changelog entry
        entry_lines = [
            f"## [{version.version_string}] - {timestamp}",
            "",
            f"**{description}**",
            "",
            f"- Games: {version.game_count}",
        ]
        
        if is_major:
            entry_lines.append("- Type: Major version (complete regeneration)")
        else:
            entry_lines.append("- Type: Minor update")
        
        if diff and diff.has_changes:
            entry_lines.append(f"- Changes: {diff.summary}")
            entry_lines.append("")
            
            if diff.added_games:
                entry_lines.append("### Added Games")
                for g in diff.added_games[:10]:  # Limit to 10 entries
                    entry_lines.append(
                        f"- Week {g['week']}: {g['matchup']} ({g['grade']}) - "
                        f"{g['date']} {g['time']} @ {g['venue']}"
                    )
                if len(diff.added_games) > 10:
                    entry_lines.append(f"- ... and {len(diff.added_games) - 10} more")
                entry_lines.append("")
            
            if diff.removed_games:
                entry_lines.append("### Removed Games")
                for g in diff.removed_games[:10]:
                    entry_lines.append(
                        f"- Week {g['week']}: {g['matchup']} ({g['grade']}) - "
                        f"{g['date']} {g['time']} @ {g['venue']}"
                    )
                if len(diff.removed_games) > 10:
                    entry_lines.append(f"- ... and {len(diff.removed_games) - 10} more")
                entry_lines.append("")
            
            if diff.modified_games:
                entry_lines.append("### Modified Games")
                for g in diff.modified_games[:10]:
                    entry_lines.append(
                        f"- Week {g['week']}: {g['matchup']} ({g['grade']})"
                    )
                    entry_lines.append(f"  - From: {g['old']}")
                    entry_lines.append(f"  - To: {g['new']}")
                if len(diff.modified_games) > 10:
                    entry_lines.append(f"- ... and {len(diff.modified_games) - 10} more")
                entry_lines.append("")
        
        entry_lines.append("")
        entry_lines.append("---")
        entry_lines.append("")
        
        new_entry = "\n".join(entry_lines)
        
        # Read existing changelog or create header
        if self.changelog_path.exists():
            existing = self.changelog_path.read_text(encoding="utf-8")
            # Insert after the header section
            if "# Draw Version History" in existing:
                parts = existing.split("\n---\n", 1)
                if len(parts) > 1:
                    # Insert new entry after header
                    content = parts[0] + "\n---\n\n" + new_entry + parts[1]
                else:
                    content = existing + "\n\n" + new_entry
            else:
                content = existing + "\n\n" + new_entry
        else:
            header = [
                "# Draw Version History",
                "",
                "This file is automatically generated. Do not edit manually.",
                "",
                "---",
                "",
            ]
            content = "\n".join(header) + new_entry
        
        self.changelog_path.write_text(content, encoding="utf-8")
    
    def _update_current(self, version_path: Path, xlsx_path: Path = None, analytics_path: Path = None):
        """Update the 'current' files to point to the latest version."""
        # Copy draw JSON
        current_json = self.base_path / "current.json"
        shutil.copy(version_path, current_json)
        
        # Copy schedule Excel if provided
        if xlsx_path and xlsx_path.exists():
            current_xlsx = self.base_path / "current.xlsx"
            shutil.copy(xlsx_path, current_xlsx)
        
        # Copy analytics Excel if provided
        if analytics_path and analytics_path.exists():
            current_analytics = self.base_path / "current_analytics.xlsx"
            shutil.copy(analytics_path, current_analytics)
    
    def save_new_draw(
        self,
        draw: DrawStorage,
        description: str,
        is_major: bool = True,
        xlsx_path: Path = None,
        analytics_path: Path = None
    ) -> DrawVersion:
        """
        Save a new draw version.
        
        Args:
            draw: The draw to save
            description: Human-readable description of this version
            is_major: If True, increments major version; else increments minor
            xlsx_path: Optional path to schedule Excel to copy into versions/
            analytics_path: Optional path to analytics Excel to copy into versions/
            
        Returns:
            DrawVersion object for the saved version
        """
        # Determine version number
        if is_major:
            major, minor = self.get_next_major_version()
        else:
            major, minor = self.get_next_minor_version()
        
        # Update draw metadata
        draw.description = description
        draw.metadata["version"] = f"v{major}.{minor}"
        draw.metadata["version_type"] = "major" if is_major else "minor"
        
        # Save the draw to versions/ subfolder
        filename = self._version_filename(major, minor)
        filepath = self.versions_path / filename
        draw.save(str(filepath))
        
        # Copy xlsx and analytics into versions/ if provided
        versioned_xlsx = None
        versioned_analytics = None
        if xlsx_path and Path(xlsx_path).exists():
            versioned_xlsx = self.versions_path / f"draw_v{major}.{minor}.xlsx"
            shutil.copy(xlsx_path, versioned_xlsx)
        if analytics_path and Path(analytics_path).exists():
            versioned_analytics = self.versions_path / f"draw_v{major}.{minor}_analytics.xlsx"
            shutil.copy(analytics_path, versioned_analytics)
        
        # Create version record
        version = DrawVersion(
            major=major,
            minor=minor,
            created_at=draw.created_at,
            description=description,
            filename=filename,
            game_count=draw.num_games
        )
        
        # Update changelog
        self._update_changelog(version, description, is_major=is_major)
        
        # Update current pointer files
        self._update_current(
            filepath,
            xlsx_path=versioned_xlsx or (Path(xlsx_path) if xlsx_path else None),
            analytics_path=versioned_analytics or (Path(analytics_path) if analytics_path else None)
        )
        
        print(f"Saved {version} to {filepath}")
        print(f"Updated current.json in {self.base_path}")
        return version
    
    def save_modified_draw(
        self,
        new_draw: DrawStorage,
        old_draw: DrawStorage,
        description: str,
        xlsx_path: Path = None,
        analytics_path: Path = None
    ) -> DrawVersion:
        """
        Save a modified draw as a minor version with diff tracking.
        
        Args:
            new_draw: The modified draw to save
            old_draw: The previous draw for comparison
            description: Description of changes
            xlsx_path: Optional path to schedule Excel to copy into versions/
            analytics_path: Optional path to analytics Excel to copy into versions/
            
        Returns:
            DrawVersion object for the saved version
        """
        # Compute diff
        diff = self.compute_diff(old_draw, new_draw)
        
        # Get next minor version
        major, minor = self.get_next_minor_version()
        
        # Update draw metadata
        new_draw.description = description
        new_draw.metadata["version"] = f"v{major}.{minor}"
        new_draw.metadata["version_type"] = "minor"
        new_draw.metadata["changes"] = {
            "added": len(diff.added_games),
            "removed": len(diff.removed_games),
            "modified": len(diff.modified_games)
        }
        
        # Save the draw to versions/
        filename = self._version_filename(major, minor)
        filepath = self.versions_path / filename
        new_draw.save(str(filepath))
        
        # Copy xlsx and analytics into versions/ if provided
        versioned_xlsx = None
        versioned_analytics = None
        if xlsx_path and Path(xlsx_path).exists():
            versioned_xlsx = self.versions_path / f"draw_v{major}.{minor}.xlsx"
            shutil.copy(xlsx_path, versioned_xlsx)
        if analytics_path and Path(analytics_path).exists():
            versioned_analytics = self.versions_path / f"draw_v{major}.{minor}_analytics.xlsx"
            shutil.copy(analytics_path, versioned_analytics)
        
        # Create version record
        version = DrawVersion(
            major=major,
            minor=minor,
            created_at=new_draw.created_at,
            description=description,
            filename=filename,
            game_count=new_draw.num_games
        )
        
        # Update changelog with diff
        self._update_changelog(version, description, diff=diff, is_major=False)
        
        # Update current pointer files
        self._update_current(
            filepath,
            xlsx_path=versioned_xlsx or (Path(xlsx_path) if xlsx_path else None),
            analytics_path=versioned_analytics or (Path(analytics_path) if analytics_path else None)
        )
        
        print(f"Saved {version} to {filepath}")
        print(f"Changes: {diff.summary}")
        print(f"Updated current.json in {self.base_path}")
        return version
    
    def load_version(self, major: int, minor: int) -> Optional[DrawStorage]:
        """Load a specific version. Checks versions/ subfolder first, then base path."""
        filename = self._version_filename(major, minor)
        # Check versions/ subfolder first (new location)
        filepath = self.versions_path / filename
        if filepath.exists():
            return DrawStorage.load(str(filepath))
        # Fallback to base path (legacy location)
        filepath = self.base_path / filename
        if filepath.exists():
            return DrawStorage.load(str(filepath))
        return None
    
    def load_latest(self) -> Optional[DrawStorage]:
        """Load the latest version."""
        current_path = self.base_path / "current.json"
        if current_path.exists():
            return DrawStorage.load(str(current_path))
        
        # Fallback to finding latest
        latest = self.get_latest_version()
        if latest:
            return self.load_version(latest.major, latest.minor)
        return None
    
    def list_versions(self) -> str:
        """Get a formatted list of all versions."""
        versions = self.get_versions()
        if not versions:
            return "No versions found."
        
        lines = ["Version History:", "=" * 50]
        for v in versions:
            lines.append(f"  {v.version_string:8} | {v.created_at[:10]} | {v.game_count:4} games | {v.description[:40]}")
        return "\n".join(lines)
    
    def save_solver_output(
        self,
        solution: dict,
        data: dict,
        description: str,
        mode: str = "staged",
        is_major: bool = True,
    ) -> DrawVersion:
        """
        Unified output saving for all solver modes.
        
        Saves:
        1. Versioned draw JSON in versions/
        2. Versioned schedule Excel in versions/
        3. Versioned analytics Excel in versions/
        4. current.json, current.xlsx, current_analytics.xlsx in year root
        5. CHANGELOG.md update
        6. Violation report if any
        
        Args:
            solution: The X solution dictionary from the solver
            data: The data dictionary
            description: Description of this solve (e.g., "Staged solve - severity 1-3")
            mode: Solve mode label ("staged", "simple", "resume", etc.)
            is_major: Whether this is a major version (True) or minor update (False)
            
        Returns:
            DrawVersion object
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils import convert_X_to_roster, export_roster_to_excel
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_description = f"{description} ({mode} mode, {timestamp})"
        
        # 1. Create DrawStorage from solution
        draw = DrawStorage.from_X_solution(solution, description=full_description)
        
        # 2. Export schedule Excel to a temp location, then version it
        roster = convert_X_to_roster(solution, data)
        temp_xlsx = self.base_path / f"_temp_schedule_{timestamp}.xlsx"
        export_roster_to_excel(roster, data, filename=str(temp_xlsx))
        
        # 3. Generate analytics Excel
        temp_analytics = None
        try:
            from .storage import DrawAnalytics
            analytics = DrawAnalytics(draw, data)
            temp_analytics = self.base_path / f"_temp_analytics_{timestamp}.xlsx"
            analytics.export_analytics_to_excel(str(temp_analytics))
        except Exception as e:
            print(f"Warning: Could not generate analytics: {e}")
        
        # 4. Save as versioned draw (handles versioning, current, changelog)
        if is_major:
            version = self.save_new_draw(
                draw, full_description, is_major=True,
                xlsx_path=temp_xlsx,
                analytics_path=temp_analytics
            )
        else:
            # Load previous draw for diff
            old_draw = self.load_latest()
            if old_draw:
                version = self.save_modified_draw(
                    draw, old_draw, full_description,
                    xlsx_path=temp_xlsx,
                    analytics_path=temp_analytics
                )
            else:
                version = self.save_new_draw(
                    draw, full_description, is_major=True,
                    xlsx_path=temp_xlsx,
                    analytics_path=temp_analytics
                )
        
        # 5. Run violation check and save if violations found
        try:
            from .tester import DrawTester
            tester = DrawTester(draw, data)
            report = tester.run_violation_check()
            print(f"Violation Check: {report.summary()}")
            
            if report.has_violations:
                violations_path = self.versions_path / f"violations_{version.version_string}.txt"
                with open(violations_path, 'w') as f:
                    f.write(report.full_report())
                # Also save as current violations
                current_violations = self.base_path / "current_violations.txt"
                shutil.copy(violations_path, current_violations)
                print(f"Violation report saved to {violations_path}")
            else:
                # Remove stale current violations file if exists
                current_violations = self.base_path / "current_violations.txt"
                if current_violations.exists():
                    current_violations.unlink()
        except Exception as e:
            print(f"Warning: Could not run violation check: {e}")
        
        # 6. Clean up temp files
        if temp_xlsx.exists():
            temp_xlsx.unlink()
        if temp_analytics and temp_analytics.exists():
            temp_analytics.unlink()
        
        print(f"\n{'='*60}")
        print(f"OUTPUT SUMMARY ({version.version_string})")
        print(f"{'='*60}")
        print(f"  Version:    {version.version_string}")
        print(f"  Games:      {version.game_count}")
        print(f"  Mode:       {mode}")
        print(f"  Draw JSON:  {self.base_path / 'current.json'}")
        print(f"  Schedule:   {self.base_path / 'current.xlsx'}")
        print(f"  Analytics:  {self.base_path / 'current_analytics.xlsx'}")
        print(f"  Versioned:  {self.versions_path / version.filename}")
        print(f"  Changelog:  {self.changelog_path}")
        
        return version
    
    def migrate_legacy_draws(self):
        """
        Migrate draw_v*.json files from base path into versions/ subfolder.
        
        Call this once to move existing versioned draws into the new structure.
        """
        moved = 0
        for path in self.base_path.glob("draw_v*.json"):
            dest = self.versions_path / path.name
            if not dest.exists():
                shutil.move(str(path), str(dest))
                print(f"  Migrated {path.name} -> versions/")
                moved += 1
        
        # Also move any matching xlsx files
        for path in self.base_path.glob("draw_v*.xlsx"):
            dest = self.versions_path / path.name
            if not dest.exists():
                shutil.move(str(path), str(dest))
                print(f"  Migrated {path.name} -> versions/")
                moved += 1
        
        if moved:
            print(f"Migrated {moved} files to versions/")
            # Update current.json to point to latest
            latest = self.get_latest_version()
            if latest:
                latest_path = self.versions_path / latest.filename
                self._update_current(latest_path)
                print(f"Updated current.json -> {latest.version_string}")
        else:
            print("No legacy draws to migrate.")
