# analytics/versioning.py
"""
Draw versioning system with automatic CHANGELOG generation.

This module provides:
1. Semantic versioning for draws (major.minor)
2. Automatic CHANGELOG generation with diffs
3. Version history tracking
4. Safe storage with version collision prevention

Version Scheme:
- Major (X.0): Complete regeneration or structural changes
- Minor (X.Y): Incremental updates (game modifications, swaps, fixes)

Usage:
    from analytics.versioning import DrawVersionManager
    
    manager = DrawVersionManager("draws/2026")
    
    # Save a new major version (fresh generation)
    version = manager.save_new_draw(draw, "Initial generation")
    
    # Save a minor update (modification to existing)
    version = manager.save_modified_draw(draw, old_draw, "Fixed Maitland clash")
"""

import re
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
        draws/2026/
        ├── CHANGELOG.md         # Auto-generated version history
        ├── draw_v1.0.json       # Major version 1
        ├── draw_v1.1.json       # Minor update
        ├── draw_v2.0.json       # New major version
        └── current.json         # Symlink/copy to latest
    """
    
    VERSION_PATTERN = re.compile(r"draw_v(\d+)\.(\d+)\.json")
    
    def __init__(self, base_dir: str, year: Optional[int] = None):
        """
        Initialize version manager.
        
        Args:
            base_dir: Base directory for versioned draws
            year: Optional year (will be appended to base_dir if provided)
        """
        if year:
            self.base_path = Path(base_dir) / str(year)
        else:
            self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.changelog_path = self.base_path / "CHANGELOG.md"
    
    def get_versions(self) -> List[DrawVersion]:
        """Get all existing versions, sorted by version number."""
        versions = []
        for path in self.base_path.glob("draw_v*.json"):
            match = self.VERSION_PATTERN.match(path.name)
            if match:
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
    
    def _update_current(self, version_path: Path):
        """Update the 'current.json' to point to the latest version."""
        current_path = self.base_path / "current.json"
        # On Windows, just copy the file; on Unix we could use symlink
        import shutil
        shutil.copy(version_path, current_path)
    
    def save_new_draw(
        self,
        draw: DrawStorage,
        description: str,
        is_major: bool = True
    ) -> DrawVersion:
        """
        Save a new draw version.
        
        Args:
            draw: The draw to save
            description: Human-readable description of this version
            is_major: If True, increments major version; else increments minor
            
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
        
        # Save the draw
        filename = self._version_filename(major, minor)
        filepath = self.base_path / filename
        draw.save(str(filepath))
        
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
        
        # Update current pointer
        self._update_current(filepath)
        
        print(f"Saved {version} to {filepath}")
        return version
    
    def save_modified_draw(
        self,
        new_draw: DrawStorage,
        old_draw: DrawStorage,
        description: str
    ) -> DrawVersion:
        """
        Save a modified draw as a minor version with diff tracking.
        
        Args:
            new_draw: The modified draw to save
            old_draw: The previous draw for comparison
            description: Description of changes
            
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
        
        # Save the draw
        filename = self._version_filename(major, minor)
        filepath = self.base_path / filename
        new_draw.save(str(filepath))
        
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
        
        # Update current pointer
        self._update_current(filepath)
        
        print(f"Saved {version} to {filepath}")
        print(f"Changes: {diff.summary}")
        return version
    
    def load_version(self, major: int, minor: int) -> Optional[DrawStorage]:
        """Load a specific version."""
        filename = self._version_filename(major, minor)
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
