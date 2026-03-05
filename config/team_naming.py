# config/team_naming.py
"""
Team Naming Conventions for Multi-Team Grades.

When a club has multiple teams in the same grade, this module provides
the standard naming conventions to use.

Usage:
    from config.team_naming import get_team_names
    
    # Get names for a club with 2 teams in a grade
    names = get_team_names('Tigers', count=2)
    # Returns: ['Tigers', 'Tigers Black']
    
    # Get names for Colts with 2 teams
    names = get_team_names('Colts', count=2)
    # Returns: ['Colts Gold', 'Colts Green']
"""

from typing import List, Optional

# Team naming conventions for clubs with multiple teams in a grade
# Format: 'ClubName': ['FirstTeamName', 'SecondTeamName', ...]
TEAM_NAME_CONVENTIONS = {
    'Tigers': ['Tigers', 'Tigers Black'],           # Black & Yellow colours
    'Wests': ['Wests', 'Wests Red'],                # Red & Green colours
    'University': ['Uni', 'Uni Seapigs'],           # Seapigs is the alternate
    'Colts': ['Colts Gold', 'Colts Green'],         # Gold & Green colours
}

# Default team name (used when club not in conventions)
DEFAULT_FIRST_NAME = None  # Use club name
DEFAULT_SUFFIX_PATTERN = '{club} {number}'  # e.g., "Maitland 2"


def get_team_names(club: str, count: int = 1) -> List[str]:
    """
    Get the standard team names for a club based on naming conventions.
    
    Args:
        club: The club name (e.g., 'Tigers', 'Colts')
        count: Number of teams in the grade (1, 2, or more)
    
    Returns:
        List of team names to use
    
    Examples:
        >>> get_team_names('Tigers', 1)
        ['Tigers']
        >>> get_team_names('Tigers', 2)
        ['Tigers', 'Tigers Black']
        >>> get_team_names('Colts', 2)
        ['Colts Gold', 'Colts Green']
        >>> get_team_names('Maitland', 2)
        ['Maitland', 'Maitland 2']
    """
    if count <= 0:
        return []
    
    # Check if club has defined naming conventions
    if club in TEAM_NAME_CONVENTIONS:
        conventions = TEAM_NAME_CONVENTIONS[club]
        if count <= len(conventions):
            return conventions[:count]
        else:
            # Need more names than defined - extend with numbered suffix
            names = list(conventions)
            for i in range(len(conventions) + 1, count + 1):
                names.append(f"{club} {i}")
            return names
    
    # Default naming: club name, then club name + number
    if count == 1:
        return [club]
    
    names = [club]
    for i in range(2, count + 1):
        names.append(f"{club} {i}")
    return names


def get_team_name(club: str, index: int = 0) -> str:
    """
    Get the team name for a specific index (0-based).
    
    Args:
        club: The club name
        index: 0 for first team, 1 for second team, etc.
    
    Returns:
        The team name to use
    
    Examples:
        >>> get_team_name('Tigers', 0)
        'Tigers'
        >>> get_team_name('Tigers', 1)
        'Tigers Black'
        >>> get_team_name('Colts', 0)
        'Colts Gold'
    """
    names = get_team_names(club, index + 1)
    return names[index] if index < len(names) else f"{club} {index + 1}"


def get_default_name(club: str) -> str:
    """
    Get the default (first) team name for a club.
    
    For most clubs this is just the club name, but for some
    (like Colts) there's a specific first team name.
    
    Examples:
        >>> get_default_name('Tigers')
        'Tigers'
        >>> get_default_name('Colts')
        'Colts Gold'
        >>> get_default_name('University')
        'Uni'
    """
    if club in TEAM_NAME_CONVENTIONS:
        return TEAM_NAME_CONVENTIONS[club][0]
    return club


def has_naming_convention(club: str) -> bool:
    """Check if a club has defined naming conventions."""
    return club in TEAM_NAME_CONVENTIONS


def list_all_conventions() -> dict:
    """Return all defined naming conventions."""
    return dict(TEAM_NAME_CONVENTIONS)


# For backwards compatibility and easy importing
__all__ = [
    'TEAM_NAME_CONVENTIONS',
    'get_team_names',
    'get_team_name', 
    'get_default_name',
    'has_naming_convention',
    'list_all_conventions',
]
