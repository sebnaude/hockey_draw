# core/__init__.py
"""
Core scheduling engine.

Provides:
- Data models (Team, Club, Grade, Timeslot, etc.)
- Constraint implementations
- Staged solver with checkpointing
- Decision variable generation
"""

from .models import (
    PlayingField,
    Grade,
    Timeslot,
    Club,
    Team,
    ClubDay,
    Game,
    WeeklyDraw,
    Roster,
)

__all__ = [
    'PlayingField',
    'Grade',
    'Timeslot',
    'Club',
    'Team',
    'ClubDay',
    'Game',
    'WeeklyDraw',
    'Roster',
]
