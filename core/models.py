# models.py
"""
Data models for scheduling system.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import time as tm

class PlayingField(BaseModel):
    name: str = Field(..., description="Name of the field")
    location: str = Field(..., description="Location of the field")

    def __getattr__(self, attr: str) -> str:
        if attr == "field_location":
            return self.location
        elif attr == "field_name":
            return self.name
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

class Grade(BaseModel):
    name: str = Field(..., description="Grade name")
    teams: List[str] = Field(..., description="List of team names in this grade")
    num_teams: int = Field(0, description="Number of teams in this grade")
    num_games: int = Field(0, description="Number of games in this grade")

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "num_teams", len(self.teams))

    def __lt__(self, other: "Grade") -> bool:
        expected_order = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
        if not isinstance(other, Grade):
            return NotImplemented
        try:
            self_index = expected_order.index(self.name)
            other_index = expected_order.index(other.name)
        except ValueError:
            raise ValueError(f"Unknown grade in comparison: {self.name} or {other.name}")
        return self_index > other_index

    def set_games(self, num_rounds: int) -> None:
        self.num_games = (num_rounds // (self.num_teams - 1)) * (self.num_teams - 1) if self.num_teams % 2 == 0 else (num_rounds // self.num_teams) * (self.num_teams - 1)

class Timeslot(BaseModel):
    date: str = Field(..., description="Date of the game (e.g., '2025-03-04')")
    day: str = Field(..., description="Day of the game (e.g., 'Saturday', 'Sunday')")
    time: str = Field(..., description="Time of the game (e.g., 14:00 for 2 PM)")
    week: int = Field(..., description="The week number for the season")
    day_slot: int = Field(..., description="The game slot for the day (e.g., 1 for first game of the day)")
    field: PlayingField = Field(..., description="Field where the game is played")
    round_no: int = Field(..., description="Round number for the season")

class Club(BaseModel):
    name: str = Field(..., description="Club name")
    home_field: str = Field(..., description="Home field")
    preferred_times: List[Timeslot] = Field(default=[], description="Preferred play times for the club")
    num_teams: int = Field(0, description="Number of teams in this club")

class Team(BaseModel):
    name: str = Field(..., description="Name of the team")
    club: Club = Field(..., description="Club the team belongs to")
    grade: str = Field(..., description="Grade the team belongs to")
    preferred_times: List[Timeslot] = Field(default=[], description="Times the team prefers to play")
    unavailable_times: List[Timeslot] = Field(default=[], description="Times the team cannot play")
    constraints: List[str] = Field(default=[], description="Special scheduling constraints for the team")

class ClubDay(BaseModel):
    date: str = Field(..., description="Date of the game (e.g., '2025-03-04')")
    day: str = Field(..., description="Day of the game (e.g., 'Saturday', 'Sunday')")
    week: int = Field(..., description="The week number for the season")
    field: PlayingField = Field(..., description="Field where the game is played")

class Game(BaseModel):
    team1: str = Field(..., description="First team playing")
    team2: str = Field(..., description="Second team playing")
    timeslot: Timeslot = Field(..., description="Scheduled time for the game")
    field: PlayingField = Field(..., description="Field where the game is played")
    grade: Grade = Field(..., description="Grade the game belongs to")

class WeeklyDraw(BaseModel):
    week: int = Field(..., description="Week number in the season")
    round_no: int = Field(..., description="Round number for the season")
    games: List[Game] = Field(..., description="Games scheduled for this week")
    bye_teams: List[str] = Field(default=[], description="Teams with a bye this week")

class Roster(BaseModel):
    weeks: List[WeeklyDraw] = Field(..., description="Complete schedule for the season")

    def save(self, path: str) -> None:
        """Save the roster to a JSON file."""
        from pathlib import Path
        import json
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.dict(), f, indent=4)
