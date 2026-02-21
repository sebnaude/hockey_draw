# Hockey Draw Scheduling System

A constraint programming system for generating hockey competition schedules using Google OR-Tools.

## Quick Start

```bash
cd refactored
pip install -r requirements.txt
python run.py generate --year 2025
```

## Usage

```bash
# Generate new draw (default: 2025)
python run.py generate --year 2025

# Generate with staged solving (recommended)
python run.py generate --year 2025 --staged

# Resume from checkpoint
python run.py generate --year 2025 --staged --resume run_1

# Test existing draw for violations
python run.py test draws/draw_2025.json

# Generate full analytics report
python run.py analyze draws/draw_2025.json

# Generate stakeholder report (Excel)
python run.py report draws/draw_2025.json --output analytics.xlsx

# Generate club-specific report
python run.py club-report draws/draw_2025.json Maitland --output reports/

# Generate compliance certificate
python run.py cert draws/draw_2025.json --output compliance.xlsx

# Get help
python run.py --help
python run.py generate --help
```

## Directory Structure

```
refactored/
├── run.py              # Main entry point (CLI)
├── main_staged.py      # Staged solver implementation
├── constraints.py      # Constraint implementations
├── models.py           # Data models
├── utils.py            # Utility functions
│
├── analytics/          # Draw analysis and testing
│   ├── storage.py      # Pliable JSON format, partial import
│   ├── reports.py      # ClubReport, GradeReport, ComplianceCertificate
│   └── tester.py       # Modification testing
│
├── config/             # Season configurations
│   ├── season_2025.py
│   └── season_2026.py
│
├── core/               # Core engine (models)
├── data/               # Input data files
├── docs/               # Documentation
├── tests/              # Test suite
├── draws/              # Output schedules
└── checkpoints/        # Solver checkpoints
```

## Documentation

- **[User Guide](docs/README.md)** - Input data requirements and configuration
- **[System Overview](docs/SYSTEM_OVERVIEW.md)** - Technical architecture
- **[Draw Rules](docs/DRAW_RULES.md)** - Constraint documentation
- **[AI Instructions](docs/claude.md)** - For AI assistants

## Requirements

- Python 3.8+
- OR-Tools
- Pandas
- Pydantic

See `requirements.txt` for full dependencies.

## License

Internal use only.
