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
# Generate new draw (staged solving - 2 stages)
python run.py generate --year 2025

# Generate with automatic constraint relaxation (if infeasible)
python run.py generate --year 2025 --relax

# Generate with simple mode (all constraints at once)
python run.py generate --year 2025 --simple

# Run only stage 1 (required constraints)
python run.py generate --year 2025 --stages stage1_required

# Resume from checkpoint
python run.py generate --year 2025 --resume run_1 stage1_required

# Diagnose infeasibility
python run.py diagnose --year 2025
python run.py diagnose --year 2025 --resolve  # Auto-relax constraints

# Test existing draw for violations
python run.py test draws/draw_2025.json --year 2025

# Generate full analytics report
python run.py analyze draws/draw_2025.json --year 2025

# Generate pre-season configuration report
python run.py preseason --year 2026

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
