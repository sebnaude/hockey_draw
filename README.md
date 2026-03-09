# Hockey Draw Scheduling System

A constraint programming system for generating hockey competition schedules using Google OR-Tools.

## Quick Start

```bash
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
├── run.py              # Main entry point (CLI)
├── main_staged.py      # Staged solver implementation
├── models.py           # Data models
├── utils.py            # Utility functions
│
├── constraints/        # Constraint modules
│   ├── original.py     # Original human-written constraints
│   ├── ai.py           # AI-enhanced constraints
│   ├── soft.py         # Soft constraint variants
│   ├── severity.py     # Severity-based relaxation
│   └── resolver.py     # Infeasibility resolver
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
├── scripts/            # Utility scripts
├── draws/              # Output schedules
├── reports/            # Generated reports
└── checkpoints/        # Solver checkpoints
```

## Documentation

### For AI Assistants
- **[AI Documentation Index](docs/ai/README.md)** - Start here
- **[Season Setup](docs/ai/SEASON_SETUP.md)** - Pre-season checklist
- **[Configuration Reference](docs/ai/CONFIGURATION_REFERENCE.md)** - All config parameters
- **[Constraint Application](docs/ai/CONSTRAINT_APPLICATION.md)** - How to add restrictions
- **[Game Time Dictionaries](docs/ai/GAME_TIME_DICTIONARIES.md)** - PHL/2nd grade variable filtering
- **[System Operation](docs/ai/SYSTEM_OPERATION.md)** - Running the solver

### For Human Operators
- **[System Overview](docs/system/SYSTEM_OVERVIEW.md)** - Technical architecture
- **[Capabilities](docs/system/CAPABILITIES.md)** - What the system can do
- **[User Guide](docs/system/USER_GUIDE.md)** - Input data requirements

### Season-Specific
- **[Draw Rules](seasons/RULES.md)** - Constraint documentation (human-readable)
- **[2026 Reports](seasons/2026/)** - Pre-season and analysis reports

## Requirements

- Python 3.8+
- OR-Tools
- Pandas
- Pydantic

See `requirements.txt` for full dependencies.

## License

Internal use only.
