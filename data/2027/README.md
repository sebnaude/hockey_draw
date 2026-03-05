# 2027 Season Data

This folder contains team and constraint data for the 2027 hockey season.

## Structure

```
data/2027/
├── teams/              # Club CSV files (one per club)
│   └── {club}.csv      # Format: Club,Grade,Team Name
├── noplay/             # Club unavailability spreadsheets (optional)
│   └── {club}_noplay.xlsx
├── field_availability/ # Field-specific availability (if needed)
└── documentation.txt   # Season notes and status
```

## IMPORTANT

All files in this folder must be created fresh for 2027.
DO NOT copy files from previous seasons without verification.

## Checklist

- [ ] Create team CSV for each club
- [ ] Verify team names and grades
- [ ] Add any noplay Excel files
- [ ] Update documentation.txt with season status
