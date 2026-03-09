# Game Time Dictionaries Guide

> **Purpose:** Detailed explanation of how PHL_GAME_TIMES and SECOND_GRADE_TIMES control variable generation.
> **When to use:** When modifying which timeslots are available for PHL or 2nd grade.

---

## The Problem These Dicts Solve

Without filtering, the solver creates decision variables for EVERY possible:
- (Team1, Team2, Grade, Day, Time, Week, Field) combination

For a typical season this could be **100,000+ variables**. Most are impossible:
- PHL shouldn't play at 8:30am
- 2nd grade can't play at Gosford (PHL-only venue)
- Lower grades can't play Friday nights

**Solution:** Filter at variable generation time. Only create variables for valid slots.

---

## The Three Time Dictionaries

### 1. DAY_TIME_MAP (Base Slots)

**Controls:** All grades 3rd-6th (lower grades)

```python
DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    },
    'Central Coast Hockey Park': {
        'Sunday': [tm(12, 0), tm(13, 30)]
    }
}
```

**Note:** Lower grades automatically excluded from:
- Gosford (via filtering in `generate_X()`)
- Friday nights (not in DAY_TIME_MAP)

### 2. PHL_GAME_TIMES (PHL Only)

**Controls:** PHL grade only

**Structure:**
```python
PHL_GAME_TIMES = {
    'venue': {
        'field': {
            'day': [times]
        }
    }
}
```

**Example with annotations:**
```python
PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # East Field
            'Friday': [tm(19, 0)],  # 7pm Friday (Junior Boys alignment)
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]  # Mid-day slots
        },
        'WF': {  # West Field (same as EF)
            'Friday': [tm(19, 0)],
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        # SF (South Field) NOT LISTED - PHL excluded from SF
    },
    'Central Coast Hockey Park': {
        'Wyong Main Field': {
            'Friday': [tm(20, 0)],  # 8pm Friday (AGM decision)
            'Sunday': [tm(12, 0), tm(13, 30)]  # Gosford Sunday times
        },
    },
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}
```

**Key exclusions:**
- SF (South Field) at NIHC - not listed, PHL can't play there
- Early morning slots (8:30, 10:00) - not listed
- Late evening (7:30pm+) on Sunday - not listed

### 3. SECOND_GRADE_TIMES (2nd Grade Only)

**Controls:** 2nd grade only

**Design principle:** Include PHL times ± one slot for back-to-back scheduling.

```python
SECOND_GRADE_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {
            # PHL: 11:30, 13:00, 14:30, 16:00
            # 2nd: Add 10:00 (before) and 17:30 (after)
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
        'WF': {
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
    },
    # Gosford NOT LISTED - 2nd grade cannot play at Gosford
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(10, 30), tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}
```

**Key exclusions:**
- Gosford (Central Coast Hockey Park) - not listed, 2nd can't play there
- Friday nights - not listed (only PHL plays Friday)
- SF at NIHC - not listed

---

## How Variables Are Generated

In `utils.py` → `generate_X()`:

```python
def generate_X(...):
    for week in weeks:
        for grade in grades:
            if grade == 'PHL':
                # Use PHL_GAME_TIMES filtering
                valid_slots = filter_by_phl_times(...)
            elif grade == '2nd':
                # Use SECOND_GRADE_TIMES filtering
                valid_slots = filter_by_second_grade_times(...)
            else:
                # Lower grades: Use DAY_TIME_MAP, exclude Gosford and Friday
                valid_slots = filter_for_lower_grades(...)
            
            for slot in valid_slots:
                for matchup in matchups:
                    X[key] = model.NewBoolVar(...)
```

---

## Modifying Time Dictionaries

### Adding a New PHL Timeslot

1. Check it exists in `DAY_TIME_MAP` (base slots)
2. Add to `PHL_GAME_TIMES` under correct venue/field/day
3. Consider if 2nd grade needs the adjacent slot added

**Example:** Add 10:00am PHL at NIHC EF Sunday

```python
# WRONG - 10:00 not adjacent to PHL slots, breaks back-to-back
PHL_GAME_TIMES['Newcastle International Hockey Centre']['EF']['Sunday'].append(tm(10, 0))

# RIGHT - but consider if this makes sense
# 10:00 is far from other PHL times (11:30+), may cause issues
```

### Removing a Timeslot

Simply remove from the list. No variables will be created for that slot.

```python
# Remove 11:30 from PHL at NIHC EF Sunday
'Sunday': [tm(13, 0), tm(14, 30), tm(16, 0)]  # 11:30 removed
```

### Adding Friday Nights

1. Add to `DAY_TIME_MAP` if not present (for consistency)
2. Add to `PHL_GAME_TIMES` under 'Friday' key
3. Update `FRIDAY_NIGHT_CONFIG` with counts and dates

```python
DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Friday': [tm(19, 0)],  # Add Friday if not present
        'Sunday': [...],
    },
}

PHL_GAME_TIMES['Newcastle International Hockey Centre']['EF']['Friday'] = [tm(19, 0)]
```

---

## Common Mistakes

### 1. Adding a time not in DAY_TIME_MAP

```python
# WRONG - 15:30 doesn't exist in base slots
PHL_GAME_TIMES['...']['EF']['Sunday'] = [tm(15, 30)]  # Error!

# RIGHT - use existing slot
PHL_GAME_TIMES['...']['EF']['Sunday'] = [tm(14, 30)]  # 14:30 exists
```

### 2. Forgetting to update both EF and WF

```python
# INCOMPLETE - only updated EF
PHL_GAME_TIMES['Newcastle International Hockey Centre']['EF']['Sunday'] = [new_times]
# Also update WF!
PHL_GAME_TIMES['Newcastle International Hockey Centre']['WF']['Sunday'] = [new_times]
```

### 3. Including SF for PHL

```python
# WRONG - PHL can't play on SF
PHL_GAME_TIMES['Newcastle International Hockey Centre']['SF'] = {...}

# RIGHT - SF not in PHL dict
```

### 4. Including Gosford for 2nd Grade

```python
# WRONG - 2nd can't play at Gosford
SECOND_GRADE_TIMES['Central Coast Hockey Park'] = {...}

# RIGHT - Gosford not in 2nd grade dict
```

---

## Verification

After modifying time dicts, run pre-season report:

```powershell
.\.venv\Scripts\python.exe run.py preseason --year 2026
```

Check:
- Variable counts are reasonable
- PHL has Friday variables (if expected)
- 2nd grade doesn't show Gosford
- Lower grades don't show Gosford or Friday
