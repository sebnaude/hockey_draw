# Friday Night Games Clarification - 2026 Season

**Created:** 2026-03-09  
**Purpose:** Document the correct understanding of Friday night games for future reference.

---

## ⚠️ CRITICAL: Friday Nights Are NOT Extra Weekends

### Correct Understanding

A Friday night game at Gosford is **PART OF that weekend**, not an additional weekend.

- If a PHL team plays Friday at Gosford, they **do NOT also play Sunday** that week
- Friday games are an **alternative slot** within the same weekend, not additional capacity
- The team has ONE game that weekend - just on Friday instead of Sunday

### Why PHL Has 22 Available Weekends (Not 28)

**Incorrect reasoning (what was previously assumed):**
- 20 Sundays + 8 Friday nights = 28 potential game slots ❌

**Correct reasoning:**
- 24 total Sundays in season (March 22 - August 30)
- 4 blocked weekends (Easter, State Championships)
- = 20 available Sundays for most grades
- **PLUS:** 2 additional weekends "rescued" via Friday option
  - These are weekends where Sunday is blocked (e.g., State Championships Sat/Sun)
  - But Friday night at Gosford is still available
  - So the weekend becomes playable for PHL via Friday
- = **22 total available weekends for PHL**

### Rescued Weekends (2026)

The following weekends are blocked for Sunday play but available Friday:

| Weekend | Sunday Status | Friday Available? | Result |
|---------|--------------|-------------------|--------|
| May 15-17 | Masters SC (blocked) | Yes (Gosford) | Rescued |
| Jun 19-21 | U16 Girls SC (blocked) | Yes (Gosford) | Rescued |

### 8 Friday Night Games Means...

The AGM decision for "8 Friday night games at Gosford" means:
- 8 of PHL's games will be on Friday evenings
- These are still within the 22 available weekends
- Not 8 EXTRA games on top of the normal schedule

### Config Implication

```python
MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 normal + 2 rescued (NOT 20 + 8!)
    '2nd': 20,   # 20 Sundays only
    ...
}
```

---

## Source Documents

- AGM Minutes 2026
- Email from HCPL re: "20 playing weekends" 
- State Championship dates confirmation

---

## Future Reference

When setting up future seasons:
1. Count available Sundays (after blocking)
2. Identify blocked weekends where Friday is still available
3. Add those "rescued" weekends to PHL total
4. Do NOT simply add Friday night count to Sunday count
