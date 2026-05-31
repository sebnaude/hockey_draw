<!-- status: handoff -->
<!-- spec: spec-035 (e2e) — deep-dive into the ClubDay × ClubVsClubAlignment infeasibility -->
<!-- author: opus session 2026-05-29 -->
<!-- supersedes the §11 summary in spec-035-e2e-infeasibility-handoff.md with a full account -->

# spec-035 — ClubDay × ClubVsClubAlignment infeasibility: FULL handoff

> **Read order:** this file is self-contained. It records (a) the goal, (b) every
> probe we ran and its verdict, (c) what both atom families are designed to do,
> (d) the exact mechanism that makes them jointly infeasible, (e) everything we
> ruled out, (f) the candidate fixes and which the convenor leans toward, and
> (g) the EXACT experimental state of the `draw-s038-fix` worktree that MUST be
> reverted. Nothing here is omitted; trust the EVIDENCE sections, treat the
> MECHANISM section as well-supported but re-checkable.

---

## 0. TL;DR

- **Goal:** spec-035 wants a raw `--core` (minus `ClubGameSpread`) e2e solve on the
  forced-free `season_test` config to get *through presolve into search*. It does
  **not** — CP-SAT proves `INFEASIBLE: 'exactly_one: empty or all false'` ~33 s into
  presolve.
- **Root, probe-proven:** the only thing making `core − ClubGameSpread` infeasible is
  the interaction **`ClubDayParticipation` (club_day group) × `ClubVsClubStackedWeekends`
  (club_alignment group)**. Every other atom group is feasible alongside both.
- **`ClubGameSpread` is NOT involved.** It is the atom we *exclude* for Run 1 of the
  e2e; it is a different atom (per-club field-spread cap). Do not confuse it with the
  `club_alignment` stacking cluster.
- **The stacking atom is sound on its own** — with NO club-day enforcement the whole
  core reaches search. The multi-team flexibility the convenor described genuinely
  satisfies the nesting across a season.
- **The club-day is the sole trigger,** specifically `ClubDayParticipation` forcing a
  club's *entire* grade-spread onto ONE Sunday. For a club that fields 2+ teams in a
  grade (University), that one day cannot satisfy the cross-club nested-superset rule,
  by *either* resolution (derby OR split). Confirmed: Crusaders (1 team/grade) →
  feasible; University (2 teams in 4th and 6th) → infeasible.
- **No ordering key fixes it** (tested four). **No slack releases it** (no slack keys).
  Only dropping the per-week coupling (count-only) or removing the nesting restores
  feasibility — i.e. the defect is the *hard per-week cross-club implication firing on
  a club-day weekend*, not the ordering and not the per-team budgets.

---

## 1. What spec-035 is and where it sits

`docs/todo/spec-035-ultimate-e2e-core-run.md` (status `in_progress`). The spec-035 plan
end-of-line plan: a real CP-SAT solve of the production `core` constraint group
(plus always-on `symmetry_breakers`) on `config/season_test` (2026 base teams +
field availability, `forced_games == []`, `blocked_games == []`, week 1 NOT fixed,
`--workers 10`). Two runs: Run 1 = `core --exclude ClubGameSpread`, Run 2 = full
`core`. Goal: reach presolve + survive 30 min of search, and read out CP-SAT's
remaining `[Symmetry]`. Units A (launcher `scripts/run_core_e2e.py`) and B (CP-SAT
log capture + `parse_symmetry_stats` in `solver_diagnostics.py`) are merged on
`final-form`. Unit C (the actual run) is BLOCKED on this infeasibility.

History: an earlier handoff (`spec-035-e2e-infeasibility-handoff.md`) blamed
`AwayClubHomeWeekendsCount` and `ClubVsClubStackedWeekends` with a `proven during
initial copy #124504` signature. **spec-037** (AwayClubHomeWeekendsCount → derived
Sunday range) and **spec-038** (ClubVsClubStackedWeekends → four-layer model) both
landed since; they cleared those, the var count dropped 143k→~121k, and the failure
signature CHANGED to `exactly_one: empty or all false`. This handoff is the current
state after 037+038.

- `final-form` HEAD at investigation time: `2ebe31e` (spec-038 Unit B merge).
- **spec-038 is still `building`** (its DoD was "make this atom feasible on
  season_test"; the rewrite changed the failure mode but did not clear it).

---

## 2. EVIDENCE — the raw symptom

`scripts/bisect_core_feasibility.py --max-time 120 --workers 10` (this harness
ALWAYS excludes `ClubGameSpread`; extra `--exclude` names stack on top):

```
#Variables: ~121'262   (model BUILDS fine)
Starting presolve at ~1s
INFEASIBLE: 'exactly_one: empty or all false'
  - rule 'at_most_one: empty or all false' was applied 3 times.
status: INFEASIBLE        (~33 s in, before search)
```

`exactly_one: empty or all false` = presolve derived a `sum(bools)==1` (or `>=1` +
`<=1`) whose literal set is empty or all-fixed-false. It is produced DURING presolve
(the vars exist at construction — see §5), so a hard atom is fixing some required
"must play" set to all-zero.

`season_test` shape: 22 distinct Sunday weeks, 21 Friday weeks, PHL num_rounds 20.
Grades & team counts: PHL 6, 2nd 4, 3rd 8, 4th 11, 5th 9, 6th 10.

---

## 3. EVIDENCE — group bisection (each row = a real solve, workers 10)

`core − ClubGameSpread` = **8 critical_feasibility fundamentals** + **3 soft
symmetry_breakers** + **home_away_balance (2)** + **club_day (5)** + **club_alignment
(2)**.

Fundamentals (always kept): `NoDoubleBookingTeams`, `NoDoubleBookingFields`,
`EqualGamesAndBalanceMatchUps`, `PHLConcurrencyAtBroadmeadow`,
`SameGradeSameClubNoConcurrency`, `ClubNoConcurrentSlot`, `PHLAnd2ndAdjacency`,
`VenueEarliestSlotFill`.
Soft symmetry_breakers (kept, can't cause infeasibility): `SoftLexMatchupOrdering`,
`NIHCFillEFBeforeSF`, `NIHCFillWFBeforeEF`.

| Probe | Set (on top of fundamentals + symmetry) | Verdict |
|---|---|---|
| A | fundamentals + symmetry only | ✅ REACHED_SEARCH |
| B | + home_away_balance only | ✅ REACHED_SEARCH |
| C | + club_day only (all 5) | ✅ REACHED_SEARCH |
| D | + club_alignment only (Stacked Weekends+CoLocation) | ✅ REACHED_SEARCH |
| E | home_away + club_alignment (no club_day) | ✅ REACHED_SEARCH |
| F | home_away + club_day (no club_alignment) | ✅ REACHED_SEARCH |
| **G** | **club_day + club_alignment (no home_away)** | ❌ INFEASIBLE |
| **H** | **ClubVsClubStackedWeekends + CoLocation + ClubDayParticipation** | ❌ INFEASIBLE |
| **I** | **ClubVsClubStackedWeekends + ClubDayParticipation (no CoLocation)** | ❌ INFEASIBLE |

**Conclusions:**
- Every group is feasible *alone* and in every *pair* EXCEPT club_day × club_alignment.
- The minimal infeasible set is **`ClubVsClubStackedWeekends` + `ClubDayParticipation`**
  (Probe I — `CoLocation` not needed; `home_away` not involved; the other 4 club_day
  atoms not needed).
- **`ClubVsClubStackedWeekends` alone (Probe D) reaches search.** The stacking atom is
  not broken by itself.

---

## 4. EVIDENCE — Crusaders vs University isolation

Temporarily restricted `CLUB_DAYS` to one club at a time (env hack, §10) and ran
`ClubVsClubStackedWeekends + ClubDayParticipation`:

| Club day kept | Club's team structure | Verdict |
|---|---|---|
| **Crusaders only** | 1 team each in 3rd,4th,5th,6th | ✅ REACHED_SEARCH |
| **University only** | 3rd(1), 4th(2: Redhogs+Seapigs), 5th(1), 6th(2: Gentlemen+Seapigs6th) | ❌ INFEASIBLE |

**The infeasibility is specific to a club that fields 2+ teams in a grade.** Single-
team-per-grade clubs stack fine on their club day.

`season_test` `CLUB_DAYS` (inherited from `season_2026.py:183`):
```python
'Crusaders':  {'date': datetime(2026, 6, 14), 'note': 'Crusaders Club Day'}   # all 4 teams back-to-back, same field
'University': {'date': datetime(2026, 7, 26), 'note': 'University Club Day'}
```
Both dates are **Sundays** (`2026-06-14`, `2026-07-26`). Both have **no opponent**
(`opponent is None`).

---

## 5. EVIDENCE — vars exist at construction; presolve fixes them

For each club-day team, candidate X-vars on the club-day date (built via
`generate_X`, `generate_games` includes intra-club same-grade games):

```
Crusaders 3rd  174 cand vars, 7 opponents, all Sunday
Crusaders 4th  246 cand, 10 opp ; 5th 198/8 ; 6th 222/9
University 3rd 174/7 ; 4th(Redhogs) 246/10 ; 4th(Seapigs) 246/10 ;
University 5th 198/8 ; 6th(Gentlemen) 222/9 ; 6th(Seapigs6th) 222/9
```

So `ClubDayParticipation`'s `sum(team's date vars) >= 1` is NOT empty at build —
each team has 174–246 candidates. The all-false set is produced **during presolve**
by `ClubVsClubStackedWeekends` fixing those vars to 0.

---

## 6. The two atom families — what they are DESIGNED to do

### 6a. club_alignment group = the "ClubVsClubAlignment" stacking (2 atoms)

Files: `constraints/atoms/club_vs_club_stacked_weekends.py`,
`constraints/atoms/club_vs_club_stacked_co_location.py`,
shared math `constraints/atoms/_club_vs_club_stacked_shared.py`.
spec: `docs/todo/spec-038-clubvsclub-stacked-team-pair-granularity.md` (status
`building`). Replaces the legacy monolithic `ClubVsClubAlignment` (the canonical
`ClubVsClubAlignment` registry entry is retained only as a tester/name anchor).

**Intent (verbatim from spec-038 "Why"):** for each unordered club pair `(A,B)` and
each grade both field:
- each grade meets the other club some number of times per season (derived);
- **stack those meetings on the same aligned weekends across grades** so a family
  with kids in 4ths and 5ths sees A-vs-B on the same Sunday for both kids;
- **nested-superset:** *the smaller meeting-count grades only play on weekends a
  larger meeting-count grade also plays.*
- Caveat 1 (PHL): forced-Friday meetings consume the matchup budget but can't satisfy
  Sunday stacking, so subtract them from PHL's Sunday budget.
- Caveat 2 (multi-team): when both clubs field 2 teams in a grade, **both** cross-club
  games play on each aligned weekend; when one side has 2 and the other 1, the model
  is **free to choose** which of the two teams plays the single opposing team each
  aligned weekend (the asymmetric case naturally yields more aligned weekends).

`ClubVsClubStackedWeekends.apply` (current spec-038 four-layer model):

- **Layer 1** — `team_pair_play[(tp, w)] = OR(Sunday X-vars for team-pair tp in week
  w)` via `get_or_create_bool` (`AddMaxEquality`, bidirectional). Fixed to 0 if no
  vars. (`tp` = an unordered cross-club team-pair.)
- **Layer 2 (per-team-pair budget, HARD)** — `tp_min <= sum_w team_pair_play[tp] <=
  tp_max` where `(tp_min,tp_max) = (base, base+1)` (`base = per_matchup = R//(T-1)`
  even, `R//T` odd). **Largely REDUNDANT** with `EqualGamesAndBalanceMatchUps`, which
  already pins every team-pair to `[base, base+1]` across the season.
- **Layer 3** — `play_pg[(pair,grade,w)] = OR over team_pair_play[tp,w] for tp in that
  pair-grade` (registered under `STACK_PLAY_PREFIX`). The per-pair-grade "≥1 game this
  weekend" indicator the nesting & co-location consume.
- **Layer 4 (per-weekend cardinality, HARD)** — `sum(team_pair_play in pair-grade) ==
  min(a,b)` when `play_pg=1`, `== 0` when `play_pg=0` (two `OnlyEnforceIf` branches).
  This is Caveat 2: on an aligned weekend the full matching of `min(a,b)` games plays.
  **Needs (club,team) granularity (min(a,b)).**
- **Layer 5 (per-pair-grade total, HARD)** — `min_budget <= sum_w play_pg <=
  max_budget` where `(min_budget,max_budget) = (max(a,b)*base, max(a,b)*(base+1))`
  (the aligned-WEEKEND count range).
- **Layer 6 (cross-grade nested-superset chain, HARD)** — grades sorted, then for each
  consecutive `(hi, lo)` and each Sunday week: `play_pg[lo, w] <= play_pg[hi, w]`
  (lo plays ⟹ hi plays). **This is the offending layer.** Current sort key is
  `-spec[2]` = `-max_budget` = `-max(a,b)*(base+1)`.

`ClubVsClubStackedCoLocation`: on weekends where ≥2 grades of a pair play, all those
games go on one field, contiguous slots, gated by `stack_active = play[second-ranked
grade]`. **Not required for the infeasibility** (Probe I excludes it).

### 6b. club_day group (5 atoms) — the "ClubDay" festival

Files under `constraints/atoms/club_day_*.py`, shared `_club_day_shared.py`. Config
key `club_days` (`config/season_2026.py:183`; `season_test` inherits it). Mirrors
the original monolithic `original.py:ClubDayConstraint`. A club picks a date (optional
opponent). The atoms:

- **`ClubDayParticipation`** — every team of the club plays ≥1 game on that date:
  `model.Add(sum(team's date vars) >= 1)` per team. **(This is the trigger.)**
- **`ClubDaySameField`** — all the club's club-day games on ONE field
  (`sum(field_used indicators) == 1`). Guards `len(field_vars) <= 1: continue`.
- **`ClubDayContiguousSlots`** — the club-day games occupy contiguous slots (no gaps).
- **`ClubDayOpponentMatchup`** — IF an opponent is named, force cross-club matchups per
  grade (`sum(cross_vars) >= min(host_grade_count, opp_grade_count)`). No-op when
  `opponent is None` (the season_test case).
- **`ClubDayIntraClubMatchup`** — for grades where the club has 2+ teams AND there is
  no opponent for that grade, force **intra-club derbies**: `sum(intra_vars) >=
  len(host_grade_teams)//2`. So University's 4th (Redhogs+Seapigs) and 6th
  (Gentlemen+Seapigs6th) play **each other** on the club day.

**Net design:** a club's whole roster plays one Sunday, on one field, back-to-back —
a festival. Multi-team grades derby; single-team grades play an external club (or the
named opponent, across grades, when one is set).

---

## 7. MECHANISM — why ClubDay × stacking is jointly infeasible (multi-team only)

Established facts feeding the mechanism:
1. **Stacking alone is feasible** (Probe D). The per-pair nested-superset is fine over
   a full season because multi-team grades give the solver freedom over *which* team
   and *which* weekend.
2. **The trigger is `ClubDayParticipation`** — it crushes a club's *entire* grade-
   spread onto ONE Sunday W. (Probe H: Participation + StackedWeekends alone is
   infeasible; the other club_day atoms are not required.)
3. A grade is **active only on the weekends it actually plays** (`play_pg` is a per-
   weekend boolean, true on ~weekend-count of the 22 Sundays), NOT every weekend.

On University's club-day Sunday W, all 6 University teams must play. University fields
2 teams in 4th and 2 in 6th. There are exactly two ways those multi-team grades can
discharge participation on W, and **both** collide with Layer 6:

- **Derby (what `ClubDayIntraClubMatchup` forces):** Redhogs vs Seapigs (4th),
  Gentlemen vs Seapigs6th (6th) — *intra*-club, so for every external club B,
  `play_pg[(Uni,B), 4th, W] = 0` and `play_pg[(Uni,B), 6th, W] = 0`. University's
  single-team grades 3rd & 5th have NO derby partner, so participation forces them to
  play a cross-club game on W. Take 5th vs some B₅. Layer 6 says `play_pg[5th] <=
  play_pg[<a higher-meet grade of (Uni,B₅)>]`. Every external B₅ shares 4th (and
  usually 3rd/6th) with University, and those grades are at 0 on W (derby) → the
  implication forces `play_pg[(Uni,B₅), 5th, W] = 0` → University cannot play 5th
  against ANY B₅ on W → but participation says it must → `exactly_one: empty/all-false`.
  (No external B₅ shares *only* 5th with University — checked: University's 5th
  opponents are Colts, Crusaders, Maitland, Norths, Tigers, Wests, all of which also
  share 3rd and/or 4th.)

- **Split (allowed when IntraClubMatchup is OFF, as in the minimal probes):** the two
  4th teams play two different external clubs B and B′. Then Layer 6 drags University's
  *single* 3rd-grade team toward both pairs on W (`4th-vs-B ⟹ 3rd-vs-B`, `4th-vs-B′ ⟹
  3rd-vs-B′`) → one team, two opponents, one day → impossible.

Either resolution is infeasible, which is why Probe H (derby *optional*) is already
infeasible — the solver has no third option. The crux: **the per-week cross-club
nested-superset keeps demanding "the higher-meet grade plays this *specific external
opponent* this weekend" on the one weekend the club is deliberately playing
everyone/derbying on a single field.** The alignment the nesting is trying to force is
already delivered by the festival itself (one day, one field) — the cross-club form of
it is both redundant and unsatisfiable there.

Why single-team clubs (Crusaders) are fine: with 1 team per grade, Crusaders can play
ONE opponent that fields all four grades across all four games on W (a fully-stacked
festival), so the nesting is trivially satisfied. University cannot, because no single
opponent fields 2 teams in *both* 4th and 6th to absorb University's doubles (Tigers
has 2 in 6th but 1 in 4th; Wests 2 in 4th but 1 in 6th; etc.).

---

## 8. EVIDENCE — what we RULED OUT (ordering, keys, slack)

**Ordering (Layer 6 sort key) — tested all four, University still INFEASIBLE for the
team-count-scaled ones; none fix it:**

| Layer-6 sort key | meaning | University feasible? |
|---|---|---|
| `max_budget = max(a,b)*(base+1)` | current code | ❌ |
| `min_budget = max(a,b)*base` | aligned-weekend count | ❌ |
| `base = per_matchup` (`(club,team)` key) | per-team-pair meets | ❌ |
| `a*b*base` (`(club)` aggregate, `per_pair_grade_meeting_counts`) | total cross-club games | ❌ (minimal AND derbies-forced) |
| count-only nesting (`sum_w play[lo] <= sum_w play[hi]`, drop per-week) | — | ✅ REACHED_SEARCH |
| remove Layer 6 entirely | — | ✅ REACHED_SEARCH (full core too) |

So the **ordering key is genuinely on the wrong quantity** (the current code sorts by
`max_budget`, a team-count-scaled weekend budget, not by meeting count — a real
correctness smell the convenor flagged), **but fixing the ordering does not fix the
feasibility.** The feasibility blocker is the *per-week strict implication itself*
firing on the club-day weekend.

**The two keys (resolved):**
- **`(club)` aggregate `a*b*base`** = total cross-club games in the grade between the
  clubs. Correct quantity for **superset ORDERING** (multi-team grades meet more →
  should be supersets). Grade seniority is irrelevant.
- **`(club,team)` `base`** = meets per unique team-pair. Genuinely needed **only for
  Layer 4's matching cardinality (`min(a,b)`)** — Caveat 2, "both games on the aligned
  weekend" / "choose which team". It is NOT needed for ordering, and Layer 2's per-tp
  budget that also uses it is largely redundant with `EqualGamesAndBalanceMatchUps`.

**Slack:** `ClubVsClubStackedWeekends`, `ClubVsClubStackedCoLocation`, and
`ClubDayParticipation` all have `slack_key=None`. The documented "core is infeasible at
slack 0, slack releases it" note (dependency tree, attributed to `ClubNoConcurrentSlot`)
is a DIFFERENT issue and does NOT apply here — `ClubNoConcurrentSlot` is a fundamental
kept in all the *feasible* probes. Slack cannot release this.

**Channeling:** `get_or_create_bool` uses `AddMaxEquality` (bidirectional OR) — correct,
no defect. `play_pg = OR(team_pair_play)`; `team_pair_play = OR(Sunday X-vars)`. Fine.

---

## 9. The fix space (no code committed yet — convenor decision pending)

The defect is the **hard per-week cross-club nested-superset (Layer 6) firing on a
club-day weekend**, where the club-day festival already self-aligns. Candidate fixes:

1. **(i) Make the "superset grade is active this weekend" indicator count ANY game the
   club plays in that grade that weekend — including an intra-club derby — not only the
   specific cross-club pair.** Then on a club-day weekend University's 4th *is* active
   (the derby), the family-alignment intent is satisfied, and the nesting stops
   demanding 4th-vs-the-single-team-grade's-opponent. This is a granularity change in
   what the nesting nests on (per-(club,grade,weekend) "club plays this grade this
   weekend" rather than per-(club-pair,grade,weekend) cross-club). **NOTE the open
   question the convenor raised:** what does "active this weekend" mean precisely — it
   is per-weekend (true only on weekends the grade actually plays), not always-true.

2. **(ii) Exempt club-day weeks (for the host club) from the cross-club nesting.** The
   club-day atoms (`SameField`, `ContiguousSlots`, participation) already co-locate the
   whole roster that day, so the cross-club superset requirement is redundant there.
   Narrowest change; "the club day is its own alignment."

3. **(iii) Reformulate Layer 6 as a CHOOSE / non-strict relationship** (the convenor's
   "stop forcing the higher grades to be a strict superset" language) — the feasible
   `count-only` test approximates this but loses same-weekend alignment, so a faithful
   version needs care.

4. **(iv) Make Layer 6 soft** — REJECTED by the convenor (they want a correct hard rule,
   not a preference). Recorded only so it isn't re-proposed.

**Independently of the feasibility fix, the ordering key SHOULD be corrected** from
`max_budget` to the `(club)` aggregate `a*b*base` (meeting count) — that is a real
correctness issue the convenor identified, even though it does not by itself fix
feasibility. Keep Layer 4's `min(a,b)` matching (the `(club,team)` key) as-is.

**The convenor's current framing (2026-05-29):** the ClubDay × ClubVsClubAlignment
conflict "is not real" — i.e. it is an artefact of the cross-club nesting wrongly
asserting itself on a self-aligned festival day, and the two should coexist. Direction
between (i) and (ii) not yet finalised; (i) ("the derby IS the grade being active")
is the one most aligned with the convenor's last messages.

---

## 10. EXPERIMENTAL STATE — `draw-s038-fix` worktree MUST be reverted

A throwaway worktree was created for the investigation:
`C:\Users\c3205\Documents\Code\python\draw-s038-fix` on branch
`spec038-clubday-stacking-fix` (off `final-form` `2ebe31e`). It contains **temporary
instrumentation that is NOT a fix and must be discarded** before any real change:

- `constraints/atoms/club_vs_club_stacked_weekends.py`:
  - added `import os` + `_SKIP_LAYERS = {…CVC_SKIP_LAYERS…}` module global;
  - env-gated `if N not in _SKIP_LAYERS:` wrappers around Layers 2, 4, 5, 6;
  - `CVC_NEST_MODE=count` branch in Layer 6 (count-only nesting);
  - Layer 6 sort key was edited several times (currently `-grade_meetings_total[grade]`,
    i.e. the `(club)` aggregate experiment). **Original is `key=lambda spec: (-spec[2],
    spec[0])`** (max_budget).
- `config/season_2026.py`: added `_os_clubday_probe` import + `_ALL_CLUB_DAYS` +
  `CLUB_DAYS_ONLY` env hack to restrict club days for isolation probes. **Original
  `CLUB_DAYS` block must be restored.**

Nothing was committed. Either `git checkout -- .` in that worktree and reuse it for the
real fix, or `git worktree remove` it and branch fresh. The harness
(`scripts/bisect_core_feasibility.py`) and launcher (`scripts/run_core_e2e.py`) on
`final-form` are clean and were not changed.

Also note: earlier this session the spec-035 plan + the original infeasibility handoff
were edited on `final-form` (the §11 re-bisection note, and the bye_spacing→spacing
follow-on phase added per convenor request). Those edits are uncommitted on
`draw-final-form`. The follow-on phase (Units D/E: add `bye_spacing`, then swap to
`spacing`) is `drafting` and blocked behind this fix.

---

## 11. REPRODUCE / CONTINUE

Venv lives ONLY in `C:\Users\c3205\Documents\Code\python\draw\.venv`. Scripts resolve
repo root from `__file__`, so run that python against whichever worktree's code you
want to test.

```powershell
$PY = "C:\Users\c3205\Documents\Code\python\draw\.venv\Scripts\python.exe"

# Reproduce the infeasibility (full core - ClubGameSpread):
& $PY <worktree>\scripts\bisect_core_feasibility.py --max-time 60 --workers 10

# Minimal infeasible pair (StackedWeekends + ClubDayParticipation):
& $PY <worktree>\scripts\bisect_core_feasibility.py --max-time 50 --workers 10 `
  --exclude AwayClubHomeWeekendsCount AwayClubPerOpponentAndAggregateHomeBalance `
  ClubDayContiguousSlots ClubDayIntraClubMatchup ClubDayOpponentMatchup `
  ClubDaySameField ClubVsClubStackedCoLocation
# -> INFEASIBLE_PRESOLVE ('exactly_one: empty or all false')

# Prove stacking alone is fine (no club-day enforcement) -> REACHED_SEARCH:
& $PY <worktree>\scripts\bisect_core_feasibility.py --max-time 60 --workers 10 `
  --exclude AwayClubHomeWeekendsCount AwayClubPerOpponentAndAggregateHomeBalance `
  ClubDayContiguousSlots ClubDayIntraClubMatchup ClubDayOpponentMatchup `
  ClubDayParticipation ClubDaySameField

# Per-pair grade specs (a, b, base, meetings=a*b*base, weekends=max(a,b)*base):
& $PY -c "from main_staged import load_data; from utils import generate_games; from constraints.atoms._club_vs_club_stacked_shared import *; d=load_data('test'); d['games']=generate_games(d['teams']); [print(p,[(g,team_pair_counts(d,p,g),_per_matchup_for_grade(d,g),per_pair_grade_meeting_counts(d,p)[g],per_pair_grade_aligned_weekends(d,p,g)) for g in per_pair_grade_meeting_counts(d,p)]) for p in enumerate_club_pairs(d) if 'University' in p]"
```

Verdict classifier lives in `bisect_core_feasibility.py::_classify`:
`INFEASIBLE_PRESOLVE` / `REACHED_SEARCH` / `FEASIBLE/OPTIMAL` / `UNKNOWN`.

---

## 12. OPEN QUESTIONS for the convenor

1. Pick the fix direction: (i) derby/any-game counts as the grade being "active this
   weekend" for the nesting, or (ii) exempt club-day weekends from the cross-club
   nesting. (Convenor leaning toward "the conflict isn't real / the derby IS the
   alignment" ⇒ (i).)
2. Confirm the ordering correctness fix (`max_budget` → `(club)` aggregate `a*b*base`)
   should be made alongside — it's a separate, real correctness issue.
3. Is `ClubVsClubStackedWeekends` Layer 2's per-team-pair budget worth keeping given it
   duplicates `EqualGamesAndBalanceMatchUps`? (Cleanup candidate; not load-bearing for
   this bug.)
4. This is constraint-SEMANTIC work owned by spec-038 (still `building`), NOT spec-035
   construction work — the fix should land as completing spec-038, then spec-035 Unit C
   (the 30-min runs + symmetry readout) resumes.
