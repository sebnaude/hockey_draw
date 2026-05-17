# TODO — Goals + Implementation Plans

**Audience:** Anyone (human or AI) about to implement system / engineering work.

**Purpose:** Living catalogue of *what we want to build* (`GOALS.md`) plus *specific implementation plans* for each piece of work (`spec-*.md`), with status tracking so multiple agents can pick up units in parallel without colliding.

## Structure

```
todo/
  README.md           — this file
  GOALS.md            — long-horizon product + engineering goals + specifications
  spec-*.md           — individual implementation plans, one per independent unit
  done/               — completed plans, kept for historical reference
```

## Plan status header (the /basic methodology)

Every `spec-*.md` carries a status header at the top:

```html
<!-- status: not_ready | ready | in_progress | done -->
<!-- owner: session=<session-id> claimed=<ISO-8601> -->
```

- `not_ready` — author still expanding it. Other agents must NOT pick it up.
- `ready` — fully scoped, free for any agent to claim.
- `in_progress` — claimed; the `owner` line names who's working it.
- `done` — completed; file moves to `done/`.

When an agent picks up a `ready` plan it flips status to `in_progress` and stamps the owner line. When the plan is fully delivered (all units shipped, tests passing, docs updated, branch merged + pushed + worktree removed — see `/basic` skill), the file is moved to `done/`.

## Definition of Done

Each plan declares its DoD up front — concrete, testable, observable. The Sonnet review agent checks the implementation against the DoD. No criterion met → unit not done.

## Doc registry per plan

Each plan lists *every doc* that needs updating once the work lands. Documentation updates are part of the unit, not a follow-up. If a doc isn't on the list at plan time, it's because the author confirmed it doesn't need changing — never because they forgot.

## Hard rule: no deferred work

A plan describes what *this* execution delivers — full stop. If during implementation more work is discovered:

- **Required for the plan's functional outcome?** Do it now, update plan + DoD.
- **Not required?** Spawn a *new* plan file. Do NOT leave a "TODO: also do X" line.

The principle: when a plan finishes, nothing is left behind in it.

## Picking a plan to work on

1. Open this directory and list `spec-*.md` files.
2. Skip anything with status `not_ready`, `in_progress`, or `done`.
3. Among `ready` files, check the dependency line in each plan's header — pick one whose deps are all `done`.
4. Flip status to `in_progress`, stamp owner.
5. Load the `/basic` skill and follow the methodology.

## What does NOT live here

- Per-season convenor / operational TODOs → `../seasonal/{year}/operational_TODO.md`
- Plain-English rule docs → `../operator-human/`
- Engineering reference docs → `../system/`
