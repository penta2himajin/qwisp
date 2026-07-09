---
name: Session Handoff
about: Long-running workstream that spans multiple Claude sessions
title: '[handoff] '
labels: ['session-handoff']
---

<!--
This template captures the current state of a workstream so a future session
can resume without re-discovering context.

Rules:
- Body holds CURRENT state only. Overwrite on each session.
- History goes in a pinned comment titled "Session log", append-only.
- Permanent decisions migrate to docs/decisions/*.md (ADR) once referenced
  in 2+ sessions.
-->

## Snapshot

- Branch: `<branch-name>`
- Last commit: `<short-sha>` @ YYYY-MM-DD
- Working tree: clean / dirty (`<file1>`, `<file2>`)
- Last session: YYYY-MM-DD HH:MM <TZ>

## Status

<!-- one of: in-progress / blocked / ready-for-review -->

## Next action

<!-- One sentence. Verb + object + expected outcome.
The receiving session reads this aloud and confirms with the user before executing. -->

## Verification

<!-- How to know the next action is complete. Command + expected output + fallback check.
Example:
- `pytest tests/test_no_leak.py -v` returns green
- If it fails, check LEAK_FEATURES registration
-->

## Context pointers

<!-- Pointers, not content. Receiving session reads only what's needed.
Use commit SHAs and line ranges where possible.
- Code: `path/to/file.py` L42-78 @ <sha>
- Doc: `docs/<...>.md` §<section>
- PR: #<n> (related)
-->

## Decisions made

<!-- Settled within this workstream; do not re-litigate.
Migrate to docs/decisions/*.md (ADR) once referenced in 2+ sessions. -->

## Failed approaches

<!-- Mandatory if anything was tried and abandoned.
Format per entry:
- Approach
- Why it failed (verbatim error message if any)
- Why the current approach is preferred -->

## Open questions for user

<!-- Items needing user input before proceeding. Delete section if empty. -->
