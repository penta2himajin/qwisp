# Session Handoff Protocol

Protocol for handing off long-running work between Claude sessions via GitHub issues.

## Premises

- **Scope**: long-running workstreams that span multiple sessions and possibly multiple interfaces (claude.ai ↔ Claude Code).
- **Medium**: GitHub issue with `session-handoff` label.
- **Granularity**: one issue per workstream. Not one per session.
- **Avoid destructive history loss**: the issue body holds current state and is overwritten on each session. History accumulates in a pinned comment titled "Session log", append-only.

## Sender procedure (end of session)

1. Commit and push outstanding work. Make the working tree clean, or record the dirty state explicitly.
2. Overwrite the issue body using `.github/ISSUE_TEMPLATE/handoff.md` format.
   - **Snapshot**: branch + latest commit SHA + timestamp.
   - **Next action**: granular enough that the next session can begin immediately.
   - **Failed approaches**: record anything tried and abandoned. Skipping this is the most common cause of duplicated effort.
3. Append this session's outcome to the pinned "Session log" comment as a single dated paragraph.
4. Reference the issue number in the final message.

## Receiver procedure (start of session)

1. List open issues labelled `session-handoff` and identify the relevant workstream.
2. Read the issue body. Internalise Snapshot, Next action, Failed approaches.
3. Verify the Snapshot matches reality.
   - Compare `git log -1 origin/<branch>` to the recorded commit SHA.
   - Check working tree state if claimed clean.
4. If drift exists, report to the user before acting (e.g., "issue records `<sha-A>`, but origin is now `<sha-B>` with these changes: ...") and request guidance.
5. If no drift, read the **Next action** aloud to the user and confirm before executing.
6. Execute. Verify against the **Verification** field. Then follow the sender procedure to update the issue.

## Promotion rule

When a Decision in the issue is referenced in 2+ later sessions, promote it to an ADR under `docs/decisions/*.md`. Replace the entry in the issue body with a link to the ADR. This keeps the issue body bounded and prevents knowledge from being lost when the issue is closed.

## SSOT precedence

When sources conflict:

1. `CLAUDE.md` (project-wide invariants)
2. `.claude/rules/` (path-scoped)
3. ADR in `docs/decisions/` (settled judgements)
4. Handoff issue body (current workstream state)

The handoff issue is the most volatile. If it conflicts with `CLAUDE.md`, the latter wins.

## Anti-patterns

- Treating the body as a chat log. Use comments — especially the pinned "Session log" — for history.
- Inlining large logs or code dumps. Use file path + SHA + line-range pointers instead.
- Ending a session with no concrete Next action.
- Updating the title without updating the body.
- Closing the issue without migrating Decisions to an ADR — knowledge is lost when the closed issue is no longer surfaced.
- Concurrent edits from claude.ai and Claude Code on the same issue. If both interfaces are active, designate one as writer and the other as reader.

## Parallel workstreams

Multiple concurrent issues are fine. Express dependencies via GitHub issue refs (`#NN`). If the dependency graph becomes complex, evaluate dedicated trackers (e.g., `gastownhall/beads`) — but for solo-plus-AI workflows, GitHub issues at this granularity are usually sufficient.
