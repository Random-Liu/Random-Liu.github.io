# Agent Instructions

The shared agent instructions for this repository live in `GEMINI.md`. Claude Code imports them here:

@GEMINI.md

## Claude-Specific Overrides

These overrides take precedence over `GEMINI.md` when this file is in effect.

### Git Commit Convention
- Use the following co-author signature instead of the Gemini one, separated by a blank line from the main message:
  ```
  Co-authored-by: Claude <noreply@anthropic.com>
  ```

### Note Authoring Workflow

For long-form notes built from external discussions (e.g., a Gemini conversation transcript):

1. **Structure**: User provides a source (URL or pasted text) plus a target topic / filename. Fetch the source, propose a chapter outline in chat, iterate until the user approves.
2. **Draft**: Write `notes/<topic>.cn.md` first (Chinese is the source per `GEMINI.md` §2), then `notes/<topic>.md` as a 1:1 English mirror. Use a feature branch named `claude/notes-<topic>`, commit per section. Open a PR via `mcp__github__create_pull_request`.
3. **Review**: After opening the PR, call `subscribe_pr_activity` for it.
    - Clear, scoped edit requests (wording, formatting, GitHub suggestion blocks): apply directly with a new commit and reply on the thread.
    - Ambiguous comments, structural changes, or scope changes: ask in this chat session before editing.
    - Net-new technical sub-discussions happen in this chat session, not in PR threads.
4. **Publish**: GitHub does not allow self-approval, so do not wait for a formal approval state. Two paths trigger publish:
    - User merges the PR directly in GitHub: just call `unsubscribe_pr_activity`. No further action.
    - User leaves a `lgtm` / `approve` / `merge` comment on the PR: fast-forward merge `main` to the PR head, push, then call `unsubscribe_pr_activity`. CI runs `mkdocs gh-deploy` automatically.

Mobile review preference: when feasible, prefer GitHub app's line comments and suggestion blocks on the PR over describing changes in chat — they map directly onto edits.
