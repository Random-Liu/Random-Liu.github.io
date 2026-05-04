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
