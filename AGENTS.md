# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

## 1. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- Write abstractions only when a second caller exists.
- Build exactly the flexibility and configurability that was requested.
- Handle the error paths that can actually occur.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

Match the length of written documents to what the task needs: cover the substance, and leave out filler sections, redundant summaries, and boilerplate.

## 2. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Leave adjacent code, comments, and formatting as they are.
- Refactor only what the task requires.
- If you notice unrelated dead code, mention it and leave it in place.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Leave pre-existing dead code alone unless asked.

The test: Every changed line should trace directly to the user's request.

## 3. Goal-Driven Execution

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

## 4. Responses

**Reply to the user in natural, polite Korean (해요체/합니다체).**

- Applies to conversational replies, not to code, identifiers, or commit messages.
- Keep technical terms in their usual form, as a Korean developer would say them aloud.
- Shorten a reply by carrying fewer items, not by compressing sentences.
