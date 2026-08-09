# STATE1.md — live project state

**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | Phase 0 — D2Q9 LBM in NumPy (`DOCS/IDEA2.md`) |
| **Current task** | `T001` |
| **Task status** | `not_started` |
| **Completed tasks** | none |
| **Milestone reached** | none (next: M1 at T002) |
| **Rung status** | R1 ⬜ · R2 ⬜ · R3 ⬜ · R4 ⬜ |
| **Last updated** | 2026-08-09 — scaffold session |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

None.

## Open questions

- **Q-001** — Wall-offset convention for bounce-back: does the wall sit on the last fluid node or
  halfway between nodes? Affects `H` in the Poiseuille analytic solution and `L` in the cavity
  benchmark. **Decide in T002 and record as a decision below.** Blocks nothing yet; will silently
  corrupt Rung 2 if T002 leaves it undocumented.
- **Q-002** — SVG rasterisation dependency (T009) not chosen. Not blocking; PNG is what M4 needs.

## Environment

Project venv: `myenv/` (gitignored). Python 3.11.15.

| Package | Version | Added by |
|---|---|---|
| numpy | 2.4.6 | pre-existing |
| matplotlib | 3.11.1 | pre-existing |
| pillow | 12.3.0 | pre-existing |

Not yet installed, needed later: `pytest` (T001), `pygame` (T007), `imageio[ffmpeg]` (T011).

Install with `myenv/Scripts/pip.exe install <pkg>` and **add a row above in the same session.**

## Performance baseline

Not yet measured. Recorded in T010.

| Grid | Cells | Baseline steps/s | Post-optimisation | Budget floor |
|---|---|---|---|---|
| 400×100 | 40k | — | — | ≥400 |
| 800×200 | 160k | — | — | ≥120 |
| 2000×500 | 1M | — | — | ≥15 |

## Decisions

Anything chosen that wasn't already specified in `DOCS/IDEA2.md` or `CLAUDE.md`. Append; never edit
a past entry — supersede it with a new one that says so.

| ID | Date | Decision | Why |
|---|---|---|---|
| D-001 | 2026-08-09 | Docs live in `DOCS/`, next-session prompts in `PROMPTS/`, agentic config in `.claude/`. `idea2.md` moved to `DOCS/IDEA2.md`. | Keeps the repo root for code and product-level `idea.md` / `README.md`. |
| D-002 | 2026-08-09 | One task per session, enforced by `/start-task` + `/checkpoint`. | Small context per session; every session boundary is a validated state. |
| D-003 | 2026-08-09 | `myenv/` is the canonical interpreter; commands in docs use `myenv/Scripts/python.exe`. | A global-python invocation would silently miss project deps on Windows. |
| D-004 | 2026-08-09 | `Navier-Fluid-Equation/` treated as read-only prior work; only its polygon-vertex logic is reused, in T004. | It is potential-flow, a different method. Mixing the two codebases would confuse both. |

## Session log

Append one entry per session. Newest at the bottom.

### 2026-08-09 — Session 0: scaffold

**Task worked:** none (setup)

**Done**
- Read `DOCS/IDEA2.md` in full; confirmed environment (`myenv`, numpy 2.4.6, matplotlib 3.11.1).
- Created `CLAUDE.md` (12 hard constraints, session protocol, conventions, module map).
- Created `DOCS/PLAN1.md` (11 tasks, dependency graph, session map, milestone gates, risks).
- Created `DOCS/TASKS1.md` (full contract per task: goal, depends-on, I/O, acceptance criteria,
  constraints that bite, notes).
- Created `DOCS/STATE1.md` (this file) and `PROMPTS/templates/session-prompt-template.md`.
- Created `.claude/`: `commands/start-task.md`, `checkpoint.md`, `new-task.md`, `validate.md`, plus
  `settings.json`. Adapted from a prior project's command set (`.claude-blueprint/`, since deleted —
  do not look for it).
- Moved `idea2.md` → `DOCS/IDEA2.md`.
- Generated `PROMPTS/001-t001-core-equilibrium.md`.

**Not done / deferred**
- No solver code at all. That is intentional — T001 is session 1.

**Decisions made**
- D-001 through D-004 above.

**Blockers**
- None.

**Next**
- Paste `PROMPTS/001-t001-core-equilibrium.md` into a fresh session. It runs `/start-task T001`.
