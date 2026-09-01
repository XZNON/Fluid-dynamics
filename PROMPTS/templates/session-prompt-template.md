# Template — next-session prompt

`/checkpoint` fills this in and writes it to `PROMPTS/NNN-<slug>.md`. It must be **paste-ready**: a
cold session with zero prior context reads it and can work. Inline everything that matters — a link
the session might not follow is not context.

Replace every `<...>`. Delete no section; write "none" where a section is empty.

---

```markdown
# Session <N> — <T2XX>: <Task title>

## What this project is

<One or two lines: a validated 2D D2Q9 lattice-Boltzmann simulator on NumPy and Warp, under a
product layer that takes a picture and three physical numbers. The live spec is `DOCS/IDEA4.md`.
The solver is not the product — see `idea.md` § Risks, "The trap".>

## Read these first, in this order

1. `CLAUDE.md` — hard constraints, session protocol, conventions.
2. `DOCS/STATE3.md` — **in full**. Snapshot, blockers, open questions, environment, decisions, and
   the last few session-log entries.
3. `DOCS/TASKS3.md` § <T2XX> — the task contract, in full.
4. `DOCS/IDEA4.md` § <the sections that task cites>.
5. `DOCS/PLAN3.md` § Session map and § Risks — <one line on where this task sits>.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** worked <T2XX-1> and <landed / half-landed> <what>.
- **Rung status:** R1-R4 <state> · A-E <state> · F <state> · G <state> · H <state> · I <state> · J <state>
- **Milestone reached:** <M0/M1/... or none>
- **Completed tasks:** <ids>

## Your task this session

**<T2XX> — <Title>.** One task, this session only.

Run this first:

    /start-task <T2XX>

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

### Acceptance criteria (restated in full — this is what marks the task done)

<Verbatim copy of the criteria checklist from DOCS/TASKS3.md. Every box. Not summarised.>

### Constraints that bite on this task

<By number and content, from CLAUDE.md. E.g. "Constraint 4 — f is (9, ny, nx), float32; every later
module inherits this.">

### Blockers, open questions and decisions that affect you

<Quote the relevant Q-ids and D-ids from DOCS/STATE3.md with their content. "None" if none.>

### Before you start

<Packages to install into myenv, files that must exist, rungs that must be green. "Nothing" if
nothing.>

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it — do not expand this
one. If it's listed under `DOCS/IDEA4.md` § Deliberately deferred, the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate` for every rung at or below this task; confirm nothing regressed.
3. **Run `/checkpoint`** — updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
```
