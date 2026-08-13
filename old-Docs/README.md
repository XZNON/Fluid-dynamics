# old-Docs — Phase 0 archive

Phase 0 closed at **M4** on 2026-08-13 (T001 → T011, all four validation rungs green). These three
documents drove it and are **frozen**: read for history, never edited, never condensed (**D-041**,
`DOCS/STATE2.md` § Decisions).

| File | What it is |
|---|---|
| `STATE1.md` | Phase 0's live state file — snapshot, environment, performance baseline, **§ Decisions D-005 … D-040**, and eleven session-log entries |
| `TASKS1.md` | Phase 0's task contracts, T001 → T011, each with its acceptance criteria and the outcome that closed it |
| `PLAN1.md` | Phase 0's plan — dependency graph, session map, milestone gates, risks |

**§ Decisions D-005 … D-040 in `STATE1.md` are still in force.** Decision numbering continues
unbroken in `DOCS/STATE2.md` at D-041, so there is exactly one D-029 in the project. Phase 1 code and
docs cite these by number; read the entry a task names rather than the whole file.

## What is not here

- **`DOCS/IDEA2.md`** — the Phase 0 spec stayed in `DOCS/`. Roughly a hundred docstrings across
  `lbm/`, `validate/`, `tests/` and `bench.py` cite it by path, per `CLAUDE.md`'s convention that
  the reasoning is one hop away, and moving it would have made every one of those wrong.
- **`DOCS/bench_baseline.json`** — read by `bench.py`; it is data, not a document.
- **`DOCS/ISSUES.jsonl`** — the live issue queue, read by `tools/issues.py`.
- **The Phase 0 session prompts** (`PROMPTS/001-…` … `011-…`) — deleted in session 12; recoverable
  from git history. `PROMPTS/012-phase1-planning.md` onward are kept.

## Where the live documents are

`DOCS/IDEA3.md` (Phase 1 spec) · `DOCS/PLAN2.md` (plan) · `DOCS/TASKS2.md` (backlog) ·
`DOCS/STATE2.md` (**live state — read this one first**).
