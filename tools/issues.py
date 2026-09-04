"""Local issue queue with an explicit push to GitHub.

Problems found while testing or running a rung get *queued* here, never filed
automatically. `sync` is the only thing that talks to GitHub, and it shells out
to the `gh` CLI so no token handling lives in this repo.

Design notes
------------
* The queue is ``DOCS/ISSUES.jsonl`` — one JSON object per line, committed, so
  the backlog survives a machine and is diffable in review.
* Every entry carries a **fingerprint** id, ``sha1(source|location|title)``.
  Re-running a failing rung bumps ``count`` and ``last_seen`` on the existing
  entry instead of appending a duplicate. This is what makes automatic capture
  from pytest safe.
* Nothing here imports numpy or touches ``lbm/`` — it is stdlib only and is not
  part of the solver.

Usage
-----
    python -m tools.issues add --title "..." [--body ...] [--label bug]
    python -m tools.issues list [--status open|pushed|dropped|all]
    python -m tools.issues show <id>
    python -m tools.issues drop <id> [--reason ...]
    python -m tools.issues sync [--dry-run] [--limit N] [--yes]
    python -m tools.issues capture --source validate -- <command ...>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO_ROOT / "DOCS" / "ISSUES.jsonl"

STATUS_OPEN = "open"
STATUS_PUSHED = "pushed"
STATUS_DROPPED = "dropped"

#: How much captured output to keep in an issue body. Long tracebacks are the
#: whole point of capture, but a 10k-line pytest log is not an issue body.
MAX_BODY_CHARS = 6000


# --------------------------------------------------------------------------
# queue file
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise(text: str) -> str:
    """Collapse whitespace and strip run-specific noise for fingerprinting.

    Numbers, hex addresses and absolute paths vary between runs of the same
    failure; folding them keeps the fingerprint stable so a re-run dedupes.
    """
    t = text.strip().lower()
    t = re.sub(r"0x[0-9a-f]+", "<addr>", t)
    t = re.sub(r"[a-z]:[\\/][^\s'\"]+", "<path>", t)
    t = re.sub(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", "<n>", t)
    t = re.sub(r"\s+", " ", t)
    return t


def fingerprint(source: str, location: str, title: str) -> str:
    key = f"{source}|{location}|{_normalise(title)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def load() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with QUEUE_PATH.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{QUEUE_PATH}:{lineno}: corrupt queue line: {exc}"
                ) from exc
    return entries


def save(entries: Iterable[dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(QUEUE_PATH)


def add_entry(
    title: str,
    *,
    body: str = "",
    labels: list[str] | None = None,
    source: str = "manual",
    location: str = "",
    quiet: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Queue an issue, or bump the existing one with the same fingerprint.

    Returns ``(entry, is_new)``. Never contacts GitHub.
    """
    entries = load()
    fid = fingerprint(source, location, title)

    for entry in entries:
        if entry.get("id") == fid:
            entry["count"] = int(entry.get("count", 1)) + 1
            entry["last_seen"] = _now()
            # A recurrence of something already filed is worth knowing about,
            # but it does not reopen the issue — that is GitHub's call.
            save(entries)
            if not quiet:
                seen = entry["count"]
                where = entry.get("github", {}).get("url", "not pushed")
                print(f"[issues] duplicate {fid} (seen {seen}x, {where})")
            return entry, False

    entry = {
        "id": fid,
        "created": _now(),
        "last_seen": _now(),
        "status": STATUS_OPEN,
        "title": title.strip(),
        "body": body.strip()[:MAX_BODY_CHARS],
        "labels": sorted(set(labels or [])),
        "source": source,
        "location": location,
        "count": 1,
    }
    entries.append(entry)
    save(entries)
    if not quiet:
        print(f"[issues] queued {fid}: {entry['title']}")
    return entry, True


# --------------------------------------------------------------------------
# GitHub, via gh
# --------------------------------------------------------------------------

def _gh_available() -> bool:
    from shutil import which

    return which("gh") is not None


def _repo_slug() -> str:
    """``owner/name`` from the origin remote."""
    out = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit("no git remote 'origin'; cannot infer repository")
    url = out.stdout.strip()
    m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"cannot parse repo slug from remote url: {url}")
    return m.group(1)


def _render_body(entry: dict[str, Any]) -> str:
    lines = [
        entry.get("body", "").strip(),
        "",
        "---",
        f"- source: `{entry.get('source', 'manual')}`",
    ]
    if entry.get("location"):
        lines.append(f"- location: `{entry['location']}`")
    lines.append(f"- first seen: {entry.get('created', '?')}")
    if int(entry.get("count", 1)) > 1:
        lines.append(f"- seen {entry['count']}x, last {entry.get('last_seen', '?')}")
    lines.append(f"- queue id: `{entry['id']}`")
    lines.append("")
    lines.append("_Queued by `tools/issues.py` and pushed with `sync`._")
    return "\n".join(lines)


def _create_issue(entry: dict[str, Any], slug: str) -> dict[str, Any]:
    """Create one GitHub issue. Retries without labels if the labels are unknown."""
    body = _render_body(entry)

    def run(labels: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [
            "gh", "issue", "create",
            "--repo", slug,
            "--title", entry["title"],
            "--body-file", "-",
        ]
        for label in labels:
            cmd += ["--label", label]
        return subprocess.run(
            cmd, input=body, capture_output=True, text=True, cwd=REPO_ROOT
        )

    labels = list(entry.get("labels", []))
    proc = run(labels)
    if proc.returncode != 0 and labels and "label" in proc.stderr.lower():
        print(f"[issues] {entry['id']}: labels {labels} rejected, retrying without")
        proc = run([])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "gh issue create failed")

    url = proc.stdout.strip().splitlines()[-1].strip()
    number = url.rsplit("/", 1)[-1]
    return {"url": url, "number": int(number) if number.isdigit() else None}


def sync(*, dry_run: bool = False, limit: int | None = None, assume_yes: bool = False) -> int:
    entries = load()
    pending = [e for e in entries if e.get("status") == STATUS_OPEN]
    if limit is not None:
        pending = pending[:limit]

    if not pending:
        print("[issues] nothing to push")
        return 0

    if not _gh_available():
        print(
            "[issues] gh CLI not found. Install it, then authenticate:\n"
            "    winget install --id GitHub.cli -e\n"
            "    gh auth login\n"
            f"[issues] {len(pending)} issue(s) stay queued in {QUEUE_PATH.name}",
            file=sys.stderr,
        )
        return 1

    slug = _repo_slug()
    print(f"[issues] {len(pending)} issue(s) -> {slug}")
    for entry in pending:
        print(f"  {entry['id']}  {entry['title']}")
    if dry_run:
        print("[issues] dry run, nothing pushed")
        return 0

    if not assume_yes and sys.stdin.isatty():
        if input(f"push {len(pending)} issue(s) to {slug}? [y/N] ").strip().lower() != "y":
            print("[issues] aborted, queue untouched")
            return 1

    failed = 0
    for entry in pending:
        try:
            info = _create_issue(entry, slug)
        except RuntimeError as exc:
            failed += 1
            print(f"[issues] {entry['id']} FAILED: {exc}", file=sys.stderr)
            continue
        entry["status"] = STATUS_PUSHED
        entry["github"] = info
        entry["pushed"] = _now()
        print(f"[issues] {entry['id']} -> {info['url']}")
        save(entries)  # after each, so a mid-run failure loses nothing

    save(entries)
    return 1 if failed else 0


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def capture(command: list[str], *, source: str, labels: list[str]) -> int:
    """Run ``command``; queue an issue if it exits non-zero. Returns its exit code."""
    if not command:
        raise SystemExit("capture needs a command after --")

    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    if proc.returncode == 0:
        return 0

    joined = " ".join(command)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    tail = output[-MAX_BODY_CHARS:]
    first_fail = next(
        (ln for ln in output.splitlines()
         if re.search(r"(\bFAILED?\b|\w*Error\b|\berror:|\bTraceback\b)", ln)),
        f"exit code {proc.returncode}",
    )

    # The command goes in `location`, not the title -- a `python -m x` invocation
    # with flags crowds out the one line that says what actually broke.
    add_entry(
        title=f"[{source}] {first_fail.strip()[:140]}",
        body=f"`{joined}` exited {proc.returncode}.\n\n```\n{tail}\n```",
        labels=labels,
        source=source,
        location=joined,
    )
    return proc.returncode


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _fmt_row(e: dict[str, Any]) -> str:
    mark = {STATUS_OPEN: " ", STATUS_PUSHED: "^", STATUS_DROPPED: "x"}.get(e.get("status", ""), "?")
    n = e.get("github", {}).get("number")
    ref = f"#{n}" if n else ""
    count = f" x{e['count']}" if int(e.get("count", 1)) > 1 else ""
    return f"{mark} {e['id']}  {e.get('source', '?'):<8} {e['title'][:70]}{count} {ref}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tools.issues", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="queue an issue")
    a.add_argument("--title", required=True)
    a.add_argument("--body", default="")
    a.add_argument("--label", action="append", default=[], dest="labels")
    a.add_argument("--source", default="manual")
    a.add_argument("--location", default="")

    l = sub.add_parser("list", help="show the queue")
    l.add_argument("--status", default=STATUS_OPEN,
                   choices=[STATUS_OPEN, STATUS_PUSHED, STATUS_DROPPED, "all"])

    s = sub.add_parser("show", help="print one entry in full")
    s.add_argument("id")

    d = sub.add_parser("drop", help="mark an entry as not worth filing")
    d.add_argument("id")
    d.add_argument("--reason", default="")

    y = sub.add_parser("sync", help="push open entries to GitHub via gh")
    y.add_argument("--dry-run", action="store_true")
    y.add_argument("--limit", type=int)
    y.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    c = sub.add_parser("capture", help="run a command, queue an issue if it fails")
    c.add_argument("--source", default="run")
    c.add_argument("--label", action="append", default=[], dest="labels")
    c.add_argument("command", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)

    if args.cmd == "add":
        add_entry(args.title, body=args.body, labels=args.labels,
                  source=args.source, location=args.location)
        return 0

    if args.cmd == "list":
        entries = load()
        if args.status != "all":
            entries = [e for e in entries if e.get("status") == args.status]
        if not entries:
            print(f"[issues] queue empty ({args.status})")
            return 0
        for e in entries:
            print(_fmt_row(e))
        print(f"\n{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}  "
              f"(^ pushed, x dropped)")
        return 0

    if args.cmd == "show":
        for e in load():
            if e["id"].startswith(args.id):
                print(json.dumps(e, indent=2, sort_keys=True))
                return 0
        print(f"[issues] no entry {args.id}", file=sys.stderr)
        return 1

    if args.cmd == "drop":
        entries = load()
        for e in entries:
            if e["id"].startswith(args.id):
                e["status"] = STATUS_DROPPED
                e["dropped_reason"] = args.reason
                save(entries)
                print(f"[issues] dropped {e['id']}")
                return 0
        print(f"[issues] no entry {args.id}", file=sys.stderr)
        return 1

    if args.cmd == "sync":
        return sync(dry_run=args.dry_run, limit=args.limit, assume_yes=args.yes)

    if args.cmd == "capture":
        cmd = args.command
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        return capture(cmd, source=args.source, labels=args.labels)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
