"""Queue a local issue for every distinct test failure.

Nothing here contacts GitHub. Failures land in ``DOCS/ISSUES.jsonl`` via
``tools.issues``, deduped by fingerprint — iterating on one broken test bumps a
counter rather than piling up entries. Pushing is a separate, explicit step
(``python -m tools.issues sync``).

Capture is **on by default**; turn it off for a run with ``--no-issue-capture``
or by setting ``LBM_ISSUE_CAPTURE=0``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import issues  # noqa: E402

#: Longest excerpt of a failure report kept in the queued body.
_REPORT_CHARS = 4000


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-issue-capture",
        action="store_true",
        default=False,
        help="do not queue failures into DOCS/ISSUES.jsonl",
    )


def _capture_enabled(config: pytest.Config) -> bool:
    if config.getoption("--no-issue-capture"):
        return False
    return os.environ.get("LBM_ISSUE_CAPTURE", "1") not in ("0", "false", "no")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    if report.when != "call" or not report.failed:
        return
    if not _capture_enabled(item.config):
        return

    text = str(report.longrepr)
    # The last non-empty line of a pytest longrepr is the assertion / exception
    # line -- the one worth putting in the title.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    headline = lines[-1] if lines else "test failed"

    issues.add_entry(
        title=f"[test] {item.nodeid}: {headline[:120]}",
        body=f"Failing test `{item.nodeid}`.\n\n```\n{text[-_REPORT_CHARS:]}\n```",
        labels=["bug", "from-tests"],
        source="pytest",
        location=item.nodeid,
        quiet=True,
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not _capture_enabled(config) or not terminalreporter.stats.get("failed"):
        return
    open_count = sum(1 for e in issues.load() if e.get("status") == issues.STATUS_OPEN)
    terminalreporter.write_line("")
    terminalreporter.write_line(
        f"[issues] {open_count} open in DOCS/ISSUES.jsonl — "
        f"review with `python -m tools.issues list`, push with `sync`",
        yellow=True,
    )
