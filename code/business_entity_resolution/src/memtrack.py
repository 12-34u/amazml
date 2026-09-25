"""Peak-memory and timing log for pipeline stages.

Every stage runs inside `with stage("name"):`. On exit it prints elapsed time and the
process peak RSS, and appends the same numbers to notes/resource_log.tsv. The budget is
5 GB per stage. Run each stage in its own process (the CLI entry points do), because
ru_maxrss is the peak over the whole life of the process.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import resource
import time
from pathlib import Path

BUDGET_GB = 5.0
LOG_PATH = Path(__file__).resolve().parents[1] / "notes" / "resource_log.tsv"


def peak_rss_gb() -> float:
    """Peak resident set size of this process so far, in GB (Linux reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def current_rss_gb() -> float:
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * resource.getpagesize() / 1024**3


@contextlib.contextmanager
def stage(name: str, log: bool = True):
    t0 = time.time()
    yield
    secs, peak = time.time() - t0, peak_rss_gb()
    flag = "  !! OVER BUDGET" if peak > BUDGET_GB else ""
    print(f"[{name}] {secs:.1f}s | peak RSS {peak:.2f} GB (budget {BUDGET_GB:.0f} GB){flag}")
    if log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG_PATH.exists()
        with open(LOG_PATH, "a") as f:
            if new:
                f.write("timestamp\tstage\tseconds\tpeak_rss_gb\n")
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}\t{name}\t{secs:.1f}\t{peak:.2f}\n")
