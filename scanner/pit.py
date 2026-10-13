"""Point-in-time S&P membership — the survivorship fix's fixable half.

Builds per-ticker membership intervals from the committed change log
(data/sp500_changes.json, gen_sp500_changes.py) so the indicator proof can
drop a name's PRE-INCLUSION history: a stock added to the index mid-window
typically got there BY running up, and sampling that run-up as if the name
had been on a 2021 practitioner's radar is exactly the backfill bias
RESEARCH.md §3 flags.

Honest limits, stated up front:
  * This CANNOT add the missing delisted losers back — their bars don't
    exist on the free tier. That half stays a (now calibrated) bound.
  * Only S&P events are known. NDX-only names in the proof universe carry
    no event history → unconstrained (never silently guessed).
  * Direction of error is conservative: dropping pre-add rows removes
    upward-biased history; a name that was legitimately visible earlier
    (e.g. NDX member before its S&P add) loses some fair rows. That trims
    the sample, never inflates the edge.

Stdlib-only, deterministic, unit-tested without network.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

# Membership intervals: ticker → list of (start, end) date pairs, end=None =
# still a member. Tickers absent from the mapping are UNCONSTRAINED.
Intervals = dict[str, list[tuple[date | None, date | None]]]


def load_events(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text()).get("events") or []
    except Exception:
        return []


def membership_intervals(events: list[dict]) -> Intervals:
    """Fold the chronological add/remove log into per-ticker intervals.

    State rule for the log's left edge: a ticker whose FIRST logged event is
    a REMOVE was a member since before the log begins (start=None = -inf); a
    ticker whose first event is an ADD was not. Duplicate adds/removes (table
    quirks) are ignored rather than guessed at.
    """
    by_ticker: dict[str, list[tuple[str, str]]] = {}
    for ev in sorted(events, key=lambda e: e.get("date") or ""):
        d = ev.get("date")
        if not d:
            continue
        if ev.get("added"):
            by_ticker.setdefault(ev["added"], []).append((d, "add"))
        if ev.get("removed"):
            by_ticker.setdefault(ev["removed"], []).append((d, "remove"))

    out: Intervals = {}
    for ticker, evs in by_ticker.items():
        intervals: list[tuple[date | None, date | None]] = []
        open_start: date | None | str = "closed"  # sentinel: no open interval
        if evs[0][1] == "remove":
            open_start = None  # member since before the log begins
        for d_str, kind in evs:
            d = date.fromisoformat(d_str)
            if kind == "add":
                if open_start == "closed":
                    open_start = d
            else:
                if open_start != "closed":
                    intervals.append((open_start, d))  # type: ignore[arg-type]
                    open_start = "closed"
        if open_start != "closed":
            intervals.append((open_start, None))  # type: ignore[arg-type]
        out[ticker] = intervals
    return out


def is_member(ticker: str, day: date, intervals: Intervals) -> bool:
    """Membership on `day`. Unknown tickers (no S&P event history — e.g.
    NDX-only names) are unconstrained → True. Interval end (the removal
    effective date) is exclusive."""
    ivs = intervals.get(ticker)
    if ivs is None:
        return True
    return any(
        (start is None or day >= start) and (end is None or day < end)
        for start, end in ivs
    )
