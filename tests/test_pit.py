"""Unit tests for scanner.pit — point-in-time S&P membership intervals.

The survivorship fix's fixable half: these tests pin the interval fold rules
(first-event-is-remove ⇒ member since before the log; add opens, remove
closes exclusive; unknown tickers unconstrained). The calibration split in
portfolio_backtest.survivorship_calibration was pinned here too until
2026-10-09, when that module was archived (deletion batch 4b; git history holds
it, and its two tests, at fdfe3266). Run: pytest tests/test_pit.py
"""
from __future__ import annotations

from datetime import date

from scanner import pit


def _ev(d, added=None, removed=None, reason=None):
    return {"date": d, "added": added, "removed": removed, "reason": reason}


def test_add_opens_membership():
    iv = pit.membership_intervals([_ev("2023-09-18", added="ABNB")])
    assert iv["ABNB"] == [(date(2023, 9, 18), None)]
    assert not pit.is_member("ABNB", date(2023, 9, 17), iv)
    assert pit.is_member("ABNB", date(2023, 9, 18), iv)


def test_first_event_remove_means_member_since_before_log():
    iv = pit.membership_intervals([_ev("2024-03-01", removed="XRAY")])
    assert iv["XRAY"] == [(None, date(2024, 3, 1))]
    assert pit.is_member("XRAY", date(2020, 1, 1), iv)   # pre-log history kept
    assert not pit.is_member("XRAY", date(2024, 3, 1), iv)  # removal day exclusive


def test_remove_then_readd_yields_two_intervals():
    iv = pit.membership_intervals([
        _ev("2022-06-01", removed="UAA"),
        _ev("2024-11-01", added="UAA"),
    ])
    assert iv["UAA"] == [(None, date(2022, 6, 1)), (date(2024, 11, 1), None)]
    assert pit.is_member("UAA", date(2021, 1, 1), iv)
    assert not pit.is_member("UAA", date(2023, 1, 1), iv)
    assert pit.is_member("UAA", date(2025, 1, 1), iv)


def test_duplicate_events_ignored_not_guessed():
    iv = pit.membership_intervals([
        _ev("2022-01-01", added="DUP"),
        _ev("2022-06-01", added="DUP"),      # add while member — table quirk
        _ev("2023-01-01", removed="DUP"),
        _ev("2023-02-01", removed="DUP"),    # remove while not member — quirk
    ])
    assert iv["DUP"] == [(date(2022, 1, 1), date(2023, 1, 1))]


def test_unknown_ticker_is_unconstrained():
    iv = pit.membership_intervals([_ev("2023-09-18", added="ABNB")])
    assert pit.is_member("NVDA", date(2015, 1, 1), iv)  # no event history → True


def test_one_row_carrying_both_add_and_remove():
    iv = pit.membership_intervals([_ev("2024-05-01", added="NEW", removed="OLD")])
    assert iv["NEW"] == [(date(2024, 5, 1), None)]
    assert iv["OLD"] == [(None, date(2024, 5, 1))]


def test_load_events_fail_soft(tmp_path):
    assert pit.load_events(tmp_path / "missing.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert pit.load_events(bad) == []
