"""Unit tests for scanner.indicators — the shared indicator library.

The whole point of scanner.indicators is that extracting the triple-implemented
RSI/MACD math into one place changed NOTHING observable. So the core tests here
are REGRESSION baselines: each old inline formula (technicals._rsi/_macd/
_macd_tail, goldencross._rsi_last/_hist_series, backtest._rsi_series/_macd_frame)
is reproduced verbatim below as a `_ref_*` reference, and the new module is
asserted equal to it across random-walk series and edge cases. If someone later
"tidies" the math in scanner.indicators, these fail.

pytest isn't installed in this venv — stdlib unittest. Run with:
    .venv/bin/python -m unittest tests.test_indicators -v
"""
from __future__ import annotations

import math
import random
import unittest

import pandas as pd

from scanner import indicators


# --------------------------------------------------------------------------
# Verbatim copies of the PRE-refactor implementations — the behavior baseline.
# --------------------------------------------------------------------------
def _ref_rsi_last(close: pd.Series, period: int = 14) -> float:
    """Exact former scanner.technicals._rsi."""
    if len(close) < period + 1:
        return math.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100 - 100 / (1 + rs))


def _ref_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Exact former scanner.backtest._rsi_series."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(avg_loss != 0, 100.0)


def _ref_macd_last(close: pd.Series) -> tuple[float, str | None]:
    """Exact former scanner.technicals._macd."""
    if len(close) < 35:
        return math.nan, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    cross = None
    if len(hist) >= 2:
        if hist.iloc[-2] <= 0 < hist.iloc[-1]:
            cross = "bullish"
        elif hist.iloc[-2] >= 0 > hist.iloc[-1]:
            cross = "bearish"
    return float(hist.iloc[-1]), cross


def _ref_macd_tail(close: pd.Series, n: int = 6) -> list[float]:
    """Exact former scanner.technicals._macd_tail."""
    if len(close) < 35:
        return []
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return [round(float(x), 4) for x in hist.tail(n).tolist()]


def _ref_macd_frame(close: pd.Series) -> pd.DataFrame:
    """Exact former scanner.backtest._macd_frame."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    prev = hist.shift(1)
    cross = pd.Series(None, index=close.index, dtype=object)
    cross = cross.mask((prev <= 0) & (hist > 0), "bullish")
    cross = cross.mask((prev >= 0) & (hist < 0), "bearish")
    return pd.DataFrame({"macd_hist": hist, "macd_cross": cross})


def _walk(n: int, seed: int, p0: float = 50.0) -> pd.Series:
    rng = random.Random(seed)
    prices = [p0]
    for _ in range(n - 1):
        prices.append(max(3.5, prices[-1] * (1 + rng.gauss(0, 0.015))))
    idx = pd.bdate_range(start="2025-01-01", periods=n)
    return pd.Series(prices, index=idx)


class TestRSIParity(unittest.TestCase):
    def test_series_matches_reference_across_walks(self):
        for seed in range(6):
            close = _walk(120, seed)
            got = indicators.rsi_series(close)
            ref = _ref_rsi_series(close)
            # NaN-safe elementwise comparison at full precision.
            pd.testing.assert_series_equal(got, ref, check_names=False)

    def test_last_matches_reference(self):
        for seed in range(6):
            close = _walk(120, seed)
            self.assertAlmostEqual(indicators.rsi_last(close), _ref_rsi_last(close), places=9)

    def test_last_series_agree(self):
        close = _walk(120, seed=3)
        self.assertAlmostEqual(
            indicators.rsi_last(close), float(indicators.rsi_series(close).iloc[-1]), places=9
        )

    def test_short_series_is_nan(self):
        close = _walk(10, seed=1)  # < period + 1
        self.assertTrue(math.isnan(indicators.rsi_last(close)))

    def test_all_gains_pins_to_100(self):
        close = pd.Series([10.0 + i for i in range(30)])  # monotone up → avg_loss 0
        self.assertEqual(indicators.rsi_last(close), 100.0)


class TestMACDParity(unittest.TestCase):
    def test_hist_series_matches_reference(self):
        for seed in range(6):
            close = _walk(120, seed)
            got = indicators.macd_hist_series(close)
            ref = _ref_macd_frame(close)["macd_hist"]
            pd.testing.assert_series_equal(got, ref, check_names=False)

    def test_cross_series_matches_reference(self):
        for seed in range(6):
            close = _walk(120, seed)
            hist = indicators.macd_hist_series(close)
            got = indicators.macd_cross_series(hist)
            ref = _ref_macd_frame(close)["macd_cross"]
            pd.testing.assert_series_equal(got, ref, check_names=False, check_dtype=False)

    def test_last_matches_reference(self):
        for seed in range(6):
            close = _walk(120, seed)
            got_hist, got_cross = indicators.macd_last(close)
            ref_hist, ref_cross = _ref_macd_last(close)
            self.assertAlmostEqual(got_hist, ref_hist, places=9)
            self.assertEqual(got_cross, ref_cross)

    def test_tail_matches_reference(self):
        for seed in range(6):
            close = _walk(120, seed)
            self.assertEqual(indicators.macd_tail(close), _ref_macd_tail(close))

    def test_short_series_floors(self):
        close = _walk(20, seed=1)  # < 35
        hist, cross = indicators.macd_last(close)
        self.assertTrue(math.isnan(hist))
        self.assertIsNone(cross)
        self.assertEqual(indicators.macd_tail(close), [])

    def test_cross_labels_on_constructed_hist(self):
        # Hand-built histogram that crosses up (-1 → +1) then down (+1 → -1).
        up = pd.Series([-2.0, -1.0, 1.0])
        self.assertEqual(indicators.macd_cross_last(up), "bullish")
        down = pd.Series([2.0, 1.0, -1.0])
        self.assertEqual(indicators.macd_cross_last(down), "bearish")
        flat = pd.Series([1.0, 2.0, 3.0])  # no zero crossing
        self.assertIsNone(indicators.macd_cross_last(flat))
        self.assertIsNone(indicators.macd_cross_last(pd.Series([1.0])))  # too short


class TestATR(unittest.TestCase):
    def test_true_range_first_bar_is_high_low(self):
        high = pd.Series([11.0, 12.0, 13.0])
        low = pd.Series([10.0, 10.5, 11.0])
        close = pd.Series([10.5, 11.5, 12.5])
        tr = indicators.true_range(high, low, close)
        self.assertAlmostEqual(tr.iloc[0], 1.0, places=9)  # H-L, no prev close
        # bar 2: max(12-10.5, |12-10.5|, |10.5-10.5|) = 1.5
        self.assertAlmostEqual(tr.iloc[1], 1.5, places=9)

    def test_atr_is_positive_and_warms(self):
        close = _walk(60, seed=5)
        high = close * 1.01
        low = close * 0.99
        a = indicators.atr(high, low, close, period=14)
        self.assertTrue(math.isnan(a.iloc[0]))          # not warmed
        self.assertGreater(float(a.iloc[-1]), 0.0)      # settled, positive
        self.assertEqual(len(a), len(close))


if __name__ == "__main__":
    unittest.main()


def test_the_macd_line_and_histogram_agree():
    """`macd_line_series` duplicates three lines of `macd_hist_series` rather
    than refactoring it — that function feeds goldencross's frozen parameters,
    the indicator proof and the backtest, and a refactor that is supposed to be
    bit-identical and quietly is not would corrupt the corpus silently.

    This is what makes the duplication safe: the histogram must equal the line
    minus the line's own 9-period signal, bar for bar. If either copy is edited
    alone, this fails.
    """
    import numpy as np
    import pandas as pd

    from scanner import indicators

    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.5, 400)))
    line = indicators.macd_line_series(close)
    signal = line.ewm(span=9, adjust=False).mean()
    pd.testing.assert_series_equal(indicators.macd_hist_series(close), line - signal)
