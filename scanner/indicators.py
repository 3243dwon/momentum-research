"""Canonical technical-indicator math — one implementation, imported everywhere.

Before this module existed, RSI-14 and the MACD histogram were each written
THREE separate times — scanner.technicals (last value, for the live scan row),
scanner.goldencross (full series + weekly resample, for the 金叉 tier and its
settled grading) and scanner.backtest (vectorized panel, for historical replay).
Three copies of the same Wilder/EWM formulas is a drift hazard: a tweak in one
(a min_periods change, a different zero-division guard) silently desynchronizes
the live signal from the number it is graded against.

The fix is a strict layering, not a rewrite of the math:

  * pure SERIES functions (rsi_series, macd_hist_series, macd_cross_series, atr)
    carry the formula and NO length guard — they return whatever pandas produces,
    exactly like the old backtest/goldencross series helpers did;
  * thin LAST-VALUE wrappers (rsi_last, macd_last, macd_tail) re-apply the
    original callers' warmup guards and sentinels so their observable behavior is
    byte-for-byte what technicals used to return.

Every extracted function reproduces its predecessor's output on warmed data by
construction (same math), and tests/test_indicators.py + the existing
tests/test_backtest.py feature-parity assertion pin that equality. Pure pandas —
no config, no network, no scanner imports — so this module is safe to import from
anywhere without a cycle and trivial to unit-test headless.

Conventions (inherited verbatim from the originals so nothing changes):
  RSI   Wilder's smoothing, EWM alpha = 1/period, min_periods = period.
  MACD  12/26/9 EWM spans, histogram = MACD line − signal line.
  ATR   Wilder's smoothing of the True Range (new here; there was no real ATR —
        levels.py fakes one from mean-abs-move and risk.hard_stop's ATR path is
        never fed).
"""
from __future__ import annotations

import math

import pandas as pd

# MACD warmup floor. The histogram needs the slower 26-EMA plus the 9-EMA of the
# MACD line to settle; below this the last value is unreliable, so the scalar
# wrappers refuse to emit one — mirroring the old technicals._macd `< 35` guard.
MACD_MIN_BARS = 35


# --------------------------------------------------------------------------
# RSI (Wilder)
# --------------------------------------------------------------------------
def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI for the whole series (the last-value form is rsi_last).

    Identical to the former scanner.backtest._rsi_series: EWM alpha=1/period
    with min_periods=period, and an all-gain window (avg_loss == 0) pinned to
    RSI 100 rather than left as a divide-by-zero NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(avg_loss != 0, 100.0)


def rsi_last(close: pd.Series, period: int = 14) -> float:
    """Last RSI value, or math.nan when not warmed (< period + 1 bars).

    Reproduces scanner.technicals._rsi exactly: the short-series floor returns
    math.nan; goldencross._rsi_last used the same math but returned None instead,
    so its call site maps nan → None after calling this.
    """
    if len(close) < period + 1:
        return math.nan
    return float(rsi_series(close, period).iloc[-1])


# --------------------------------------------------------------------------
# MACD (12/26/9)
# --------------------------------------------------------------------------
def macd_hist_series(close: pd.Series) -> pd.Series:
    """MACD histogram over a close series (12/26/9) — no warmup guard.

    Identical to the former scanner.goldencross._hist_series and the histogram
    column of scanner.backtest._macd_frame.
    """
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def macd_cross_series(hist: pd.Series) -> pd.Series:
    """Per-bar zero-line cross label from a histogram series.

    bullish where the histogram was <= 0 and is now > 0; bearish on the mirror.
    Vectorized form of macd_cross_last — identical to the former
    scanner.backtest._macd_frame cross column.
    """
    prev = hist.shift(1)
    cross = pd.Series(None, index=hist.index, dtype=object)
    cross = cross.mask((prev <= 0) & (hist > 0), "bullish")
    cross = cross.mask((prev >= 0) & (hist < 0), "bearish")
    return cross


def macd_cross_last(hist: pd.Series) -> str | None:
    """Zero-line cross on the final bar of a histogram series, else None.

    The scalar counterpart of macd_cross_series, matching the last-two-bar test
    in the former scanner.technicals._macd.
    """
    if len(hist) < 2:
        return None
    if hist.iloc[-2] <= 0 < hist.iloc[-1]:
        return "bullish"
    if hist.iloc[-2] >= 0 > hist.iloc[-1]:
        return "bearish"
    return None


def macd_last(close: pd.Series) -> tuple[float, str | None]:
    """Last MACD-histogram value + its zero-line cross label.

    Reproduces scanner.technicals._macd: (math.nan, None) when not warmed
    (< MACD_MIN_BARS bars).
    """
    if len(close) < MACD_MIN_BARS:
        return math.nan, None
    hist = macd_hist_series(close)
    return float(hist.iloc[-1]), macd_cross_last(hist)


def macd_tail(close: pd.Series, n: int = 6) -> list[float]:
    """The last `n` MACD-histogram values, oldest→newest, at 4dp.

    Reproduces scanner.technicals._macd_tail: [] when not warmed
    (< MACD_MIN_BARS bars), so a thin-data name never yields a phantom tail.
    """
    if len(close) < MACD_MIN_BARS:
        return []
    hist = macd_hist_series(close)
    return [round(float(x), 4) for x in hist.tail(n).tolist()]


# --------------------------------------------------------------------------
# ATR (Wilder) — new; there was no true ATR in the codebase
# --------------------------------------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range: max(H−L, |H−prevClose|, |L−prevClose|) per bar.

    The first bar has no previous close, so its TR is simply H−L (the |·prevClose|
    terms drop out under the NaN, leaving the high−low range).
    """
    prev_close = close.shift(1)
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range via Wilder smoothing (EWM alpha=1/period).

    A real, volatility-based ATR the codebase previously lacked — levels.py
    approximates one from mean-abs-move and risk.hard_stop's ATR argument is
    never supplied. Requires OHLC (High/Low), which the historical bar store
    provides; the live scan row currently carries only Close, so this feeds the
    backtest/research path, not the live row, until OHLC is plumbed through.
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
