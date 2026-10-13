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


def macd_line_series(close: pd.Series) -> pd.Series:
    """The MACD LINE itself (ema12 - ema26), which is not the histogram.

    Every scan already computes this — `macd_hist_series` builds it on the way to
    the histogram and then discards it — and nothing in the repo could read it,
    which mattered because the owner's 波段 setup is DEFINED by its sign. 轴下
    means the LINE is below zero at a bullish cross; `goldencross._below_zero`
    tests the HISTOGRAM below zero, which is a different population. The two are
    easy to conflate and `scanner/axis_cross_proof.py` grades the first.

    DELIBERATELY NOT REFACTORED INTO `macd_hist_series`. That function has nine
    call sites including `goldencross` (frozen pre-registered parameters),
    `indicator_proof` (the corpus itself) and `backtest`; a refactor that is
    *supposed* to be bit-identical and quietly is not would corrupt the record
    with no visible symptom. The three duplicated lines are pinned instead by
    `test_indicators.py::test_the_macd_line_and_histogram_agree`, which asserts
    the histogram equals this line minus its own 9-period signal — so the copies
    cannot drift even though neither calls the other.
    """
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


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


# --- the wide-window reads: SMA / Bollinger / Donchian / volume ---------------
# Added 2026-09-22 on the owner's call. These need far more history than the
# scan's legacy 90-day window (SMA200 alone wants 200 sessions), which is why
# technicals.py now fetches wide and slices the legacy window back out for
# every pre-existing field — see technicals.LEGACY_HISTORY_DAYS.
#
# All four return None rather than a partial dict when the series is too short:
# a half-warmed SMA200 is a number that looks fine and means nothing, and the
# card would print it with the same confidence as a real one.

SMA_FAST, SMA_SLOW = 50, 200
# How far back `golden_cross` looks to decide the cross is FRESH. The minimum
# frame is SMA_SLOW + CROSS_LOOKBACK + 1, not SMA_SLOW + 1, because the bar we
# compare against must itself be warm — see sma_read.
SMA_CROSS_LOOKBACK = 5
BB_PERIOD, BB_STD, BB_RANK_WINDOW = 20, 2.0, 120
DC_FAST, DC_SLOW = 20, 55
VOL_FAST, VOL_SLOW = 20, 50


def sma_read(close: pd.Series) -> dict | None:
    """50/200 SMA state: the stack, distance to each, and a fresh-cross flag.

    `golden_cross` is the EVENT (50 crossed above 200 within the last 5 bars),
    not the state — the state is `stack`, and conflating them is how a card ends
    up announcing a months-old cross as news every single day.
    """
    if len(close) < SMA_SLOW + SMA_CROSS_LOOKBACK + 1:
        return None
    fast = close.rolling(SMA_FAST).mean()
    slow = close.rolling(SMA_SLOW).mean()
    f, s, last = float(fast.iloc[-1]), float(slow.iloc[-1]), float(close.iloc[-1])
    if pd.isna(f) or pd.isna(s) or not s:
        return None
    # THE COMPARISON BAR MUST BE WARM, and `fast > slow` cannot tell you that:
    # a NaN comparison is False, which is indistinguishable from a genuine "was
    # not stacked". On a frame barely longer than SMA_SLOW the 5-bars-ago entry
    # is still unwarmed, so every stacked name read as having JUST crossed.
    # Measured 2026-09-22 on a monotonic riser, where the 50 is above the 200
    # for the whole warm period and a real cross is impossible: `golden_cross`
    # came back True at 201 and 203 bars. That is a card announcing news that
    # did not happen, on exactly the names with the least history to judge it.
    warm = fast.notna() & slow.notna()
    stacked = (fast > slow).where(warm)
    prior = stacked.iloc[-(SMA_CROSS_LOOKBACK + 1)]
    cross_up = bool(stacked.iloc[-1]) and not pd.isna(prior) and not bool(prior)
    return {
        "sma50": round(f, 2),
        "sma200": round(s, 2),
        "above_50": bool(last > f),
        "above_200": bool(last > s),
        "stack": bool(f > s),
        "dist_50_pct": round((last / f - 1) * 100, 2) if f else None,
        "dist_200_pct": round((last / s - 1) * 100, 2),
        "golden_cross": cross_up,
    }


def bollinger_read(close: pd.Series) -> dict | None:
    """Bollinger(20, 2): %B, bandwidth, and whether bandwidth is in a squeeze.

    `squeeze` is bandwidth in the bottom fifth of its OWN last 120 sessions, not
    an absolute threshold — band width scales with a name's volatility, so a
    fixed cutoff calls every low-vol megacap permanently squeezed.
    """
    if len(close) < BB_PERIOD + 1:
        return None
    mid = close.rolling(BB_PERIOD).mean()
    sd = close.rolling(BB_PERIOD).std()
    upper, lower = mid + BB_STD * sd, mid - BB_STD * sd
    m, u, lo, last = (float(mid.iloc[-1]), float(upper.iloc[-1]),
                      float(lower.iloc[-1]), float(close.iloc[-1]))
    if pd.isna(m) or pd.isna(u) or not m or u == lo:
        return None
    width = (upper - lower) / mid
    rank = None
    if len(width.dropna()) >= BB_RANK_WINDOW:
        r = width.rolling(BB_RANK_WINDOW).rank(pct=True).iloc[-1]
        rank = round(float(r), 3) if not pd.isna(r) else None
    return {
        "pct_b": round((last - lo) / (u - lo), 3),
        "bandwidth_pct": round(float(width.iloc[-1]) * 100, 2),
        "bw_rank": rank,
        "squeeze": (rank is not None and rank <= 0.20),
        "above_upper": bool(last > u),
        "below_lower": bool(last < lo),
    }


def donchian_read(high: pd.Series, low: pd.Series, close: pd.Series) -> dict | None:
    """Donchian 20/55 channel position and fresh breakouts.

    Channels are taken from bars BEFORE today (`.shift(1)`) so "broke the 20-day
    high" means today cleared a level that already existed, not that today's own
    high set it — which is true of every bar and would fire constantly.
    """
    if len(close) < DC_FAST + 2:
        return None
    hi20, lo20 = high.rolling(DC_FAST).max().shift(1), low.rolling(DC_FAST).min().shift(1)
    h, lw, last = float(hi20.iloc[-1]), float(lo20.iloc[-1]), float(close.iloc[-1])
    if pd.isna(h) or pd.isna(lw) or h == lw:
        return None
    prev_break = bool(close.iloc[-2] > hi20.iloc[-2]) if len(close) >= 3 and not pd.isna(hi20.iloc[-2]) else False
    out = {
        "hi20": round(h, 2),
        "lo20": round(lw, 2),
        "pos20": round((last - lw) / (h - lw), 3),
        "break20": bool(last > h) and not prev_break,
        "breakdown20": bool(last < lw),
        "hi55": None,
        "break55": False,
    }
    if len(close) >= DC_SLOW + 2:
        hi55 = high.rolling(DC_SLOW).max().shift(1)
        h55 = float(hi55.iloc[-1])
        if not pd.isna(h55):
            prev55 = bool(close.iloc[-2] > hi55.iloc[-2]) if not pd.isna(hi55.iloc[-2]) else False
            out["hi55"] = round(h55, 2)
            out["break55"] = bool(last > h55) and not prev55
    return out


def volume_read(close: pd.Series, high: pd.Series, low: pd.Series,
                volume: pd.Series) -> dict | None:
    """Volume beyond the single-bar `rel_volume` the row already carries.

    `trend` is the 20d/50d average-volume ratio — whether participation is
    building or draining over weeks rather than in one session. `cmf` is Chaikin
    Money Flow(20), which weights each bar's volume by where it closed inside
    its own range, so it separates "heavy and bought" from merely "heavy".
    `ud_ratio` is up-day volume over down-day volume across the last 20.
    """
    if len(close) < VOL_SLOW + 1:
        return None
    v20, v50 = volume.rolling(VOL_FAST).mean(), volume.rolling(VOL_SLOW).mean()
    a, b = float(v20.iloc[-1]), float(v50.iloc[-1])
    if pd.isna(a) or pd.isna(b) or not b:
        return None
    rng = (high - low).replace(0, float("nan"))
    mfm = ((close - low) - (high - close)) / rng
    cmf_s = (mfm * volume).rolling(VOL_FAST).sum() / volume.rolling(VOL_FAST).sum()
    cmf = float(cmf_s.iloc[-1]) if not pd.isna(cmf_s.iloc[-1]) else None
    up = close > close.shift(1)
    tail_v, tail_up = volume.iloc[-VOL_FAST:], up.iloc[-VOL_FAST:]
    up_v, dn_v = float(tail_v[tail_up].sum()), float(tail_v[~tail_up].sum())
    return {
        "trend": round(a / b, 2),
        "rising": bool(a / b >= 1.2),
        "draining": bool(a / b <= 0.8),
        "cmf": round(cmf, 3) if cmf is not None else None,
        "accumulating": (cmf is not None and cmf > 0.10),
        "distributing": (cmf is not None and cmf < -0.10),
        "ud_ratio": round(up_v / dn_v, 2) if dn_v else None,
    }


# Card rendering for the four reads above, kept beside them for the same reason
# structure.read/structure.phrase live together: the thresholds that decide
# "notable" and the words that announce it drift apart the moment they are in
# different files.
WIDE_PHRASE_MAX = 3


def wide_phrase(ta: dict | None, cap: int = WIDE_PHRASE_MAX) -> str:
    """A short Chinese phrase for the NOTABLE part of a wide read, or "".

    Absence is the message, exactly as in structure.phrase: an ordinary name in
    the middle of its channel with average participation earns no words, because
    a line that always prints something teaches a reader to stop reading it.

    ORDER IS BY NEWSWORTHINESS, NOT BY FAMILY, and the cap makes that ordering
    load-bearing rather than cosmetic. EVENTS first (a channel break, a fresh
    50/200 cross) because they are true today and were not true yesterday; then
    conditions a reader would act on (a squeeze, a band break, being under the
    200); then flow. The standing trend state (`均线多头`) is LAST because it is
    true of most names in an uptrend at once — it is the least differentiating
    thing here, so it is the one that should yield when the cap bites.

    That ordering was wrong in the first draft: the standing state outranked
    flow, so on a name where all four families had something to say, the volume
    read — the one the card exists to add — was the bit that got dropped.
    """
    if not ta:
        return ""
    sma, bb, dc, vol = ta.get("sma"), ta.get("bb"), ta.get("dc"), ta.get("vol")
    bits: list[str] = []

    # 1. Channel events.
    if dc:
        if dc.get("break55"):
            bits.append("破55日高")
        elif dc.get("break20"):
            bits.append("破20日高")
        elif dc.get("breakdown20"):
            bits.append("破20日低")
    # 2. Moving-average EVENT, or the one standing state worth interrupting for.
    if sma:
        if sma.get("golden_cross"):
            bits.append("50/200 金叉")
        elif not sma.get("above_200"):
            bits.append("200日线下")
    # 3. Volatility condition.
    if bb:
        if bb.get("squeeze"):
            bits.append("带宽挤压")
        elif bb.get("above_upper"):
            bits.append("破布林上轨")
        elif bb.get("below_lower"):
            bits.append("破布林下轨")
    # 4. Flow.
    if vol:
        if vol.get("accumulating"):
            bits.append("资金流入")
        elif vol.get("distributing"):
            bits.append("资金流出")
        elif vol.get("rising"):
            bits.append("量能抬升")
    # 5. The standing stack, last — first to be dropped by the cap.
    if sma and sma.get("above_200") and sma.get("stack") and sma.get("above_50"):
        bits.append("均线多头")
    return " · ".join(bits[:cap])


def wide_stamp(ta: dict | None) -> dict | None:
    """The gradeable part of a wide read, for the append-only log.

    Mirrors structure.stamp: keeps the SIGNALS and drops the price LEVELS
    (sma50/sma200/hi20/lo20/hi55). A grader reads forward returns against the
    state at the call, so a level is dead weight on a file that is committed on
    every scan — and levels are what make an entry grow without bound.

    Additive by construction: no existing key moves, and no grader reads this
    yet, so nothing already logged is regraded.
    """
    if not ta:
        return None
    out: dict = {}
    sma, bb, dc, vol = ta.get("sma"), ta.get("bb"), ta.get("dc"), ta.get("vol")
    if sma:
        out["sma"] = {k: sma.get(k) for k in
                      ("above_50", "above_200", "stack", "golden_cross",
                       "dist_50_pct", "dist_200_pct")}
    if bb:
        out["bb"] = {k: bb.get(k) for k in
                     ("pct_b", "bw_rank", "squeeze", "above_upper", "below_lower")}
    if dc:
        out["dc"] = {k: dc.get(k) for k in
                     ("pos20", "break20", "break55", "breakdown20")}
    if vol:
        out["vol"] = {k: vol.get(k) for k in
                      ("trend", "cmf", "rising", "accumulating", "distributing", "ud_ratio",
                       "feed")}
    return out or None


# --- MACD as physics: DIF is speed, the histogram is acceleration -------------
# Added 2026-09-22. The card used to say "daily MACD still below zero and
# rising (轴下)" about every name, and that sentence reads the HISTOGRAM. 轴下 in
# the owner's setup is defined by the DIF LINE (EMA12 - EMA26) being below zero,
# which is a different population: on the 2026-09-22 cross-section 3 of the 10
# candidates that could be checked had DIF ABOVE zero (MRK, FCX, PLTR) and were
# still announced as 轴下 reversals when they were pullbacks inside an uptrend.
#
# The physical reading is the useful one. Price is position; DIF, a difference of
# a fast and a slow average, is a smoothed slope — speed; the histogram (DIF
# minus its own 9-EMA) is how that speed is changing — acceleration. So the
# histogram turning up through zero marks the moment the move STOPPED SPEEDING
# UP, not the moment it turned. Below the axis, price keeps falling until DIF
# itself reaches zero. 减速不等于掉头.

MACD_DIF_ZERO_HORIZON = 15   # beyond this a linear bars-to-zero is not a read


def macd_read(close: pd.Series) -> dict | None:
    """DIF, DEA and histogram with their slopes, scaled to price, plus phase.

    Computed on the SAME close series the row's `macd_hist` comes from, so the
    histogram printed here equals the one the trigger read. `dif_pct` and
    `hist_pct` divide by price because a raw MACD value is in dollars: AMGN's
    histogram of -1.90 and GRAB's of -0.014 are not comparable numbers.
    """
    if len(close) < MACD_MIN_BARS:
        return None
    dif = macd_line_series(close)
    hist = macd_hist_series(close)
    d, dp = float(dif.iloc[-1]), float(dif.iloc[-2])
    h, hp = float(hist.iloc[-1]), float(hist.iloc[-2])
    px = float(close.iloc[-1])
    if any(pd.isna(x) for x in (d, dp, h, hp)) or not px:
        return None
    slope = d - dp
    below = d < 0
    # Velocity sign x acceleration sign — the four phases of a swing.
    if below:
        phase = "falling_decelerating" if h >= 0 else "falling_accelerating"
    else:
        phase = "rising_accelerating" if h >= 0 else "rising_decelerating"
    to_zero = None
    if below and slope > 0:
        est = -d / slope
        if est <= MACD_DIF_ZERO_HORIZON:
            to_zero = round(est, 1)
    return {
        "dif": round(d, 4),
        "dea": round(d - h, 4),
        "hist": round(h, 4),
        "dif_pct": round(d / px * 100, 3),
        "hist_pct": round(h / px * 100, 3),
        "axis": "below" if below else "above",
        "dif_rising": bool(slope > 0),
        "hist_rising": bool(h > hp),
        "phase": phase,
        "dif_bars_to_zero": to_zero,
    }


def macd_phrase(read: dict | None) -> str:
    """The per-name MACD phrase for a golden-cross card line, or "".

    Every name on that card is, by the trigger, a histogram below zero and
    rising — so the card-level sentence already says "acceleration is turning".
    What differs between names is SPEED, and that is what this names: which
    side of the axis DIF is on, and whether DIF itself has turned.
    """
    if not read:
        return ""
    if read.get("axis") == "below":
        if read.get("dif_rising"):
            n = read.get("dif_bars_to_zero")
            tail = f"·DIF约{n:.0f}根回零轴" if n is not None and n >= 1 else ""
            return f"轴下·跌势减速{tail}"
        return "轴下·仍在加速跌"
    if read.get("dif_rising"):
        return "轴上·回踩后再加速"
    return "轴上·涨速仍放缓"


def macd_stamp(read: dict | None) -> dict | None:
    """The gradeable part for the append-only log: scale-free numbers and the
    axis/phase, no dollar levels. Makes 轴下 vs 轴上 a forward-gradeable split —
    the distinction the trigger itself cannot see."""
    if not read:
        return None
    return {k: read.get(k) for k in
            ("axis", "phase", "dif_pct", "hist_pct", "dif_rising", "hist_rising",
             "dif_bars_to_zero")}
