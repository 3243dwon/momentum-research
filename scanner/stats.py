"""Small-sample statistics for signal grading — the honesty layer.

The memory's root-cause note for the owner's low confidence is blunt: the ledger
reports hit-rates on tiny samples with no way to tell a 52% edge from a 50% coin
flip. A bare "54% at 5d" reads as signal when it is often noise dressed as one.

This module supplies the missing rigor as pure, dependency-free functions:

  * wilson_interval — a binomial confidence interval that, unlike the naive
    normal approximation, stays inside [0, 1] and behaves sanely at small n and
    extreme proportions (exactly the regime our per-bucket counts live in);
  * coin_flip_verdict — turns that interval into a plain call: is the whole 95%
    interval above 0.5 ("edge"), entirely below ("adverse"), or does it straddle
    0.5 so the result is statistically indistinguishable from a coin flip?
  * block_bootstrap_mean_ci — a CI on a MEAN return that resamples whole
    calendar dates, so overlapping forward-return windows and whole-universe
    co-firing (both massively autocorrelated) stop being counted as independent
    trials. This is the fix for the single worst over-claim: a Wilson CI on
    n raw ticker-days is ~an order of magnitude too tight here.
  * benjamini_hochberg — FDR control across a family of tests, so that with
    35 signal×horizon cells at 95% we don't bless the 1-2 that pass by chance.
  * sharpe_moments / probabilistic_sharpe / expected_max_sharpe /
    deflated_sharpe — the Bailey–López de Prado selection-aware layer: PSR
    corrects a Sharpe for short samples and fat tails, E[max SR] is the hurdle
    that N trials of pure noise would set, and DSR asks whether the SELECTED
    strategy clears the hurdle its own selection created;
  * pbo_cscv — Probability of Backtest Overfitting via combinatorially
    symmetric cross-validation: how often the in-sample best of the family
    lands below median out-of-sample.

No pandas, no I/O — trivially testable. random is seeded by callers for
reproducible bootstrap CIs.
"""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from statistics import NormalDist

# 95% two-sided normal quantile. Kept as a named default so callers can widen to
# 99% (z≈2.576) for a stricter "edge" bar without magic numbers at call sites.
Z_95 = 1.959963984540054


def wilson_interval(hits: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion hits/n.

    Returns (low, high) bounds on the true hit-rate at the given confidence, or
    None when n == 0 (no evidence → no interval). Bounds are clamped to [0, 1].
    """
    if n <= 0:
        return None
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def coin_flip_verdict(hits: int, n: int, baseline: float = 0.5, z: float = Z_95) -> str:
    """Is hits/n distinguishable from `baseline` (default a 50/50 coin flip)?

    "edge"      — the whole confidence interval sits above baseline (real signal)
    "adverse"   — the whole interval sits below baseline (reliably worse)
    "coin_flip" — the interval straddles baseline (can't distinguish from chance)
    "no_data"   — n == 0
    """
    ci = wilson_interval(hits, n, z)
    if ci is None:
        return "no_data"
    low, high = ci
    if low > baseline:
        return "edge"
    if high < baseline:
        return "adverse"
    return "coin_flip"


def binom_summary(hits: int, n: int, baseline: float = 0.5, z: float = Z_95) -> dict:
    """Bundle the count, rate, Wilson CI and coin-flip verdict for one bucket.

    The shape a grading report attaches next to a hit-rate so the reader sees the
    number AND whether it means anything: {n, hits, hit_rate, ci_low, ci_high,
    verdict}. hit_rate/ci are None when n == 0.
    """
    ci = wilson_interval(hits, n, z)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "ci_low": round(ci[0], 3) if ci else None,
        "ci_high": round(ci[1], 3) if ci else None,
        "verdict": coin_flip_verdict(hits, n, baseline, z),
    }


# --------------------------------------------------------------------------
# Overlap-aware inference — resample DATES, not ticker-days
# --------------------------------------------------------------------------
def block_bootstrap_mean_ci(
    # Sequence, not list: the block KEY is only ever hashed/str()'d here, so any
    # `list[tuple[str, list[float]]]` a caller already has should pass. list is
    # invariant, so `list[tuple[object, ...]]` would reject exactly that — and
    # every caller (goldencross, opening, axis_cross_proof) groups by date-string.
    # None of the bootstrap functions mutate `blocks`, so read-only is honest.
    blocks: Sequence[tuple[object, list[float]]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI + two-sided p-value on the MEAN of values that arrive in dependent blocks.

    `blocks` is a list of (block_key, values) — group every observation by its
    calendar DATE, so each block is one day's worth of (already market-demeaned)
    returns. We resample the DATES with replacement (not the individual
    observations), pool the drawn days' values, and take the pooled mean; over
    n_boot draws that gives a percentile CI that honours BOTH the serial overlap
    of consecutive forward windows AND the whole-universe co-firing within a day
    — the dependence a Wilson-on-raw-n CI ignores, making it far too tight.

    Returns {mean, ci_low, ci_high, p_value, n_obs, n_blocks, n_eff}. The verdict
    is against a null mean of ZERO (these are demeaned/edge returns): p_value is
    the two-sided bootstrap tail past 0. n_eff (≈ number of days) is the honest
    effective sample size to show beside the inflated raw n_obs.
    """
    days = [(k, v) for k, v in blocks if v]
    n_obs = sum(len(v) for _, v in days)
    n_blocks = len(days)
    if n_obs == 0 or n_blocks == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_value": None,
                "n_obs": 0, "n_blocks": 0, "n_eff": 0}

    # Precompute each day's (sum, count) so a bootstrap draw is O(n_blocks)
    # additions of cached scalars, not a re-sum of every observation.
    day_sum = [math.fsum(v) for _, v in days]
    day_cnt = [len(v) for _, v in days]
    observed = math.fsum(day_sum) / n_obs
    rng = random.Random(seed)
    population = range(n_blocks)
    means: list[float] = []
    for _ in range(n_boot):
        draw = rng.choices(population, k=n_blocks)  # n_blocks days, with replacement
        cnt = sum(day_cnt[i] for i in draw)
        if cnt:
            means.append(sum(day_sum[i] for i in draw) / cnt)
    if not means:
        return {"mean": round(observed, 4), "ci_low": None, "ci_high": None,
                "p_value": None, "n_obs": n_obs, "n_blocks": n_blocks, "n_eff": n_blocks}
    means.sort()
    lo = means[max(0, int((alpha / 2) * len(means)))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    # Two-sided bootstrap p vs a null of 0: how often the resampled mean lands on
    # the opposite side of 0 from the observed mean (×2, clamped).
    frac_le0 = sum(1 for m in means if m <= 0) / len(means)
    frac_ge0 = sum(1 for m in means if m >= 0) / len(means)
    p = min(1.0, 2 * min(frac_le0, frac_ge0))
    return {
        "mean": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(p, 4),
        "n_obs": n_obs,
        "n_blocks": n_blocks,   # ≈ effective sample size (days), not ticker-days
        "n_eff": n_blocks,
    }


def moving_block_bootstrap_mean_ci(
    blocks: Sequence[tuple[object, list[float]]],
    block_len: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI + p-value on a mean where consecutive DATES are themselves dependent.

    block_bootstrap_mean_ci resamples single dates i.i.d., which honours
    within-day co-firing but NOT the cross-date dependence of overlapping
    forward windows: at horizon h, adjacent firing dates share ~(h-1)/h of
    their return window, so i.i.d.-date CIs can be several times too tight when
    firings cluster into episodes. This variant resamples CIRCULAR MOVING
    BLOCKS of `block_len` consecutive entries of the date-ordered sequence
    (choose block_len >= the forward horizon), so whole dependent stretches
    move together and the CI widens to match the true information content.

    `blocks` must be in date order (the caller sorts). Returns the same shape
    as block_bootstrap_mean_ci plus n_blocks_drawn. Coin-flip results from the
    tighter i.i.d. variant are conservative and safe; any EDGE claim on
    overlapping horizons should clear THIS interval instead.
    """
    days = [(k, v) for k, v in blocks if v]
    n_obs = sum(len(v) for _, v in days)
    n = len(days)
    if n_obs == 0 or n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_value": None,
                "n_obs": 0, "n_blocks": 0, "n_eff": 0}
    day_sum = [math.fsum(v) for _, v in days]
    day_cnt = [len(v) for _, v in days]
    observed = math.fsum(day_sum) / n_obs
    if n <= block_len:
        # Degenerate: the whole sample fits inside ONE dependence block, so
        # every resample is the full sample and the CI collapses to zero width
        # — false confidence. No interval can honestly be estimated here.
        return {"mean": round(observed, 4), "ci_low": None, "ci_high": None,
                "p_value": None, "n_obs": n_obs, "n_blocks": n, "n_eff": 1}
    L = max(1, block_len)
    n_draws = max(1, math.ceil(n / L))       # blocks per resample to cover ~n days
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        tot = 0.0
        cnt = 0
        for _ in range(n_draws):
            s = rng.randrange(n)
            for j in range(L):                # circular block of L consecutive days
                i = (s + j) % n
                tot += day_sum[i]
                cnt += day_cnt[i]
        if cnt:
            means.append(tot / cnt)
    if not means:
        return {"mean": round(observed, 4), "ci_low": None, "ci_high": None,
                "p_value": None, "n_obs": n_obs, "n_blocks": n, "n_eff": n}
    means.sort()
    lo = means[max(0, int((alpha / 2) * len(means)))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    frac_le0 = sum(1 for m in means if m <= 0) / len(means)
    frac_ge0 = sum(1 for m in means if m >= 0) / len(means)
    return {
        "mean": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(min(1.0, 2 * min(frac_le0, frac_ge0)), 4),
        "n_obs": n_obs,
        "n_blocks": n,
        "n_eff": max(1, n // L),              # honest independent-unit count
    }


def moving_block_bootstrap_diff_ci(
    a_blocks: Sequence[tuple[object, list[float]]],
    b_blocks: Sequence[tuple[object, list[float]]],
    block_len: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Moving-block version of block_bootstrap_diff_ci: mean(a) − mean(b) with
    circular blocks of consecutive dates drawn over the UNION date sequence, so
    both cohorts move through time together and overlapping-window dependence
    is honoured on both legs of the difference."""
    a_map = {k: (math.fsum(v), len(v)) for k, v in a_blocks if v}
    b_map = {k: (math.fsum(v), len(v)) for k, v in b_blocks if v}
    dates = sorted(set(a_map) | set(b_map), key=str)
    n_a = sum(c for _, c in a_map.values())
    n_b = sum(c for _, c in b_map.values())
    n = len(dates)
    if not n_a or not n_b or n == 0:
        return {"edge": None, "ci_low": None, "ci_high": None, "p_value": None,
                "n_a": n_a, "n_b": n_b, "n_days": n}
    a_sum = [a_map.get(d, (0.0, 0))[0] for d in dates]
    a_cnt = [a_map.get(d, (0.0, 0))[1] for d in dates]
    b_sum = [b_map.get(d, (0.0, 0))[0] for d in dates]
    b_cnt = [b_map.get(d, (0.0, 0))[1] for d in dates]
    observed = (math.fsum(a_sum) / n_a) - (math.fsum(b_sum) / n_b)
    if n <= block_len:
        # Same degenerate case as the mean version: one block = whole sample.
        return {"edge": round(observed, 4), "ci_low": None, "ci_high": None,
                "p_value": None, "n_a": n_a, "n_b": n_b, "n_days": n, "n_eff": 1}
    L = max(1, block_len)
    n_draws = max(1, math.ceil(n / L))
    rng = random.Random(seed)
    edges: list[float] = []
    for _ in range(n_boot):
        at = bt = 0.0
        ac = bc = 0
        for _ in range(n_draws):
            s = rng.randrange(n)
            for j in range(L):
                i = (s + j) % n
                at += a_sum[i]
                ac += a_cnt[i]
                bt += b_sum[i]
                bc += b_cnt[i]
        if ac and bc:
            edges.append(at / ac - bt / bc)
    if not edges:
        return {"edge": round(observed, 4), "ci_low": None, "ci_high": None,
                "p_value": None, "n_a": n_a, "n_b": n_b, "n_days": n}
    edges.sort()
    lo = edges[max(0, int((alpha / 2) * len(edges)))]
    hi = edges[min(len(edges) - 1, int((1 - alpha / 2) * len(edges)))]
    frac_le0 = sum(1 for e in edges if e <= 0) / len(edges)
    frac_ge0 = sum(1 for e in edges if e >= 0) / len(edges)
    return {
        "edge": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(min(1.0, 2 * min(frac_le0, frac_ge0)), 4),
        "n_a": n_a,
        "n_b": n_b,
        "n_days": n,
        "n_eff": max(1, n // L),
    }


def block_bootstrap_diff_ci(
    a_blocks: Sequence[tuple[object, list[float]]],
    b_blocks: Sequence[tuple[object, list[float]]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI + two-sided p-value on the DIFFERENCE of two cohorts' means (a − b),
    resampling whole calendar DATES so the market factor common to both cohorts
    on a given day — and the overlap of forward windows — is handled honestly.

    Both inputs group observations by DATE (block_key, values). We resample the
    UNION of dates with replacement; on each draw we pool the drawn days' a-values
    and b-values and take mean(a_pool) − mean(b_pool). Because the two cohorts are
    compared on the SAME resampled days, the shared daily market move largely
    cancels in the difference — this is the market-neutral "does gate A beat
    cohort B" estimate (e.g. does the VWAP-hold fired cohort out-earn the raw
    eligible-gapper control cohort).

    Returns {edge, ci_low, ci_high, p_value, n_a, n_b, n_days}. n_days is the
    effective sample (distinct dates in the union) — the honest count to show
    beside the inflated raw observation counts. Pair with zero_crossing_verdict.
    """
    a_map = {k: (math.fsum(v), len(v)) for k, v in a_blocks if v}
    b_map = {k: (math.fsum(v), len(v)) for k, v in b_blocks if v}
    dates = sorted(set(a_map) | set(b_map), key=str)
    n_a = sum(c for _, c in a_map.values())
    n_b = sum(c for _, c in b_map.values())
    n_days = len(dates)
    if not n_a or not n_b or n_days == 0:
        return {"edge": None, "ci_low": None, "ci_high": None, "p_value": None,
                "n_a": n_a, "n_b": n_b, "n_days": n_days}
    a_sum = [a_map.get(d, (0.0, 0))[0] for d in dates]
    a_cnt = [a_map.get(d, (0.0, 0))[1] for d in dates]
    b_sum = [b_map.get(d, (0.0, 0))[0] for d in dates]
    b_cnt = [b_map.get(d, (0.0, 0))[1] for d in dates]
    observed = (math.fsum(a_sum) / n_a) - (math.fsum(b_sum) / n_b)
    rng = random.Random(seed)
    population = range(n_days)
    edges: list[float] = []
    for _ in range(n_boot):
        draw = rng.choices(population, k=n_days)  # n_days dates, with replacement
        ac = sum(a_cnt[i] for i in draw)
        bc = sum(b_cnt[i] for i in draw)
        if ac and bc:  # a draw with either cohort empty yields no defined diff
            edges.append(sum(a_sum[i] for i in draw) / ac - sum(b_sum[i] for i in draw) / bc)
    if not edges:
        return {"edge": round(observed, 4), "ci_low": None, "ci_high": None,
                "p_value": None, "n_a": n_a, "n_b": n_b, "n_days": n_days}
    edges.sort()
    lo = edges[max(0, int((alpha / 2) * len(edges)))]
    hi = edges[min(len(edges) - 1, int((1 - alpha / 2) * len(edges)))]
    frac_le0 = sum(1 for e in edges if e <= 0) / len(edges)
    frac_ge0 = sum(1 for e in edges if e >= 0) / len(edges)
    p = min(1.0, 2 * min(frac_le0, frac_ge0))
    return {
        "edge": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(p, 4),
        "n_a": n_a,
        "n_b": n_b,
        "n_days": n_days,   # ≈ effective sample size (distinct dates)
    }


def zero_crossing_verdict(ci_low: float | None, ci_high: float | None) -> str:
    """Plain call on a mean-return CI relative to a null of ZERO (demeaned edge).

    "edge" — whole interval above 0 (adds return over the market/control);
    "adverse" — whole interval below 0; "coin_flip" — straddles 0 (indistinct
    from no edge); "no_data" — no interval.
    """
    if ci_low is None or ci_high is None:
        return "no_data"
    if ci_low > 0:
        return "edge"
    if ci_high < 0:
        return "adverse"
    return "coin_flip"


# moving_block_bootstrap_mean_ci (and its diff/edge siblings) report p as
# round(2 * min(frac_le0, frac_ge0), BOOTSTRAP_P_DECIMALS), where the fractions
# are counts over n_boot resamples. Two separate floors follow, and a p of 0.0
# means "below whichever binds", NEVER "essentially zero".
BOOTSTRAP_P_DECIMALS = 4


def bootstrap_p_floor(n_boot: int) -> float:
    """Smallest NON-ZERO p-value a bootstrap of `n_boot` resamples can report.

    Two floors, and the binding one is the larger:

      * RESAMPLE resolution. The tail count is an integer k out of n_boot, and
        the reported p is 2k/n_boot, so the achievable values are
        {0, 2/n_boot, 4/n_boot, ...}. Nothing lives between 0 and 2/n_boot.
      * REPORTING precision. p is rounded to BOOTSTRAP_P_DECIMALS, so anything
        under 5e-5 prints as 0.0 however many resamples were drawn. Past
        n_boot = 2/1e-4 = 20_000 extra resamples buy no visible resolution.

    Use it as an UPPER BOUND on the true p of a cell whose p came back 0.0: the
    count was zero, so the true tail is somewhere below this, and how far below
    is exactly what the bootstrap did not measure. A threshold beneath this
    floor cannot be cleared on evidence — only by the floored value sorting
    first. Callers that gate on a p must compare the threshold against this.
    """
    resample = 2.0 / n_boot if n_boot > 0 else 1.0
    reporting = 10.0 ** (-BOOTSTRAP_P_DECIMALS)
    return max(resample, reporting)


def benjamini_hochberg_detail(pvalues: list[float | None], q: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR with the step-up CRITICAL VALUE exposed.

    Same rejections as benjamini_hochberg (which delegates here, so the two can
    never drift), plus the number the rejections actually rest on:

        {"survives": [bool], "m": int, "k_max": int, "critical_value": float|None}

    critical_value = (k_max/m)*q, the largest rank's threshold — and under BH's
    step-up rule that is the effective cut for the WHOLE rejected set, not just
    for the cell sitting at k_max. It is None when nothing is rejected.

    Why a caller wants it: a p estimated by simulation carries a resolution
    floor (bootstrap_p_floor), and a rejection is only evidence-backed when the
    critical value sits AT OR ABOVE that floor. Below it, a p that floored at
    0.0 is rejected for the arithmetic reason that 0 <= anything — the same
    verdict it would get if its true p were ten times the threshold.
    """
    indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    m = len(indexed)
    out = [False] * len(pvalues)
    if m == 0:
        return {"survives": out, "m": 0, "k_max": 0, "critical_value": None}
    indexed.sort(key=lambda t: t[1])
    # Largest rank k (1-based) with p_(k) <= (k/m)*q; reject all up to k.
    k_max = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * q:
            k_max = rank
    for rank, (orig_i, _) in enumerate(indexed, start=1):
        if rank <= k_max:
            out[orig_i] = True
    return {"survives": out, "m": m, "k_max": k_max,
            "critical_value": (k_max / m) * q if k_max else None}


def benjamini_hochberg(pvalues: list[float | None], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control across a family of tests.

    Returns a pass/fail flag per input position (True = reject null / survives at
    false-discovery-rate q). None p-values (untested cells) are never rejected.
    With ~35 signal×horizon cells at nominal 95%, ~1-2 pass by chance; BH keeps
    the EXPECTED false-discovery fraction among the passers at q, which is the
    right family-wide honesty gate for a trust-first product.

    NOTE for callers whose p-values come from a BOOTSTRAP: this function cannot
    tell a resolved small p from one that floored at 0.0 for want of resamples,
    and 0.0 clears every threshold by arithmetic. Gate on
    benjamini_hochberg_detail()["critical_value"] vs bootstrap_p_floor(n_boot)
    as well — see indicator_proof._resolve_fdr.
    """
    return benjamini_hochberg_detail(pvalues, q)["survives"]


# --------------------------------------------------------------------------
# Selection-aware inference — deflating a Sharpe for the trials that found it
# (Bailey & López de Prado 2014, "The Deflated Sharpe Ratio"; Bailey, Borwein,
#  López de Prado & Zhu 2015, "The Probability of Backtest Overfitting")
# --------------------------------------------------------------------------
EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni, E[max] of Gaussian order stats

_NORM = NormalDist()


def sharpe_moments(returns: list[float]) -> dict:
    """Per-period Sharpe plus the higher moments PSR needs, from a raw series.

    Returns {sr, t, skew, kurt} — sr = mean/sample-std (NOT annualised; keep
    every trial at the same period frequency), skew/kurt from population
    central moments (kurt non-excess: a normal reads 3.0). All None when the
    series is too short (< 3) or has zero variance — no verdict without spread.
    """
    t = len(returns)
    if t < 3:
        return {"sr": None, "t": t, "skew": None, "kurt": None}
    mean = math.fsum(returns) / t
    d = [x - mean for x in returns]
    m2 = math.fsum(x * x for x in d) / t
    if m2 <= 0:
        return {"sr": None, "t": t, "skew": None, "kurt": None}
    m3 = math.fsum(x * x * x for x in d) / t
    m4 = math.fsum(x * x * x * x for x in d) / t
    sample_std = math.sqrt(m2 * t / (t - 1))
    return {
        "sr": mean / sample_std,
        "t": t,
        "skew": m3 / m2**1.5,
        "kurt": m4 / (m2 * m2),
    }


def probabilistic_sharpe(sr: float, t: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float | None:
    """PSR: P[true Sharpe > sr_benchmark | observed sr over t periods].

    The plain z-test on a Sharpe assumes normal returns and a long sample; PSR
    keeps the same shape but widens the estimator's standard error for
    skewness and fat tails (kurt is non-excess), so a Sharpe built on a lucky
    tail stops looking significant. None when t < 2 or the variance-inflation
    term goes non-positive (moments outside the approximation's domain).
    """
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if t < 2 or denom <= 0:
        return None
    z = (sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)
    return _NORM.cdf(z)


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """E[max estimated SR] across n_trials strategies whose TRUE Sharpe is zero.

    The selection hurdle: try N configurations whose estimated Sharpes scatter
    with cross-trial variance var_trials and the best of pure noise still
    shows ≈ this Sharpe. Asymptotic Gaussian order-statistic form (the DSR
    paper's E[max_N]); 0.0 when there was no real selection (n_trials <= 1)
    or no dispersion to select from (var_trials <= 0).

    Treating correlated trials (e.g. one signal at 5 horizons) as independent
    OVERSTATES the hurdle — callers get a conservative deflation, which is the
    right direction for a trust-first grader.
    """
    if n_trials <= 1 or var_trials <= 0:
        return 0.0
    sd = math.sqrt(var_trials)
    return sd * (
        (1.0 - EULER_GAMMA) * _NORM.inv_cdf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(returns: list[float], n_trials: int, var_trials: float) -> dict:
    """DSR: does the SELECTED strategy's Sharpe clear the noise-max hurdle?

    PSR answers "is this Sharpe real?"; DSR answers the honest version of the
    question after a search: "is it real GIVEN that it is the best of
    n_trials attempts?" — PSR evaluated against sr_star = expected_max_sharpe
    instead of zero. DSR ≥ 0.95 ⇒ significant at 5% even after paying for the
    selection. Returns {sr, t, skew, kurt, sr_star, psr, dsr, n_trials};
    psr is the undeflated (single-trial) reference so the cost of selection
    is visible as psr − dsr.
    """
    mom = sharpe_moments(returns)
    out = {**mom, "sr_star": None, "psr": None, "dsr": None, "n_trials": n_trials}
    if mom["sr"] is None:
        return out
    sr_star = expected_max_sharpe(n_trials, var_trials)
    out["sr_star"] = round(sr_star, 4)
    psr = probabilistic_sharpe(mom["sr"], mom["t"], mom["skew"], mom["kurt"], 0.0)
    dsr = probabilistic_sharpe(mom["sr"], mom["t"], mom["skew"], mom["kurt"], sr_star)
    out["psr"] = round(psr, 4) if psr is not None else None
    out["dsr"] = round(dsr, 4) if dsr is not None else None
    out["sr"] = round(mom["sr"], 4)
    out["skew"] = round(mom["skew"], 4)
    out["kurt"] = round(mom["kurt"], 4)
    return out


def _sr_from_aggregates(s1: float, s2: float, n: int) -> float:
    """Sharpe from cached (Σx, Σx², n) — the O(1) inner step of pbo_cscv.

    Flat/degenerate series (n < 2 or zero variance) rank at -inf: a strategy
    that never varies has no defensible Sharpe and must never win selection.
    """
    if n < 2:
        return float("-inf")
    mean = s1 / n
    var = (s2 - s1 * s1 / n) / (n - 1)
    if var <= 0:
        return float("-inf")
    return mean / math.sqrt(var)


def pbo_cscv(matrix: list[list[float]], n_blocks: int = 10,
             block_ids: list[int] | None = None,
             focus: int | None = None) -> dict:
    """Probability of Backtest Overfitting via combinatorially symmetric CV.

    `matrix` is trials × periods: one aligned per-period return series per
    trial (a trial that is flat on a period contributes 0.0 there). Periods
    are cut into n_blocks CONTIGUOUS blocks (or grouped by an explicit
    `block_ids` label per period — the caller's hook for purging/embargoing
    boundary overlap before the split). For every C(n_blocks, n_blocks/2)
    train/test bisection: rank trials by in-sample Sharpe, take the IS best,
    find its OUT-of-sample relative rank ω = r/(N+1), λ = logit(ω).

    PBO = fraction of bisections with λ < 0 — how often the family's in-sample
    winner lands below median out of sample. ≈ 0.5 means selecting the best
    backtest carries no information; near 0 means the winner keeps winning.

    Returns {pbo, n_combos, n_trials, n_periods, n_blocks, median_logit,
    prob_oos_loss (IS-best's OOS Sharpe < 0), focus: {picked_frac,
    oos_rank_median} for the `focus` trial index or None}. pbo is None when
    the setup is degenerate (< 2 trials, odd/insufficient blocks).
    """
    n_trials = len(matrix)
    if n_trials < 2 or n_blocks < 2 or n_blocks % 2:
        return {"pbo": None, "n_combos": 0, "n_trials": n_trials, "n_periods": 0,
                "n_blocks": n_blocks, "median_logit": None, "prob_oos_loss": None,
                "focus": None}
    n_periods = len(matrix[0])
    if any(len(row) != n_periods for row in matrix) or n_periods < 2 * n_blocks:
        return {"pbo": None, "n_combos": 0, "n_trials": n_trials, "n_periods": n_periods,
                "n_blocks": n_blocks, "median_logit": None, "prob_oos_loss": None,
                "focus": None}

    if block_ids is None:
        # Even contiguous partition; the first (n_periods % n_blocks) blocks
        # absorb the remainder one period each.
        base, rem = divmod(n_periods, n_blocks)
        block_ids = []
        for b in range(n_blocks):
            block_ids.extend([b] * (base + (1 if b < rem else 0)))
    labels = sorted(set(block_ids))
    if len(block_ids) != n_periods or len(labels) != n_blocks:
        return {"pbo": None, "n_combos": 0, "n_trials": n_trials, "n_periods": n_periods,
                "n_blocks": n_blocks, "median_logit": None, "prob_oos_loss": None,
                "focus": None}
    b_index = {lab: i for i, lab in enumerate(labels)}

    # Cache per-(block, trial) Σx / Σx² / n so each bisection is scalar sums.
    s1 = [[0.0] * n_trials for _ in range(n_blocks)]
    s2 = [[0.0] * n_trials for _ in range(n_blocks)]
    cnt = [0] * n_blocks
    for p, lab in enumerate(block_ids):
        b = b_index[lab]
        cnt[b] += 1
        for i in range(n_trials):
            x = matrix[i][p]
            s1[b][i] += x
            s2[b][i] += x * x

    logits: list[float] = []
    oos_losses = 0
    picked = 0
    focus_ranks: list[float] = []
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    for train in combos:
        train_set = set(train)
        test = [b for b in range(n_blocks) if b not in train_set]
        is_sr = []
        oos_sr = []
        for i in range(n_trials):
            is_sr.append(_sr_from_aggregates(
                sum(s1[b][i] for b in train), sum(s2[b][i] for b in train),
                sum(cnt[b] for b in train)))
            oos_sr.append(_sr_from_aggregates(
                sum(s1[b][i] for b in test), sum(s2[b][i] for b in test),
                sum(cnt[b] for b in test)))
        best = max(range(n_trials), key=lambda i: is_sr[i])
        # OOS relative rank of the IS winner: 1 = worst … N = best (ties count
        # half so two identical trials share a rank instead of both inflating).
        r = 1 + sum(1 for v in oos_sr if v < oos_sr[best]) \
            + 0.5 * (sum(1 for v in oos_sr if v == oos_sr[best]) - 1)
        omega = r / (n_trials + 1)
        logits.append(math.log(omega / (1 - omega)))
        if oos_sr[best] < 0:
            oos_losses += 1
        if focus is not None:
            if best == focus:
                picked += 1
            rf = 1 + sum(1 for v in oos_sr if v < oos_sr[focus]) \
                + 0.5 * (sum(1 for v in oos_sr if v == oos_sr[focus]) - 1)
            focus_ranks.append(rf)

    logits.sort()
    n_c = len(logits)
    median_logit = logits[n_c // 2] if n_c % 2 else (logits[n_c // 2 - 1] + logits[n_c // 2]) / 2
    focus_out = None
    if focus is not None and focus_ranks:
        focus_ranks.sort()
        m = len(focus_ranks)
        med = focus_ranks[m // 2] if m % 2 else (focus_ranks[m // 2 - 1] + focus_ranks[m // 2]) / 2
        focus_out = {"picked_frac": round(picked / n_c, 3), "oos_rank_median": med,
                     "n_trials": n_trials}
    return {
        "pbo": round(sum(1 for x in logits if x < 0) / n_c, 4),
        "n_combos": n_c,
        "n_trials": n_trials,
        "n_periods": n_periods,
        "n_blocks": n_blocks,
        "median_logit": round(median_logit, 4),
        "prob_oos_loss": round(oos_losses / n_c, 4),
        "focus": focus_out,
    }


# --- Exact-level tests for overlapping-window instruments -------------------
# Added 2026-08-08 after the rotation sentinel's re-registration. The moving-
# block bootstrap above is correctly calibrated on the historical panel (~1100
# dates), but a LIVE instrument reaches its floor at a few dozen dates, and
# there the bootstrap's asymptotics have not arrived: at n = 11 with
# block_len = 10 its realised false-positive rate under a true null measures
# 69%, against a nominal 5%. Widening the interval does not fix that — the
# failure is that a resample of one block carries no information about the
# sampling distribution. What fixes it is a test whose level holds BY
# CONSTRUCTION at any n, which is what these two provide.


def _block_means(values: Sequence[float], block_len: int) -> list[float]:
    """Chop a DATE-ORDERED series into NON-OVERLAPPING blocks and average each.

    Non-overlapping is the whole point: with a `block_len`-day forward window,
    two observations more than `block_len` apart share no future, so block
    means are the coarsest unit that is plausibly independent. A trailing
    partial block is DROPPED rather than kept — a short block has a larger
    variance than a full one, and mixing the two silently breaks the
    exchangeability the sign-flip test below relies on.
    """
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    k = len(values) // block_len
    return [math.fsum(values[i * block_len:(i + 1) * block_len]) / block_len
            for i in range(k)]


def sign_flip_block_test(values: Sequence[float], block_len: int,
                         n_perm: int = 20000, seed: int = 0,
                         alpha: float = 0.05, ci: bool = True) -> dict:
    """Randomisation test on the mean of a series with overlapping windows.

    Chops the series into non-overlapping blocks (`_block_means`), then tests
    H0: the block means are distributed symmetrically about zero, by
    enumerating or sampling sign flips. Under that null the test is EXACT at
    any number of blocks — its level does not depend on an asymptotic
    approximation, which is precisely what fails at live-instrument sample
    sizes.

    The interval is obtained by INVERTING the test: it is the set of shifts the
    same randomisation would not reject, so interval and p-value can never
    disagree (a real failure mode when a CI comes from one procedure and a
    p-value from another).

    Enumerates all 2**k sign patterns when that is cheaper than `n_perm`, so
    the small-k case — the one that matters — is exact rather than simulated.

    Returns mean, ci_low, ci_high, p_value, n_blocks, n_obs_used, exact.
    `p_value` is None when there are no blocks; note the smallest attainable
    two-sided p on k blocks is 2/2**k, so k < 6 cannot reach 0.05 at all — that
    is a property of the evidence, not a defect, and callers should floor on
    `n_blocks` accordingly.
    """
    means = _block_means(values, block_len)
    k = len(means)
    if k == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_value": None,
                "n_blocks": 0, "n_obs_used": 0, "block_sd": None, "exact": False}

    observed = math.fsum(means) / k
    exact = k <= 20 and 2 ** k <= n_perm

    def _signs():
        if exact:
            for combo in itertools.product((1.0, -1.0), repeat=k):
                yield combo
        else:
            rng = random.Random(seed)
            for _ in range(n_perm):
                yield tuple(1.0 if rng.random() < 0.5 else -1.0 for _ in range(k))

    patterns = list(_signs())

    def _p_for(shift: float) -> float:
        """Two-sided randomisation p for H0: mean == shift."""
        centred = [m - shift for m in means]
        stat = abs(math.fsum(centred) / k)
        # +1 in numerator and denominator: the observed assignment is itself one
        # of the equally likely sign patterns, so excluding it biases p downward.
        hits = sum(1 for s in patterns
                   if abs(math.fsum(c * sg for c, sg in zip(centred, s))) / k >= stat - 1e-12)
        return min(1.0, hits / len(patterns)) if exact else min(1.0, (hits + 1) / (len(patterns) + 1))

    p_value = _p_for(0.0)

    # Invert the test for the interval: bisect outward from the observed mean
    # for the largest shift still accepted at `alpha`. Monotone in |shift|.
    spread = max((max(means) - min(means)), abs(observed), 1e-9) * 4 + 1e-9

    def _edge(direction: int) -> float | None:
        lo, hi = 0.0, spread
        if _p_for(observed + direction * hi) > alpha:
            return None  # unbounded within a sane search window
        for _ in range(40):
            mid = (lo + hi) / 2
            if _p_for(observed + direction * mid) > alpha:
                lo = mid
            else:
                hi = mid
        return observed + direction * lo

    # `ci=False` skips the ~80 extra randomisation passes the inversion costs.
    # Used by sign_flip_mde, which needs only the p-value and calls this
    # thousands of times; with the interval on, an MDE would take minutes.
    ci_low, ci_high = (_edge(-1), _edge(1)) if ci else (None, None)
    # block_sd rides along so callers can size an MDE without re-chopping the
    # series through a private helper — the two must never disagree about what
    # a block is.
    sd = (math.sqrt(math.fsum((m - observed) ** 2 for m in means) / (k - 1))
          if k > 1 else None)
    return {
        "mean": round(observed, 4),
        "ci_low": round(ci_low, 4) if ci_low is not None else None,
        "ci_high": round(ci_high, 4) if ci_high is not None else None,
        "p_value": round(p_value, 4),
        "n_blocks": k,
        "n_obs_used": k * block_len,
        "block_sd": round(sd, 4) if sd is not None else None,
        "exact": exact,
    }


def sign_flip_mde(block_sd: float, n_blocks: int, alpha: float = 0.05,
                  power: float = 0.80, n_sim: int = 2000, seed: int = 0) -> float | None:
    """Smallest true mean `sign_flip_block_test` would detect `power` of the time.

    Published BESIDE every verdict, so that a "coin_flip" from an instrument
    that cannot see anything smaller than 3pp is not read as evidence of
    absence. That misreading is the failure this whole layer exists to prevent.

    Returns None when no effect is detectable at any size — which is the honest
    answer below 6 blocks, where the smallest attainable p (2/2**k) exceeds
    0.05 no matter how large the effect.
    """
    if n_blocks < 1 or block_sd <= 0:
        return None
    if 2.0 / (2 ** n_blocks) > alpha:
        return None

    rng = random.Random(seed)

    def _power_at(shift: float) -> float:
        hits = 0
        for _ in range(n_sim):
            sample = [rng.gauss(shift, block_sd) for _ in range(n_blocks)]
            # The test statistic on already-blocked data: reuse the same
            # machinery with block_len 1 so the two paths cannot drift apart.
            # 512 draws resolve p to ~0.002, which is ample when the only
            # question is "p <= alpha". Enumerating 2**k here made the k=20
            # case take 20s per scan, on a path that runs 11x per weekday.
            r = sign_flip_block_test(sample, 1, n_perm=min(2 ** n_blocks, 512),
                                     alpha=alpha, ci=False)
            if r["p_value"] is not None and r["p_value"] <= alpha:
                hits += 1
        return hits / n_sim

    lo, hi = 0.0, block_sd * 6
    if _power_at(hi) < power:
        return None
    for _ in range(9):
        mid = (lo + hi) / 2
        if _power_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)
