# Does the signal actually work? — a pre-registered honesty audit

*A research note on the momentum scanner in this repository. Every number below is
produced by code in `scanner/` and stored as a committed artifact in `data/`; file
paths are cited inline so the claims are checkable, not decorative.*

---

## TL;DR (the finding, stated against interest)

I built a scanner that fires momentum signals. Then I spent as much effort trying to
**prove the signals are noise** as building them. The honest result:

1. Of **35** signal×horizon cells tested, **30 are statistically indistinguishable
   from a coin flip.** Five clear zero as nominal edges; after correcting for the fact
   that testing 35 things will hand you ~1.8 false positives by chance
   (Benjamini–Hochberg FDR, q = 0.05), **exactly one survives.**
2. The survivor is **overnight gap ≥ +2%, held 21 sessions**: a market-demeaned
   **+1.60%** edge, 95% CI **[0.72, 2.43]**, and — importantly — it holds in *both* a
   risk-on and a risk-off regime, so it is not beta wearing a costume.
3. **That same edge does not survive contact with execution.** Turned into a 5-slot
   portfolio a real account could actually place, it returns **1.432× vs SPY's
   1.454×** over the window — it *loses* — its entire profit is the top 5 trades, and
   forcing six of the winners to −60% (a crude survivorship correction) sinks it to
   **0.804×**.
4. Two selection-aware audits split the verdict — and the split is the finding. **PBO
   (CSCV) = 0.05**: across 252 train/test bisections of history, the family's in-sample
   winner keeps winning out of sample (median OOS rank: top of 35) — the *choice* of
   signal is real. But the **Deflated Sharpe Ratio = 0.52**: after paying for the fact
   that it was the best of 35 trials, its *absolute* Sharpe sits exactly at the hurdle
   pure noise would set. The right signal was selected; the size of its edge is unproven.
5. The only test that matters is therefore the future. A **live forward ledger** grades
   every call in real time; the matured short-horizon calls sit at **143 hit / 165 miss**
   (46%, as of 2026-07-18 — the Wilson CI straddles 50%: a coin flip) and the decisive
   21-day verdict is **pre-committed for ~September 2026** with a coded kill-switch.

The point of this project is not "an AI that beats the market." It is a working
apparatus for finding out whether a signal is real **and reporting the answer even when
the answer is no.**

---

## 1. Why this document exists

Most tools in this space are optimised to *look* like they work: a confident pick, a
green backtest curve, a big hit-rate. That is the form of rigour without the substance —
what Feynman called cargo-cult science. The failure mode is specific and well studied in
finance: try enough variations and one will look brilliant by luck (selection under
multiple testing), then present the survivor as if it were the hypothesis you started
with.

This note is the antidote. It states one pre-registered question, applies the standard
statistical machinery for *not fooling yourself*, and reports what came back — including
the parts that argue against the product.

## 2. The question (pre-registered and falsifiable)

> **Do the scanner's technical signals earn a positive, market-neutral forward return
> that survives multiple-testing correction, out-of-sample splitting, and realistic
> execution costs — and does the one signal that does survive remain an edge when a real
> account has to trade it?**

Falsifiable by construction: each signal gets a confidence interval against a null of
zero market-demeaned return, and a plain verdict — `edge`, `adverse`, or `coin_flip`
(`scanner/stats.py`, `zero_crossing_verdict`). "Looks good" is not an available answer.

## 3. Data & universe (biases named up front)

| | Single-signal proof | Executable-portfolio proof |
|---|---|---|
| Source | Alpaca IEX daily OHLCV | same |
| Symbols | **510** | **513** |
| Window | 2021-08-30 → 2026-07-09 (~5y) | 2024-08-08 → 2026-07-09 |
| Universe pin | `symbols_sha256 f767eaa3…` | frozen list, `data/proof_universe.json` |
| Adjustment | split + dividend adjusted | same |

Known biases, stated before the results rather than buried after them
(`data/indicator_performance.json → caveats`, `data/portfolio_backtest.json →
in_sample_banner`):

- **Survivorship.** The universe is *current* membership, so delisted losers are absent
  — every edge is biased **up** and every drawdown is biased **shallow**. Unfixable on
  this data source, so it is quantified instead (§6.3 stress test).
- **One regime.** ~2021–2026 is effectively a single macro episode; the OOS split and
  regime split (§4) are partial mitigations, not a cure.
- **In-sample discovery.** The backtest compounds the *same* data that discovered the
  edge. It is a translation of a hypothesis, not new evidence — which is exactly why the
  real trigger is the live ledger (§8), not this document.
- **Indicative vs tradeable inputs.** RVOL and gap use IEX partial-tape volume and a
  non-auction open, so they are flagged `indicative_only`; close-based RSI/MACD are not.

## 4. Method — the machinery for not fooling yourself

All of it is dependency-free and unit-tested (`scanner/stats.py`, `tests/test_stats.py`).
The names in **bold** are the canonical techniques, so the mapping to the literature is
explicit:

- **No look-ahead.** Signal reads `close[i]`; entry is next bar's `open[i+1]`; exit is
  `close[i+h]`. The signal can never see the bar it trades on.
- **Beta-neutral benchmark.** Every forward return is *market-demeaned* by the same-day
  universe mean. An "edge" is return **above the market that day**, not with it.
- **Overlap-aware inference (moving-block bootstrap).** Overlapping forward windows and
  whole-universe co-firing are massively autocorrelated, so a textbook CI on *n*
  ticker-days reads several times too tight. Instead the CI resamples **circular moving
  blocks of calendar dates**, `block_len = max(5, horizon)`, and reports the honest
  effective sample size `n_indep` (≈ number of independent day-blocks) next to the
  inflated raw *n*.
- **Multiple-testing control (Benjamini–Hochberg FDR).** Across the family of 35
  signal×horizon cells at q = 0.05, BH keeps the expected false-discovery fraction among
  the survivors bounded — so we do not bless the one or two cells that pass by chance.
- **Out-of-sample split.** A held-out 30% tail (split 2025-01-23); a cell only "holds
  OOS" if the full-window edge did not flip adverse there. Underpowered on one regime,
  and reported as such.
- **Regime split.** `risk_on` = SPY above *both* its 50- and 200-day SMA, else `other`.
  An edge that exists only in `risk_on` "is beta in disguise" and is discounted.
- **Pre-registration.** The portfolio study's parameters (signal, threshold, hold, slots,
  sizing, costs, benchmark) are frozen and hashed (`prereg_hash 3383e1d15a8f…`) *before*
  the run, so the spec cannot be quietly tuned to the result.
- **Selection-aware deflation (DSR) and Probability of Backtest Overfitting (PBO/CSCV).**
  BH corrects p-values for multiplicity; these two price the *selection itself* — the
  Sharpe-space hurdle the search created, and the out-of-sample stability of the family's
  ranking. Implemented in `scanner/stats.py` (`deflated_sharpe`, `pbo_cscv`), applied in §7.

## 5. Result A — the single-signal screen

*(`data/indicator_performance.json`; 35 cells; market-demeaned; moving-block CIs.)*

| Outcome | Cells |
|---|---|
| `coin_flip` (CI straddles 0) | **30** |
| nominal `edge` (CI > 0) | 5 |
| **survives BH-FDR (q = 0.05)** | **1** |
| expected false positives across family | ~1.8 |

**The lone survivor:**

| Signal | Horizon | Market-demeaned edge | 95% CI | `n_indep` | risk_on | other |
|---|---|---|---|---|---|---|
| `gap_up` — 跳空 ≥ +2% · 高开 (long) | 21d | **+1.60%** | **[0.72, 2.43]** | 53 | +1.93% `edge` | +1.28% `edge` |

Two things make this more than a lucky cell: the interval clears zero comfortably, and it
stays an edge in **both** regimes — the check that most often kills a "signal" (it turns
out to be long-the-market) does not kill this one.

## 6. Result B — does the edge survive being *traded*?

A statistically real per-event edge and a strategy you can run are different things. The
pre-registered portfolio (`data/portfolio_backtest.json`) trades exactly that gap signal:
enter, hold 21 sessions, 5 concurrent slots, random selection among that day's qualifiers,
200 Monte-Carlo seeds for selection luck.

### 6.1 It loses to SPY on the executable spec

| Spec | Strategy | SPY | Per-trade net (95% CI) | Excess vs SPY (95% CI) | Verdict |
|---|---|---|---|---|---|
| Idealised — open entry, 0.5% cost | 1.546× | 1.461× | +2.87% [−0.71, +6.41] | +1.94% [−1.31, +5.16] | `coin_flip` |
| **Executable — next-close entry, 1.0% cost** | **1.432×** | **1.454×** | +2.71% [−0.54, +6.57] | +2.33% [−0.59, +5.84] | `coin_flip` |

The executable row is the one a human could actually place (the broker has no
market-on-open order). It **trails SPY**, and the per-trade and excess verdicts are both
coin flips — the CIs straddle zero.

### 6.2 The profit is a handful of trades

On the executable spec the **top 5 trades account for 100%** of the P&L; remove the single
best trade and the multiple falls from 1.432× to **1.261×**. This is not a broad edge; it
is a fat tail.

### 6.3 It is fragile to the survivorship bias we already flagged

Forcing the biggest winners down to −60% (a crude proxy for the delisted names the
current-membership universe cannot contain): two winners → 1.368×; **six winners →
0.804×** (a 20% loss). The headline depends on names that survived *because* they
survived.

### 6.4 Where the edge *is* real: per-event, not per-portfolio

The honest positive result lives at the per-**event** level, market-demeaned
(`portfolio_backtest.json → marginal_band`):

| Horizon | gap ∈ [2%, 3%) | gap ≥ 3% | difference |
|---|---|---|---|
| 21d | **+2.27%** [1.54, 3.03] `edge` | **+3.33%** [2.25, 4.39] `edge` | coin_flip |

So the resolution of the apparent contradiction: **the gap signal earns a real
market-neutral premium per event, but at the concentration and frictions of a 5-slot
portfolio, selection luck and costs eat it.** Both statements are true; reporting only the
first would be the lie.

## 7. Result C — is the survivor real *given the search that found it?*

*(`data/overfit_audit.json`, generated by `scripts/run_overfit_audit.py`; machinery in
`scanner/stats.py` — `deflated_sharpe`, `pbo_cscv` — unit-tested in `tests/test_stats.py`.
Same frozen panel as §5: `matches_indicator_proof: true`.)*

Sections 5–6 corrected for multiplicity in *p-value* space (BH-FDR). The
backtest-overfitting canon has two sharper instruments, and both are implemented here.
Deliberately conservative choices throughout: every trial's fired-date demeaned returns
are aggregated to non-overlapping 21-trading-day bins (the family's longest horizon)
so Sharpe scale and independence are uniform; the 35 trials are treated as independent
though they are correlated; and the cross-trial variance that sets the noise hurdle
necessarily includes the family's *real* edge dispersion. All three biases push the
hurdle **up** — this audit can under-claim, never over-claim.

### 7.1 Deflated Sharpe Ratio — the absolute bar

Treat `gap_up@21d` as what it actually is: the *selected best* of 35 trials. The noise
hurdle E[max SR] is what the luckiest of 35 true-zero strategies would show; PSR is then
evaluated **at that hurdle** instead of at zero:

| | value |
|---|---|
| Survivor Sharpe (per 21d bin, 58 bins) | 0.407 (≈ 1.41 annualized) |
| PSR vs 0 — no selection penalty | **0.999** |
| Noise hurdle SR\* — best of 35 null trials | 0.399 |
| **DSR — PSR vs SR\*** | **0.52** |

Undeflated, the Sharpe looks unimpeachable. Deflated, it sits **almost exactly at the
hurdle**: the survivor's absolute Sharpe is about what the luckiest of 35 noise trials
would have shown. Given the conservatisms above, 0.52 is an honest floor, not a
debunking — but it plainly fails the canonical DSR ≥ 0.95 bar. (The naive
overlap-inflated variant reads 0.81 — still short. Reported for contrast, not for use.)

### 7.2 Probability of Backtest Overfitting — the relative bar

CSCV: cut the panel into 10 contiguous blocks (21-day embargo at every block tail, so no
forward window straddles a train/test boundary), and for each of the 252 bisections ask
whether the family's **in-sample winner keeps winning out of sample**:

| | value |
|---|---|
| **PBO** — IS winner below median OOS | **0.048** |
| P[OOS Sharpe of IS winner < 0] | 0.032 |
| Survivor = IS winner in | 68% of bisections |
| Survivor's median OOS rank | **top of 35** |

The opposite verdict — and that is the finding. Rank stability is exceptional: whichever
half of history you select on, the same signal tops the other half.

### 7.3 Purged fold consistency — where the edge lives in time

Five contiguous ~year folds, 21-day purge at each tail, moving-block CI per fold:

| Fold | Window | Edge (95% CI) | Verdict |
|---|---|---|---|
| 1 | 2021-08 → 2022-07 | −0.22% [−2.01, +1.71] | coin_flip |
| 2 | 2022-08 → 2023-07 | +2.89% [+0.34, +6.00] | edge |
| 3 | 2023-08 → 2024-06 | +0.74% [−1.24, +2.80] | coin_flip |
| 4 | 2024-07 → 2025-06 | +1.75% [+0.16, +3.44] | edge |
| 5 | 2025-07 → 2026-06 | +4.53% [+2.36, +6.81] | edge |

4/5 folds positive, 3/5 individually significant — and the one dead fold is the 2021–22
drawdown tape, consistent with §5's regime split. The most recent year is the strongest,
which is encouraging for the live test and exactly the recency a skeptic should discount.

### 7.4 Reading the disagreement

DSR ≈ 0.5 and PBO ≈ 0.05 are not a contradiction; they answer different questions. PBO
validates the **choice** — ordinally, this signal genuinely dominates its family out of
sample. DSR withholds validation of the **level** — cardinally, the absolute Sharpe,
after paying for the search that found it, is not yet distinguishable from selection
luck. Put together: *the right signal was selected; the size of its edge remains
unproven.* Which is precisely the question the live ledger below is pre-registered to
settle.

## 8. The real test — a live, pre-committed forward ledger

In-sample results are a hypothesis. The verdict is out-of-sample and in the future.
Every pick, alert, prediction, golden-cross and catch-list call is logged the moment it
fires and graded against its own forward outcome (`data/ledger.json`, `scanner/state.py`):

- **500** live entries in the rolling window; matured short-horizon calls stand at
  **143 hit / 165 miss** (46%, as of 2026-07-18; Wilson CI [41%, 52%] straddles 50%) — a
  coin flip, in the wild, exactly as §5–6 predict. The `/research` page reads these counts
  live from `ledger.json`, so the site can never quote a stale number even when this
  paragraph does.
- The decisive **21-day** cohort is **not yet matured**; its pre-committed verdict lands
  **~September 2026.**
- **Allocation rule (Popper, in code):** deploy the ≤20% satellite sleeve *only if*
  (a) the per-trade 21d net verdict is not adverse **and** (b) the live ledger's ~Sept-2026
  fired-cohort excess verdict is `edge` with a CI overlapping the backtest. Either fails →
  do not allocate (`portfolio_backtest.json → decision_rules`).
- **Kill-switch:** a separate early-entry catch-list is *already failing* its forward test
  — control-relative net edge of **−4.08% at 10d** (`data/early_entry_performance.json`).
  It carries a coded kill criterion and a review date; a signal that cannot beat "buy any
  gapper" is scheduled for retirement, not quietly kept.

A prediction with a date and a pre-registered pass/fail bar is a falsifiable claim. That is
the difference between a track record and a horoscope.

## 9. Threats to validity (what would make me wrong)

- **Survivorship** biases every in-sample number up — partially bounded in §6.3, not
  eliminated.
- **One macro regime**: the OOS and regime splits are underpowered; a genuine bear market
  is untested.
- **FX and tax** are unmodelled in absolute paths (the vs-SPY comparison is FX-neutral to
  first order; a GBP ISA loses 15% US dividend withholding).
- **Multiplicity beyond these 35 cells**: every *other* variant I ever looked at is an
  implicit test. BH and the DSR hurdle (§7) price the *registered* family's selection;
  no statistic can price undocumented curiosity. The honest mitigations are the frozen
  universe, the pre-registration hashes, and the live ledger — which pays no selection
  penalty at all.

## 10. What I would strengthen next (the honest to-do)

The first version of this note listed the Deflated Sharpe Ratio, purged/embargoed CV,
and PBO as the canonical gaps. They are now built and run (§7) — seeded, unit-tested,
and committed as `data/overfit_audit.json`. What honestly remains:

1. **Strategy-level DSR on the executable portfolio.** §7 deflates the per-event signal;
   deflating the 5-slot portfolio's trade sequence (across its 200 Monte-Carlo selection
   seeds) would price the same search at the level a real account experiences. Needs
   per-seed trade series retained by the backtester.
   *Status (2026-07-20): built and run — the audit's `portfolio_dsr` block
   (`data/overfit_audit.json`) feeds every family member's fired events through the
   identical executable five-slot simulator (close entry, 1.0% RT cost, random fill;
   per-seed series retained as `seed_sr_band`) and evaluates the survivor's daily
   SPY-excess account series against the family's account-level SR dispersion.
   Result: survivor SR 0.022/day (≈0.35 annualized, 1,217 sessions) vs a noise
   hurdle SR\* of 0.64/day → **DSR ≈ 0.00**. The §7 split deepens at account level:
   the choice stays validated (PBO), but the level — once real costs and slot
   capacity price the search (short-hold trials are shredded by fees, and that real
   dispersion raises the hurdle; conservative by construction) — is indistinguishable
   from selection noise. This strengthens the September rule rather than weakening
   it: the pre-registered live ledger is the only evidence that can move the
   allocation decision.*
2. **The same discipline on the live cohort.** When the forward ledger's 21d cohort
   matures (~Sep 2026), run PSR on the *live* series with `n_trials = 1` — the whole
   point of pre-registration is that the live test pays **no selection penalty**. That
   number, not any in this document, is the deployable one.
   *Status (2026-07-20): the analysis itself is now pre-committed —
   `scanner/live_psr.py` compiles daily into `data/live_psr.json` (cohort, horizon,
   date-block aggregation, maturity floors and pass bar all frozen in code, pinned by
   `tests/test_live_psr.py`). Verdicts read `accumulating` until the floors are met, so
   the September number falls out of a script written before the data existed.*
3. **A bear regime.** No statistic here can synthesize one. Fold 1 (2021–22) is the only
   adverse-tape evidence and it reads flat; the coded kill-criterion floors (§8) are the
   standing insurance until the market supplies the missing experiment.

## 11. Reproducibility & provenance

- **Frozen universe** `data/proof_universe.json` (516 symbols, refreshed quarterly by hand
  with a reviewed diff, so membership churn can't flip a verdict between runs).
- **Pre-registration hash** `3383e1d15a8f…`; **universe hash** `f767eaa3…`; bootstrap
  **seed = 0** throughout → byte-reproducible CIs.
- Rerun: `scripts/run_indicator_proof.py`, `scripts/run_portfolio_backtest.py`,
  `scripts/run_axis_proof.py`, `scripts/run_overfit_audit.py`; regen the universe with
  `scripts/gen_proof_universe.py`.
- The §7 audit runs on the byte-identical panel as the committed proof (it asserts the
  `symbols_sha256` match and records it), with bin span, embargo, CSCV blocks, and seed
  all pinned in its `config` block.
- A **commit gate** (`scripts/proof_commit_gate.py`, `tests/test_proof_commit_gate.py`)
  refuses to publish a proof if the universe collapses or a verdict silently flips without
  an explicit override — the results in this repo cannot drift unnoticed.

## 12. Intellectual lineage

The method here is deliberately unoriginal — it is the standard toolkit for not deceiving
yourself in empirical finance, and using it well is the point:

- **R. Feynman, "Cargo Cult Science" (1974)** — *"The first principle is that you must not
  fool yourself — and you are the easiest person to fool."* This project's in-sample
  disclaimers and null results are that principle written in code.
- **M. López de Prado, *Advances in Financial Machine Learning* (2018)** — backtest
  overfitting, purged/embargoed CV, the Deflated Sharpe Ratio, and the discipline that
  most published strategies are false discoveries. The FDR gate, frozen universe, and
  forward ledger are independent implementations of the same immune system; the DSR and
  PBO in §7 are direct implementations of his two flagship diagnostics — and they came
  back split (choice validated, level not), which is more informative than either verdict
  alone.
- **K. Popper** (falsifiability) and **N. Taleb, *Fooled by Randomness*** — the reasons the
  system ships a *coded kill-switch* and a *control cohort* rather than a marketing number.

---

*Generated from committed artifacts under `data/`. If a claim here disagrees with the JSON,
the JSON wins — regenerate and correct this note.*
