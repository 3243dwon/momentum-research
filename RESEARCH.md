# Does the signal actually work? — a pre-registered honesty audit

*A research note on the momentum scanner in this repository. Every number below is
produced by code in `scanner/` and stored as a committed artifact in `data/`; file
paths are cited inline so the claims are checkable, not decorative.*

---

## TL;DR (the finding, stated against interest)

I built a scanner that fires momentum signals. Then I spent as much effort trying to
**prove the signals are noise** as building them. The honest result:

1. Of **35** signal×horizon cells tested, **all 35 are statistically indistinguishable
   from a coin flip** after correcting for the fact that testing 35 things will hand you
   ~1.8 false positives by chance (Benjamini–Hochberg FDR, q = 0.05). **Nothing survives.**
   A few `gap_up` cells clear zero as a nominal edge, against a family bar of 0.0014
   that no cell has passed on any panel since 2026-07-09. *Which* ones changes from one
   weekly regeneration to the next: across the six committed panels from 2026-08-08 to
   2026-10-03 the set changed on three of the five steps, and `gap_up` @10d's CI low
   sat within ±0.04 of zero on all six. This note therefore does not name them. The
   current set is each cell's `verdict` in `data/indicator_performance.json`, and the
   site's `/research` page counts them live.
2. The largest cell is still **overnight gap ≥ +2%, held 21 sessions**: a
   market-demeaned **+1.19%**, 95% CI **[0.16, 2.16]**, `n_indep` 53, graded
   `coin_flip` (panel of 2026-10-03). That interval **crossed zero and re-cleared
   it from one weekly regeneration to the next** — [0.08, 2.11] on the panel of
   2026-08-08, [−0.00, 2.05] on 2026-08-15, [0.04, 2.04] on 2026-08-22 — so *does
   this cell clear zero* is a question whose answer can change weekly at this
   sample size, which is a stronger statement against the cell than either answer
   is for it. It is **not an edge in both regimes** — risk-on is +0.92% CI
   [−0.51, 2.24] (`coin_flip`) and the out-of-sample tail is +1.94% CI
   [−0.40, 4.07] (`coin_flip`). (These are the
   **point-in-time-conditioned** numbers — names added to the index mid-window
   contribute no pre-inclusion history, §3.)

   > **⚠️ This corrects a previous version of this note**, which reported *"32 coin
   > flips, exactly one survives: +1.42%, CI [0.61, 2.32], and it holds in both
   > regimes."* Every one of those figures was correctly computed — on a panel ending
   > **2026-07-09**. Re-running the identical code on the identical frozen universe today
   > returns the numbers above. **The reconstruction of how that claim was born and died
   > is §5.1, and it is a more useful result than the claim ever was.**
   >
   > **⚠️ And it has now happened a second time, which is the point.** The figures
   > above were, when this notice was written, the panel of **2026-08-15** (`data/indicator_performance.json`,
   > `git_sha` a665884). The version this note carried until then quoted @21d at
   > +1.12%, CI [0.08, 2.11], p = 0.037 — correctly computed on the panel of
   > 2026-08-08. One weekly regeneration moved @21d's interval across zero AND
   > promoted @3d into the nominal slot it vacated. The *identity* of "the one cell
   > that still clears zero" is therefore not stable at weekly resolution, which is a
   > stronger statement against the cell than any single p-value is for it.
   >
   > `.github/workflows/quant-proof.yml` regenerated that artifact every Saturday until
   > the owner deleted it on 2026-10-09 (deletion batch 5; the artifact is static at its
   > 2026-10-03 panel, code at `fdfe3266`) and no workflow regenerates a paragraph. Read
   > the figures from the artifact itself,
   > with its `generated_at` and `git_sha`, rather than from this sentence, which went
   > stale twice. (`quant/evidence.py`, which printed them, was retired 2026-10-02.)
3. **That same edge does not survive contact with execution.** Turned into a 5-slot
   portfolio a real account could actually place, it returns **1.432× vs SPY's
   1.454×** over the window — it *loses* — its entire profit is the top 5 trades, and
   stressing 5% of trades to −60% sinks it to **0.804×**. That 5% is no longer a guess:
   the observed index-removal-for-weakness rate over the same window is **5.2%**
   (26 of 503 names, §6.3), so the 0.804× cell is the calibrated survivorship bound.
4. Two selection-aware audits split the verdict — and the split is the finding. **PBO
   (CSCV) = 0.12**: across 252 train/test bisections of history, the family's in-sample
   winner keeps winning out of sample (median OOS rank: top of 35) — the *choice* of
   signal is real. But the **Deflated Sharpe Ratio = 0.41**: after paying for the fact
   that it was the best of 35 trials, its *absolute* Sharpe sits below the hurdle
   pure noise would set. The right signal was selected; the size of its edge is unproven.
5. The only test that matters is therefore the future. A **live forward ledger** grades
   every call in real time; on 2026-10-04 its matured calls read **157 hit / 179 miss**
   (46.7% — Wilson CI [41.5%, 52.1%] straddles 50%: a coin flip). The coded kill-switch
   has since **fired** on the early-entry tier, retiring it (§8). And the pre-committed
   ~September verdict grades that same retired tier, **not** the `gap_up` @21d cell of
   item 2 — so that cell still has no live forward test. §8 states this plainly rather
   than letting the September date imply a decision it cannot deliver. Since the
   correction in §5.1, "build a forward test for the survivor" is no longer the obvious
   next move: a forward instrument on a ~+1.2pp effect with a per-block sd of ~2.6pp
   needs roughly **28 independent blocks — about 2.3 years — for 80% power**, and the
   floor as drafted (10 matured entry-dates ≈ half of *one* block) would fill in two
   weeks and emit a confident-looking verdict that can falsify nothing.

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
| Source | Alpaca SIP (consolidated) daily OHLCV; IEX before 2026-09-22 (#394) | Alpaca IEX daily OHLCV |
| Symbols | the 516 frozen in `data/proof_universe.json`; **511** had bars on the 2026-10-03 panel | **513** |
| Window | a trailing 1,825-calendar-day fetch, re-run weekly; graded dates 2021-11-23 → 2026-10-02 on the 2026-10-03 panel | 2024-08-08 → 2026-07-09 |
| Universe pin | the frozen list; `universe.symbols_sha256` fingerprints which of it had bars, so it moves with that set (five values since 2026-07-10; `38024bfe…` on 2026-10-03) | frozen list, `data/proof_universe.json` |
| Adjustment | split + dividend adjusted | same |

Known biases, stated before the results rather than buried after them
(`data/indicator_performance.json → caveats`, `data/portfolio_backtest.json →
in_sample_banner`):

- **Survivorship.** Two distinct halves, treated separately since 2026-07-21:
  - *Backfill (fixed).* The panel is **point-in-time conditioned**: the committed S&P
    change log (`data/sp500_changes.json`, scraped from the index's published
    constituent changes) drives a per-(name, date) membership filter in
    `scanner/pit.py`, so names added to the index mid-window contribute no
    pre-inclusion history and the beta baseline is computed on members only. The
    window rolls weekly, so both counts move: on the 2026-10-03 panel it was 77 names
    (`sp500_changes.json` read against that panel's dates) and 7.5% of panel rows
    (`universe.pit`: 569,940 of 616,219 kept). NDX-only names carry no event history
    and pass unconstrained — stated, not guessed.
  - *Missing losers (bounded).* Delisted names' bars do not exist on this data source,
    so their absence still biases every edge **up** and every drawdown **shallow**.
    That half is now **calibrated** rather than guessed: the observed in-window
    removal-for-weakness rate sets the §6.3 stress fraction.
- **One regime.** ~2021–2026 is effectively a single macro episode; the OOS split and
  regime split (§4) are partial mitigations, not a cure.
- **In-sample discovery.** The backtest compounds the *same* data that discovered the
  edge. It is a translation of a hypothesis, not new evidence — which is exactly why the
  real trigger is the live ledger (§8), not this document.
- **Indicative vs tradeable inputs.** Only RVOL is flagged `indicative_only`
  (`_INDICATIVE_KEYS` in `scanner/indicator_proof.py`), for IEX partial-tape volume.
  Gap was flagged for its IEX non-auction open until a 2026-07 robustness study found
  its edge survives next-close entry and is strongest in the most-liquid tercile, so
  `gap_up` reads `indicative_only: false`; the artifact's `caveats` string still names
  both, and is stale. Panels from 2026-09-26 are fetched from the consolidated SIP tape
  (#394), which removes the partial-tape reason for the RVOL flag too; the flag has not
  been revisited.

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
- **Out-of-sample split.** A held-out 30% tail (`out_of_sample.split_date`, which moves
  forward with the window each regeneration: 2025-04-21 on the 2026-10-03 panel); a cell only "holds
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
| tested | **35** |
| **survives BH-FDR (q = 0.05)** | **0**, on every panel since 2026-07-09 |
| expected false positives across family | ~1.8 |
| nominal `edge` (CI > 0, uncorrected) | 1 to 3 `gap_up` cells across the six panels from 2026-08-08 to 2026-10-03, a different set on three of five weekly steps; the rest are `coin_flip` |

**There is no survivor. The best cell:**

| Signal | Horizon | Market-demeaned edge | 95% CI | p | `n_indep` | risk_on | other | OOS |
|---|---|---|---|---|---|---|---|---|
| `gap_up` — 跳空 ≥ +2% · 高开 (long) | 21d | +1.19% | [0.16, 2.16] | 0.024 | 53 | +0.92% `coin_flip` | +1.50% `edge` | +1.94% `coin_flip` |

*(Panel of 2026-10-03, `git_sha` f9cd7fa.)* On that panel the interval cleared zero; on
the panel of 2026-08-15 it did not (TL;DR item 2 lists the crossings). It does not stay an
edge in both regimes: the check that most often kills a "signal" (it turns out to be
long-the-market) does not pass, since what edge there is sits in `other` and not in
`risk_on`. The family bar is 0.05/35 = **0.0014**; on that panel this cell's p was 17×
above it.

### 5.1 The correction — how a survivor was born, and how it died

An earlier version of this note reported **+1.42%, CI [0.61, 2.32], "exactly one survives",
"an edge in both regimes."** Those numbers were correctly computed from a panel ending
**2026-07-09** and published on 2026-07-21. They are not true of the same panel today, and
the reason is worth more than the original claim.

**What changed — five cutoffs of the same frozen universe, same seed, same code:**

| corpus ends | edge | 95% CI | p | survives FDR | family passed |
|---|---|---|---|---|---|
| 2026-06-30 | +1.490% | [0.656, 2.321] | 0.000 | ✅ | 2 of 35 |
| 2026-07-09 | +1.405% | [0.561, 2.261] | 0.000 | ✅ | **1** of 35 ← published |
| 2026-07-16 | +1.300% | [0.444, 2.260] | 0.006 | ❌ | 0 of 35 |
| 2026-07-23 | +1.185% | [0.281, 2.166] | 0.012 | ❌ | 0 of 35 |
| 2026-07-29 | +1.116% | [0.086, 2.189] | 0.036 | ❌ | 0 of 35 |

This is **not** corpus drift: truncating today's store to the published cutoff reproduces
the published verdict exactly. It is new evidence.

**Three findings, in order of how much they should change your reading.**

**(a) The whole flip is one month.** Deleting calendar-month June 2026 (696 fired rows over
20 signal dates, mean −7.22%, the worst of 58 months; second-worst −4.71%) from the current
panel returns `n_obs` 15,527 and +1.4901, CI [0.656, 2.321], p = 0.000 — bit-for-bit the
2026-06-30 row. That is an arithmetic identity, not a model.

**(b) One month was always enough, because the study is thinner than it looks.** With
`block_len = max(5, h) = 21`, a five-year daily panel carries **~52 independent
observations**, so one calendar month is roughly one of them. The pass/fail boundary sat
about **1.2 t-units** away. A study that a single month can reverse was never the "five
years of evidence" its row count suggested.

**(c) The FDR pass was, in part, a Monte Carlo draw.** `scanner/stats.py` computes
`p = 2·min(frac_le0, frac_ge0)`, so at the then-shipped `N_BOOT = 1000` the attainable
p-grid is {0, 0.002, 0.004, …} while the BH rank-1 bar is 0.05/35 = **0.001429** —
strictly *between* two grid points. A lone survivor in this family therefore had to print
exactly `0.0` to pass. Re-running the identical published panel across **12 seeds**, one
(seed 3) prints 0.002 and grades the family **0 of 35**. The shipped seed 0 was a passing
draw. This note's panel is now generated at `n_boot = 10000`, where the grid resolves the
bar; that change is a disclosed methodology fix, not a definition change (`n_boot` is a CLI
and dispatch argument, never a frozen pre-registered parameter — §3.3 of `docs/SOP.md`).

**What is *not* claimed.** An earlier draft of this correction argued the headline had been
"manufactured" by the Feb–May 2026 AI-hardware run-up and handed back in June. That was
tested and **refuted**: the window 2022-07-01 → 2025-07-01, containing no 2026 data at all,
grades **+1.31%, CI [0.48, 2.29], p = 0.000**, and the half-year series sat near +1.4% for
eight consecutive half-years from 2022H2. The rising expanding mean is dilution of a weak
2021H2/2022H1 head, not accretion of a lucky tail.

**The honest characterisation.** A roughly **+1.3 to +1.5pp / 21d gap-momentum drift,
present at about that level since 2022H2, resting on ~52 independent observations** —
*underpowered against its own left tail*. Not a fabrication; not an edge that decayed. The
framing "35 tests, 32 coin flips, exactly one survives" overstated what 52 observations can
support, and the correct response is to say so rather than to condition on June and
re-register a variant. (No conditioning rescues it, and that was checked: the correlation
between a slice's normal edge and its June loss is −0.78, the only slice unhurt in June is
the one whose normal edge is negative, and dropping the top realised-volatility quartile
collapses the pooled figure from +1.12% to −0.06%. The payoff and the drawdown are one
object. Those two figures are the **2026-08-08 panel's**, like the rest of this
reconstruction, and are left as computed: the slice analysis behind them was a one-off
and no artifact carries it, so re-stating it against a later panel would be an invented
number rather than a corrected one.)

**Why this went unnoticed for three weeks.** `.github/workflows/quant-proof.yml` ran
monthly *and* dark (`QUANT_PROOF_COMMIT` defaulted to `false` — compute, print, never
commit), so `data/indicator_performance.json` sat frozen at its 2026-07-09 panel while
every prose surface quoted it as current. The regeneration is now weekly and live, and
`scripts/proof_commit_gate.py` holds any future verdict flip for human review instead of
publishing it silently. (Superseded 2026-10-09: the owner deleted the weekly job, the gate
and the two runners in deletion batch 5. `data/indicator_performance.json` is now static at
its 2026-10-03 panel and the code is in git history at `fdfe3266`.)

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

Stressing a seeded sample of trades to −60% (a proxy for the delisted names the
current-membership universe cannot contain): 2% of trades → 1.368×; **5% → 0.804×**
(a 20% loss). The 5% bracket is now **calibrated, not guessed**: over this exact
backtest window the index removed **26 of 503 names for market-cap weakness = 5.2%**
(`data/sp500_changes.json`; M&A exits counted separately — an acquisition premium is
not a missing loser). The headline depends on names that survived *because* they
survived, and the observed attrition rate says the honest stress is the one that
erases the edge.

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

## 7. Result C — is the best cell real *given the search that found it?*

*(`data/overfit_audit.json`, generated by `scripts/run_overfit_audit.py`, archived 2026-10-09 and
recoverable with `git show fdfe3266:scripts/run_overfit_audit.py`; machinery in
`scanner/stats.py` — `deflated_sharpe`, `pbo_cscv` — unit-tested in `tests/test_stats.py`.)*

> **⚠️ Dated 2026-08-08: this section has NOT been regenerated.** `overfit_audit.json`
> still describes the **2026-07-09** panel — the pre-correction one — so
> `matches_indicator_proof` is no longer true and every figure below prices the cell as it
> read *before* §5.1. It is left as-is deliberately rather than quietly re-run:
> `scripts/run_overfit_audit.py` was on `docs/SOP.md` §3.3's frozen list and hardcoded
> `SURVIVOR = ('gap_up', 21)`, so regenerating it was a decision to take on its own, not a
> side effect of correcting this note. It is now archived in git history at `fdfe3266`
> (2026-10-09, deletion batch 4b; owner), and the result stays frozen as it is. Read what follows as *"what the selection-aware
> audit said about the cell at its strongest"* — which, if anything, makes its verdict
> below more damning, not less.

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
| Survivor Sharpe (per 21d bin, 58 bins) | 0.308 (≈ 1.07 annualized) |
| PSR vs 0 — no selection penalty | **0.992** |
| Noise hurdle SR\* — best of 35 null trials | 0.338 |
| **DSR — PSR vs SR\*** | **0.41** |

Undeflated, the Sharpe still looks strong. Deflated, it sits **below the hurdle**: the
survivor's absolute Sharpe is less than what the luckiest of 35 noise trials would have
shown. The point-in-time correction moved this number *against* the edge (pre-PIT it
read 0.52, sitting exactly at the hurdle) — which is what a survivorship fix should do,
and the honest direction for a document stated against interest. It plainly fails the
canonical DSR ≥ 0.95 bar. (The naive overlap-inflated variant reads 0.68 — still short.
Reported for contrast, not for use.)

### 7.2 Probability of Backtest Overfitting — the relative bar

CSCV: cut the panel into 10 contiguous blocks (21-day embargo at every block tail, so no
forward window straddles a train/test boundary), and for each of the 252 bisections ask
whether the family's **in-sample winner keeps winning out of sample**:

| | value |
|---|---|
| **PBO** — IS winner below median OOS | **0.119** |
| P[OOS Sharpe of IS winner < 0] | 0.119 |
| Survivor = IS winner in | 54% of bisections |
| Survivor's median OOS rank | **top of 35** |

The opposite verdict — and that is the finding, though the PIT correction softened it
too (pre-PIT: PBO 0.048, IS winner in 68% of bisections). Rank stability remains the
strong read: whichever half of history you select on, the same signal's median OOS rank
is still the top of its family.

### 7.3 Purged fold consistency — where the edge lives in time

Five contiguous ~year folds, 21-day purge at each tail, moving-block CI per fold:

| Fold | Window | Edge (95% CI) | Verdict |
|---|---|---|---|
| 1 | 2021-08 → 2022-07 | −0.62% [−2.19, +1.18] | coin_flip |
| 2 | 2022-08 → 2023-07 | +1.90% [+0.45, +3.26] | edge |
| 3 | 2023-08 → 2024-06 | +0.24% [−1.56, +2.15] | coin_flip |
| 4 | 2024-07 → 2025-06 | +1.65% [+0.16, +3.04] | edge |
| 5 | 2025-07 → 2026-06 | +3.40% [+1.23, +5.65] | edge |

4/5 folds positive, 3/5 individually significant — and the one dead fold is the 2021–22
drawdown tape, consistent with §5's regime split. The most recent year is the strongest,
which is encouraging for the live test and exactly the recency a skeptic should discount.

### 7.4 Reading the disagreement

DSR ≈ 0.4 and PBO ≈ 0.12 are not a contradiction; they answer different questions. PBO
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

- Matured calls read **157 hit / 179 miss** on 2026-10-04 (46.7%; Wilson CI
  [41.5%, 52.1%] straddles 50%) — a coin flip, in the wild, exactly as §5–6 predict.
  **Coverage, stated honestly:** `ledger.json` advertises a 30-day retention window but is
  capped at 500 newest entries, and at current firing rates those 500 span a handful of
  days, not 30: **5** on that reading (2026-09-29 → 10-04; 3,177 entries before the cap),
  4 when this was first measured on 2026-08-01. The file now publishes a `coverage` block carrying the
  measured span next to the requested window, so no consumer — this document included —
  can quote the 30 without the 4 beside it. The `/research` page reads the counts live, so
  the site cannot go stale even when this paragraph does.
- **Kill-switch — FIRED. The catch-list tier is retired, not scheduled for retirement.**
  `data/early_entry_performance.json → kill_criterion.status` reads **`killed`**: the 10d
  tripwire tripped at 18 matured entry dates (floor 15) with a control-relative excess of
  **−4.32%**, CI **[−7.43, −0.98]**, p = 0.012. The 21d leg reads **−16.11%**, CI
  **[−21.60, −8.62]**, p = 0.000 on 8 of a required 10 dates — adverse, not yet at floor.
  `scanner/opening.py` refused to publish the tier by its own frozen rule, and on
  2026-10-06 the tier's code was deleted (SOP §3.2.5 as amended). Logging stopped there;
  the record stays whole in the static `data/early_entry_performance.json` and
  `data/logs/early_entry_log.jsonl`, and the verdict in `data/verdicts.json`.
- **The 21-day cohort under §10.2 is NOT the `gap_up` @21d cell — and this matters.**
  `scanner/live_psr.py` draws its cohorts from `opening._excess_blocks`, i.e. the
  **catch-list** construct (gap ≥ 3% *plus* an intraday VWAP-hold, entered ~10:00 ET). The
  §5 cell is a different signal: gap ≥ 2%, entered at the **next day's open**, held
  21 sessions, with no intraday condition. The live instrument therefore grades the tier
  that has just been killed, and **the gap cell, which passed FDR on the panel ending
  2026-07-09 and on none since (§5.1), has no live forward test at all.** That is the gap
  this document should be read as admitting.
- **New registration — the agent desk's take/pass decision (registered 2026-08-05, review
  2026-11-02).** `scanner/performance.py` `_desk_instrument`. Claim: the desk's
  take-versus-pass call separates winners from losers on SPY-excess at **5d**,
  block-bootstrapped over entry *dates* so a day's co-fired names count once. **Stated
  against interest:** the rolling take−pass figures were already visible when this was
  registered (2026-08-04: +0.46 / +0.60 / +0.75 / +0.09 / +0.28 at 1/3/5/10/21d), so the
  verdict counts only decisions made on or after the registration date — the data that made
  the desk look worth grading never grades it. Two design points worth naming: the maturity
  floor (15 entry-dates) binds on the **weaker arm**, because 20 take-days against 3
  pass-days is three days of contrast; and the decision rule is an **LLM**, so the model is
  the definition — every row is stamped with the model that decided it, and if live config
  diverges from the frozen value the trial voids rather than silently grading a moving rule.
  The 21d leg is a **falsification arm**: it can kill the claim, never confirm it. No
  verdict yet — it reads `accumulating` until the floor is met.
- **New registration — the premarket edge list (registered 2026-08-08, review
  2026-11-02).** `scanner/premarket_edge.py`. Claim: overnight **high-confidence bullish**
  ripple calls that clear a **falling-knife veto**, two frozen technical vetoes (RSI ≥ 80;
  MACD histogram negative and falling) and a `risk_on` regime gate beat a same-day
  random-universe control on SPY-excess from the entry day's open to the **next session's
  settled close (1d)**, date-blocked. **Stated against interest, twice.** (1) The stream
  was chosen because it already looked good — the rolling 30d window ending 2026-08-06 read
  70.3% gross / 67.6% net 1d hit on n = 37 — and those figures pool BOTH directions while
  this trial is long-only, a cell never separately measured. (2) The falling-knife leg
  (definition **v2**, same registration day — see also the **v3** note below) was added
  *after* inspecting the bucket it
  acts on: of 31 `contradicted` calls in the 30d window to 2026-08-08 — calls made on names
  whose tape was already moving against the thesis — the 21 bullish graded 38% hit /
  −2.08% mean at 1d, against 7 graded bearish at +0.04%. **Twenty-one observations
  establish nothing.** It is included because it restates a desk rule predating this
  corpus, uses a mechanically-computed field rather than a tuned threshold, and can only
  remove candidates — and it is graded as its own arm so it can be shown to have been
  wrong. **v1 logged zero rows before being superseded**, which is the only condition under
  which replacing a registration discards no evidence; once rows exist, a gate change is a
  new family. Only rows logged on/after registration count. Design
  points worth naming: the gate is **conjunctive with no fitted weights** (the §5.1
  seed-fragility reconstruction is the standing warning against tuning on samples this
  size);
  VWAP / premarket-volume / premarket-gap legs are deliberately absent (IEX premarket tape
  is indicative-only, and the one VWAP-gated morning list this shop ran is the tier killed
  above); entry is the IEX non-auction open, disclosed as indicative; the floor (15
  entry-dates) binds on the weaker arm; an adverse 1d interval at floor **kills the list
  mechanically**; and the frozen block is fingerprinted onto every log row — an edited
  definition orphans its own rows and the compile refuses a verdict (`definition_drift`)
  rather than grading a moving rule. **On pushing, stated precisely** (a daily card was
  added 2026-08-09 at the owner's request and is dark by default): pushing this list
  **cannot** contaminate the trial — grading runs on settled closes against a same-day
  random control, so what was or was not sent never enters a number. What a push costs is
  a different thing, and it is not a statistical cost: acting on the list before the
  review date means the verdict, when it lands, is retrospective on positions already
  taken. The flag (`PREMARKET_EDGE_PUSH`) is therefore fail-closed and owner-only, and the
  card is composed and audited every morning whether or not it is sent.

  **v3 (2026-08-09, still before the first fire) — a projection found the trial could not
  have concluded, and this is the most useful thing it has produced so far.** Measured
  rather than assumed: the frozen overnight window admits a high-confidence bullish ripple
  row on 11 of 31 recent sessions after the knife veto (not the ~20 of 31 the roadmap
  quoted, which counted rows at any hour and so overstated supply 3–5×); ~45% of ripple
  targets are absent from the scan cross-section and so arm `tech_unknown`; the regime gate
  removes another ~20% of days; and the premarket scan actually lands inside the frozen
  08:15–09:30 ET fire window on ~80% of sessions. Compounded, ≈**0.097 pass-arm entry-dates
  per session** — putting the floor of 15 no earlier than **2027-01** and realistically into
  **2027-04**, and meaning the 2026-11-02 appointment will read `accumulating`, not a
  verdict. Worse, v2's **120-day log retention held only ~83 trading days**, capping
  reachable entry-dates at ~8 against a floor of 15: the log would have evicted dates faster
  than they accrued and the instrument would have printed `accumulating` *forever*, unable
  to conclude in either direction.

  **Corrected 2026-08-10, on first-fire morning:** this paragraph said ≈0.14, which was not
  the product of its own legs — they multiply to 0.097, and the four legs that reached the
  code (the landing rate was dropped in transit, while the module comment claimed it was
  included) multiply to 0.122. The trial still concludes: 500-day retention reaches ~34
  entry-dates against a required 30. But the margin is **12%, not 61%**, which makes the
  landing rate load-bearing — and unlike every other leg it is a property of this project's
  cron schedule rather than of the market. Measured from the full commit history, it was
  **0.368** before a mid-July scheduling change and **0.842** after; **break-even is 0.714**.
  The instrument was structurally unconcludable for the two months preceding registration
  and nobody would have known. One correction runs the other way and belongs in the same
  breath: the `~55% present in the cross-section` leg was measured while a bar-fetch defect
  was dropping roughly 40% of the universe from every scan, so that leg is too low and the
  true rate is **above** 0.097 once it is fixed — 0.097 is the honest arithmetic of the legs
  as measured, not a floor on the clock. The lesson beyond the accrual-rate one: a
  projection stated as a single compounded constant cannot be checked against its own legs,
  so the retention test now multiplies the five named legs instead of asserting a bare
  `0.14`.
  Retention is now 500 days, and `PMEDGE_REVIEW_DATE` / `PMEDGE_RETENTION_DAYS` — both of
  which had silently escaped the fingerprint, so editing either latched no drift — are now
  inside it. Nothing about *what* is graded changed: same cohort, same gates, same floor,
  same kill rule. **The transferable lesson: size a trial's log retention against its
  ACCRUAL RATE, never against `review − registered`.** v2's retention test passed at
  120 ≥ 86 while the trial was structurally broken. No verdict yet — `accumulating` until
  floor.
- **New registration — the 盘前精选 selection test (registered 2026-09-23, checkpoint
  2027-07-01).** `scanner/premarket_selection.py`; design in
  `docs/premarket-selection-registration.md`. A SEPARATE family on v3's pass arm, because
  three reviews found v3 is a kill-screen and not a confirmation: at its 15-date floor a true
  +2% edge clears zero only 29-46% of the time, its random control is itself -0.77% vs SPY
  (a size mix), and 19 of 22 pass rows were AI/semis names. Claim: the published picks beat
  their **own price-matched peers** (the 10 most-correlated public names over the prior 60
  sessions, fixed at fire from prior-close bars) from the entry day's SIP open to the next
  session's settled close, **one observation per entry-date**. Tested with an exact one-sided
  sign-flip at 30/45/60 dates with O'Brien-Fleming boundaries; the verdict **locks** the first
  time a boundary is crossed. **Null-calibrated through the real test**: 2.45% realised
  false-confirm rate (2.65% under t3 tails) against 2.5% nominal, versus 9.3% for the same
  test re-read at every date the way v3 reads its verdict. Power at the measured per-date SD
  of 5: 85% for +2%; it cannot confirm +1%, said up front. **Stated against interest:** the
  design was drawn after looking at v3's first 12 entry-dates (+1.61% vs price-matched peers,
  SD 5.2 on 10 dates), so none of them count and nothing was fitted to them. First look
  projected 2027-01..02, final 2027-05..06 — projections, not appointments.
- **New registration — the gainer-board watch (registered 2026-09-24, review 2026-12-15).** *(Abandoned 2026-10-06; `scanner/board_watch.py` deleted 2026-10-09, in git history at 9886e2db.)*
  `scanner/board_watch.py`. Log-only: no card, no push, nothing served. Claim under test: a name
  on Alpaca's market-wide top-gainers board at ≥ +100% on the day, priced ≥ $1 and not a
  warrant/right/unit, bought at that session's **settled close**, returns ≠ 0 by the **next
  session's settled close** (SIP tape), one mean per entry-date, sign-flip test, floor 20
  entry-dates; 5d is descriptive. **Stated against interest:** it was prompted by one
  anecdote pointing one way (JAGX, +1,296% on 2026-09-22, then roughly −70% from that close
  by midday the next session), so the expected reading is `fades` and a `fades` verdict
  will confirm what motivated it — which is why 1d was fixed before any row exists and
  why registration is the day *after* the board was first read by hand. Two things this
  record cannot see: names halted or delisted before their horizon (counted as
  `unresolved`, never dropped), and names that spike and fade between two scan slots.
  The universe is untouched — `config.MIN_PRICE` feeds frozen instruments.
- **Allocation rule (Popper, in code):** deploy the ≤20% satellite sleeve *only if*
  (a) the per-trade 21d net verdict is not adverse **and** (b) the live ledger's ~Sept-2026
  fired-cohort excess verdict is `edge` with a CI overlapping the backtest. Either fails →
  do not allocate (`portfolio_backtest.json → decision_rules`). Given the bullet above,
  condition (b) is keyed to the killed cohort and its verdict is already `adverse` — so
  **the rule already reads "do not allocate", and September cannot change that.** The rule
  fired correctly; it was simply pointed at the wrong cohort when it was written.

A prediction with a date and a pre-registered pass/fail bar is a falsifiable claim. That is
the difference between a track record and a horoscope.

## 9. Threats to validity (what would make me wrong)

- **Survivorship** — the backfill half is corrected (point-in-time membership, §3);
  the missing-delisted-losers half is calibrated-bounded (§6.3), not eliminated. A full
  fix needs a survivorship-free data source (delisted bars), which this free stack
  does not have.
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
   *Correction (2026-07-31): the sentence above — "that number is the deployable one" —
   is **wrong as written**, and the error is mine, not the instrument's. `live_psr.py`
   draws its cohorts from `opening._excess_blocks`, so the frozen analysis grades the
   **catch-list** tier (gap ≥ 3% + intraday VWAP-hold), which the kill criterion has now
   retired — not the `gap_up` @21d cell of §5 (gap ≥ 2%, next-day open, 21-session hold).
   The pre-registration is intact and did exactly its job; it was simply aimed at the wrong
   cohort. September will therefore deliver an honest verdict on a dead tier, and
   **nothing at all on the gap cell.** That used to be the top item on this list. After
   §5.1 it is not: there is no surviving edge to confirm, and the instrument as drafted is
   underpowered by roughly a factor of eight — its 10-matured-entry-date floor is about
   half of **one** independent block, giving a minimum detectable effect near 10.7pp
   against a target of ~1.2pp. Entry-dates are the wrong unit; independent blocks are the
   binding one. Any future registration must state its floor in blocks (≥28 for 80% power
   at α=0.05, ≈2.3 years) and must not describe itself as confirming a survivor.*
3. **A bear regime.** No statistic here can synthesize one. Fold 1 (2021–22) is the only
   adverse-tape evidence and it reads flat; the coded kill-criterion floors (§8) are the
   standing insurance until the market supplies the missing experiment.

## 11. Reproducibility & provenance

- **Frozen universe** `data/proof_universe.json` (516 symbols, refreshed quarterly by hand
  with a reviewed diff, so membership churn can't flip a verdict between runs).
- **Pre-registration hash** `3383e1d15a8f…`; bootstrap **seed = 0** throughout →
  byte-reproducible CIs for a given set of bars. The proof's `universe.symbols_sha256`
  is a per-run fingerprint of which frozen symbols had bars, not a pin: it moves when
  that set does (`38024bfe…` on the 2026-10-03 panel).
- Rerun: none. `scripts/run_indicator_proof.py` and `scripts/run_backtest.py` were archived on
  2026-10-09 (deletion batch 5), and `scripts/run_overfit_audit.py` with the
  `scanner/portfolio_backtest.py` machinery the same day (deletion batch 4b); all are
  recoverable with `git show fdfe3266:<path>`. The universe is regenerated with
  `scripts/gen_proof_universe.py`. The axis-proof and portfolio-backtest
  runners were deleted (2026-10-01 and 2026-10-06, sustainability plan W2); their
  committed results stay and the code is in git history.
- The §7 audit asserts that it ran on the byte-identical panel as the proof of its day
  and records the `symbols_sha256` match, with bin span, embargo, CSCV blocks, and seed
  all pinned in its `config` block. Its day was 2026-07-21, so it does not describe
  the current proof (§7's banner).
- A **commit gate** (`scripts/proof_commit_gate.py`, `tests/test_proof_commit_gate.py`;
  deleted 2026-10-09 with the weekly job, code at `fdfe3266`) refused to publish a proof if
  the universe collapsed or a `verdict_final` flipped without an explicit override, and
  since 2026-10-04 the proof tested its own commit before pushing
  (`scripts/check_committed_tree.py`), parking a red on branch `proof-held-<run id>`.
  Neither stopped the numbers moving week to week. On 2026-10-03 a regeneration moved
  `gap_up` @10d above zero under a sentence here saying it no longer cleared, and
  nothing noticed until main went red 13 hours later (`docs/incidents.md`, 2026-10-04). That is
  why TL;DR item 1 no longer names the cells that clear zero.

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

## 13. The file drawer, opened — hypotheses tested and rejected

§9 prices the multiplicity of the *registered* family. This section is the other half of
the same discipline: external hypotheses tested here and rejected, written down so the
rejection costs something to reverse and so the next session does not re-derive it. A note
that publishes only what survived is a selection effect wearing a lab coat.

### 13.1 Session liquidity sweeps + fair value gaps ("ICT" / Smart Money Concepts) — rejected 2026-09-11

**The claim.** Mark the Asia and London session high/low; wait for one to be swept; drop to
the 1-minute chart; take a fair-value-gap reversal back through the range; target 1:2 RR.

**Why it was tested rather than dismissed.** Strip the vocabulary and the parent idea is a
failed breakout of a prior reference range — the shape Raschke & Connors called "turtle
soup" (*Street Smarts*, 1995) and the ancestor of the opening-range-breakout-failure
literature. That is a respectable lineage, and this repo has no reversal instrument of any
kind: §5–§8 are all continuation tiers, so "we already cover it" was not available as an
answer.

**Measured — the sessions the strategy names.** Probed live against the account's own
Alpaca credentials. US equities do trade 20:00–04:00 ET, on the Blue Ocean ATS
(`DataFeed.BOATS`) — but the tape is degenerate: SPY prints a median 110–338 shares/minute
and **56–74% of overnight 1-minute bars have zero range** (high = low). A fair value gap is
defined by displacement between three consecutive candles; on that tape the construct is
undefined more often than not. Pre-market (04:00–09:30 ET, the honest "London killzone"
stand-in) runs 783–2,915 shares/minute against 37,000–158,000 in regular hours. Separately,
the free tier hard-refuses SIP data inside 15 minutes (`subscription does not permit
querying recent SIP data`), so a 1-minute signal is unexecutable live even where it is
researchable historically.

**Measured — does the shape pay, anywhere?** Four backtests, three reference-range
definitions, each against a control. Breakeven at 1:2 RR is 33.33%.

| reference range | n | span | win rate | expectancy | control |
| --- | --- | --- | --- | --- | --- |
| pre-market (04:00–09:30 ET) | 807 | 128 sessions, 8 names | 31.4% | −0.005R | paired edge −0.0044R, P(edge>0) = 0.470 |
| prior RTH session | 712 | 126 days, 10 names | 26.0% | −0.084R | random-direction 27.8% (sd 1.2%, 200 seeds) |
| prior 5/10/20 days, daily bars | 632,250 bars | 511 names, 1,098 dates | 31.2–31.8% | +0.014 to +0.023R | see below |

All eleven intraday variants are negative; none was withheld. **Making the sweep more
textbook makes it monotonically worse**: requiring 5bps of penetration beyond the level
gives 24.7% and −0.107R, 15bps gives 22.4% and −0.127R (t = −2.38). The only significant
result in the study is the claimed mechanism running backwards — a deeper, cleaner sweep is
*continuation*, not exhaustion.

**The 轴下 conditional — the one cell with a prior in this repo rather than in ICT.** The
house technique (weekly-DOWN + a daily trigger, `david-axis-golden-cross-technique`) applied
to a range instead of a MACD cross: fade only sweeps that oppose the 30-week trend. Paired
edge against its own mirror at K = 5/10/20 days: **−0.0036R, +0.0180R, −0.0121R**, with
P(edge>0) = 0.448 / 0.713 / 0.380. The sign flips with the lookback, every CI straddles
zero, and the conditional cell is indistinguishable from the cell it was meant to beat
(contrast: −0.0067R, +0.0066R, −0.0047R).

**What the controls actually revealed, and it is worth keeping.** At K = 10, fading the
sweep returns +0.0168R and *taking* it returns +0.0016R — while a random entry with
identical stop/target geometry on the same price paths returns **−0.1815R**. Both directions
beat random by a wide margin; neither beats the other. The sweep bar is a good place to put
a **stop** — it sits at a real local extreme, so it is hit less often — and carries no
information about **direction**. That single fact explains the whole family's apparent edge
in the vendor literature, where the control is invariably absent.

**Disclosed weaknesses.** Effective n is sessions, not trades (128–238 independent days
intraday), and the day-clustered CIs are wide enough to hide a true effect under ~0.1R; this
is a rejection on a dead control, not a tight zero. The daily panel is `data/bars/`, frozen
and survivorship-biased upward (§3, §9) — which biases long outcomes *up*, making the null
conservative rather than flattering.

**What would reverse it.** Not cost, not data, not venue: a fade that beats its own
randomized directional control, day-clustered, on a held-out split. Nothing here suggests
where to look for one.

**What was kept.** Only the level. The search established that a swing extreme read off
HIGHS and LOWS is a materially better stop anchor than one read off closes, which is how
`scanner/levels.py` had been placing every published stop — fixed in the same PR that
recorded this null.

---

---

*Generated from committed artifacts under `data/`. If a claim here disagrees with the JSON,
the JSON wins — regenerate and correct this note.*
